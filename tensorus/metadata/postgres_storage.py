import psycopg2
import psycopg2.pool
import psycopg2.extras  # For dict cursor
import json
from typing import List, Optional, Dict, Any
from uuid import UUID
import logging

from .storage_abc import MetadataStorage
from .schemas import (
    TensorDescriptor, SemanticMetadata, DataType, StorageFormat, AccessControl, CompressionInfo,
    LineageMetadata, ComputationalMetadata, QualityMetadata,
    RelationalMetadata, UsageMetadata # Import other extended types for method signatures
)
import copy # Ensure copy is imported, as it was added to InMemoryStorage and might be useful here too.

# Configure module level logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

__all__ = ["PostgresMetadataStorage"]

# --- DDL Comments ---
#
# CREATE TYPE data_type_enum AS ENUM (
#     'float32', 'float64', 'float16', 'int32', 'int64', 'int16', 'int8', 'uint8',
#     'boolean', 'string', 'complex64', 'complex128', 'other'
# );
#
# CREATE TYPE storage_format_enum AS ENUM (
#     'raw', 'numpy_npz', 'hdf5', 'compressed_zlib', 'compressed_gzip', 'custom'
# );
#
# CREATE TABLE IF NOT EXISTS tensor_descriptors (
#     tensor_id UUID PRIMARY KEY,
#     dimensionality INTEGER NOT NULL,
#     shape INTEGER[] NOT NULL,
#     data_type TEXT NOT NULL, -- Could use data_type_enum
#     storage_format TEXT NOT NULL, -- Could use storage_format_enum
#     creation_timestamp TIMESTAMPTZ NOT NULL,
#     last_modified_timestamp TIMESTAMPTZ NOT NULL,
#     owner TEXT,
#     access_control JSONB,
#     byte_size BIGINT,
#     checksum TEXT,
#     compression_info JSONB,
#     tags TEXT[],
#     metadata JSONB,
#     CONSTRAINT shape_dimensionality_check CHECK (array_length(shape, 1) = dimensionality OR dimensionality = 0 AND shape IS NULL OR array_length(shape,1) = 0)
# );
# CREATE INDEX IF NOT EXISTS idx_td_owner ON tensor_descriptors(owner);
# CREATE INDEX IF NOT EXISTS idx_td_data_type ON tensor_descriptors(data_type);
# CREATE INDEX IF NOT EXISTS idx_td_tags ON tensor_descriptors USING GIN(tags);
#
# -- Table for SemanticMetadata (example, one-to-many with TensorDescriptor)
# CREATE TABLE IF NOT EXISTS semantic_metadata_entries (
#     id SERIAL PRIMARY KEY, -- Or UUID primary key if preferred
#     tensor_id UUID NOT NULL REFERENCES tensor_descriptors(tensor_id) ON DELETE CASCADE,
#     name TEXT NOT NULL,
#     description TEXT,
#     -- other fields from SemanticMetadata schema ...
#     UNIQUE (tensor_id, name) -- Ensure name is unique per tensor_id
# );
#
# -- Generic table structure for 1-to-1 extended metadata (Lineage, Computational, etc.)
# -- Replace <metadata_name> with lineage, computational, quality, relational, usage
# CREATE TABLE IF NOT EXISTS <metadata_name>_metadata (
#    tensor_id UUID PRIMARY KEY REFERENCES tensor_descriptors(tensor_id) ON DELETE CASCADE,
#    data JSONB NOT NULL -- Store the entire Pydantic model as JSONB
# );
#

class PostgresMetadataStorage(MetadataStorage):
    def __init__(self, dsn: Optional[str] = None, min_conn: int = 1, max_conn: int = 5, **kwargs):
        self.dsn = dsn
        self.pool = None
        if dsn: # Allow DSN or individual params
             self.pool = psycopg2.pool.SimpleConnectionPool(min_conn, max_conn, dsn=dsn)
        elif kwargs.get('database') and kwargs.get('user'):
            self.pool = psycopg2.pool.SimpleConnectionPool(min_conn, max_conn, **kwargs)
        else:
            raise ValueError("PostgreSQL connection parameters (DSN or host/db/user etc.) not provided.")

    def _execute_query(self, query: str, params: tuple = None, fetch: str = None):
        """Helper to execute queries with connection pooling."""
        if not self.pool:
            raise ConnectionError("Connection pool not initialized.")
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(query, params)
                conn.commit()
                if fetch == "one":
                    return cur.fetchone()
                if fetch == "all":
                    return cur.fetchall()
                # For INSERT/UPDATE/DELETE, rowcount might be useful
                return cur.rowcount
        except Exception as e:
            if conn: conn.rollback()
            # Log error e
            raise # Re-raise after logging or wrap in custom exception
        finally:
            if conn and self.pool:
                self.pool.putconn(conn)

    # --- TensorDescriptor Methods ---
    def add_tensor_descriptor(self, descriptor: TensorDescriptor) -> None:
        query = """
            INSERT INTO tensor_descriptors (
                tensor_id, dimensionality, shape, data_type, storage_format,
                creation_timestamp, last_modified_timestamp, owner, access_control,
                byte_size, checksum, compression_info, tags, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (tensor_id) DO UPDATE SET
                dimensionality = EXCLUDED.dimensionality,
                shape = EXCLUDED.shape,
                data_type = EXCLUDED.data_type,
                storage_format = EXCLUDED.storage_format,
                last_modified_timestamp = EXCLUDED.last_modified_timestamp,
                owner = EXCLUDED.owner,
                access_control = EXCLUDED.access_control,
                byte_size = EXCLUDED.byte_size,
                checksum = EXCLUDED.checksum,
                compression_info = EXCLUDED.compression_info,
                tags = EXCLUDED.tags,
                metadata = EXCLUDED.metadata;
        """
        params = (
            descriptor.tensor_id,
            descriptor.dimensionality,
            descriptor.shape,
            descriptor.data_type.value,
            descriptor.storage_format.value,
            descriptor.creation_timestamp,
            descriptor.last_modified_timestamp,
            descriptor.owner,
            descriptor.access_control.model_dump_json() if descriptor.access_control else None, # Pydantic v2
            # json.dumps(descriptor.access_control.dict()) if descriptor.access_control else None, # Pydantic v1
            descriptor.byte_size,
            descriptor.checksum,
            descriptor.compression_info.model_dump_json() if descriptor.compression_info else None, # Pydantic v2
            # json.dumps(descriptor.compression_info.dict()) if descriptor.compression_info else None, # Pydantic v1
            descriptor.tags,
            json.dumps(descriptor.metadata) if descriptor.metadata else None,
        )
        self._execute_query(query, params)

    def get_tensor_descriptor(self, tensor_id: UUID) -> Optional[TensorDescriptor]:
        query = "SELECT * FROM tensor_descriptors WHERE tensor_id = %s;"
        row = self._execute_query(query, (tensor_id,), fetch="one")
        if row:
            # Pydantic models expect enums, not their string values directly from DB for some fields
            # Need to handle this transformation carefully.
            # Also, JSONB fields need to be parsed back.
            data = dict(row)
            data['data_type'] = DataType(data['data_type'])
            data['storage_format'] = StorageFormat(data['storage_format'])
            if data.get('access_control'): # JSONB field
                 data['access_control'] = AccessControl(**data['access_control'])
            if data.get('compression_info'): # JSONB field
                 data['compression_info'] = CompressionInfo(**data['compression_info'])
            # metadata is also JSONB but can be any dict, so direct assignment is fine
            return TensorDescriptor(**data)
        return None

    def delete_tensor_descriptor(self, tensor_id: UUID) -> bool:
        # ON DELETE CASCADE should handle related metadata in other tables
        query = "DELETE FROM tensor_descriptors WHERE tensor_id = %s;"
        rowcount = self._execute_query(query, (tensor_id,))
        return rowcount > 0

    # --- SemanticMetadata Methods (Example for one-to-many) ---
    # Assuming semantic_metadata_entries table as defined in DDL comments
    def add_semantic_metadata(self, metadata: SemanticMetadata) -> None:
        # Check if TD exists
        if not self.get_tensor_descriptor(metadata.tensor_id):
             raise ValueError(f"TensorDescriptor with ID {metadata.tensor_id} not found.")

        # Upsert based on (tensor_id, name)
        query = """
            INSERT INTO semantic_metadata_entries (tensor_id, name, description)
            VALUES (%s, %s, %s)
            ON CONFLICT (tensor_id, name) DO UPDATE SET
                description = EXCLUDED.description;
        """
        # Add other fields from SemanticMetadata schema to query and params as needed
        params = (metadata.tensor_id, metadata.name, metadata.description)
        self._execute_query(query, params)

    def get_semantic_metadata(self, tensor_id: UUID) -> List[SemanticMetadata]:
        query = "SELECT tensor_id, name, description FROM semantic_metadata_entries WHERE tensor_id = %s;"
        rows = self._execute_query(query, (tensor_id,), fetch="all")
        return [SemanticMetadata(**dict(row)) for row in rows]


    # --- Placeholder for other methods ---
    def update_tensor_descriptor(self, tensor_id: UUID, **kwargs) -> Optional[TensorDescriptor]:
        # Complex: involves fetching, updating fields, then writing back.
        # Need to handle partial updates carefully.
        # Example: SELECT ... FOR UPDATE, then construct UPDATE statement.
        current_td = self.get_tensor_descriptor(tensor_id)
        if not current_td:
            return None

        update_data = current_td.model_dump() # Pydantic v2
        # update_data = current_td.dict() # Pydantic v1

        for key, value in kwargs.items():
            if key in update_data: # only update valid fields
                update_data[key] = value

        # Re-validate before saving
        try:
            updated_td = TensorDescriptor(**update_data)
            updated_td.last_modified_timestamp = datetime.utcnow() # Explicitly update timestamp
            self.add_tensor_descriptor(updated_td) # Use add which does upsert
            return updated_td
        except Exception: # Should be pydantic.ValidationError
            # Log error or raise custom validation error
            raise

    def list_tensor_descriptors(
        self,
        owner: Optional[str] = None,
        data_type: Optional[DataType] = None,
        tags_contain: Optional[List[str]] = None,
        lineage_version: Optional[str] = None,
        # Add other filter params as needed, matching the API layer
        # For brevity, only a few are shown here.
    ) -> List[TensorDescriptor]:
        base_query = "SELECT DISTINCT td.* FROM tensor_descriptors td"
        joins: List[str] = []
        conditions: List[str] = []
        params: Dict[str, Any] = {} # Using dict for named parameters with %(name)s style

        if owner:
            conditions.append("td.owner = %(owner)s")
            params["owner"] = owner
        if data_type:
            conditions.append("td.data_type = %(data_type)s")
            params["data_type"] = data_type.value
        if tags_contain is not None and len(tags_contain) > 0:
            conditions.append("td.tags @> %(tags_contain)s") # Array contains operator
            params["tags_contain"] = tags_contain

        # Example: Filtering by lineage.version
        if lineage_version:
            if "lm" not in [j.split()[1] for j in joins if len(j.split()) > 1]: # Avoid duplicate joins
                joins.append("LEFT JOIN lineage_metadata lm ON td.tensor_id = lm.tensor_id")
            conditions.append("lm.data->>'version' = %(lineage_version)s")
            params["lineage_version"] = lineage_version

        # Construct final query
        if joins:
            base_query += " " + " ".join(joins)
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        base_query += ";"

        rows = self._execute_query(base_query, params, fetch="all") # type: ignore # psycopg2 params can be dict
        results = []
        for row in rows:
            data = dict(row)
            data['data_type'] = DataType(data['data_type'])
            data['storage_format'] = StorageFormat(data['storage_format'])
            if data.get('access_control'): data['access_control'] = AccessControl(**data['access_control'])
            if data.get('compression_info'): data['compression_info'] = CompressionInfo(**data['compression_info'])
            results.append(TensorDescriptor(**data))
        return results

    def get_semantic_metadata_by_name(self, tensor_id: UUID, name: str) -> Optional[SemanticMetadata]:
        query = "SELECT tensor_id, name, description FROM semantic_metadata_entries WHERE tensor_id = %s AND name = %s;"
        row = self._execute_query(query, (tensor_id, name), fetch="one")
        return SemanticMetadata(**dict(row)) if row else None

    def update_semantic_metadata(self, tensor_id: UUID, name: str, new_description: Optional[str] = None, new_name: Optional[str] = None) -> Optional[SemanticMetadata]:
        current_sm = self.get_semantic_metadata_by_name(tensor_id, name)
        if not current_sm:
            return None

        description_to_set = new_description if new_description is not None else current_sm.description
        name_to_set = new_name if new_name is not None else current_sm.name

        # If name is being changed, check for conflict first
        if new_name and new_name != name:
            existing_with_new_name = self.get_semantic_metadata_by_name(tensor_id, new_name)
            if existing_with_new_name:
                raise ValueError(f"SemanticMetadata with name '{new_name}' already exists for tensor {tensor_id}.")

        query = "UPDATE semantic_metadata_entries SET name = %s, description = %s WHERE tensor_id = %s AND name = %s;"
        self._execute_query(query, (name_to_set, description_to_set, tensor_id, name))

        # Return the updated object by fetching it again
        return self.get_semantic_metadata_by_name(tensor_id, name_to_set)


    def delete_semantic_metadata(self, tensor_id: UUID, name: str) -> bool:
        query = "DELETE FROM semantic_metadata_entries WHERE tensor_id = %s AND name = %s;"
        rowcount = self._execute_query(query, (tensor_id, name)) # type: ignore # rowcount is int
        return rowcount > 0

    # --- Implementations for other extended metadata (using JSONB 'data' column pattern) ---
    def _add_jsonb_metadata(self, table_name: str, metadata_obj: Any) -> None:
        # Check if the parent TensorDescriptor exists
        # This check is good practice, though add_tensor_descriptor also checks.
        # Redundant if _add_jsonb_metadata is only called internally after a TD check.
        # For direct calls or future refactoring, it's safer.
        parent_td = self.get_tensor_descriptor(metadata_obj.tensor_id)
        if not parent_td:
             raise ValueError(f"TensorDescriptor with ID {metadata_obj.tensor_id} not found. Cannot add {metadata_obj.__class__.__name__}.")

        query = f"""
            INSERT INTO {table_name} (tensor_id, data) VALUES (%(tensor_id)s, %(data)s)
            ON CONFLICT (tensor_id) DO UPDATE SET data = EXCLUDED.data;
        """
        params = {
            "tensor_id": metadata_obj.tensor_id,
            "data": metadata_obj.model_dump_json() # Pydantic v2
            # "data": metadata_obj.json() # Pydantic v1
        }
        self._execute_query(query, params)


    def _get_jsonb_metadata(self, table_name: str, tensor_id: UUID, model_class: type) -> Optional[Any]:
        query = f"SELECT data FROM {table_name} WHERE tensor_id = %(tensor_id)s;"
        params = {"tensor_id": tensor_id}
        row = self._execute_query(query, params, fetch="one")
        if row and row['data']:
            return model_class.model_validate_json(row['data']) # Pydantic v2
            # return model_class.parse_raw(row['data']) # Pydantic v1
        return None


    def _update_jsonb_metadata(self, table_name: str, tensor_id: UUID, model_class: type, **kwargs) -> Optional[Any]:
        current_obj = self._get_jsonb_metadata(table_name, tensor_id, model_class)
        if not current_obj:
            return None

        current_data = current_obj.model_dump() # Pydantic v2
        # current_data = current_obj.dict() # Pydantic v1

        # Perform a deep update for nested dictionaries if necessary
        # For simple top-level field updates, direct assignment is fine.
        # This example does a simple top-level merge.
        for key, value in kwargs.items():
            current_data[key] = value
        # updated_data = {**current_data, **kwargs} # This does a shallow merge

        try:
            new_obj = model_class.model_validate(current_data) # Pydantic v2
            # new_obj = model_class(**current_data) # Pydantic v1
            self._add_jsonb_metadata(table_name, new_obj) # Use add for upsert
            return new_obj
        except Exception as e: # Should be Pydantic ValidationError
            # Log e
            raise ValueError(f"Update for {model_class.__name__} failed validation: {e}")


    def _delete_jsonb_metadata(self, table_name: str, tensor_id: UUID) -> bool:
        query = f"DELETE FROM {table_name} WHERE tensor_id = %(tensor_id)s;"
        params = {"tensor_id": tensor_id}
        rowcount = self._execute_query(query, params) # type: ignore # rowcount is int
        return rowcount > 0

    def add_lineage_metadata(self, m: LineageMetadata): self._add_jsonb_metadata("lineage_metadata", m)
    def get_lineage_metadata(self, tid: UUID): return self._get_jsonb_metadata("lineage_metadata", tid, LineageMetadata)
    def update_lineage_metadata(self, tid: UUID, **kw): return self._update_jsonb_metadata("lineage_metadata", tid, LineageMetadata, **kw)
    def delete_lineage_metadata(self, tid: UUID): return self._delete_jsonb_metadata("lineage_metadata", tid)

    def add_computational_metadata(self, m: ComputationalMetadata): self._add_jsonb_metadata("computational_metadata", m)
    def get_computational_metadata(self, tid: UUID): return self._get_jsonb_metadata("computational_metadata", tid, ComputationalMetadata)
    def update_computational_metadata(self, tid: UUID, **kw): return self._update_jsonb_metadata("computational_metadata", tid, ComputationalMetadata, **kw)
    def delete_computational_metadata(self, tid: UUID): return self._delete_jsonb_metadata("computational_metadata", tid)

    def add_quality_metadata(self, m: QualityMetadata): self._add_jsonb_metadata("quality_metadata", m)
    def get_quality_metadata(self, tid: UUID): return self._get_jsonb_metadata("quality_metadata", tid, QualityMetadata)
    def update_quality_metadata(self, tid: UUID, **kw): return self._update_jsonb_metadata("quality_metadata", tid, QualityMetadata, **kw)
    def delete_quality_metadata(self, tid: UUID): return self._delete_jsonb_metadata("quality_metadata", tid)

    def add_relational_metadata(self, m: RelationalMetadata): self._add_jsonb_metadata("relational_metadata", m)
    def get_relational_metadata(self, tid: UUID): return self._get_jsonb_metadata("relational_metadata", tid, RelationalMetadata)
    def update_relational_metadata(self, tid: UUID, **kw): return self._update_jsonb_metadata("relational_metadata", tid, RelationalMetadata, **kw)
    def delete_relational_metadata(self, tid: UUID): return self._delete_jsonb_metadata("relational_metadata", tid)

    def add_usage_metadata(self, m: UsageMetadata): self._add_jsonb_metadata("usage_metadata", m)
    def get_usage_metadata(self, tid: UUID): return self._get_jsonb_metadata("usage_metadata", tid, UsageMetadata)
    def update_usage_metadata(self, tid: UUID, **kw): return self._update_jsonb_metadata("usage_metadata", tid, UsageMetadata, **kw)
    def delete_usage_metadata(self, tid: UUID): return self._delete_jsonb_metadata("usage_metadata", tid)


    def get_parent_tensor_ids(self, tensor_id: UUID) -> List[UUID]:
        # Assumes lineage_metadata table with 'data' JSONB column
        # and data column has 'parent_tensors' list like: [{"tensor_id": "uuid", ...}]
        query = """
            SELECT parent.value->>'tensor_id' AS parent_id
            FROM lineage_metadata lm, jsonb_array_elements(lm.data->'parent_tensors') AS parent
            WHERE lm.tensor_id = %(tensor_id)s;
        """
        params = {"tensor_id": tensor_id}
        rows = self._execute_query(query, params, fetch="all")
        return [UUID(row['parent_id']) for row in rows if row['parent_id']]

    def get_child_tensor_ids(self, tensor_id: UUID) -> List[UUID]:
        # Searches all lineage_metadata entries
        query = """
            SELECT lm.tensor_id
            FROM lineage_metadata lm, jsonb_array_elements(lm.data->'parent_tensors') AS parent
            WHERE parent.value->>'tensor_id' = %(target_parent_id)s;
        """
        params = {"target_parent_id": str(tensor_id)} # Ensure UUID is string for JSONB comparison
        rows = self._execute_query(query, params, fetch="all")
        return [row['tensor_id'] for row in rows]

    def search_tensor_descriptors(self, text_query: str, fields: List[str]) -> List[TensorDescriptor]:
        if not fields:
            return []

        # Base query
        select_clause = "SELECT DISTINCT td.* FROM tensor_descriptors td"
        joins: List[str] = []
        where_conditions: List[str] = []
        query_params: Dict[str, Any] = {"text_query": f"%{text_query}%"} # For ILIKE

        # Helper to add joins only once
        joined_aliases = {"td"}

        for field_path in fields:
            parts = field_path.split('.', 1)
            field_prefix = parts[0]
            field_suffix = parts[1] if len(parts) > 1 else None

            # Default to tensor_descriptors table if no prefix or prefix is 'tensor_descriptor'
            table_alias = "td"
            column_or_json_path = field_prefix # If no suffix, field_prefix is the column name

            if field_prefix == "semantic":
                if "sm" not in joined_aliases:
                    joins.append("LEFT JOIN semantic_metadata_entries sm ON td.tensor_id = sm.tensor_id")
                    joined_aliases.add("sm")
                table_alias = "sm"
                column_or_json_path = field_suffix if field_suffix else "name" # Default search semantic name
            elif field_prefix in ["lineage", "computational", "quality", "relational", "usage"]:
                # Assumes extended metadata tables are named e.g. "lineage_metadata"
                # and have a JSONB 'data' column.
                ext_table_name = f"{field_prefix}_metadata"
                ext_alias = f"{field_prefix[0]}m" # e.g., lm, cm
                if ext_alias not in joined_aliases:
                    joins.append(f"LEFT JOIN {ext_table_name} {ext_alias} ON td.tensor_id = {ext_alias}.tensor_id")
                    joined_aliases.add(ext_alias)
                table_alias = ext_alias
                # Construct JSON path, e.g., data->'source'->>'identifier'
                # This requires parsing field_suffix if it's nested.
                # For simplicity, assume field_suffix is a top-level key in the JSONB 'data' field for now.
                # e.g. if field_suffix = "source.identifier", this becomes data->'source'->>'identifier'
                if field_suffix:
                    json_path_parts = field_suffix.split('.')
                    json_op_path = "->".join([f"'{p}'" for p in json_path_parts[:-1]])
                    if json_op_path:
                         column_or_json_path = f"{table_alias}.data->{json_op_path}->>'{json_path_parts[-1]}'"
                    else: # Top level key in JSON
                         column_or_json_path = f"{table_alias}.data->>'{json_path_parts[-1]}'"
                else: # Searching the whole JSON blob (less efficient, but possible)
                    column_or_json_path = f"{table_alias}.data::text" # Cast JSONB to text to search
            else: # Assumed to be a direct column on tensor_descriptors
                 column_or_json_path = field_path # e.g. "owner" or "tags"

            # Add condition for this field
            # For array fields like 'tags', use a different operator or unnest
            if table_alias == "td" and column_or_json_path == "tags":
                 # This is a basic way; for tags, often unnesting or specific array ops are better
                where_conditions.append(f"array_to_string({table_alias}.{column_or_json_path}, ' ') ILIKE %(text_query)s")
            else:
                where_conditions.append(f"{column_or_json_path} ILIKE %(text_query)s")

        if not where_conditions:
            return [] # No valid fields to search

        query = f"{select_clause} {' '.join(joins)} WHERE {' OR '.join(where_conditions)};"

        rows = self._execute_query(query, query_params, fetch="all")
        results = []
        for row in rows:
            data = dict(row)
            data['data_type'] = DataType(data['data_type'])
            data['storage_format'] = StorageFormat(data['storage_format'])
            if data.get('access_control'): data['access_control'] = AccessControl(**data['access_control'])
            if data.get('compression_info'): data['compression_info'] = CompressionInfo(**data['compression_info'])
            results.append(TensorDescriptor(**data))
        return results


    def aggregate_tensor_descriptors(self, group_by_field: str, agg_function: str, agg_field: Optional[str]=None) -> Dict[Any, Any]:
        # Simplified initial implementation: Group by direct TD fields, count only
        if not group_by_field or not hasattr(TensorDescriptor, group_by_field.split('.')[0]): # Basic check
             raise ValueError(f"Invalid group_by_field: {group_by_field}")

        # Only supporting count and direct TD fields for now
        # A full implementation needs dynamic JOINs and path resolution similar to search.
        if agg_function.lower() != "count":
            raise NotImplementedError(f"Aggregation function '{agg_function}' not yet fully implemented for all fields.")

        # Assuming group_by_field is a direct column on tensor_descriptors table for this simplified version
        # e.g. "data_type", "owner"
        sql_group_by_field = group_by_field # Sanitize this if it comes from user input directly!

        query = f"SELECT {sql_group_by_field}, COUNT(*) as count FROM tensor_descriptors GROUP BY {sql_group_by_field};"

        rows = self._execute_query(query, {}, fetch="all") # type: ignore # Pass empty dict for params if none
        return {row[sql_group_by_field]: row['count'] for row in rows}

    # --- Export/Import Methods ---
    def get_export_data(self, tensor_ids: Optional[List[UUID]] = None) -> 'TensorusExportData': # type: ignore
        # This would require complex JOINs or multiple queries per tensor_id
        # to gather all related metadata from different tables / JSONB columns.
        # For a full implementation:
        # 1. Determine list of tensor_ids to export (all if None).
        # 2. For each tensor_id:
        #    a. Fetch TensorDescriptor.
        #    b. Fetch SemanticMetadata list.
        #    c. Fetch LineageMetadata (from its JSONB column).
        #    d. Fetch ComputationalMetadata (from its JSONB column).
        #    e. ... and so on for all extended types.
        #    f. Construct TensorusExportEntry.
        # 3. Assemble into TensorusExportData.
        raise NotImplementedError("get_export_data is not yet fully implemented for PostgresMetadataStorage.")

    def import_data(self, data: 'TensorusExportData', conflict_strategy: str = "skip") -> Dict[str, int]: # type: ignore
        # This is highly complex for SQL due to potential conflicts, different table structures,
        # and the need for transactional integrity.
        # For each entry in data.entries:
        # 1. Handle TensorDescriptor:
        #    - If conflict_strategy is "skip": INSERT ... ON CONFLICT (tensor_id) DO NOTHING.
        #    - If conflict_strategy is "overwrite":
        #        - DELETE from tensor_descriptors WHERE tensor_id = ... (CASCADE should handle related data).
        #        - INSERT new TensorDescriptor.
        #    - Or use complex UPSERT that updates all fields if overwriting.
        # 2. Handle SemanticMetadata (one-to-many, specific columns):
        #    - If overwriting, delete existing semantic entries for the tensor_id first.
        #    - INSERT new semantic entries. Handle (tensor_id, name) conflicts.
        # 3. Handle other extended metadata (one-to-one, JSONB column):
        #    - Use INSERT ... ON CONFLICT (tensor_id) DO UPDATE SET data = EXCLUDED.data for upsert.
        #    - If "skip" and conflict, DO NOTHING.
        #    - If "overwrite" and it's part of a larger TD overwrite, cascade delete handles old, then insert new.
        #
        # All of this should ideally happen within a single transaction for the whole import,
        # or at least per TensorusExportEntry.
        raise NotImplementedError("import_data is not yet fully implemented for PostgresMetadataStorage.")

    # --- Analytics Methods (Postgres Implementations) ---
    def get_co_occurring_tags(self, min_co_occurrence: int = 2, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        # This query is complex and can be inefficient on large datasets without specific optimizations.
        # It unnests tags, creates pairs, counts them, then formats the output.
        query = """
            WITH tensor_tags AS (
                SELECT tensor_id, unnest(tags) AS tag FROM tensor_descriptors WHERE cardinality(tags) >= 2
            ),
            tag_pairs AS (
                SELECT
                    t1.tensor_id,
                    LEAST(t1.tag, t2.tag) AS tag_a,
                    GREATEST(t1.tag, t2.tag) AS tag_b
                FROM tensor_tags t1
                JOIN tensor_tags t2 ON t1.tensor_id = t2.tensor_id AND t1.tag < t2.tag
            ),
            pair_counts AS (
                SELECT tag_a, tag_b, COUNT(*) AS co_occurrence_count
                FROM tag_pairs
                GROUP BY tag_a, tag_b
                HAVING COUNT(*) >= %(min_co_occurrence)s
            ),
            ranked_pairs AS (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY tag_a ORDER BY co_occurrence_count DESC, tag_b) as rn_a,
                       ROW_NUMBER() OVER (PARTITION BY tag_b ORDER BY co_occurrence_count DESC, tag_a) as rn_b
                FROM pair_counts
            )
            -- This final SELECT is tricky to get into the desired nested Dict structure directly from SQL.
            -- It's often easier to process the pair_counts or ranked_pairs in Python.
            -- For now, let's fetch ranked pairs and process in Python.
            SELECT tag_a, tag_b, co_occurrence_count FROM ranked_pairs
            WHERE rn_a <= %(limit)s OR rn_b <= %(limit)s;
            -- This limit logic is not perfect for the desired output structure directly.
            -- A simpler approach: just get all pairs above min_co_occurrence and limit/process in Python.
            -- Simpler query for pair counts:
            -- SELECT tag_a, tag_b, COUNT(*) AS co_occurrence_count
            -- FROM tag_pairs
            -- GROUP BY tag_a, tag_b
            -- HAVING COUNT(*) >= %(min_co_occurrence)s
            -- ORDER BY co_occurrence_count DESC;
            -- Then process this result in Python to build the nested dict and apply limits.
        """
        # Using the simpler query for pair counts
        simpler_query = """
            WITH tensor_tags AS (
                SELECT tensor_id, unnest(tags) AS tag FROM tensor_descriptors WHERE cardinality(tags) >= 2
            ),
            tag_pairs AS (
                SELECT LEAST(t1.tag, t2.tag) AS tag_a, GREATEST(t1.tag, t2.tag) AS tag_b
                FROM tensor_tags t1 JOIN tensor_tags t2 ON t1.tensor_id = t2.tensor_id AND t1.tag < t2.tag
            )
            SELECT tag_a, tag_b, COUNT(*) AS co_occurrence_count
            FROM tag_pairs GROUP BY tag_a, tag_b
            HAVING COUNT(*) >= %(min_co_occurrence)s
            ORDER BY tag_a, co_occurrence_count DESC;
        """
        params = {"min_co_occurrence": min_co_occurrence, "limit": limit} # Limit used in Python processing

        rows = self._execute_query(simpler_query, params, fetch="all") # type: ignore

        co_occurrence_map: Dict[str, List[Dict[str, Any]]] = {}
        if rows:
            for row in rows:
                tag_a, tag_b, count = row['tag_a'], row['tag_b'], row['co_occurrence_count']
                # Add for tag_a
                if tag_a not in co_occurrence_map: co_occurrence_map[tag_a] = []
                if len(co_occurrence_map[tag_a]) < limit:
                    co_occurrence_map[tag_a].append({"tag": tag_b, "count": count})
                # Add for tag_b
                if tag_b not in co_occurrence_map: co_occurrence_map[tag_b] = []
                if len(co_occurrence_map[tag_b]) < limit:
                     co_occurrence_map[tag_b].append({"tag": tag_a, "count": count})

        # Sort internal lists (already sorted by query for tag_a, but not for tag_b's list)
        for tag_key in co_occurrence_map:
            co_occurrence_map[tag_key].sort(key=lambda x: x["count"], reverse=True)

        return {k: v for k, v in co_occurrence_map.items() if v} # Filter out tags with no co-occurrences meeting criteria


    def get_stale_tensors(self, threshold_days: int, limit: int = 100) -> List[TensorDescriptor]:
        query = """
            SELECT td.*
            FROM tensor_descriptors td
            LEFT JOIN usage_metadata um ON td.tensor_id = um.tensor_id
            WHERE COALESCE( (um.data->>'last_accessed_at')::TIMESTAMPTZ, td.last_modified_timestamp ) < (NOW() - INTERVAL '1 day' * %(threshold_days)s)
            ORDER BY COALESCE( (um.data->>'last_accessed_at')::TIMESTAMPTZ, td.last_modified_timestamp ) ASC
            LIMIT %(limit)s;
        """
        params = {"threshold_days": threshold_days, "limit": limit}
        rows = self._execute_query(query, params, fetch="all") # type: ignore

        results = []
        if rows:
            for row in rows:
                data = dict(row)
                data['data_type'] = DataType(data['data_type'])
                data['storage_format'] = StorageFormat(data['storage_format'])
                if data.get('access_control'): data['access_control'] = AccessControl(**data['access_control'])
                if data.get('compression_info'): data['compression_info'] = CompressionInfo(**data['compression_info'])
                results.append(TensorDescriptor(**data))
        return results

    def get_complex_tensors(self, min_parent_count: Optional[int] = None, min_transformation_steps: Optional[int] = None, limit: int = 100) -> List[TensorDescriptor]:
        if min_parent_count is None and min_transformation_steps is None:
            raise ValueError("At least one criterion (min_parent_count or min_transformation_steps) must be provided.")

        conditions = []
        params: Dict[str, Any] = {"limit": limit}

        base_query = "SELECT td.* FROM tensor_descriptors td LEFT JOIN lineage_metadata lm ON td.tensor_id = lm.tensor_id"

        if min_parent_count is not None:
            conditions.append("jsonb_array_length(lm.data->'parent_tensors') >= %(min_parent_count)s")
            params["min_parent_count"] = min_parent_count

        if min_transformation_steps is not None:
            conditions.append("jsonb_array_length(lm.data->'transformation_history') >= %(min_transformation_steps)s")
            params["min_transformation_steps"] = min_transformation_steps

        query = f"{base_query} WHERE ({' OR '.join(conditions)}) LIMIT %(limit)s;"

        rows = self._execute_query(query, params, fetch="all") # type: ignore
        results = []
        if rows:
            for row in rows:
                data = dict(row)
                data['data_type'] = DataType(data['data_type'])
                data['storage_format'] = StorageFormat(data['storage_format'])
                if data.get('access_control'): data['access_control'] = AccessControl(**data['access_control'])
                if data.get('compression_info'): data['compression_info'] = CompressionInfo(**data['compression_info'])
                results.append(TensorDescriptor(**data))
        return results

    # --- Health and Metrics Methods (Postgres Implementations) ---
    def check_health(self) -> tuple[bool, str]:
        try:
            self._execute_query("SELECT 1;", fetch=None)
            return True, "postgres"
        except Exception as e:
            logger.error(f"Postgres health check failed: {e}")
            return False, "postgres"

    def get_tensor_descriptors_count(self) -> int:
        query = "SELECT COUNT(*) as count FROM tensor_descriptors;"
        row = self._execute_query(query, fetch="one")
        return row['count'] if row else 0

    def get_extended_metadata_count(self, metadata_model_name: str) -> int:
        # Map Pydantic model names to table names
        # This assumes a specific naming convention for tables, e.g., lowercase with underscores.
        table_name_map = {
            "LineageMetadata": "lineage_metadata",
            "ComputationalMetadata": "computational_metadata",
            "QualityMetadata": "quality_metadata",
            "RelationalMetadata": "relational_metadata",
            "UsageMetadata": "usage_metadata",
            "SemanticMetadata": "semantic_metadata_entries" # Special case for semantic
        }
        table_name = table_name_map.get(metadata_model_name)

        if not table_name:
            # Or raise error, or log warning
            logger.warning(
                f"get_extended_metadata_count called for unmapped model name '{metadata_model_name}' in Postgres."
            )
            return 0

        query = f"SELECT COUNT(*) as count FROM {table_name};" # Ensure table_name is not from user input directly
        row = self._execute_query(query, fetch="one")
        return row['count'] if row else 0

    def clear_all_data(self) -> None:
        # In a real scenario, might TRUNCATE tables or use a specific test DB
        self._execute_query("DELETE FROM semantic_metadata_entries;") # Order matters due to FKs
        self._execute_query("DELETE FROM lineage_metadata;")
        self._execute_query("DELETE FROM computational_metadata;")
        self._execute_query("DELETE FROM quality_metadata;")
        self._execute_query("DELETE FROM relational_metadata;") # Corrected self_execute_query
        self._execute_query("DELETE FROM usage_metadata;")
        self._execute_query("DELETE FROM tensor_descriptors;")
        logger.info("Postgres tables cleared (conceptually).")

    def close_pool(self):
        if self.pool:
            self.pool.closeall()
            self.pool = None
            logger.info("PostgreSQL connection pool closed.")
