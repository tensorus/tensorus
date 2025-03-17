import numpy as np
import logging
import os
import json
from typing import Dict, List, Any, Tuple, Optional, Union, Callable

from tensor_agent import TensorAgent
from tensor_blockchain import TensorBlockchain
from tensor_query_language import TQLParser
from tensor_distributed import TensorDistributedNode, FederatedLearning
from tensor_security import TensorSecurity, TensorDifferentialPrivacy, secure_operation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorAdvanced:
    """
    Integration module that connects all advanced features of Tensorus.
    """
    
    def __init__(self, database_ref, config_path: Optional[str] = "config/advanced_config.json"):
        """
        Initialize advanced features integration.
        
        Args:
            database_ref: Reference to the TensorDatabase instance
            config_path: Path to configuration file
        """
        self.database = database_ref
        self.config = self._load_config(config_path)
        
        # Initialize advanced components based on configuration
        self._init_components()
        
        logger.info("TensorAdvanced initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from JSON file or use defaults."""
        default_config = {
            "enable_agent": True,
            "enable_blockchain": True,
            "enable_tql": True,
            "enable_distributed": False,
            "enable_security": True,
            "enable_privacy": False,
            "agent": {
                "learning_rate": 0.01,
                "exploration_rate": 0.2,
                "reward_decay": 0.9
            },
            "blockchain": {
                "difficulty": 2,
                "auto_mine": True
            },
            "distributed": {
                "role": "standalone",
                "port": 5050,
                "coordinator_url": None
            },
            "security": {
                "token_expiry": 3600,
                "enable_audit_log": True
            },
            "privacy": {
                "epsilon": 1.0,
                "delta": 1e-5,
                "mechanism": "laplace"
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded advanced configuration from {config_path}")
                return {**default_config, **config}  # Merge with defaults
            except Exception as e:
                logger.warning(f"Error loading advanced config: {e}. Using defaults.")
        
        logger.info("Using default advanced configuration")
        return default_config
    
    def _init_components(self):
        """Initialize advanced components based on configuration."""
        # Initialize agent if enabled
        if self.config.get("enable_agent", True):
            agent_config = self.config.get("agent", {})
            self.agent = TensorAgent(
                config_path=agent_config.get("config_path"),
                learning_rate=agent_config.get("learning_rate", 0.01),
                exploration_rate=agent_config.get("exploration_rate", 0.2),
                reward_decay=agent_config.get("reward_decay", 0.9),
                checkpoint_path=agent_config.get("checkpoint_path")
            )
            self.agent.start_monitoring(self.database)
            logger.info("TensorAgent initialized and monitoring started")
        else:
            self.agent = None
            logger.info("TensorAgent disabled")
        
        # Initialize blockchain if enabled
        if self.config.get("enable_blockchain", True):
            blockchain_config = self.config.get("blockchain", {})
            self.blockchain = TensorBlockchain(
                chain_file=blockchain_config.get("chain_file"),
                difficulty=blockchain_config.get("difficulty", 2),
                auto_mine=blockchain_config.get("auto_mine", True)
            )
            logger.info("TensorBlockchain initialized")
        else:
            self.blockchain = None
            logger.info("TensorBlockchain disabled")
        
        # Initialize TQL parser if enabled
        if self.config.get("enable_tql", True):
            self.tql_parser = TQLParser(self.database)
            logger.info("TQL parser initialized")
        else:
            self.tql_parser = None
            logger.info("TQL parser disabled")
        
        # Initialize distributed node if enabled
        if self.config.get("enable_distributed", False):
            distributed_config = self.config.get("distributed", {})
            self.distributed_node = TensorDistributedNode(
                database_ref=self.database,
                node_id=distributed_config.get("node_id"),
                address=distributed_config.get("address"),
                port=distributed_config.get("port", 5050),
                coordinator_url=distributed_config.get("coordinator_url"),
                role=distributed_config.get("role", "standalone")
            )
            
            # Start the node
            self.distributed_node.start()
            
            # Initialize federated learning if role is coordinator or hybrid
            if self.distributed_node.role in ["coordinator", "hybrid"]:
                self.federated = FederatedLearning(self.distributed_node)
                logger.info("Federated learning initialized")
            else:
                self.federated = None
                
            logger.info(f"Distributed node initialized with role {self.distributed_node.role}")
        else:
            self.distributed_node = None
            self.federated = None
            logger.info("Distributed node disabled")
        
        # Initialize security if enabled
        if self.config.get("enable_security", True):
            security_config = self.config.get("security", {})
            self.security = TensorSecurity(
                config_path=security_config.get("config_path"),
                secret_key=security_config.get("secret_key"),
                token_expiry=security_config.get("token_expiry", 3600),
                enable_audit_log=security_config.get("enable_audit_log", True)
            )
            # Add security reference to database
            self.database.security = self.security
            logger.info("Security initialized")
        else:
            self.security = None
            logger.info("Security disabled")
        
        # Initialize differential privacy if enabled
        if self.config.get("enable_privacy", False):
            privacy_config = self.config.get("privacy", {})
            self.privacy = TensorDifferentialPrivacy(
                epsilon=privacy_config.get("epsilon", 1.0),
                delta=privacy_config.get("delta", 1e-5),
                mechanism=privacy_config.get("mechanism", "laplace")
            )
            logger.info("Differential privacy initialized")
        else:
            self.privacy = None
            logger.info("Differential privacy disabled")
    
    def execute_tql(self, query: str, token: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a TQL query with proper security checks.
        
        Args:
            query: TQL query string
            token: Security token (if security is enabled)
            
        Returns:
            Query results
        """
        if not self.tql_parser:
            raise ValueError("TQL parser is not enabled")
        
        # Check security if enabled
        if self.security and token:
            # Parse the query to determine the operation type
            try:
                parsed_query = self.tql_parser.parse(query)
                operation_type = parsed_query.get("operation")
                
                # Map TQL operations to security operations
                security_operations = {
                    "SELECT": "read",
                    "SEARCH": "search",
                    "CREATE": "write",
                    "UPDATE": "write",
                    "DELETE": "delete",
                    "TRANSFORM": "process",
                    "DECOMPOSE": "process"
                }
                
                operation = security_operations.get(operation_type, "read")
                
                # Check authorization
                if not self.security.authorize(token, "tensor", operation):
                    raise PermissionError(f"Not authorized for tensor:{operation}")
                
                # Log the query execution
                payload = self.security._verify_token(token)
                user_id = payload.get("user_id") if payload else "unknown"
                
                self.security.log_audit_event(
                    user_id, 
                    "tql_execution", 
                    "attempt", 
                    {"query": query, "operation": operation}
                )
            except Exception as e:
                logger.error(f"TQL security check failed: {e}")
                raise
        
        # Execute the query
        try:
            result = self.tql_parser.execute(query)
            
            # Apply differential privacy if enabled
            if self.privacy and isinstance(result.get("result"), np.ndarray):
                # Determine sensitivity based on operation (simplified)
                sensitivity = 1.0
                result["result"] = self.privacy.privatize_tensor(result["result"], sensitivity)
                result["privacy_applied"] = True
            
            # Log blockchain record if enabled
            if self.blockchain:
                # Extract tensor ID if available
                tensor_id = None
                if "tensor_id" in result:
                    tensor_id = result["tensor_id"]
                elif "tensors" in result and len(result["tensors"]) > 0:
                    tensor_id = result["tensors"][0]
                
                # Log operation if tensor ID is available
                if tensor_id:
                    # Map TQL operations to blockchain operations
                    operation_mapping = {
                        "SELECT": "read",
                        "SEARCH": "search",
                        "CREATE": "create",
                        "UPDATE": "update",
                        "DELETE": "delete",
                        "TRANSFORM": "transform",
                        "DECOMPOSE": "decompose"
                    }
                    
                    # Get user ID from token
                    user_id = None
                    if self.security and token:
                        payload = self.security._verify_token(token)
                        user_id = payload.get("user_id") if payload else None
                    
                    # Add operation
                    if "operation" in result:
                        operation_type = operation_mapping.get(result["operation"], "query")
                        self.blockchain.add_operation(
                            operation_type=operation_type,
                            tensor_id=tensor_id,
                            user_id=user_id,
                            metadata={"query": query}
                        )
            
            # Log successful execution
            if self.security and token:
                payload = self.security._verify_token(token)
                user_id = payload.get("user_id") if payload else "unknown"
                
                self.security.log_audit_event(
                    user_id, 
                    "tql_execution", 
                    "success", 
                    {"query": query}
                )
            
            return result
            
        except Exception as e:
            # Log failure
            if self.security and token:
                payload = self.security._verify_token(token)
                user_id = payload.get("user_id") if payload else "unknown"
                
                self.security.log_audit_event(
                    user_id, 
                    "tql_execution", 
                    "failure", 
                    {"query": query, "error": str(e)}
                )
            
            raise
    
    def save_tensor(self, tensor: np.ndarray, metadata: Optional[Dict[str, Any]] = None, 
                   distributed: bool = False, token: Optional[str] = None) -> str:
        """
        Save a tensor with advanced features.
        
        Args:
            tensor: Tensor data
            metadata: Optional metadata
            distributed: Whether to use distributed storage
            token: Security token (if security is enabled)
            
        Returns:
            Tensor ID
        """
        if metadata is None:
            metadata = {}
        
        # Check security if enabled
        if self.security and token:
            if not self.security.authorize(token, "tensor", "write"):
                raise PermissionError("Not authorized for tensor:write")
        
        # Save tensor (distributed or local)
        if distributed and self.distributed_node:
            tensor_id = self.distributed_node.save_distributed(tensor, metadata)
        else:
            tensor_id = self.database.save(tensor, metadata)
        
        # Log to blockchain if enabled
        if self.blockchain:
            # Get user ID from token
            user_id = None
            if self.security and token:
                payload = self.security._verify_token(token)
                user_id = payload.get("user_id") if payload else None
            
            # Add operation
            self.blockchain.add_operation(
                operation_type="create",
                tensor_id=tensor_id,
                user_id=user_id,
                metadata={"distributed": distributed, "shape": list(tensor.shape)}
            )
        
        return tensor_id
    
    def get_tensor(self, tensor_id: str, distributed: bool = False, 
                  apply_privacy: bool = False, token: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get a tensor with advanced features.
        
        Args:
            tensor_id: Tensor ID
            distributed: Whether to use distributed storage
            apply_privacy: Whether to apply differential privacy
            token: Security token (if security is enabled)
            
        Returns:
            Tensor data and metadata
        """
        # Check security if enabled
        if self.security and token:
            if not self.security.authorize(token, "tensor", "read"):
                raise PermissionError("Not authorized for tensor:read")
        
        # Get tensor (distributed or local)
        if distributed and self.distributed_node:
            tensor, metadata = self.distributed_node.get_distributed(tensor_id)
        else:
            tensor, metadata = self.database.get(tensor_id)
        
        # Apply differential privacy if requested and enabled
        if apply_privacy and self.privacy:
            # Simple sensitivity of 1.0 (should be customized based on use case)
            tensor = self.privacy.privatize_tensor(tensor, 1.0)
            metadata["privacy_applied"] = True
        
        # Log to blockchain if enabled
        if self.blockchain:
            # Get user ID from token
            user_id = None
            if self.security and token:
                payload = self.security._verify_token(token)
                user_id = payload.get("user_id") if payload else None
            
            # Add operation
            self.blockchain.add_operation(
                operation_type="read",
                tensor_id=tensor_id,
                user_id=user_id,
                metadata={"distributed": distributed, "privacy_applied": apply_privacy}
            )
        
        return tensor, metadata
    
    def search_similar(self, query_tensor: np.ndarray, k: int = 5, 
                      distributed: bool = False, token: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar tensors with advanced features.
        
        Args:
            query_tensor: Query tensor
            k: Number of results to return
            distributed: Whether to use distributed search
            token: Security token (if security is enabled)
            
        Returns:
            List of search results
        """
        # Check security if enabled
        if self.security and token:
            if not self.security.authorize(token, "tensor", "search"):
                raise PermissionError("Not authorized for tensor:search")
        
        # Search (distributed or local)
        if distributed and self.distributed_node:
            results = self.distributed_node.search_distributed(query_tensor, k)
        else:
            results = self.database.search_similar(query_tensor, k)
        
        # Apply differential privacy to results if enabled
        if self.privacy:
            # For similarity search, we might add noise to the scores
            for result in results:
                if "similarity" in result:
                    # Simple sensitivity of 0.1 for similarity scores
                    noise = np.random.laplace(0, 0.1 / self.privacy.epsilon)
                    result["similarity"] += noise
                    result["privacy_applied"] = True
        
        # Log to blockchain if enabled
        if self.blockchain and results:
            # Get user ID from token
            user_id = None
            if self.security and token:
                payload = self.security._verify_token(token)
                user_id = payload.get("user_id") if payload else None
            
            # Add operation (using the first result as a representative)
            self.blockchain.add_operation(
                operation_type="search",
                tensor_id=results[0].get("id", "unknown"),
                user_id=user_id,
                metadata={"num_results": len(results), "distributed": distributed}
            )
        
        return results
    
    def get_tensor_history(self, tensor_id: str, token: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the history of operations for a tensor.
        
        Args:
            tensor_id: Tensor ID
            token: Security token (if security is enabled)
            
        Returns:
            List of operations on the tensor
        """
        if not self.blockchain:
            raise ValueError("Blockchain is not enabled")
        
        # Check security if enabled
        if self.security and token:
            if not self.security.authorize(token, "tensor", "read"):
                raise PermissionError("Not authorized for tensor:read")
        
        return self.blockchain.get_tensor_history(tensor_id)
    
    def create_federated_model(self, model_id: str, initial_weights: Dict[str, np.ndarray],
                              config: Dict[str, Any], token: Optional[str] = None) -> bool:
        """
        Create a new federated learning model.
        
        Args:
            model_id: Unique identifier for the model
            initial_weights: Initial weights for the model
            config: Training configuration
            token: Security token (if security is enabled)
            
        Returns:
            True if model was created successfully
        """
        if not self.federated:
            raise ValueError("Federated learning is not enabled")
        
        # Check security if enabled
        if self.security and token:
            if not self.security.authorize(token, "model", "write"):
                raise PermissionError("Not authorized for model:write")
        
        return self.federated.create_federated_model(model_id, initial_weights, config)
    
    def start_federated_training(self, model_id: str, nodes: List[str] = None,
                                rounds: int = 1, token: Optional[str] = None) -> bool:
        """
        Start federated training for a model.
        
        Args:
            model_id: ID of the model
            nodes: List of node IDs to include (None for all available)
            rounds: Number of training rounds
            token: Security token (if security is enabled)
            
        Returns:
            True if training was started successfully
        """
        if not self.federated:
            raise ValueError("Federated learning is not enabled")
        
        # Check security if enabled
        if self.security and token:
            if not self.security.authorize(token, "model", "process"):
                raise PermissionError("Not authorized for model:process")
        
        return self.federated.start_federated_training(model_id, nodes, rounds)
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """
        Authenticate a user and generate a security token.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            JWT token or None if authentication fails
        """
        if not self.security:
            raise ValueError("Security is not enabled")
        
        return self.security.authenticate(username, password)
    
    def create_user(self, admin_token: str, username: str, password: str, roles: List[str]) -> bool:
        """
        Create a new user.
        
        Args:
            admin_token: Admin JWT token
            username: New user's username
            password: New user's password
            roles: List of roles to assign
            
        Returns:
            True if user was created successfully
        """
        if not self.security:
            raise ValueError("Security is not enabled")
        
        return self.security.create_user(admin_token, username, password, roles)
    
    def save_config(self):
        """Save all component configurations."""
        # Save agent state if enabled
        if self.agent:
            self.agent.save_state()
        
        # Save security config if enabled
        if self.security:
            self.security.save_config()
    
    def shutdown(self):
        """Gracefully shut down all components."""
        # Stop agent if enabled
        if self.agent:
            self.agent.stop_monitoring()
            self.agent.save_state()
        
        # Stop distributed node if enabled
        if self.distributed_node:
            self.distributed_node.stop()
        
        # Save security config if enabled
        if self.security:
            self.security.save_config()
        
        logger.info("TensorAdvanced shutdown complete") 