from tensorus.metadata import storage_instance as globally_configured_storage_instance
from tensorus.metadata.storage_abc import MetadataStorage
from ..tensor_storage import TensorStorage
from ..nql_agent import NQLAgent
from ..embedding_agent import EmbeddingAgent
from ..hybrid_search import HybridSearchEngine
from ..tensor_ops import TensorOps

def get_storage_instance() -> MetadataStorage:
    return globally_configured_storage_instance

_tensor_storage_instance = None
_nql_agent_instance = None
_embedding_agent_instance = None
_hybrid_search_instance = None
_tensor_ops_instance = None

def get_tensor_storage() -> TensorStorage:
    global _tensor_storage_instance
    if _tensor_storage_instance is None:
        _tensor_storage_instance = TensorStorage()
    return _tensor_storage_instance

def get_nql_agent() -> NQLAgent:
    global _nql_agent_instance
    if _nql_agent_instance is None:
        _nql_agent_instance = NQLAgent(get_tensor_storage())
    return _nql_agent_instance

def get_embedding_agent() -> EmbeddingAgent:
    global _embedding_agent_instance
    if _embedding_agent_instance is None:
        _embedding_agent_instance = EmbeddingAgent(get_tensor_storage())
    return _embedding_agent_instance

def get_tensor_ops() -> TensorOps:
    global _tensor_ops_instance
    if _tensor_ops_instance is None:
        _tensor_ops_instance = TensorOps()
    return _tensor_ops_instance

def get_hybrid_search() -> HybridSearchEngine:
    global _hybrid_search_instance
    if _hybrid_search_instance is None:
        _hybrid_search_instance = HybridSearchEngine(get_tensor_storage(), get_embedding_agent(), get_tensor_ops())
    return _hybrid_search_instance