import hashlib
import json
import time
import logging
import os
from typing import List, Dict, Any, Optional, Union
import threading
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Block:
    """
    Block in the tensor blockchain.
    Each block contains a hash of the previous block, a timestamp, 
    operations performed on tensors, and a nonce for proof-of-work.
    """
    
    def __init__(self, 
                 index: int, 
                 timestamp: float, 
                 operations: List[Dict[str, Any]], 
                 previous_hash: str, 
                 nonce: int = 0):
        """
        Initialize a block.
        
        Args:
            index: Block index in the chain
            timestamp: Block creation time
            operations: List of tensor operations
            previous_hash: Hash of the previous block
            nonce: Nonce for proof-of-work
        """
        self.index = index
        self.timestamp = timestamp
        self.operations = operations
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """
        Calculate SHA-256 hash of the block.
        
        Returns:
            Hexadecimal hash string
        """
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "operations": self.operations,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True).encode()
        
        return hashlib.sha256(block_string).hexdigest()
    
    def mine_block(self, difficulty: int):
        """
        Mine the block (proof-of-work).
        
        Args:
            difficulty: Number of leading zeros required in the hash
        """
        target = "0" * difficulty
        
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert block to dictionary representation.
        
        Returns:
            Dictionary representation of the block
        """
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "operations": self.operations,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "hash": self.hash
        }

class TensorBlockchain:
    """
    Blockchain for tracking tensor data provenance.
    Provides immutable logging of all operations performed on tensors.
    """
    
    def __init__(self, 
                 chain_file: Optional[str] = "data/tensor_blockchain.json",
                 difficulty: int = 2,
                 auto_mine: bool = True):
        """
        Initialize the blockchain.
        
        Args:
            chain_file: Path to store the blockchain
            difficulty: Proof-of-work difficulty (number of leading zeros)
            auto_mine: Whether to automatically mine blocks
        """
        self.chain_file = chain_file
        self.difficulty = difficulty
        self.auto_mine = auto_mine
        
        # Create directory if it doesn't exist
        if chain_file:
            os.makedirs(os.path.dirname(chain_file), exist_ok=True)
        
        # Initialize chain with genesis block
        self.chain = []
        self.pending_operations = []
        self.lock = threading.Lock()
        
        # Load chain from file or create genesis block
        if chain_file and os.path.exists(chain_file):
            self.load_chain()
        else:
            self.create_genesis_block()
        
        logger.info("TensorBlockchain initialized")
    
    def create_genesis_block(self):
        """Create the first block in the chain."""
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            operations=[{"type": "genesis", "data": "Tensorus Genesis Block"}],
            previous_hash="0"
        )
        
        if self.auto_mine:
            genesis_block.mine_block(self.difficulty)
            
        self.chain.append(genesis_block)
        
        # Save the chain
        if self.chain_file:
            self.save_chain()
    
    def get_latest_block(self) -> Block:
        """
        Get the most recent block in the chain.
        
        Returns:
            The latest block
        """
        return self.chain[-1]
    
    def add_operation(self, 
                      operation_type: str, 
                      tensor_id: str, 
                      user_id: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None,
                      auto_mine: Optional[bool] = None) -> str:
        """
        Add a tensor operation to the pending operations.
        
        Args:
            operation_type: Type of operation (e.g., "create", "read", "update", "delete")
            tensor_id: ID of the tensor
            user_id: ID of the user performing the operation
            metadata: Additional metadata about the operation
            auto_mine: Whether to automatically mine a block (overrides instance setting)
            
        Returns:
            Operation ID
        """
        if metadata is None:
            metadata = {}
        
        operation_id = str(uuid.uuid4())
        timestamp = time.time()
        
        operation = {
            "id": operation_id,
            "type": operation_type,
            "tensor_id": tensor_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "metadata": metadata
        }
        
        with self.lock:
            self.pending_operations.append(operation)
            
            # Automatically mine a block if enough operations are pending
            should_mine = auto_mine if auto_mine is not None else self.auto_mine
            if should_mine and len(self.pending_operations) >= 5:
                self.mine_pending_operations()
        
        return operation_id
    
    def mine_pending_operations(self) -> Optional[Block]:
        """
        Create a new block with pending operations and add it to the chain.
        
        Returns:
            The newly created block, or None if no operations are pending
        """
        with self.lock:
            if not self.pending_operations:
                return None
            
            previous_block = self.get_latest_block()
            new_block = Block(
                index=previous_block.index + 1,
                timestamp=time.time(),
                operations=self.pending_operations.copy(),
                previous_hash=previous_block.hash
            )
            
            # Mine the block
            new_block.mine_block(self.difficulty)
            
            # Add to chain
            self.chain.append(new_block)
            
            # Clear pending operations
            self.pending_operations = []
            
            # Save the chain
            if self.chain_file:
                self.save_chain()
            
            logger.info(f"Mined block {new_block.index} with {len(new_block.operations)} operations")
            
            return new_block
    
    def validate_chain(self) -> bool:
        """
        Validate the integrity of the blockchain.
        
        Returns:
            True if valid, False otherwise
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # Check if the current block's hash is valid
            if current_block.hash != current_block.calculate_hash():
                logger.warning(f"Block {current_block.index} has an invalid hash")
                return False
            
            # Check if the current block's previous_hash matches the previous block's hash
            if current_block.previous_hash != previous_block.hash:
                logger.warning(f"Block {current_block.index} has an invalid previous hash")
                return False
        
        return True
    
    def save_chain(self):
        """Save the blockchain to a file."""
        try:
            with open(self.chain_file, 'w') as f:
                chain_data = [block.to_dict() for block in self.chain]
                json.dump(chain_data, f, indent=2)
            
            logger.info(f"Blockchain saved to {self.chain_file}")
        except Exception as e:
            logger.error(f"Error saving blockchain: {e}")
    
    def load_chain(self):
        """Load the blockchain from a file."""
        try:
            with open(self.chain_file, 'r') as f:
                chain_data = json.load(f)
            
            self.chain = []
            for block_data in chain_data:
                block = Block(
                    index=block_data["index"],
                    timestamp=block_data["timestamp"],
                    operations=block_data["operations"],
                    previous_hash=block_data["previous_hash"],
                    nonce=block_data["nonce"]
                )
                block.hash = block_data["hash"]
                self.chain.append(block)
            
            logger.info(f"Loaded blockchain with {len(self.chain)} blocks from {self.chain_file}")
            
            # Validate the loaded chain
            if not self.validate_chain():
                logger.warning("Loaded blockchain is invalid")
                raise ValueError("Invalid blockchain")
                
        except Exception as e:
            logger.error(f"Error loading blockchain: {e}")
            self.create_genesis_block()
    
    def get_tensor_history(self, tensor_id: str) -> List[Dict[str, Any]]:
        """
        Get the history of operations for a specific tensor.
        
        Args:
            tensor_id: ID of the tensor
            
        Returns:
            List of operations performed on the tensor
        """
        history = []
        
        # Search all blocks for operations on this tensor
        for block in self.chain:
            for operation in block.operations:
                if operation.get("tensor_id") == tensor_id:
                    history_entry = dict(operation)
                    history_entry["block_index"] = block.index
                    history_entry["block_hash"] = block.hash
                    history.append(history_entry)
        
        # Add any pending operations for this tensor
        for operation in self.pending_operations:
            if operation.get("tensor_id") == tensor_id:
                history_entry = dict(operation)
                history_entry["pending"] = True
                history.append(history_entry)
        
        # Sort by timestamp
        history.sort(key=lambda x: x["timestamp"])
        
        return history
    
    def get_block_by_index(self, index: int) -> Optional[Block]:
        """
        Get a block by its index.
        
        Args:
            index: Block index
            
        Returns:
            Block object or None if not found
        """
        if 0 <= index < len(self.chain):
            return self.chain[index]
        return None
    
    def get_block_by_hash(self, block_hash: str) -> Optional[Block]:
        """
        Get a block by its hash.
        
        Args:
            block_hash: Block hash
            
        Returns:
            Block object or None if not found
        """
        for block in self.chain:
            if block.hash == block_hash:
                return block
        return None
    
    def get_chain_size(self) -> int:
        """
        Get the number of blocks in the chain.
        
        Returns:
            Number of blocks
        """
        return len(self.chain)
    
    def get_chain_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the blockchain.
        
        Returns:
            Dictionary with blockchain summary information
        """
        return {
            "blocks": len(self.chain),
            "pending_operations": len(self.pending_operations),
            "last_block_hash": self.get_latest_block().hash,
            "last_block_time": self.get_latest_block().timestamp,
            "is_valid": self.validate_chain()
        } 