import numpy as np
import logging
import time
from typing import Dict, List, Any, Tuple, Optional, Union
import threading
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorAgent:
    """
    Agentic component of Tensorus that uses reinforcement learning to 
    dynamically reconfigure database parameters and strategies.
    """
    
    def __init__(self, 
                 config_path: str = "config/agent_config.json",
                 learning_rate: float = 0.01,
                 exploration_rate: float = 0.2,
                 reward_decay: float = 0.9,
                 checkpoint_path: Optional[str] = "data/agent_checkpoints"):
        """
        Initialize the tensor agent.
        
        Args:
            config_path: Path to the agent configuration
            learning_rate: Learning rate for the RL algorithm
            exploration_rate: Initial exploration rate for epsilon-greedy policy
            reward_decay: Discount factor for future rewards
            checkpoint_path: Path to save/load agent state
        """
        self.config = self._load_config(config_path)
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.reward_decay = reward_decay
        self.checkpoint_path = checkpoint_path
        
        # Create checkpoint directory if it doesn't exist
        if checkpoint_path:
            os.makedirs(checkpoint_path, exist_ok=True)
        
        # Initialize state space, action space, and Q-table
        self.state_features = [
            "query_rate",
            "insert_rate",
            "avg_query_time",
            "avg_insert_time",
            "memory_usage",
            "index_type",
            "index_size"
        ]
        
        self.action_space = {
            "index_type": ["flat", "ivf", "hnsw"],
            "metric": ["l2", "ip"],
            "compression_level": [0, 1, 2, 3],
            "batch_size": [1, 10, 50, 100],
            "use_gpu": [True, False]
        }
        
        # Initialize Q-table with small random values
        self.q_table = {}
        
        # Track performance metrics
        self.metrics_history = []
        
        # Agent state
        self.current_state = None
        self.current_action = None
        self.active = False
        
        logger.info("TensorAgent initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load agent configuration from JSON file or use defaults."""
        default_config = {
            "monitoring_interval": 60,  # seconds
            "reconfiguration_cooldown": 300,  # seconds
            "min_samples_before_learning": 10,
            "reward_weights": {
                "query_time_improvement": 0.4,
                "insert_time_improvement": 0.3,
                "memory_efficiency": 0.2,
                "stability": 0.1
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded agent configuration from {config_path}")
                return {**default_config, **config}  # Merge with defaults
            except Exception as e:
                logger.warning(f"Error loading agent config: {e}. Using defaults.")
        
        logger.info("Using default agent configuration")
        return default_config
    
    def start_monitoring(self, database_ref):
        """
        Start monitoring the database and reconfiguring as needed.
        
        Args:
            database_ref: Reference to the TensorDatabase instance
        """
        if self.active:
            logger.warning("Agent is already active")
            return
        
        self.database = database_ref
        self.active = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Agent monitoring started")
    
    def stop_monitoring(self):
        """Stop the agent from monitoring and reconfiguring."""
        self.active = False
        logger.info("Agent monitoring stopped")
    
    def _monitoring_loop(self):
        """Background loop for monitoring database performance."""
        last_reconfiguration_time = 0
        
        while self.active:
            try:
                # Get current database metrics
                current_metrics = self.database.get_metrics()
                
                # Enrich metrics with additional information
                current_metrics.update({
                    "timestamp": time.time(),
                    "index_type": self.database.config.get("index_type", "flat"),
                    "memory_usage": self._get_memory_usage()
                })
                
                # Add to history
                self.metrics_history.append(current_metrics)
                
                # Convert to state representation
                current_state = self._metrics_to_state(current_metrics)
                self.current_state = current_state
                
                # Decide if reconfiguration is needed
                current_time = time.time()
                cooldown_passed = (current_time - last_reconfiguration_time) > self.config["reconfiguration_cooldown"]
                enough_samples = len(self.metrics_history) >= self.config["min_samples_before_learning"]
                
                if cooldown_passed and enough_samples:
                    # Decide on action using epsilon-greedy policy
                    if np.random.random() < self.exploration_rate:
                        # Explore: Random action
                        action = self._get_random_action()
                    else:
                        # Exploit: Best action according to Q-table
                        action = self._get_best_action(current_state)
                    
                    # Apply configuration
                    self._apply_configuration(action)
                    self.current_action = action
                    last_reconfiguration_time = current_time
                    
                    # Decay exploration rate
                    self.exploration_rate = max(0.05, self.exploration_rate * 0.95)
                
                # Sleep until next monitoring cycle
                time.sleep(self.config["monitoring_interval"])
                
                # After waiting, compute reward and update Q-table
                if self.current_action is not None:
                    new_metrics = self.database.get_metrics()
                    new_state = self._metrics_to_state(new_metrics)
                    reward = self._compute_reward(current_metrics, new_metrics)
                    
                    # Update Q-table
                    self._update_q_value(current_state, self.current_action, reward, new_state)
                    
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _metrics_to_state(self, metrics: Dict[str, Any]) -> str:
        """
        Convert metrics dictionary to a discretized state representation.
        
        Args:
            metrics: Dictionary of database metrics
            
        Returns:
            State representation as a string
        """
        # Extract and discretize relevant metrics
        state_values = []
        
        # Query rate (queries per second)
        if "queries" in metrics and "timestamp" in metrics and len(self.metrics_history) > 1:
            prev_metrics = self.metrics_history[-2]
            time_diff = metrics["timestamp"] - prev_metrics["timestamp"]
            query_diff = metrics["queries"] - prev_metrics["queries"]
            query_rate = query_diff / max(1, time_diff)
            state_values.append(self._discretize_value(query_rate, [0, 1, 10, 100]))
        else:
            state_values.append("low")
        
        # Insert rate (inserts per second)
        if "inserts" in metrics and "timestamp" in metrics and len(self.metrics_history) > 1:
            prev_metrics = self.metrics_history[-2]
            time_diff = metrics["timestamp"] - prev_metrics["timestamp"]
            insert_diff = metrics["inserts"] - prev_metrics["inserts"]
            insert_rate = insert_diff / max(1, time_diff)
            state_values.append(self._discretize_value(insert_rate, [0, 1, 10, 100]))
        else:
            state_values.append("low")
        
        # Average query time
        if "avg_search_time" in metrics:
            state_values.append(self._discretize_value(metrics["avg_search_time"], [0, 0.01, 0.1, 1.0]))
        else:
            state_values.append("medium")
        
        # Average insert time
        if "avg_insert_time" in metrics:
            state_values.append(self._discretize_value(metrics["avg_insert_time"], [0, 0.01, 0.1, 1.0]))
        else:
            state_values.append("medium")
        
        # Memory usage
        if "memory_usage" in metrics:
            state_values.append(self._discretize_value(metrics["memory_usage"], [0, 100, 500, 1000]))
        else:
            state_values.append("medium")
        
        # Index type
        if "index_type" in metrics:
            state_values.append(metrics["index_type"])
        else:
            state_values.append("flat")
        
        # Index size (number of tensors)
        if "inserts" in metrics:
            state_values.append(self._discretize_value(metrics["inserts"], [0, 100, 1000, 10000]))
        else:
            state_values.append("small")
        
        # Combine values into a state string
        return "|".join(str(v) for v in state_values)
    
    def _discretize_value(self, value: float, thresholds: List[float]) -> str:
        """
        Discretize a continuous value into a category.
        
        Args:
            value: The value to discretize
            thresholds: List of thresholds for discretization
            
        Returns:
            Discretized category as a string
        """
        categories = ["very_low", "low", "medium", "high", "very_high"]
        for i, threshold in enumerate(thresholds):
            if value < threshold:
                return categories[i]
        return categories[-1]
    
    def _get_random_action(self) -> Dict[str, Any]:
        """
        Generate a random action from the action space.
        
        Returns:
            Dictionary representing a random action
        """
        action = {}
        for key, values in self.action_space.items():
            action[key] = np.random.choice(values)
        return action
    
    def _get_best_action(self, state: str) -> Dict[str, Any]:
        """
        Get the best action for the current state according to Q-table.
        
        Args:
            state: Current state representation
            
        Returns:
            Dictionary representing the best action
        """
        if state not in self.q_table:
            return self._get_random_action()
        
        best_q = -float('inf')
        best_action = None
        
        for action_str, q_value in self.q_table[state].items():
            if q_value > best_q:
                best_q = q_value
                best_action = json.loads(action_str)
        
        if best_action is None:
            return self._get_random_action()
            
        return best_action
    
    def _apply_configuration(self, action: Dict[str, Any]):
        """
        Apply the selected configuration changes to the database.
        
        Args:
            action: Dictionary of configuration values to apply
        """
        logger.info(f"Applying configuration: {action}")
        
        # Update database configuration
        for key, value in action.items():
            if key == "index_type":
                self.database.config["index_type"] = value
                # If we have an existing indexer, we'll need to recreate it
                if self.database.indexer is not None:
                    # Save the current indexed tensors
                    existing_tensor_ids = self.database.indexer.tensor_ids
                    
                    # Create new indexer with the new index type
                    self.database.indexer = None
                    self.database._init_indexer(self.database.indexer.dimension)
                    
                    # Re-index all tensors
                    for tensor_id in existing_tensor_ids:
                        tensor, _ = self.database.storage.load_tensor(tensor_id)
                        self.database.indexer.add_tensor(tensor, tensor_id)
                    
                    logger.info(f"Re-indexed {len(existing_tensor_ids)} tensors with index type: {value}")
            
            elif key == "metric":
                # Metric change requires reindexing as well
                self.database.config["metric"] = value
                if self.database.indexer is not None:
                    existing_tensor_ids = self.database.indexer.tensor_ids
                    self.database.indexer = None
                    self.database._init_indexer(self.database.indexer.dimension)
                    
                    for tensor_id in existing_tensor_ids:
                        tensor, _ = self.database.storage.load_tensor(tensor_id)
                        self.database.indexer.add_tensor(tensor, tensor_id)
                    
                    logger.info(f"Re-indexed {len(existing_tensor_ids)} tensors with metric: {value}")
            
            elif key == "use_gpu":
                # Update GPU usage for processor and indexer
                if self.database.use_gpu != value:
                    self.database.use_gpu = value
                    
                    # Update processor
                    self.database.processor = None
                    self.database.processor = TensorProcessor(use_gpu=value)
                    
                    # Update indexer (requires reindexing)
                    if self.database.indexer is not None:
                        existing_tensor_ids = self.database.indexer.tensor_ids
                        dimension = self.database.indexer.dimension
                        self.database.indexer = None
                        self.database._init_indexer(dimension)
                        
                        for tensor_id in existing_tensor_ids:
                            tensor, _ = self.database.storage.load_tensor(tensor_id)
                            self.database.indexer.add_tensor(tensor, tensor_id)
                        
                        logger.info(f"Updated GPU usage to {value} and re-indexed {len(existing_tensor_ids)} tensors")
            
            elif key == "batch_size":
                # Update batch processing size
                self.database.config["batch_size"] = value
                logger.info(f"Updated batch size to {value}")
            
            elif key == "compression_level":
                # Update compression level for storage
                self.database.config["compression_level"] = value
                logger.info(f"Updated compression level to {value}")
    
    def _compute_reward(self, old_metrics: Dict[str, Any], new_metrics: Dict[str, Any]) -> float:
        """
        Compute reward based on improvement in metrics.
        
        Args:
            old_metrics: Metrics before reconfiguration
            new_metrics: Metrics after reconfiguration
            
        Returns:
            Reward value
        """
        reward = 0
        weights = self.config["reward_weights"]
        
        # Query time improvement (lower is better)
        if "avg_search_time" in old_metrics and "avg_search_time" in new_metrics:
            old_time = old_metrics["avg_search_time"]
            new_time = new_metrics["avg_search_time"]
            if old_time > 0:
                improvement = (old_time - new_time) / old_time
                reward += weights["query_time_improvement"] * improvement
        
        # Insert time improvement (lower is better)
        if "avg_insert_time" in old_metrics and "avg_insert_time" in new_metrics:
            old_time = old_metrics["avg_insert_time"]
            new_time = new_metrics["avg_insert_time"]
            if old_time > 0:
                improvement = (old_time - new_time) / old_time
                reward += weights["insert_time_improvement"] * improvement
        
        # Memory efficiency (lower is better)
        if "memory_usage" in old_metrics and "memory_usage" in new_metrics:
            old_memory = old_metrics["memory_usage"]
            new_memory = new_metrics["memory_usage"]
            if old_memory > 0:
                improvement = (old_memory - new_memory) / old_memory
                reward += weights["memory_efficiency"] * improvement
        
        # Stability (fewer errors is better)
        # If we had no errors during reconfiguration, add a small reward
        reward += weights["stability"] * 0.1
        
        return reward
    
    def _update_q_value(self, state: str, action: Dict[str, Any], reward: float, next_state: str):
        """
        Update Q-value using Q-learning algorithm.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Convert action to string for dictionary key
        action_str = json.dumps(action, sort_keys=True)
        
        # Ensure state exists in Q-table
        if state not in self.q_table:
            self.q_table[state] = {}
        
        # Ensure action exists for this state
        if action_str not in self.q_table[state]:
            self.q_table[state][action_str] = 0
        
        # Find max Q-value for next state
        max_next_q = 0
        if next_state in self.q_table:
            max_next_q = max(self.q_table[next_state].values(), default=0)
        
        # Q-learning update
        current_q = self.q_table[state][action_str]
        new_q = current_q + self.learning_rate * (reward + self.reward_decay * max_next_q - current_q)
        self.q_table[state][action_str] = new_q
        
        logger.debug(f"Updated Q-value: state={state}, action={action_str}, reward={reward:.4f}, new_q={new_q:.4f}")
    
    def _get_memory_usage(self) -> float:
        """
        Estimate memory usage of the database.
        
        Returns:
            Memory usage in MB
        """
        # This is a simple estimate; in a real implementation, we'd use more
        # sophisticated memory tracking
        memory_usage = 0
        
        try:
            # Estimate storage memory
            tensors = self.database.list_tensors()
            for tensor_info in tensors:
                shape = tensor_info["metadata"].get("shape", [])
                if shape:
                    # Rough estimate of tensor size in MB
                    tensor_size = np.prod(shape) * 4 / (1024 * 1024)  # Assuming float32
                    memory_usage += tensor_size
            
            # Add index memory estimate
            if self.database.indexer is not None:
                # Rough estimate based on index type and dimension
                index_memory = len(self.database.indexer.tensor_ids) * self.database.indexer.dimension * 4 / (1024 * 1024)
                if self.database.indexer.index_type == "hnsw":
                    # HNSW index typically uses more memory
                    index_memory *= 1.5
                memory_usage += index_memory
        except Exception as e:
            logger.error(f"Error estimating memory usage: {e}")
        
        return memory_usage
    
    def save_state(self, filepath: Optional[str] = None):
        """
        Save agent state to disk.
        
        Args:
            filepath: Path to save the state (default: use checkpoint_path)
        """
        if filepath is None:
            if self.checkpoint_path is None:
                logger.warning("No checkpoint path specified, cannot save agent state")
                return
            filepath = os.path.join(self.checkpoint_path, f"agent_state_{int(time.time())}.json")
        
        try:
            state = {
                "q_table": self.q_table,
                "learning_rate": self.learning_rate,
                "exploration_rate": self.exploration_rate,
                "reward_decay": self.reward_decay,
                "metrics_history": self.metrics_history[-100:],  # Only save recent history
                "config": self.config
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Agent state saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving agent state: {e}")
    
    def load_state(self, filepath: str):
        """
        Load agent state from disk.
        
        Args:
            filepath: Path to load the state from
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.q_table = state.get("q_table", {})
            self.learning_rate = state.get("learning_rate", 0.01)
            self.exploration_rate = state.get("exploration_rate", 0.2)
            self.reward_decay = state.get("reward_decay", 0.9)
            self.metrics_history = state.get("metrics_history", [])
            self.config.update(state.get("config", {}))
            
            logger.info(f"Agent state loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading agent state: {e}") 