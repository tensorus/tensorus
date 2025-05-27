# rl_agent.py
"""
Implements a Deep Q-Network (DQN) Reinforcement Learning agent for Tensorus.

The agent interacts with an environment, stores experiences (S, A, R, S', D)
in TensorStorage, samples experiences, and trains its Q-network.

Note on Experience Storage:
- Large tensors (state, next_state) are stored individually in a 'rl_states' dataset.
- Experience tuples containing IDs of states/next_states and scalar action/reward/done
  are stored as metadata in a placeholder tensor within the 'rl_experiences' dataset.
- This approach balances tensor-native storage with manageable metadata, but sampling
  requires retrieving linked state tensors, which might be slow depending on storage backend.
"""
from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import logging
from typing import Tuple, Optional, Dict, Any

# Import necessary Tensorus components
from .tensor_storage import TensorStorage
from .dummy_env import DummyEnv # Import our dummy environment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- DQN Network Definition ---
class DQN(nn.Module):
    """Simple MLP Q-Network."""
    def __init__(self, n_observations: int, n_actions: int, hidden_size: int = 128):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns Q-values for each action."""
        # Ensure input is float
        if x.dtype != torch.float32:
             x = x.float()
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# --- RL Agent Class ---
class RLAgent:
    """DQN Agent interacting with TensorStorage."""

    def __init__(self,
                 tensor_storage: TensorStorage,
                 state_dim: int,
                 action_dim: int,
                 hidden_size: int = 128,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 0.9,
                 epsilon_end: float = 0.05,
                 epsilon_decay: int = 10000,
                 target_update_freq: int = 500,
                 batch_size: int = 128,
                 experience_dataset: str = "rl_experiences",
                 state_dataset: str = "rl_states"):
        """
        Initializes the RL Agent.

        Args:
            tensor_storage: Instance of TensorStorage.
            state_dim: Dimensionality of the environment state.
            action_dim: Number of discrete actions.
            hidden_size: Hidden layer size for the DQN.
            lr: Learning rate for the optimizer.
            gamma: Discount factor for future rewards.
            epsilon_*: Epsilon-greedy exploration parameters.
            target_update_freq: How often (in steps) to update the target network.
            batch_size: Number of experiences to sample for training.
            experience_dataset: Name of the dataset to store experience metadata.
            state_dataset: Name of the dataset to store state/next_state tensors.
        """
        if not isinstance(tensor_storage, TensorStorage):
            raise TypeError("tensor_storage must be an instance of TensorStorage")

        self.tensor_storage = tensor_storage
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.experience_dataset = experience_dataset
        self.state_dataset = state_dataset

        # Ensure datasets exist
        for ds_name in [self.experience_dataset, self.state_dataset]:
             try:
                 self.tensor_storage.get_dataset(ds_name)
             except ValueError:
                 logger.info(f"Dataset '{ds_name}' not found. Creating it.")
                 self.tensor_storage.create_dataset(ds_name)

        # Device configuration (use GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"RL Agent using device: {self.device}")

        # Networks
        self.policy_net = DQN(state_dim, action_dim, hidden_size).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is only for inference

        # Optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

        self.steps_done = 0 # Counter for epsilon decay and target updates


    def select_action(self, state: torch.Tensor) -> int:
        """Selects an action using epsilon-greedy strategy."""
        sample = random.random()
        # Calculate current epsilon
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)

        self.steps_done += 1 # Increment step counter here

        if sample > eps_threshold:
            # Exploit: choose the best action from the policy network
            with torch.no_grad():
                # Ensure state is on the correct device and has batch dimension
                state = state.unsqueeze(0).to(self.device) if state.ndim == 1 else state.to(self.device)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].view(1, 1).item() # Get index of max Q value
                logger.debug(f"Exploiting: Q-Values={q_values.cpu().numpy()}, Chosen Action={action}")
                return action
        else:
            # Explore: choose a random action
            action = random.randrange(self.action_dim)
            logger.debug(f"Exploring: Chosen Action={action}")
            return action


    def store_experience(self, state: torch.Tensor, action: int, reward: float, next_state: Optional[torch.Tensor], done: bool) -> None:
        """Stores an experience tuple in TensorStorage."""
        if state is None:
             logger.error("Cannot store experience with None state.")
             return

        # 1. Store state tensor
        state_metadata = {"component": "state", "created_by": "RLAgent"}
        state_id = self.tensor_storage.insert(self.state_dataset, state.cpu(), metadata=state_metadata)

        # 2. Store next_state tensor (if not None)
        next_state_id = None
        if next_state is not None:
            next_state_metadata = {"component": "next_state", "created_by": "RLAgent"}
            next_state_id = self.tensor_storage.insert(self.state_dataset, next_state.cpu(), metadata=next_state_metadata)

        # 3. Create experience metadata
        experience_metadata = {
            "state_id": state_id,
            "action": action, # Store action directly (assuming discrete & scalar)
            "reward": reward, # Store reward directly
            "next_state_id": next_state_id, # Can be None if done
            "done": int(done), # Store boolean as int
            "created_by": "RLAgent" # Mark the experience record itself
        }

        # 4. Store placeholder tensor with experience metadata
        # Using a small tensor as the primary data for this record
        placeholder_tensor = torch.tensor([1.0])
        exp_record_id = self.tensor_storage.insert(self.experience_dataset, placeholder_tensor, experience_metadata)
        logger.debug(f"Stored experience record {exp_record_id}: state_id={state_id}, action={action}, reward={reward:.2f}, next_state_id={next_state_id}, done={done}")


    def optimize_model(self) -> None:
        """Performs one step of optimization on the policy network."""
        # Check if enough samples are available in the experience dataset
        try:
            # A bit inefficient to get the full list just for the count,
            # TensorStorage could be enhanced with a count method.
            experience_count = len(self.tensor_storage.get_dataset(self.experience_dataset))
        except ValueError:
             experience_count = 0

        if experience_count < self.batch_size:
            logger.debug(f"Not enough experiences ({experience_count}/{self.batch_size}) to optimize yet.")
            return

        # 1. Sample experience metadata records
        try:
             sampled_metadata_records = self.tensor_storage.sample_dataset(self.experience_dataset, self.batch_size)
        except ValueError:
             logger.error(f"Could not sample from dataset {self.experience_dataset}")
             return
        except Exception as e:
             logger.error(f"Error sampling experiences: {e}", exc_info=True)
             return

        # 2. Retrieve actual state/next_state tensors based on IDs in metadata
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        non_final_mask_list = [] # Keep track of which next_states are not None

        for record in sampled_metadata_records:
            meta = record['metadata']
            state_id = meta.get('state_id')
            next_state_id = meta.get('next_state_id')
            action = meta.get('action')
            reward = meta.get('reward')
            done_flag = bool(meta.get('done', 1)) # Default to True if missing? Risky. Assume present.

            # Basic validation
            if state_id is None or action is None or reward is None:
                logger.warning(f"Skipping invalid sampled record: {meta}")
                continue

            # Retrieve state
            state_record = self.tensor_storage.get_tensor_by_id(self.state_dataset, state_id)
            if state_record is None:
                logger.warning(f"Could not find state tensor with ID {state_id} for experience {meta.get('record_id')}. Skipping sample.")
                continue
            states.append(state_record['tensor'])

            # Retrieve next state if it exists (i.e., not a terminal state)
            current_next_state = None
            if not done_flag and next_state_id:
                 next_state_record = self.tensor_storage.get_tensor_by_id(self.state_dataset, next_state_id)
                 if next_state_record:
                      current_next_state = next_state_record['tensor']
                 else:
                      # This shouldn't happen if storage is consistent, but handle it
                      logger.warning(f"Could not find next_state tensor with ID {next_state_id} for non-terminal experience {meta.get('record_id')}. Treating as terminal.")
                      done_flag = True # Treat as done if next state is missing

            next_states.append(current_next_state) # Will be None for terminal states
            non_final_mask_list.append(not done_flag)

            actions.append(torch.tensor([[action]], dtype=torch.long)) # Action needs to be [[action]] for gather()
            rewards.append(torch.tensor([reward], dtype=torch.float32))
            dones.append(done_flag) # Keep Python bool for now, convert later if needed


        # If not enough valid samples were retrieved after lookup
        if not states:
             logger.warning("No valid samples retrieved after state lookup. Optimization step skipped.")
             return

        # 3. Batch the retrieved data
        # Filter out None values in next_states for target Q calculation
        non_final_next_states = torch.cat([ns for ns in next_states if ns is not None]).to(self.device) if any(non_final_mask_list) else None
        state_batch = torch.cat(states).to(self.device)
        action_batch = torch.cat(actions).to(self.device)
        reward_batch = torch.cat(rewards).to(self.device)
        non_final_mask = torch.tensor(non_final_mask_list, dtype=torch.bool, device=self.device)


        # 4. Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        # Ensure state_batch has correct dimensions if needed (e.g., B x C x H x W for images)
        # Our dummy env state is simple (B x 1)
        if state_batch.ndim == 1: # Ensure batch dimension exists
             state_batch = state_batch.unsqueeze(-1)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)


        # 5. Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        next_state_values = torch.zeros(len(states), device=self.device) # Start with zeros for all
        if non_final_next_states is not None and non_final_next_states.numel() > 0:
            with torch.no_grad():
                if non_final_next_states.ndim == 1: # Ensure batch dimension
                     non_final_next_states = non_final_next_states.unsqueeze(-1)
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]


        # 6. Compute the expected Q values (Bellman equation)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # 7. Compute loss
        criterion = nn.SmoothL1Loss() # Huber loss
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)) # Ensure target has same shape


        # 8. Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        logger.debug(f"Optimization step done. Loss: {loss.item():.4f}")


        # 9. Periodically update the target network
        if self.steps_done % self.target_update_freq == 0:
            self._update_target_network()


    def _update_target_network(self):
        """Copies weights from policy_net to target_net."""
        logger.info(f"Updating target network at step {self.steps_done}")
        self.target_net.load_state_dict(self.policy_net.state_dict())


    def train(self, env: DummyEnv, num_episodes: int):
        """Runs the training loop for a number of episodes."""
        logger.info(f"--- Starting Training for {num_episodes} episodes ---")
        episode_rewards = []

        for i_episode in range(num_episodes):
            state = env.reset()
            # state should be a tensor from env.reset()
            # state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0) # Add batch dim
            # Note: env.reset now returns a tensor, no need to convert here

            done = False
            current_episode_reward = 0
            steps_in_episode = 0

            while not done:
                # Select action
                action = self.select_action(state) # state is already a tensor

                # Perform action in environment
                next_state, reward, done, _ = env.step(action) # next_state is a tensor
                current_episode_reward += reward
                steps_in_episode += 1

                # Store experience in TensorStorage
                # Note: store_experience handles moving tensors to CPU for storage if needed
                self.store_experience(state, action, reward, next_state if not done else None, done)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Check if env forces done (e.g. max steps reached in DummyEnv)
                if done:
                     break # Exit loop if env says done

            episode_rewards.append(current_episode_reward)
            logger.info(f"Episode {i_episode+1}/{num_episodes} finished after {steps_in_episode} steps. Reward: {current_episode_reward:.2f}. Epsilon: {self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.steps_done / self.epsilon_decay):.3f}")

            # Optional: Add plotting or saving logic here
            if (i_episode + 1) % 50 == 0: # Log average reward periodically
                 avg_reward = sum(episode_rewards[-50:]) / len(episode_rewards[-50:])
                 logger.info(f"  Average reward over last 50 episodes: {avg_reward:.2f}")


        logger.info("--- Training Finished ---")
        return episode_rewards


# --- Example Usage ---
if __name__ == "__main__":
    logger.info("--- Starting RL Agent Example ---")

    # 1. Setup TensorStorage
    storage = TensorStorage()

    # 2. Setup Dummy Environment
    env = DummyEnv(max_steps=100) # Episodes are max 100 steps

    # 3. Create the RL Agent
    agent = RLAgent(
        tensor_storage=storage,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_size=64, # Smaller network for simple env
        lr=5e-4,
        gamma=0.95,
        epsilon_start=0.95,
        epsilon_end=0.05,
        epsilon_decay=20000, # Slower decay for demonstration
        target_update_freq=200, # Update target net less frequently
        batch_size=64, # Smaller batch size
        experience_dataset="dummy_env_experiences", # Use specific names
        state_dataset="dummy_env_states"
    )

    # 4. Run the training loop
    num_episodes_to_run = 200 # Adjust as needed
    rewards = agent.train(env, num_episodes=num_episodes_to_run)


    # 5. Optional: Check TensorStorage contents
    print("\n--- Checking TensorStorage contents (Sample) ---")
    try:
        exp_count = len(storage.get_dataset(agent.experience_dataset))
        state_count = len(storage.get_dataset(agent.state_dataset))
        print(f"Found {exp_count} experience records in '{agent.experience_dataset}'.")
        print(f"Found {state_count} state records in '{agent.state_dataset}'.")

        if exp_count > 0:
            print("\nExample experience record (metadata):")
            sample_exp = storage.sample_dataset(agent.experience_dataset, 1)
            if sample_exp:
                 print(sample_exp[0]['metadata'])
                 state_id = sample_exp[0]['metadata'].get('state_id')
                 if state_id:
                      state_rec = storage.get_tensor_by_id(agent.state_dataset, state_id)
                      if state_rec:
                           print(f" -> Corresponding state tensor (retrieved): {state_rec['tensor']}")

    except ValueError as e:
        print(f"Could not retrieve datasets: {e}")
    except Exception as e:
         print(f"An error occurred checking storage: {e}")


    logger.info("--- RL Agent Example Finished ---")

    # Optional: Plot rewards
    try:
         import matplotlib.pyplot as plt
         plt.figure(figsize=(10, 5))
         plt.plot(rewards)
         # Calculate a simple moving average
         moving_avg = [sum(rewards[max(0, i-20):i+1])/len(rewards[max(0, i-20):i+1]) for i in range(len(rewards))]
         plt.plot(moving_avg, linestyle='--', label='Moving Avg (20 episodes)')
         plt.title("Episode Rewards over Time")
         plt.xlabel("Episode")
         plt.ylabel("Total Reward")
         plt.grid(True)
         plt.legend()
         print("\nPlotting rewards... Close the plot window to exit.")
         plt.show()

    except ImportError:
         print("\nMatplotlib not found. Skipping reward plot. Install with: pip install matplotlib")