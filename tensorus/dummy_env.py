# dummy_env.py
"""
A simple dummy environment for testing the RL agent.
State: Position (1D)
Action: Move left (-1), Stay (0), Move right (+1) (Discrete actions)
Goal: Reach position 0
Reward: -abs(position), +10 if at goal
"""
from typing import Tuple, Dict
import torch
import random
import numpy as np # Use numpy for state representation convenience

class DummyEnv:
    def __init__(self, max_steps=50):
        self.state_dim = 1  # Position is the only state variable
        self.action_dim = 3 # Actions: 0 (left), 1 (stay), 2 (right)
        self.max_steps = max_steps
        self.current_pos = 0.0
        self.steps_taken = 0
        self.goal_pos = 0.0
        self.max_pos = 5.0 # Boundaries

    def reset(self) -> torch.Tensor:
        """Resets the environment to a random starting position."""
        self.current_pos = random.uniform(-self.max_pos, self.max_pos)
        self.steps_taken = 0
        # Return state as a PyTorch tensor
        return torch.tensor([self.current_pos], dtype=torch.float32)

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Takes an action, updates the state, and returns results."""
        if not isinstance(action, int) or action not in [0, 1, 2]:
             raise ValueError(f"Invalid action: {action}. Must be 0, 1, or 2.")

        # Update position based on action
        if action == 0: # Move left
            self.current_pos -= 0.5
        elif action == 2: # Move right
             self.current_pos += 0.5
        # Action 1 (stay) does nothing to position

        # Clip position to boundaries
        self.current_pos = np.clip(self.current_pos, -self.max_pos, self.max_pos)

        self.steps_taken += 1

        # Calculate reward
        # Higher reward closer to the goal, large penalty for being far
        reward = -abs(self.current_pos - self.goal_pos) * 0.1 # Small penalty for distance
        done = False

        # Check if goal is reached (within a small tolerance)
        if abs(self.current_pos - self.goal_pos) < 0.1:
             reward += 10.0 # Bonus for reaching goal
             done = True

        # Check if max steps exceeded
        if self.steps_taken >= self.max_steps:
            done = True
            # Optional: small penalty for running out of time
            # reward -= 1.0

        # Return next state, reward, done flag, and info dict
        next_state = torch.tensor([self.current_pos], dtype=torch.float32)
        info = {} # Empty info dict for now

        return next_state, float(reward), done, info

# Example Usage
if __name__ == "__main__":
    env = DummyEnv()
    state = env.reset()
    print(f"Initial state: {state.item()}")
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = random.choice([0, 1, 2]) # Take random action
        next_state, reward, done, _ = env.step(action)
        print(f"Step {steps+1}: Action={action}, Next State={next_state.item():.2f}, Reward={reward:.2f}, Done={done}")
        state = next_state
        total_reward += reward
        steps += 1
        if steps > env.max_steps + 5: # Safety break
             print("Exceeded max steps significantly, breaking.")
             break


    print(f"\nEpisode finished after {steps} steps. Total reward: {total_reward:.2f}")