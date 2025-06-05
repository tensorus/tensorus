import random
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base import TensorusModel


class ReplayBuffer:
    """Simple replay buffer for storing transitions."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Tuple[Any, int, float, Any, bool]] = []
        self.position = 0

    def push(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(list, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self) -> int:
        return len(self.buffer)


class QLearningModel(TensorusModel):
    """Tabular Q-learning for discrete state/action spaces."""

    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float32)

    def fit(self, env: Any, episodes: int = 10) -> None:
        for _ in range(episodes):
            state = int(env.reset())
            done = False
            while not done:
                if random.random() < self.epsilon:
                    action = random.randrange(self.n_actions)
                else:
                    action = int(np.argmax(self.q_table[state]))
                next_state, reward, done, _ = env.step(action)
                next_state = max(0, min(self.n_states - 1, int(next_state)))
                best_next = np.max(self.q_table[next_state])
                td_target = reward + self.gamma * best_next * (1 - done)
                td_error = td_target - self.q_table[state, action]
                self.q_table[state, action] += self.alpha * td_error
                state = next_state

    def predict(self, state: Any) -> int:
        state = int(state)
        return int(np.argmax(self.q_table[state]))

    def save(self, path: str) -> None:
        np.save(path, self.q_table)

    def load(self, path: str) -> None:
        self.q_table = np.load(path)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.float()
    else:
        t = torch.tensor(x, dtype=torch.float32)
    if t.dim() == 0:
        t = t.unsqueeze(0)
    return t


class DQNModel(TensorusModel):
    """Deep Q-Network implementation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64, lr: float = 1e-3,
                 gamma: float = 0.99, epsilon: float = 0.1, batch_size: int = 32, buffer_size: int = 10000,
                 target_update: int = 10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_update = target_update

        self.policy_net = QNetwork(state_dim, action_dim, hidden_size)
        self.target_net = QNetwork(state_dim, action_dim, hidden_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.steps = 0

    def _optimize(self) -> None:
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.stack([_to_tensor(s) for s in states])
        actions = torch.tensor(actions).long().unsqueeze(1)
        rewards = torch.tensor(rewards).float()
        next_states = torch.stack([_to_tensor(s) for s in next_states])
        dones = torch.tensor(dones).float()

        q_values = self.policy_net(states).gather(1, actions).squeeze()
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)
        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def fit(self, env: Any, episodes: int = 10) -> None:
        for ep in range(episodes):
            state = env.reset()
            done = False
            while not done:
                if random.random() < self.epsilon:
                    action = random.randrange(self.action_dim)
                else:
                    with torch.no_grad():
                        q_vals = self.policy_net(_to_tensor(state))
                        action = int(torch.argmax(q_vals).item())
                next_state, reward, done, _ = env.step(action)
                self.buffer.push(_to_tensor(state), action, reward, _to_tensor(next_state), done)
                state = next_state
                self._optimize()
                if self.steps % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                self.steps += 1

    def predict(self, state: Any) -> int:
        with torch.no_grad():
            q_vals = self.policy_net(_to_tensor(state))
            return int(torch.argmax(q_vals).item())

    def save(self, path: str) -> None:
        torch.save({"policy": self.policy_net.state_dict(),
                    "target": self.target_net.state_dict()}, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        self.policy_net.load_state_dict(data["policy"])
        self.target_net.load_state_dict(data["target"])


def _actor_critic_networks(state_dim: int, action_dim: int, hidden: int = 64):
    actor = nn.Sequential(
        nn.Linear(state_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, action_dim)
    )
    critic = nn.Sequential(
        nn.Linear(state_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1)
    )
    return actor, critic


class A2CModel(TensorusModel):
    """Simplified Advantage Actor-Critic."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64, lr: float = 1e-3, gamma: float = 0.99):
        self.gamma = gamma
        self.actor, self.critic = _actor_critic_networks(state_dim, action_dim, hidden_size)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

    def fit(self, env: Any, episodes: int = 10) -> None:
        for _ in range(episodes):
            state = env.reset()
            done = False
            while not done:
                state_t = _to_tensor(state)
                probs = torch.softmax(self.actor(state_t), dim=-1)
                action = torch.multinomial(probs, 1).item()
                next_state, reward, done, _ = env.step(action)
                next_state_t = _to_tensor(next_state)
                value = self.critic(state_t)
                next_value = self.critic(next_state_t).detach()
                advantage = reward + self.gamma * (1 - done) * next_value - value
                actor_loss = -torch.log(probs[action]) * advantage.detach()
                critic_loss = advantage.pow(2)
                loss = actor_loss + critic_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                state = next_state

    def predict(self, state: Any) -> int:
        with torch.no_grad():
            probs = torch.softmax(self.actor(_to_tensor(state)), dim=-1)
            return int(torch.argmax(probs).item())

    def save(self, path: str) -> None:
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])


class PPOModel(A2CModel):
    """Proximal Policy Optimization - minimal variant."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64, lr: float = 1e-3,
                 gamma: float = 0.99, clip_epsilon: float = 0.2):
        super().__init__(state_dim, action_dim, hidden_size, lr, gamma)
        self.clip_epsilon = clip_epsilon

    def fit(self, env: Any, episodes: int = 10) -> None:
        for _ in range(episodes):
            state = env.reset()
            done = False
            while not done:
                state_t = _to_tensor(state)
                probs = torch.softmax(self.actor(state_t), dim=-1)
                action = torch.multinomial(probs, 1).item()
                log_prob_old = torch.log(probs[action])
                next_state, reward, done, _ = env.step(action)
                next_state_t = _to_tensor(next_state)
                value = self.critic(state_t)
                next_value = self.critic(next_state_t).detach()
                advantage = reward + self.gamma * (1 - done) * next_value - value
                probs_new = torch.softmax(self.actor(state_t), dim=-1)
                log_prob_new = torch.log(probs_new[action])
                ratio = torch.exp(log_prob_new - log_prob_old.detach())
                actor_loss = -torch.min(ratio * advantage.detach(),
                                       torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage.detach())
                critic_loss = advantage.pow(2)
                loss = actor_loss + critic_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                state = next_state


class TRPOModel(A2CModel):
    """Placeholder Trust Region Policy Optimization."""

    def fit(self, env: Any, episodes: int = 1) -> None:
        # For brevity we reuse A2C updates. Real TRPO requires complex trust region steps.
        super().fit(env, episodes)


class SACModel(TensorusModel):
    """Simple Soft Actor Critic for discrete actions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64, lr: float = 1e-3,
                 gamma: float = 0.99, alpha: float = 0.2, buffer_size: int = 10000, batch_size: int = 32):
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.policy_net, self.q_net = _actor_critic_networks(state_dim, action_dim, hidden_size)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) +
                                    list(self.q_net.parameters()) +
                                    list(self.value_net.parameters()), lr=lr)
        self.replay = ReplayBuffer(buffer_size)

    def _sample_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        logits = self.policy_net(state)
        prob = torch.softmax(logits, dim=-1)
        action = torch.multinomial(prob, 1).item()
        log_prob = torch.log(prob[action])
        return action, log_prob

    def _update(self):
        if len(self.replay) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states = torch.stack([_to_tensor(s) for s in states])
        actions = torch.tensor(actions).long()
        rewards = torch.tensor(rewards).float()
        next_states = torch.stack([_to_tensor(s) for s in next_states])
        dones = torch.tensor(dones).float()

        q_vals = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_logits = self.policy_net(next_states)
            next_prob = torch.softmax(next_logits, dim=-1)
            next_log_prob = torch.log(next_prob + 1e-8)
            next_q = self.q_net(next_states)
            next_v = (next_prob * (next_q - self.alpha * next_log_prob)).sum(dim=1)
            target_q = rewards + self.gamma * next_v * (1 - dones)
        q_loss = F.mse_loss(q_vals, target_q)

        logits = self.policy_net(states)
        prob = torch.softmax(logits, dim=-1)
        log_prob = torch.log(prob + 1e-8)
        q_new = self.q_net(states)
        policy_loss = (prob * (self.alpha * log_prob - q_new)).sum(dim=1).mean()

        self.optimizer.zero_grad()
        (q_loss + policy_loss).backward()
        self.optimizer.step()

    def fit(self, env: Any, episodes: int = 10) -> None:
        for _ in range(episodes):
            state = env.reset()
            done = False
            while not done:
                state_t = _to_tensor(state)
                action, log_prob = self._sample_action(state_t)
                next_state, reward, done, _ = env.step(action)
                self.replay.push(state_t, action, reward, _to_tensor(next_state), done)
                self._update()
                state = next_state

    def predict(self, state: Any) -> int:
        with torch.no_grad():
            logits = self.policy_net(_to_tensor(state))
            return int(torch.argmax(logits).item())

    def save(self, path: str) -> None:
        torch.save({"policy": self.policy_net.state_dict(),
                    "q_net": self.q_net.state_dict(),
                    "value_net": self.value_net.state_dict()}, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        self.policy_net.load_state_dict(data["policy"])
        self.q_net.load_state_dict(data["q_net"])
        self.value_net.load_state_dict(data["value_net"])
