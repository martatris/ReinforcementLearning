import random, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# ----------------------------
# Environment setup
# ----------------------------
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
state_size = env.observation_space.n
action_size = env.action_space.n
print(f"FrozenLake-v1 | states={state_size} | actions={action_size}\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Neural network model
# ----------------------------
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------
# Prioritized Replay Buffer
# ----------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights,
        )

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

# ----------------------------
# Double DQN Agent
# ----------------------------
class DoubleDQNAgent:
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.policy_net = QNetwork(state_size, action_size).to(device)
        self.target_net = QNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = PrioritizedReplayBuffer(5000)
        self.batch_size = 64
        self.gamma = 0.99
        self.beta = 0.4
        self.tau = 0.01  # soft target update

    def one_hot(self, state):
        s = np.zeros(self.state_size, dtype=np.float32)
        s[state] = 1.0
        return s

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        s = torch.FloatTensor(self.one_hot(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qvals = self.policy_net(s)
        return int(torch.argmax(qvals).item())

    def store(self, s, a, r, s2, done):
        self.memory.push(s, a, r, s2, done)

    def soft_update_target(self):
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(
            self.batch_size, beta=self.beta
        )

        # Convert to tensors
        s = torch.FloatTensor([self.one_hot(x) for x in states]).to(self.device)
        s2 = torch.FloatTensor([self.one_hot(x) for x in next_states]).to(self.device)
        a = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        d = torch.FloatTensor(dones.astype(np.float32)).unsqueeze(1).to(self.device)
        w = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # --- Double DQN Update ---
        # 1. Current Q estimates
        q_values = self.policy_net(s).gather(1, a)

        # 2. Action selection by policy_net
        next_actions = self.policy_net(s2).argmax(1, keepdim=True)

        # 3. Q target from target_net
        with torch.no_grad():
            next_q = self.target_net(s2).gather(1, next_actions)
            q_target = r + self.gamma * next_q * (1 - d)

        # 4. Compute TD error
        td_error = q_target - q_values

        # 5. Loss weighted by importance sampling
        loss = (w * td_error.pow(2)).mean()

        # 6. Gradient update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 7. Update priorities
        new_priorities = torch.abs(td_error).detach().cpu().numpy() + 1e-5
        self.memory.update_priorities(indices, new_priorities.flatten())

        # 8. Soft target update
        self.soft_update_target()

        return loss.item()

# ----------------------------
# Training loop
# ----------------------------
def train(agent, episodes=2000, max_steps=100, eps_start=1.0, eps_end=0.01, eps_decay=0.001):
    epsilon = eps_start
    rewards_hist, losses = [], []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        for _ in range(max_steps):
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store(state, action, reward, next_state, done)

            loss = agent.train_step()
            if loss:
                losses.append(loss)

            total_reward += reward
            state = next_state

            if done:
                break

        epsilon = max(eps_end, epsilon * math.exp(-eps_decay))
        rewards_hist.append(total_reward)

        if ep % 200 == 0 and ep > 0:
            avg_r = np.mean(rewards_hist[-200:])
            print(f"Episode {ep}/{episodes}, ε={epsilon:.3f}, avg_reward={avg_r:.3f}")

    return rewards_hist, losses

# ----------------------------
# Run training
# ----------------------------
agent = DoubleDQNAgent(state_size, action_size, DEVICE)
print("Training Double DQN + PER ...\n")
rewards, losses = train(agent, episodes=2000)
print("\n✅ Training complete!")

# ----------------------------
# Evaluate performance
# ----------------------------
def evaluate(agent, episodes=200):
    wins = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        for _ in range(100):
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            if done:
                wins += int(reward == 1)
                break
    return wins / episodes

win_rate = evaluate(agent, episodes=200)
print(f"\n🏁 Win rate (Double DQN + PER): {win_rate*100:.2f}%")

# ----------------------------
# Render demo episode
# ----------------------------
print("\nDemo Episode:\n")
state, _ = env.reset()
for step in range(50):
    print(env.render())
    time.sleep(0.3)
    action = agent.select_action(state, epsilon=0.0)
    state, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        print(env.render())
        if reward == 1:
            print("🎯 Reached the goal!")
        else:
            print("💀 Fell into a hole!")
        break

env.close()