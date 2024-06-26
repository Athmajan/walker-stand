import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dm_control import suite
from torch.distributions import Normal
import matplotlib.pyplot as plt

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim * 2)  # *2 for mean and std
        self.action_dim = action_dim
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        mean, log_std = torch.chunk(x, 2, dim=-1)  # Split into mean and log_std
        std = torch.exp(log_std)  # Standard deviation is exponential of log_std
        return mean, std

# Function to sample actions from Gaussian distribution
def sample_action(mean, std):
    normal = Normal(mean, std)
    action = normal.sample()
    return action.clamp(-1.0, 1.0)  # Clamp action to ensure it stays within [-1, 1]

def select_action(policy, state):
    if not isinstance(state, torch.Tensor):
        state = torch.from_numpy(state).float()
    mean, std = policy(state)
    action = sample_action(mean, std)
    return action, mean, std

def normalize_reward(reward):
    return reward / 100.0

def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)

# Compute discounted rewards
def compute_discounted_rewards(rewards, gamma):
    discounted_rewards = []
    cumulative_reward = 0
    for reward in reversed(rewards):
        cumulative_reward = reward + gamma * cumulative_reward
        discounted_rewards.insert(0, cumulative_reward)
    return discounted_rewards

def visualize(env):
    frameA = np.hstack([env.physics.render(480, 480, camera_id=0),
                        env.physics.render(480, 480, camera_id=1)])
    plt.imshow(frameA)
    plt.pause(0.01)  # Need min display time > 0.0.
    plt.axis('off') 
    plt.draw()      
    plt.close() 
    return


# Training the policy network
def train(policy, optimizer, episodes=1000, gamma=0.99, save_path='policy_network.pth'):
    env = suite.load(domain_name="walker", task_name="stand")
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        log_probs = []
        rewards = []
        states = []
        
        while not state.last():
            obs = state.observation
            orientations_tensor = torch.tensor(obs['orientations'], dtype=torch.float32)
            height_tensor = torch.tensor([obs['height']], dtype=torch.float32)
            velocity_tensor = torch.tensor(obs['velocity'], dtype=torch.float32)

            input_tensor = torch.cat((orientations_tensor, height_tensor, velocity_tensor), dim=0)
            
            action, mean, std = select_action(policy, input_tensor)
            next_state = env.step(action.numpy())
            reward = normalize_reward(next_state.reward)
            
            log_prob = Normal(mean, std).log_prob(action).sum()
            log_probs.append(log_prob)
            rewards.append(reward)
            episode_reward += reward
            state = next_state
            print(state.last())
        #visualize(env)
        
        episode_rewards.append(episode_reward)
        
        discounted_rewards = compute_discounted_rewards(rewards, gamma)
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = normalize(discounted_rewards)
        
        policy_loss = torch.stack([-log_prob * reward for log_prob, reward in zip(log_probs, discounted_rewards)]).sum()
        
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        print(f'Episode {episode}, Reward: {episode_reward:.3f}')
        print(f'Episode {episode}, Reward: {policy_loss:.3f}')
        
        if episode % 10 == 0:
            torch.save(policy.state_dict(), save_path)
    
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

if __name__ == '__main__':
    env = suite.load(domain_name="walker", task_name="stand")
    obsSpec = env.observation_spec()
    orientDim = obsSpec['orientations'].shape[0]
    heightDim = len(obsSpec['height'].shape) + 1 
    velocityDim = obsSpec['velocity'].shape[0]
    input_dim = orientDim + heightDim + velocityDim

    hidden_dim = 128
    output_dim = env.action_spec().shape[0]

    policy = PolicyNetwork(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    train(policy, optimizer)
    env.close()
