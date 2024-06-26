import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from dm_control import suite


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
    normal = torch.distributions.Normal(mean, std)
    action = normal.sample()
    return action.clamp(-1.0, 1.0)  # Clamp action to ensure it stays within [-1, 1]


# Function to select action based on policy network's output
def select_action(policy, state):
    state = torch.from_numpy(state).float()
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)

def normalize_reward(reward):
    return reward / 100.0 

def normalize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


# Training the policy network
def train(policy, optimizer, episodes=1000, gamma=0.99, save_path='policy_network.pth'):
    env = suite.load(domain_name="walker", task_name="stand")
    for episode in range(episodes):
        running_loss = .0
        time_step = env.reset()
        while not time_step.last():

            state_observation = time_step.observation

            # Convert each component to PyTorch tensors
            orientations_tensor = torch.tensor(state_observation['orientations'], dtype=torch.float32, requires_grad=True)
            height_tensor = torch.tensor([state_observation['height']], dtype=torch.float32, requires_grad=True)  # Wrap scalar in list for 1D tensor
            velocity_tensor = torch.tensor(state_observation['velocity'], dtype=torch.float32, requires_grad=True)
            #reward_tensor = torch.tensor(state_reward, dtype=torch.float32)

            input_tensor = torch.cat((orientations_tensor, height_tensor, velocity_tensor), dim=0)

            # Now pass these tensors as tuple (or list) to your policy network
            mean, std = policy(input_tensor)

            action = sample_action(mean, std)

            time_step = env.step(action)
            state_reward = time_step.reward


            



            mse_loss = nn.MSELoss()
            #loss = mse_loss(action, mean)
            loss = mse_loss(action, mean) * state_reward

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()



        print(f'[{episode}] {running_loss :.3f}.')

        # save interim results
        if episode % 10 == 0:
            torch.save(policy.state_dict(), save_path)

        

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
