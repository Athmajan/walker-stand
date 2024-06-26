
import torch
import torch.nn as nn
import torchvision.models as models


class FccEncoder(nn.Module):
    def __init__(self, state_dim, latent_dim):
        super(FccEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, s):
        return self.net(s)