import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

# --------------------------------------------------------------------------- #
# 2. Actor-Critic network                                                     #
# --------------------------------------------------------------------------- #
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Common layers
        self.fc_common = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU()
        )
        
        # Actor network (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Critic network (value function)
        self.critic_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        x = self.fc_common(x)
        
        # Actor: Action probabilities
        action_probs = torch.softmax(self.actor_head(x), dim=-1)
        
        # Critic: State value
        state_value = self.critic_head(x)
        
        return action_probs, state_value
    
    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.forward(state)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1).item()
        else:
            dist = Categorical(action_probs)
            action = dist.sample().item()
            
        return action
    
    def evaluate(self, states, actions):
        action_probs, state_values = self.forward(states)
        dist = Categorical(action_probs)
        
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        
        return action_log_probs, state_values, dist_entropy