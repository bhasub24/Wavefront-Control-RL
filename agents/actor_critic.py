import torch
import torch.nn as nn
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
    
        action_logits = self.actor_head(x)
        
        # Critic: State value
        state_value = self.critic_head(x)
        
        # return action_probs, state_value
        return action_logits, state_value
    
    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)

        action_logits, _ = self.forward(state)         # [1, 128]
    
        dist = torch.distributions.Bernoulli(logits=action_logits)
        
        if deterministic:
            # Use threshold 0.5 to binarize
            action = (torch.sigmoid(action_logits) > 0.5).float()
        else:
            action = dist.sample()
        
        # Convert to list of indices where action is 1
        action = action.squeeze(0).detach().cpu().numpy().astype(int)
        flip_indices = np.where(action == 1)[0].tolist()
        
        return flip_indices, action
    
    def evaluate(self, states, actions):
        action_logits, state_values = self.forward(states)
        dist = torch.distributions.Bernoulli(logits=action_logits)
        
        # Element-wise log-probabilities: shape [T, 128]
        action_log_probs = dist.log_prob(actions)

        # Sum over action dimensions to get [T]
        log_probs = action_log_probs.sum(dim=1)

        # Entropy per sample (sum over 128 dimensions): shape [T]
        dist_entropy = dist.entropy().sum(dim=1)

        return log_probs, state_values, dist_entropy