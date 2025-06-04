import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from torch.distributions import Categorical
from agents.actor_critic import ActorCritic
import torch.nn.functional as F


# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, actor_lr=3e-4, critic_lr=3e-4, gamma=0.99, 
                 clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01, gae_lambda=0.95):
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.policy.actor_head.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.policy.critic_head.parameters(), lr=critic_lr)
        
        self.old_policy = copy.deepcopy(self.policy)
        self.old_policy.eval()  # Set to evaluation mode
    
    def update(self, states, actions, rewards, next_states, steps_per_epoch=128, epochs=10):
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))

        kl_divergences = []

        with torch.no_grad():
            old_action_log_probs, old_values, _ = self.old_policy.evaluate(states, actions)
            old_values = old_values.squeeze(-1)
            next_values = self.old_policy.forward(next_states)[1].squeeze(-1)
            values = torch.cat([old_values, next_values[-1].unsqueeze(0)], dim=0)

        # Calculate advantages using GAE (Generalized Advantage Estimation)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value - old_values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
        
        # Compute returns (used for value function loss)
        returns = advantages + old_values.detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update loop
        for _ in range(epochs):
            # Create random indices for minibatches
            indices = np.random.permutation(len(states))
            
            # Iterate through mini-batches
            for start in range(0, len(states), steps_per_epoch):
                end = start + steps_per_epoch
                if end > len(states):
                    end = len(states)
                    
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_action_log_probs = old_action_log_probs[batch_indices]
                
                # Evaluate current policy
                action_log_probs, values, entropy = self.policy.evaluate(batch_states, batch_actions)
                
                # Calculate ratios
                ratios = torch.exp(action_log_probs - batch_old_action_log_probs)

                kl = (batch_old_action_log_probs - action_log_probs).mean().item()  
                kl_divergences.append(kl)
                
                # Compute surrogate losses
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                
                # Calculate loss components
                policy_loss = -torch.min(surr1, surr2).mean()
                # value_loss = nn.MSELoss()(values.squeeze(-1), batch_returns)
                value_loss = F.smooth_l1_loss(values.squeeze(-1), batch_returns)
                entropy_loss = -entropy.mean()

                actor_loss = policy_loss + self.entropy_coef * entropy_loss
                critic_loss = self.value_coef * value_loss

                # Update actor and critic separately
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.actor_head.parameters(), 0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.critic_head.parameters(), 0.5)
                self.critic_optimizer.step()
        
        with torch.no_grad():
            new_values = self.policy.forward(states)[1].squeeze(-1)

        # Update old policy
        self.old_policy.load_state_dict(self.policy.state_dict())

        average_kl = np.mean(kl_divergences)
        
        return policy_loss.item(), value_loss.item(), entropy_loss.item(), advantages, old_values, new_values, returns, average_kl
    
    def get_action(self, state, deterministic=False):
        return self.policy.get_action(state, deterministic)
    
    def save(self, path):
        torch.save(self.policy.state_dict(), path)
        
    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.old_policy.load_state_dict(torch.load(path))