from tqdm import tqdm
import numpy as np

# Training function
def train_ppo(env, agent, max_episodes=1000, steps_per_episode=128, 
              update_interval=10, eval_interval=50, verbose=True):
    
    # Initialize logging variables
    episode_rewards = []
    eval_rewards = []
    best_intensity = 0
    best_mask = None

    # Storage for the all states, actions, rewards, and next_states
    states = []
    actions = []
    rewards = []
    next_states = []

    all_advantage_means = []

    value_losses = []
    policy_losses = []
    entropy_losses = []
    all_old_values = []
    all_new_values = []
    all_returns = []
    all_kl_divergences = []
    
    # Training loop
    for episode in tqdm(range(1, max_episodes + 1)):
        
        episode_reward = 0
        state = env.reset()
        
        # Generate trajectory
        for step in range(steps_per_episode):
            # Select action
            action, action_mask = agent.get_action(state)
            
            # Execute action
            next_state, reward = env.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action_mask)
            rewards.append(reward)
            next_states.append(next_state)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
        # Store episode reward
        episode_rewards.append(episode_reward)
        
        # Update agent
        if episode % update_interval == 0:
            policy_loss, value_loss, entropy_loss, advantages, old_values, new_values, returns, kl_divergence = agent.update(
                states, actions, rewards, next_states
            )
            
            advantages = advantages.detach().cpu().numpy().reshape(update_interval, steps_per_episode)
            per_episode_advantage_means = advantages.mean(axis=1)

            all_advantage_means.extend(per_episode_advantage_means)

            # Append scalar value loss
            value_losses.append(value_loss)
            policy_losses.append(policy_loss)
            entropy_losses.append(entropy_loss)

            # Detach and move tensors to CPU
            all_old_values.append(old_values.detach().cpu().numpy())
            all_new_values.append(new_values.detach().cpu().numpy())
            all_returns.append(returns.detach().cpu().numpy())
            all_kl_divergences.append(kl_divergence)

            if verbose:
                print(f"Episode {episode}, Reward: {episode_reward:.4f}")
                print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy Loss: {entropy_loss:.4f}")
                print(f"Current Intensity: {env.I_prev:.4f}, Max Intensity: {env.I_max:.4f}")
                print("---")
            
            states.clear()
            actions.clear()
            rewards.clear()
            next_states.clear()
        
        # Track best solution found
        if env.I_max > best_intensity:
            best_intensity = env.I_max
            best_mask = env.block_mask.copy()
        
        # Evaluate performance
        if episode % eval_interval == 0:
            eval_reward = evaluate_agent(env, agent, num_episodes=5)
            eval_rewards.append(eval_reward)
            
            if verbose:
                print(f"Evaluation at episode {episode}: Average Reward = {eval_reward:.4f}")
                print("===================================")
    
    old_values_all = np.concatenate(all_old_values)
    new_values_all = np.concatenate(all_new_values)
    returns_all = np.concatenate(all_returns)

    # Return results
    return {
        'episode_rewards': episode_rewards,
        'eval_rewards': eval_rewards,
        'best_intensity': best_intensity,
        'best_mask': best_mask,
        'all_advantage_means': all_advantage_means,
        'value_losses': value_losses,
        'old_values_all': old_values_all,
        'new_values_all': new_values_all,
        'returns_all': returns_all,
        'policy_losses': policy_losses,
        'entropy_losses': entropy_losses,
        'all_kl_divergences': all_kl_divergences,
    }

# Evaluation function
def evaluate_agent(env, agent, num_episodes=5, steps_per_episode=128):
    total_rewards = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for _ in range(steps_per_episode):
            action, _ = agent.get_action(state, deterministic=True)  # Use deterministic policy for evaluation
            next_state, reward = env.step(action)
            
            state = next_state
            episode_reward += reward
                
        total_rewards += episode_reward
        
    return total_rewards / num_episodes