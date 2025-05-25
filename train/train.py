from tqdm import tqdm

# Training function
def train_ppo(env, agent, max_episodes=1000, steps_per_episode=128, 
              update_interval=10, eval_interval=50, verbose=True):
    
    # Initialize logging variables
    episode_rewards = []
    eval_rewards = []
    best_intensity = 0
    best_mask = None
    
    # Training loop
    for episode in tqdm(range(1, max_episodes + 1)):
        # Storage for the current episode
        states = []
        actions = []
        rewards = []
        next_states = []
        
        episode_reward = 0
        state = env.reset()
        
        # Generate trajectory
        for step in range(steps_per_episode):
            # Select action
            action = agent.get_action(state)
            
            # Execute action
            next_state, reward = env.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
        # Store episode reward
        episode_rewards.append(episode_reward)
        
        # Update agent
        if episode % update_interval == 0:
            policy_loss, value_loss, entropy_loss = agent.update(
                states, actions, rewards, next_states
            )
            
            if verbose:
                print(f"Episode {episode}, Reward: {episode_reward:.4f}")
                print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy Loss: {entropy_loss:.4f}")
                print(f"Current Intensity: {env.I_prev:.4f}, Max Intensity: {env.I_max:.4f}")
                print("---")
        
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
    
    # Return results
    return {
        'episode_rewards': episode_rewards,
        'eval_rewards': eval_rewards,
        'best_intensity': best_intensity,
        'best_mask': best_mask
    }

# Evaluation function
def evaluate_agent(env, agent, num_episodes=5, steps_per_episode=128):
    total_rewards = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for _ in range(steps_per_episode):
            action = agent.get_action(state, deterministic=True)  # Use deterministic policy for evaluation
            next_state, reward, done, _ = env.step(action)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
                
        total_rewards += episode_reward
        
    return total_rewards / num_episodes