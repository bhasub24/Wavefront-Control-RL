import numpy as np
import matplotlib.pyplot as plt

# Visualization functions 
def plot_training_curve(results, run): 
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(16, 16))

    # Plot episode rewards
    plt.subplot(7, 1, 1)
    plt.plot(results['episode_rewards'], label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.grid(True)
    plt.legend()
    
    # Plot evaluation rewards if available
    if len(results['eval_rewards']) > 0:
        plt.subplot(7, 1, 2)
        eval_x = np.linspace(0, len(results['episode_rewards']), len(results['eval_rewards']))
        plt.plot(eval_x, results['eval_rewards'], label='Evaluation Reward', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Evaluation Rewards')
        plt.grid(True)
        plt.legend()

    # Plot mean advantage per episode
    if 'all_advantage_means' in results and len(results['all_advantage_means']) > 0:
        plt.subplot(7, 1, 3)
        update_x = np.linspace(0, len(results['episode_rewards']), len(results['all_advantage_means']))
        plt.plot(update_x, results['all_advantage_means'], label='Mean Advantage', color='green')
        plt.xlabel('Episode')
        plt.ylabel('Mean Advantage')
        plt.title('Advantage Trend Over Updates')
        plt.grid(True)
        plt.legend()

    # Plot value loss over updates
    if 'value_losses' in results and len(results['value_losses']) > 0:
        plt.subplot(7, 1, 4)
        plt.plot(results['value_losses'], label='Value Loss', color='red')
        plt.xlabel('Update Step')
        plt.ylabel('Value Loss')
        plt.title('Value Loss Trend Over Updates')
        plt.grid(True)
        plt.legend()

    # Plot policy loss over updates
    if 'policy_losses' in results and len(results['policy_losses']) > 0:
        plt.subplot(7, 1, 5)
        plt.plot(results['policy_losses'], label='Policy Loss', color='blue')
        plt.xlabel('Update Step')
        plt.ylabel('Policy Loss')
        plt.title('Policy Loss Trend Over Updates')
        plt.grid(True)
        plt.legend()

    # Plot entropy loss over updates
    if 'entropy_losses' in results and len(results['entropy_losses']) > 0:
        plt.subplot(7, 1, 6)
        plt.plot(results['entropy_losses'], label='Entropy Loss', color='purple')
        plt.xlabel('Update Step')
        plt.ylabel('Entropy Loss')
        plt.title('Entropy Loss Trend Over Updates')
        plt.grid(True)
        plt.legend()
    
    # Plot KL divergence over updates
    if 'all_kl_divergences' in results and len(results['all_kl_divergences']) > 0:
        plt.subplot(7, 1, 7)
        plt.plot(results['all_kl_divergences'], label='KL Divergence', color='brown')
        plt.xlabel('Update Step')
        plt.ylabel('KL Divergence')
        plt.title('KL Divergence Trend Over Updates')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(f'results/run_{run}_training_curve.png')
    plt.close()

    # Plot value estimates vs returns
    if 'old_values_all' in results and 'returns_all' in results:
        plt.figure(figsize=(8, 6))
        plt.scatter(results['returns_all'], results['old_values_all'], alpha=0.5, label='Old Value Estimates')
        plt.scatter(results['returns_all'], results['new_values_all'], alpha=0.5, label='New Value Estimates')
        plt.plot(results['returns_all'], results['returns_all'], 'k--', label='Ideal Match')
        plt.xlabel('Returns')
        plt.ylabel('Value Estimates')
        plt.title('Value Estimates vs Returns')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/run_{run}_value_vs_returns.png')
        plt.close()


def visualize_best_mask(env, best_mask, run):
    # Store original mask
    original_mask = env.block_mask.copy()
    
    # Set the best mask
    env.block_mask = best_mask.copy()
    
    # Get pixel mask
    pixel_mask = env._blocks_to_pixels()
    
    # Get intensity
    intensity = env._intensity()
    
    # Plot mask
    plt.figure(figsize=(10, 8))
    plt.imshow(pixel_mask, cmap='viridis')
    plt.colorbar(label='Mask Value')
    plt.title(f'Best Mask (Intensity: {intensity:.4f})')
    plt.savefig(f'results/run_{run}_best_mask.png')
    plt.close()
    
    # Restore original mask
    env.block_mask = original_mask