import numpy as np
import matplotlib.pyplot as plt

# Visualization functions
def plot_training_curve(results):
    plt.figure(figsize=(12, 8))
    
    # Plot episode rewards
    plt.subplot(2, 1, 1)
    plt.plot(results['episode_rewards'], label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.grid(True)
    plt.legend()
    
    # Plot evaluation rewards if available
    if len(results['eval_rewards']) > 0:
        plt.subplot(2, 1, 2)
        eval_x = np.linspace(0, len(results['episode_rewards']), len(results['eval_rewards']))
        plt.plot(eval_x, results['eval_rewards'], label='Evaluation Reward', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Evaluation Rewards')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.close()

def visualize_best_mask(env, best_mask):
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
    plt.savefig('best_mask.png')
    plt.close()
    
    # Restore original mask
    env.block_mask = original_mask