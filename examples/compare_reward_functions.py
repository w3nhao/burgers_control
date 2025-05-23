import numpy as np
import matplotlib.pyplot as plt

# Define MSE range for comparison
mse_values = np.linspace(0, 10, 1000)

# Calculate rewards using both methods
exp_rewards = np.exp(-mse_values)  # Original exponential method
inverse_rewards = 1.0 / (1.0 + mse_values)  # New inverse method

# Create the comparison plot
plt.figure(figsize=(12, 8))

# Plot both reward functions
plt.subplot(2, 2, 1)
plt.plot(mse_values, exp_rewards, 'b-', linewidth=2, label='exp(-MSE)')
plt.plot(mse_values, inverse_rewards, 'r-', linewidth=2, label='1/(1+MSE)')
plt.xlabel('MSE')
plt.ylabel('Reward')
plt.title('Reward Functions Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 10)
plt.ylim(0, 1)

# Zoom in on the low MSE region
plt.subplot(2, 2, 2)
mse_zoom = np.linspace(0, 2, 1000)
exp_zoom = np.exp(-mse_zoom)
inverse_zoom = 1.0 / (1.0 + mse_zoom)
plt.plot(mse_zoom, exp_zoom, 'b-', linewidth=2, label='exp(-MSE)')
plt.plot(mse_zoom, inverse_zoom, 'r-', linewidth=2, label='1/(1+MSE)')
plt.xlabel('MSE')
plt.ylabel('Reward')
plt.title('Zoomed View (MSE 0-2)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 2)
plt.ylim(0, 1)

# Plot the difference between the two functions
plt.subplot(2, 2, 3)
difference = exp_rewards - inverse_rewards
plt.plot(mse_values, difference, 'g-', linewidth=2)
plt.xlabel('MSE')
plt.ylabel('exp(-MSE) - 1/(1+MSE)')
plt.title('Difference Between Functions')
plt.grid(True, alpha=0.3)
plt.xlim(0, 10)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)

# Plot gradients (derivatives) to show learning signal strength
plt.subplot(2, 2, 4)
# Numerical derivatives
dmse = mse_values[1] - mse_values[0]
exp_grad = np.gradient(exp_rewards, dmse)
inverse_grad = np.gradient(inverse_rewards, dmse)

plt.plot(mse_values, np.abs(exp_grad), 'b-', linewidth=2, label='|d/dMSE exp(-MSE)|')
plt.plot(mse_values, np.abs(inverse_grad), 'r-', linewidth=2, label='|d/dMSE 1/(1+MSE)|')
plt.xlabel('MSE')
plt.ylabel('Absolute Gradient')
plt.title('Learning Signal Strength (Gradient Magnitude)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 10)
plt.yscale('log')

plt.tight_layout()
plt.savefig('reward_functions_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print some key comparison points
print("Reward Function Comparison:")
print("=" * 50)
print(f"{'MSE':<8} {'exp(-MSE)':<12} {'1/(1+MSE)':<12} {'Difference':<12}")
print("-" * 50)

test_mse_values = [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
for mse in test_mse_values:
    exp_reward = np.exp(-mse)
    inv_reward = 1.0 / (1.0 + mse)
    diff = exp_reward - inv_reward
    print(f"{mse:<8.1f} {exp_reward:<12.4f} {inv_reward:<12.4f} {diff:<12.4f}")

print("\nKey Observations:")
print("- Both functions give reward = 1.0 when MSE = 0")
print("- exp(-MSE) drops much faster for small MSE values")
print("- 1/(1+MSE) provides more stable gradients across all MSE ranges")
print("- 1/(1+MSE) gives non-zero rewards even for large MSE values")
print("- exp(-MSE) approaches 0 very quickly, potentially causing vanishing gradients") 