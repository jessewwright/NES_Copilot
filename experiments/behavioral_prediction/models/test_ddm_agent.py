from experiments.behavioral_prediction.models.ddm_agent import DDMAgent
import numpy as np

# Test instantiation
agent = DDMAgent(seed=42)
print("DDMAgent instantiated.")

# Test choose_action with equal rewards
reward_A = 1.0
reward_B = 1.0
choices = []
drifts = []
for _ in range(1000):
    choice, drift = agent.choose_action(reward_A, reward_B)
    choices.append(choice)
    drifts.append(drift)

print(f"Testing with reward_A={reward_A}, reward_B={reward_B}")
print(f"Mean choice: {np.mean(choices):.3f} (expected close to 0.5)")
print(f"Mean drift: {np.mean(drifts):.3f} (expected 0.0)")
assert np.mean(drifts) == 0.0, "Drift should be 0"
# Check if choices are roughly 50/50
counts = np.bincount(choices)
print(f"Choice counts (0, 1): {counts}")
assert len(counts) == 2 and 400 < counts[0] < 600 and 400 < counts[1] < 600, "Choices should be roughly balanced for 0 drift"

print("DDMAgent test completed.")
