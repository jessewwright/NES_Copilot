from experiments.behavioral_prediction.models.mvnes_agent import NESAgent
import numpy as np

# Test instantiation
w_s = 1.0
w_n = 1.0
agent = NESAgent(w_s=w_s, w_n=w_n, seed=42)
print(f"NESAgent instantiated with w_s={w_s}, w_n={w_n}")

# Test cases for choose_action
test_lambdas = {
    "lambda_0.0 (expect B)": 0.0,   # drift = 1*(1-0) - 1*0 = 1.0  -> choice B (1)
    "lambda_0.25 (expect B)": 0.25, # drift = 1*(0.75) - 1*0.25 = 0.5 -> choice B (1)
    "lambda_0.5 (expect random)": 0.5, # drift = 1*(0.5) - 1*0.5 = 0.0 -> choice random
    "lambda_0.75 (expect A)": 0.75, # drift = 1*(0.25) - 1*0.75 = -0.5 -> choice A (0)
    "lambda_1.0 (expect A)": 1.0    # drift = 1*(0) - 1*1 = -1.0 -> choice A (0)
}

expected_choices = {
    "lambda_0.0 (expect B)": 1,
    "lambda_0.25 (expect B)": 1,
    "lambda_0.5 (expect random)": "random", # Special case
    "lambda_0.75 (expect A)": 0,
    "lambda_1.0 (expect A)": 0
}

for desc, lambda_val in test_lambdas.items():
    choice, drift = agent.choose_action(lambda_val)
    print(f"Test: {desc}, Lambda: {lambda_val:.2f}, Drift: {drift:.2f}, Choice: {choice}")

    expected = expected_choices[desc]
    if expected == "random":
        # For random, we can't assert a specific choice but can check drift is 0
        assert drift == 0.0, f"Test {desc} failed: Drift was {drift}, expected 0.0"
        # To be more robust for random, run multiple times if not seeded, but with seed it's deterministic
        # For now, we just check drift for the random case.
        # With seed 42, rng.choice([0,1]) first gives 0 then 1. So first random choice will be 0.
        # Let's re-initialize for the random test to ensure it's the first call to rng.choice
        agent_for_random_test = NESAgent(w_s=w_s, w_n=w_n, seed=42)
        choice_random, _ = agent_for_random_test.choose_action(lambda_val)
        print(f"Test: {desc}, Lambda: {lambda_val:.2f}, First random choice with seed 42: {choice_random}")
        assert choice_random == 0, f"Test {desc} failed: First random choice with seed 42 was {choice_random}, expected 0"

    else:
        assert choice == expected, f"Test {desc} failed: Choice was {choice}, expected {expected}"

# Test with different w_s, w_n
w_s_2 = 2.0
w_n_2 = 0.5
agent2 = NESAgent(w_s=w_s_2, w_n=w_n_2, seed=123)
print(f"NESAgent instantiated with w_s={w_s_2}, w_n={w_n_2}")
lambda_val_2 = 0.6 # drift = 2*(1-0.6) - 0.5*0.6 = 2*0.4 - 0.3 = 0.8 - 0.3 = 0.5 -> choice B (1)
choice2, drift2 = agent2.choose_action(lambda_val_2)
print(f"Test: Custom w_s, w_n, Lambda: {lambda_val_2:.2f}, Drift: {drift2:.2f}, Choice: {choice2}")
assert choice2 == 1, "Test with custom w_s, w_n failed"


print("NESAgent test completed.")
