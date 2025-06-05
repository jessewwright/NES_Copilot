import numpy as np

class DDMAgent:
    def __init__(self, seed=None):
        """
        Drift-Diffusion Model (DDM) Agent.
        Makes choices based on the difference in rewards.
        In this specific simulation, rewards are equal, leading to random choice.

        Args:
            seed (int, optional): Seed for the random number generator for reproducibility.
        """
        self.rng = np.random.RandomState(seed)

    def choose_action(self, reward_A: float, reward_B: float) -> tuple[int, float]:
        """
        Chooses an action based on the rewards.

        Args:
            reward_A (float): Reward for action A.
            reward_B (float): Reward for action B.

        Returns:
            tuple[int, float]: A tuple containing the chosen action (0 for A, 1 for B)
                               and the calculated drift.
        """
        drift = reward_B - reward_A  # This will be 0 in the simulation

        # Simulate choice: if drift is 0, choose randomly.
        # Otherwise, choice probability could be a function of drift (e.g., softmax).
        # For this simulation, with drift = 0, it's a 50/50 random choice.
        if drift == 0:
            choice = self.rng.choice([0, 1]) # 0 for A, 1 for B
        elif drift > 0: # Should not happen in this simulation setup
            choice = 1 # Choose B
        else: # Should not happen in this simulation setup
            choice = 0 # Choose A

        return choice, drift
