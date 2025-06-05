import numpy as np

class NESAgent:
    def __init__(self, w_s: float, w_n: float, seed: int = None):
        """
        Simplified Normative Executive System (NES) Agent.
        Calculates drift based on salience, norm weights, and conflict level.

        Args:
            w_s (float): Weight for salience.
            w_n (float): Weight for norm violation.
            seed (int, optional): Seed for the random number generator for reproducibility.
        """
        self.w_s = w_s
        self.w_n = w_n
        self.rng = np.random.RandomState(seed)

    def choose_action(self, lambda_conflict: float) -> tuple[int, float]:
        """
        Chooses an action based on the conflict level lambda.
        Action A is neutral (0), Action B is conflict-laden (1).

        Args:
            lambda_conflict (float): The current conflict level (λ), ranging from 0.0 to 1.0.

        Returns:
            tuple[int, float]: A tuple containing the chosen action (0 for A, 1 for B)
                               and the calculated drift.
        """
        # Drift towards B (conflict action) is positive
        # Drift towards A (neutral action) is negative
        # drift = w_s * (reward_B - reward_A) + w_s * (1 - λ) - w_n * λ
        # Since reward_A and reward_B are both 1.0, (reward_B - reward_A) is 0.
        # The problem states: drift = w_s * (1 - λ) - w_n * λ
        # This implies that positive drift favors the conflict-laden action B.

        drift = self.w_s * (1 - lambda_conflict) - self.w_n * lambda_conflict

        if drift > 0:
            choice = 1  # Choose B (conflict action)
        elif drift < 0:
            choice = 0  # Choose A (neutral action)
        else: # drift == 0
            choice = self.rng.choice([0, 1]) # Random choice if drift is exactly zero

        return choice, drift
