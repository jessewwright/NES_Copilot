# Behavioral Prediction Simulation: NES vs. DDM

This simulation investigates the behavioral divergence between two hypothetical agents—Normative Executive System (NES) and Drift-Diffusion Model (DDM)—when faced with a decision involving a conflict-laden option.

## Core Goal
The primary goal is to demonstrate that, even with equal material rewards for two choices, the NES agent's choice behavior is sensitive to the level of normative conflict (λ) associated with one of the options, while the DDM agent's behavior is not (or is random). Specifically, as conflict λ increases, the NES agent is expected to increasingly avoid the norm-violating option, whereas the DDM agent's choices should remain largely unaffected by λ.

A secondary variant allows for skewed rewards to test if NES can suppress a norm-violating choice even when it's materially incentivized.

## Agent Logic

### `DDMAgent` (Drift-Diffusion Model)
*   **Behavior**: This agent makes choices based on a simple drift-diffusion process where the drift rate is determined by the difference in rewards between two options (Action A and Action B).
*   **Drift Calculation**: `drift = reward_B - reward_A`
*   **Conflict Sensitivity**: In the primary simulation where `reward_A = reward_B`, the drift is zero, leading to random choices (approximately 50% for A and 50% for B), irrespective of the conflict level λ.
*   **In Reward-Skewed Condition**: If `reward_B` is significantly higher than `reward_A`, the DDM agent will consistently choose Action B.

### `NESAgent` (Normative Executive System)
*   **Behavior**: This agent incorporates an internal "normative conflict" signal into its decision-making process.
*   **Drift Calculation**: `drift = w_s * (1 - λ) - w_n * λ`
    *   `w_s`: Weight for salience (e.g., the drive to choose an option based on its non-normative properties, potentially related to reward but modeled distinctly here).
    *   `w_n`: Weight for norm sensitivity (how much the agent is deterred by norm violation).
    *   `λ`: Conflict level associated with Action B (the norm-violating option), ranging from 0.0 (no conflict) to 1.0 (maximum conflict).
*   **Conflict Sensitivity**: As λ increases, the `w_n * λ` term increasingly inhibits the choice of Action B. The `w_s * (1 - λ)` term represents diminishing drive towards B as conflict makes it less appealing overall or as (1-λ) gates the salience.
*   **Expected Behavior**: The probability of choosing Action B is expected to decrease as λ increases. Even if Action B offers a higher material reward (in the skewed condition), the NES agent might still suppress this choice at high λ if `w_n` is sufficiently large.

## Configuration
The simulation is controlled via `configs/prediction_task.yaml` and/or command-line arguments.

### YAML Configuration (`configs/prediction_task.yaml`)
Example:
```yaml
agent_type: BOTH  # NES, DDM, or BOTH
ws: 1.0           # Salience weight for NES
wn: 1.0           # Norm weight for NES
n_trials_per_lambda: 100
lambdas: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # List of conflict levels
seed: 42          # For reproducibility
reward_A: 1.0     # Reward for Action A (neutral/control)
reward_B: 1.0     # Reward for Action B (norm-violating/conflict-laden)
```

### Command-Line Overrides
All parameters in the YAML file can be overridden via CLI arguments when running `run_prediction_task.py`. For example:
```bash
python experiments/behavioral_prediction/run_prediction_task.py --agent_type NES --ws 1.2 --wn 0.8 --n_trials_per_lambda 200 --reward_B 1.5
```
Use `python experiments/behavioral_prediction/run_prediction_task.py --help` for a full list of options.

## Output Files
Output files are saved in the `experiments/behavioral_prediction/outputs/` directory. Filenames include a unique `run_id` and key simulation parameters.

1.  **`trial_data_[run_id]_[params].csv`**:
    *   Contains one row per trial.
    *   Columns include:
        *   `agent`: Agent type (NES or DDM).
        *   `lambda`: Conflict level for the trial.
        *   `trial`: Trial number within that lambda block.
        *   `drift`: Calculated drift rate for the trial.
        *   `choice`: Agent's choice (0 for Action A, 1 for Action B).
        *   `conflict`: Same as `lambda` (repeated for clarity).
        *   `config_reward_A`: Configured reward for Action A for this run.
        *   `config_reward_B`: Configured reward for Action B for this run.
        *   `chosen_reward`: Actual reward received by the agent for its choice.

2.  **`summary_[run_id]_[params].csv`**:
    *   Contains one row per agent per lambda value.
    *   Columns include:
        *   `agent`: Agent type.
        *   `lambda`: Conflict level.
        *   `pct_B_chosen`: Percentage of trials where Action B was chosen.
        *   `mean_drift`: Average drift rate for that agent at that lambda.
        *   `std_drift`: Standard deviation of drift rates.
        *   `nes_violation_slope` (for NES agent rows): Overall slope of `pct_B_chosen` vs. `lambda` for the NES agent.
        *   `nes_lambda_cutoff_lt20pct` (for NES agent rows): Lambda value where NES `pct_B_chosen` drops below 20% (after being >= 50%).

3.  **`prediction_plot_[run_id]_[params].png`**:
    *   A plot showing "% Norm-Violating Choices (Action B)" on the Y-axis vs. "λ (Conflict Level)" on the X-axis.
    *   Separate lines are shown for NES and DDM agents.
    *   Includes 95% bootstrapped confidence intervals as shaded regions around the mean choice percentages.
    *   The NES curve may have an annotation indicating its approximate inflection point.

## Expected Pattern of Results
*   **DDM Agent**: The line on the plot for DDM should be relatively flat, hovering around 50% choice of Action B when `reward_A == reward_B`. If rewards are skewed (e.g., `reward_B > reward_A`), DDM will shift its preference accordingly, but still flatly across λ values.
*   **NES Agent**: The line on the plot for NES should show a downward trend. As λ increases, the percentage of Action B choices should decrease, demonstrating sensitivity to normative conflict. This trend should hold even if `reward_B > reward_A`, showcasing potential "moral override."
