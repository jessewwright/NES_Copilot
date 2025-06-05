import argparse
import yaml
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt # Added for plotting in next step

# Ensure the models directory is in the Python path
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from experiments.behavioral_prediction.models.ddm_agent import DDMAgent
from experiments.behavioral_prediction.models.mvnes_agent import NESAgent

def load_config(config_path):
    """Loads YAML configuration file."""
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. Using default values from script.")
        # Provide some defaults if config is missing, matching prediction_task.yaml structure
        return {
            'agent_type': 'NES',
            'ws': 1.0,
            'wn': 1.0,
            'n_trials_per_lambda': 100,
            'lambdas': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'seed': int(datetime.now().timestamp())
        }
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_simulation(config, agent_to_simulate, master_seed_offset=0):
    """
    Runs the behavioral prediction simulation for a single agent type.

    Args:
        config (dict): Dictionary containing simulation parameters.
        agent_to_simulate (str): 'NES' or 'DDM'.
        master_seed_offset (int): Offset for seeding to ensure different random sequences.

    Returns:
        pd.DataFrame: DataFrame containing trial-by-trial data for the agent.
    """
    ws = config.get('ws', 1.0)
    wn = config.get('wn', 1.0)
    n_trials_per_lambda = config.get('n_trials_per_lambda', 100)
    lambdas_config = config.get('lambdas', "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")

    if isinstance(lambdas_config, str):
        lambdas = [float(l.strip()) for l in lambdas_config.split(',')]
    elif isinstance(lambdas_config, list):
        lambdas = [float(l) for l in lambdas_config]
    else:
        raise ValueError("Lambdas must be a comma-separated string or a list of floats.")

    reward_A = 1.0
    reward_B = 1.0
    trial_data_records = []

    base_seed = config.get('seed', int(datetime.now().timestamp())) + master_seed_offset

    print(f"Running simulation for agent: {agent_to_simulate}")
    print(f"Parameters: ws={ws}, wn={wn}, n_trials_per_lambda={n_trials_per_lambda}")
    print(f"Lambdas: {lambdas}")
    print(f"Base seed for this agent run: {base_seed}")

    for i_lambda, lambda_conflict in enumerate(lambdas):
        # Seed for the agent, ensuring it's different for each lambda and agent type combination
        agent_seed = base_seed + i_lambda + 1

        if agent_to_simulate.upper() == 'NES':
            agent_instance = NESAgent(w_s=ws, w_n=wn, seed=agent_seed)
        elif agent_to_simulate.upper() == 'DDM':
            agent_instance = DDMAgent(seed=agent_seed)
        else:
            raise ValueError(f"Unknown agent type: {agent_to_simulate}.")

        # print(f"  Processing lambda: {lambda_conflict:.2f} ({i_lambda+1}/{len(lambdas)}) with agent seed {agent_seed}")

        for trial_num in range(1, n_trials_per_lambda + 1):
            if agent_to_simulate.upper() == 'NES':
                choice, drift = agent_instance.choose_action(lambda_conflict)
            elif agent_to_simulate.upper() == 'DDM':
                choice, drift = agent_instance.choose_action(reward_A, reward_B)

            current_reward = 1.0

            trial_data_records.append({
                'agent': agent_to_simulate.upper(),
                'lambda': lambda_conflict,
                'trial': trial_num,
                'drift': drift,
                'choice': choice,
                'conflict': lambda_conflict,
                'reward': current_reward
            })

            # Reduce print frequency
            # if trial_num % (n_trials_per_lambda // 5 or 1) == 0 :
            #      print(f"    Lambda {lambda_conflict:.2f}, Trial {trial_num}/{n_trials_per_lambda} completed.")
        print(f"  Lambda {lambda_conflict:.2f} completed for {agent_to_simulate}.")


    df_trial_data = pd.DataFrame(trial_data_records)
    return df_trial_data

def plot_results(summary_df, output_filename):
    """
    Plots the percentage of B choices vs. lambda for each agent.
    Saves the plot to the specified filename.
    """
    if summary_df.empty:
        print("Summary data is empty. Skipping plot generation.")
        return

    # Ensure matplotlib is available within this function too, or rely on global import
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    agents = summary_df['agent'].unique()
    # Sort agents for consistent plot legend order, e.g., DDM then NES
    # This ensures DDM (often the baseline) might appear first or consistently.
    agents = sorted(agents, key=lambda x: (x.upper() != 'DDM', x))


    for agent_name in agents:
        agent_data = summary_df[summary_df['agent'] == agent_name]
        plt.plot(agent_data['lambda'], agent_data['pct_B_chosen'], marker='o', linestyle='-', label=agent_name)

    plt.xlabel("λ (Conflict Level)")
    plt.ylabel("% Choice B (Conflict Action)")
    plt.title("Agent Choice Behavior vs. Conflict Level")

    plt.grid(True)
    plt.ylim(-0.05, 1.05)

    unique_lambdas = sorted(summary_df['lambda'].unique())
    plt.xticks(unique_lambdas)

    # Add a horizontal line at y=0.5 for DDM reference if DDM data exists
    if 'DDM' in [a.upper() for a in agents]: # Check if DDM is one of the agents
        plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, label='DDM Random (0.5)')

    plt.legend() # Call legend after all plot elements are added

    try:
        plt.savefig(output_filename)
        print(f"Plot saved to {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close() # Close the figure to free memory


def main():
    parser = argparse.ArgumentParser(description="Run NES vs DDM Behavioral Prediction Simulation")
    parser.add_argument('--agent_type', type=str, help="Agent: NES, DDM, or BOTH.")
    parser.add_argument('--ws', type=float, help="Salience weight (w_s) for NES.")
    parser.add_argument('--wn', type=float, help="Norm weight (w_n) for NES.")
    parser.add_argument('--n_trials_per_lambda', type=int, help="Number of trials per lambda.")
    parser.add_argument('--config_file', type=str, default='experiments/behavioral_prediction/configs/prediction_task.yaml', help="Path to YAML config.")
    parser.add_argument('--output_dir', type=str, default='experiments/behavioral_prediction/outputs', help="Output directory.")
    parser.add_argument('--run_id', type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'), help="Unique ID for run filenames.")

    args = parser.parse_args()
    yaml_config = load_config(args.config_file)
    cli_args = {k: v for k, v in vars(args).items() if v is not None and k not in ['config_file', 'output_dir', 'run_id']}
    merged_config = {**yaml_config, **cli_args}

    if args.ws is not None: merged_config['ws'] = float(args.ws)
    if args.wn is not None: merged_config['wn'] = float(args.wn)
    if args.n_trials_per_lambda is not None: merged_config['n_trials_per_lambda'] = int(args.n_trials_per_lambda)

    os.makedirs(args.output_dir, exist_ok=True)
    run_id = args.run_id
    print(f"Starting simulation run: {run_id}")
    print("Effective Configuration:")
    for key, value in merged_config.items():
        print(f"  {key}: {value}")

    all_trial_data_dfs = []
    agent_types_to_run = []
    requested_agent_type = merged_config.get('agent_type', 'NES').upper()

    if requested_agent_type == 'BOTH':
        agent_types_to_run = ['NES', 'DDM']
    elif requested_agent_type in ['NES', 'DDM']:
        agent_types_to_run = [requested_agent_type]
    else:
        print(f"Warning: Invalid agent_type '{requested_agent_type}' in config. Defaulting to NES.")
        agent_types_to_run = ['NES']
        merged_config['agent_type'] = 'NES' # Correct the config for the run

    master_seed_counter = 0
    for agent_type_sim in agent_types_to_run:
        print(f"--- Running for Agent: {agent_type_sim} ---")
        # Pass a copy of merged_config to run_simulation
        df_agent_trial_data = run_simulation(merged_config.copy(), agent_type_sim, master_seed_offset=master_seed_counter * 1000)
        all_trial_data_dfs.append(df_agent_trial_data)
        master_seed_counter +=1


    if not all_trial_data_dfs:
        print("No simulation data generated.")
        return

    df_all_trials = pd.concat(all_trial_data_dfs, ignore_index=True)

    trial_data_filename = os.path.join(args.output_dir, f'trial_data_{run_id}.csv')
    df_all_trials.to_csv(trial_data_filename, index=False)
    print(f"Full trial data saved to {trial_data_filename}")

    df_all_trials['choice'] = pd.to_numeric(df_all_trials['choice'])
    summary_df = df_all_trials.groupby(['agent', 'lambda']).agg(
        pct_B_chosen=('choice', 'mean'),
        mean_drift=('drift', 'mean')
    ).reset_index()

    summary_filename = os.path.join(args.output_dir, f'summary_{run_id}.csv')
    summary_df.to_csv(summary_filename, index=False)
    print(f"Summary statistics saved to {summary_filename}")

    # Call plotting function (currently a placeholder)
    plot_filename = os.path.join(args.output_dir, f'prediction_plot_{run_id}.png')
    plot_results(summary_df, plot_filename) # Defined above, currently just prints

    print(f"Simulation and data logging complete for run {run_id}.")

if __name__ == "__main__":
    main()
