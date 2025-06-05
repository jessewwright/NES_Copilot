import argparse
import yaml
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Ensure the models directory is in the Python path
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from experiments.behavioral_prediction.models.ddm_agent import DDMAgent
from experiments.behavioral_prediction.models.mvnes_agent import NESAgent

# Helper functions for summary metrics
def calculate_nes_slope(nes_summary_data):
    if nes_summary_data.empty or len(nes_summary_data) < 2:
        return np.nan
    nes_summary_data = nes_summary_data.dropna(subset=['lambda', 'pct_B_chosen'])
    if len(nes_summary_data) < 2:
        return np.nan
    try:
        nes_summary_data = nes_summary_data.sort_values(by='lambda')
        slope, _ = np.polyfit(nes_summary_data['lambda'], nes_summary_data['pct_B_chosen'], 1)
    except (np.linalg.LinAlgError, TypeError):
        slope = np.nan
    return slope

def find_nes_lambda_cutoff(nes_summary_data, threshold=0.5, target_lt_threshold=0.2):
    if nes_summary_data.empty:
        return np.nan
    nes_summary_data = nes_summary_data.sort_values(by='lambda')

    was_at_or_above_threshold = not nes_summary_data[nes_summary_data['pct_B_chosen'] >= threshold].empty
    if not was_at_or_above_threshold:
        return np.nan

    specific_cutoff_data = nes_summary_data[nes_summary_data['pct_B_chosen'] < target_lt_threshold]
    if not specific_cutoff_data.empty:
        return specific_cutoff_data['lambda'].iloc[0]

    return np.nan

def generate_output_filenames(output_dir, run_id, config):
    agent_type = config.get('agent_type', 'UNKNOWN').upper()
    ws = config.get('ws', 'na')
    wn = config.get('wn', 'na')
    reward_A = config.get('reward_A', 1.0)
    reward_B = config.get('reward_B', 1.0)

    base_name_parts = [f"{run_id}"]
    if agent_type != 'BOTH':
        base_name_parts.append(f"agent{agent_type}")
        if agent_type == 'NES':
            base_name_parts.append(f"ws{ws}")
            base_name_parts.append(f"wn{wn}")

    base_name_parts.append(f"rA{reward_A}")
    base_name_parts.append(f"rB{reward_B}")
    base_filename = "_".join(base_name_parts)

    return {
        'trial_data': os.path.join(output_dir, f'trial_data_{base_filename}.csv'),
        'summary': os.path.join(output_dir, f'summary_{base_filename}.csv'),
        'plot': os.path.join(output_dir, f'prediction_plot_{base_filename}.png')
    }

def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. Using default values.")
        return {
            'agent_type': 'NES', 'ws': 1.0, 'wn': 1.0,
            'n_trials_per_lambda': 100,
            'lambdas': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'seed': int(datetime.now().timestamp()),
            'reward_A': 1.0, 'reward_B': 1.0
        }
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_simulation(config, agent_to_simulate, master_seed_offset=0):
    ws = config.get('ws', 1.0)
    wn = config.get('wn', 1.0)
    n_trials_per_lambda = config.get('n_trials_per_lambda', 100)
    lambdas_config = config.get('lambdas', "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")

    reward_A = config.get('reward_A', 1.0)
    reward_B = config.get('reward_B', 1.0)

    if isinstance(lambdas_config, str):
        lambdas = [float(l.strip()) for l in lambdas_config.split(',')]
    elif isinstance(lambdas_config, list):
        lambdas = [float(l) for l in lambdas_config]
    else:
        raise ValueError("Lambdas must be a comma-separated string or a list of floats.")

    trial_data_records = []
    base_seed = config.get('seed', int(datetime.now().timestamp())) + master_seed_offset

    print(f"Running simulation for agent: {agent_to_simulate}")
    print(f"Parameters: ws={ws}, wn={wn}, n_trials_per_lambda={n_trials_per_lambda}, reward_A={reward_A}, reward_B={reward_B}")
    print(f"Lambdas: {lambdas}")
    print(f"Base seed for this agent run: {base_seed}")

    for i_lambda, lambda_conflict in enumerate(lambdas):
        agent_seed = base_seed + i_lambda + 1

        if agent_to_simulate.upper() == 'NES':
            agent_instance = NESAgent(w_s=ws, w_n=wn, seed=agent_seed)
        elif agent_to_simulate.upper() == 'DDM':
            agent_instance = DDMAgent(seed=agent_seed)
        else:
            raise ValueError(f"Unknown agent type: {agent_to_simulate}.")

        for trial_num in range(1, n_trials_per_lambda + 1):
            if agent_to_simulate.upper() == 'NES':
                choice, drift = agent_instance.choose_action(lambda_conflict)
            elif agent_to_simulate.upper() == 'DDM':
                choice, drift = agent_instance.choose_action(reward_A, reward_B)

            chosen_reward = reward_A if choice == 0 else reward_B

            trial_data_records.append({
                'agent': agent_to_simulate.upper(),
                'lambda': lambda_conflict,
                'trial': trial_num,
                'drift': drift,
                'choice': choice,
                'conflict': lambda_conflict,
                'config_reward_A': reward_A,
                'config_reward_B': reward_B,
                'chosen_reward': chosen_reward
            })
        print(f"  Lambda {lambda_conflict:.2f} completed for {agent_to_simulate}.")
    return pd.DataFrame(trial_data_records)

def plot_results(summary_df, trial_data_df, output_filename):
    if summary_df.empty:
        print("Summary data is empty. Skipping plot generation.")
        return
    # Plotting can still proceed if trial_data_df is empty, but CIs will be skipped.
    # if trial_data_df.empty:
    #     print("Trial data is empty. Skipping CIs for plot.")

    plt.figure(figsize=(12, 7))
    agents = sorted(summary_df['agent'].unique(), key=lambda x: (x.upper() != 'DDM', x))

    def bootstrap_ci(data, n_bootstrap=1000, ci_level=0.95):
        if len(data) == 0: return (np.nan, np.nan, np.nan)
        bootstrap_means = np.array([np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)])
        lower_bound = np.percentile(bootstrap_means, (1 - ci_level) / 2 * 100)
        upper_bound = np.percentile(bootstrap_means, (1 + ci_level) / 2 * 100)
        return np.mean(data), lower_bound, upper_bound

    for agent_name in agents:
        agent_summary_data = summary_df[summary_df['agent'] == agent_name]
        mean_pct_B_values = agent_summary_data.set_index('lambda')['pct_B_chosen']
        plot_lambdas = sorted(agent_summary_data['lambda'].unique())
        ordered_mean_pct_B = [mean_pct_B_values.get(l, np.nan) for l in plot_lambdas]

        line, = plt.plot(plot_lambdas, ordered_mean_pct_B, marker='o', linestyle='-', label=agent_name, linewidth=2)

        if not trial_data_df.empty: # Only attempt CIs if trial_data_df is available
            agent_trial_data = trial_data_df[trial_data_df['agent'] == agent_name]
            cis_lower, cis_upper = [], []
            valid_cis = False
            for l_val in plot_lambdas:
                lambda_choices = agent_trial_data[agent_trial_data['lambda'] == l_val]['choice'].values
                if len(lambda_choices) > 0 :
                    _, ci_low, ci_up = bootstrap_ci(lambda_choices)
                    cis_lower.append(ci_low); cis_upper.append(ci_up)
                    if not np.isnan(ci_low): valid_cis = True # Mark if at least one CI is valid
                else:
                    cis_lower.append(np.nan); cis_upper.append(np.nan)

            if valid_cis: # Only plot fill_between if there are valid CI numbers
                 plt.fill_between(plot_lambdas, cis_lower, cis_upper, color=line.get_color(), alpha=0.2)

        if agent_name.upper() == 'NES':
            nes_summary = agent_summary_data.sort_values(by='lambda')
            if 'mean_drift' in nes_summary.columns: # Check if mean_drift is available
                closest_to_zero_drift_lambda = nes_summary.iloc[(nes_summary['mean_drift']).abs().argsort()[:1]]
                if not closest_to_zero_drift_lambda.empty:
                    inflection_lambda = closest_to_zero_drift_lambda['lambda'].values[0]
                    inflection_pct_B = closest_to_zero_drift_lambda['pct_B_chosen'].values[0]
                    if 0.2 < inflection_pct_B < 0.8:
                        plt.annotate(f'Inflection approx. λ={inflection_lambda:.2f}',
                                     xy=(inflection_lambda, inflection_pct_B),
                                     xytext=(inflection_lambda + 0.1, inflection_pct_B + 0.2),
                                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.xlabel("λ (Conflict Level)")
    plt.ylabel("% Norm-Violating Choices (Action B)")
    plt.title("Agent Choice Behavior vs. Conflict Level (with 95% CI)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(-0.05, 1.05)
    unique_lambdas_overall = sorted(summary_df['lambda'].unique())
    plt.xticks(unique_lambdas_overall)
    if 'DDM' in [a.upper() for a in agents]:
        plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='DDM Random (0.5)')
    plt.legend()
    try:
        plt.savefig(output_filename); print(f"Plot saved to {output_filename}")
    except Exception as e: print(f"Error saving plot: {e}")
    finally: plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run NES vs DDM Behavioral Prediction Simulation")
    parser.add_argument('--agent_type', type=str, help="Agent: NES, DDM, or BOTH.")
    parser.add_argument('--ws', type=float, help="Salience weight (w_s) for NES.")
    parser.add_argument('--wn', type=float, help="Norm weight (w_n) for NES.")
    parser.add_argument('--n_trials_per_lambda', type=int, help="Number of trials per lambda.")
    parser.add_argument('--config_file', type=str, default='experiments/behavioral_prediction/configs/prediction_task.yaml', help="Path to YAML config.")
    parser.add_argument('--output_dir', type=str, default='experiments/behavioral_prediction/outputs', help="Output directory.")
    parser.add_argument('--run_id', type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'), help="Unique ID for run filenames.")
    parser.add_argument('--reward_A', type=float, help="Reward for action A (neutral/control)")
    parser.add_argument('--reward_B', type=float, help="Reward for action B (norm-violating/conflict)")

    args = parser.parse_args()
    yaml_config = load_config(args.config_file)
    cli_args = {k: v for k, v in vars(args).items() if v is not None and k not in ['config_file', 'output_dir', 'run_id']}
    merged_config = {**yaml_config, **cli_args}

    if args.ws is not None: merged_config['ws'] = float(args.ws)
    if args.wn is not None: merged_config['wn'] = float(args.wn)
    if args.n_trials_per_lambda is not None: merged_config['n_trials_per_lambda'] = int(args.n_trials_per_lambda)
    if args.reward_A is not None: merged_config['reward_A'] = float(args.reward_A)
    if args.reward_B is not None: merged_config['reward_B'] = float(args.reward_B)

    os.makedirs(args.output_dir, exist_ok=True)
    run_id = args.run_id
    print(f"Starting simulation run: {run_id}")
    print("Effective Configuration:")
    for key, value in merged_config.items(): print(f"  {key}: {value}")

    all_trial_data_dfs = []
    agent_types_to_run = []
    requested_agent_type = merged_config.get('agent_type', 'NES').upper()

    if requested_agent_type == 'BOTH':
        agent_types_to_run = ['NES', 'DDM']
    elif requested_agent_type in ['NES', 'DDM']:
        agent_types_to_run = [requested_agent_type]
    else:
        print(f"Warning: Invalid agent_type '{requested_agent_type}' in config. Defaulting to NES.")
        agent_types_to_run = ['NES']; merged_config['agent_type'] = 'NES'

    master_seed_counter = 0
    for agent_type_sim in agent_types_to_run:
        print(f"--- Running for Agent: {agent_type_sim} ---")
        df_agent_trial_data = run_simulation(merged_config.copy(), agent_type_sim, master_seed_offset=master_seed_counter * 1000)
        all_trial_data_dfs.append(df_agent_trial_data)
        master_seed_counter +=1

    if not all_trial_data_dfs:
        print("No simulation data generated."); return

    df_all_trials = pd.concat(all_trial_data_dfs, ignore_index=True)
    output_files = generate_output_filenames(args.output_dir, run_id, merged_config)

    trial_data_filename = output_files['trial_data']
    df_all_trials.to_csv(trial_data_filename, index=False)
    print(f"Full trial data saved to {trial_data_filename}")

    summary_df = df_all_trials.groupby(['agent', 'lambda']).agg(
        pct_B_chosen=('choice', 'mean'),
        mean_drift=('drift', 'mean'),
        std_drift=('drift', 'std')
    ).reset_index()

    if 'NES' in summary_df['agent'].unique():
        nes_summary_data = summary_df[summary_df['agent'] == 'NES'].copy()
        nes_slope = calculate_nes_slope(nes_summary_data)
        summary_df.loc[summary_df['agent'] == 'NES', 'nes_violation_slope'] = nes_slope
        nes_cutoff_lambda = find_nes_lambda_cutoff(nes_summary_data, threshold=0.5, target_lt_threshold=0.2)
        summary_df.loc[summary_df['agent'] == 'NES', 'nes_lambda_cutoff_lt20pct'] = nes_cutoff_lambda

    for col_name in ['nes_violation_slope', 'nes_lambda_cutoff_lt20pct']:
        if col_name not in summary_df.columns:
            summary_df[col_name] = np.nan
        summary_df[col_name] = summary_df[col_name].astype(float)

    summary_filename = output_files['summary']
    summary_df.to_csv(summary_filename, index=False)
    print(f"Summary statistics saved to {summary_filename}")

    plot_filename = output_files['plot']
    plot_results(summary_df, df_all_trials, plot_filename)
    print(f"Simulation and data logging complete for run {run_id}.")

if __name__ == "__main__":
    main()
