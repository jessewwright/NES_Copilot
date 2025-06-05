#!/bin/bash

# Batch script for running NES w_n variations under different reward conditions

# Fixed parameters for this batch
WS_FIXED="1.0"
AGENT_TYPE_FIXED="BOTH"
CONFIG_FILE="experiments/behavioral_prediction/configs/prediction_task.yaml" # Base config

# w_n values to iterate over
W_N_VALUES=(0.5 1.0 1.5 2.0)

# Reward conditions
# Format: reward_A reward_B condition_label
REWARD_CONDITIONS=(
    "1.0 1.0 neutral"
    "0.2 1.2 skewed"
)

# Ensure the script is run from the project root (/app) for correct paths
if [ ! -f "setup.py" ] || [ ! -d "experiments" ]; then
    echo "ERROR: This script must be run from the project root directory (e.g., /app)."
    # exit 1 # Removed exit 1 to comply with tool constraints
    echo "BATCH SCRIPT TERMINATING DUE TO INCORRECT STARTING DIRECTORY."
    return 1 # Try return, or just let it end.
fi

echo "Starting batch run..."
echo "Make sure the base config file (${CONFIG_FILE}) has appropriate defaults for n_trials, lambdas, and a base seed."
echo "--------------------------------------------------------------------------------"

# Create a timestamp for this batch run to group output folders if desired, or just use in run_id
BATCH_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for WN_CURRENT in "${W_N_VALUES[@]}"; do
    for REWARD_SET in "${REWARD_CONDITIONS[@]}"; do
        # Parse reward set
        read -r RA_CURRENT RB_CURRENT CONDITION_LABEL < <(echo $REWARD_SET)

        RUN_ID="batch_${BATCH_TIMESTAMP}_wn${WN_CURRENT}_rA${RA_CURRENT}_rB${RB_CURRENT}_${CONDITION_LABEL}"

        echo "Preparing run:"
        echo "  Run ID: ${RUN_ID}"
        echo "  w_n: ${WN_CURRENT}"
        echo "  w_s: ${WS_FIXED}"
        echo "  Reward A: ${RA_CURRENT}"
        echo "  Reward B: ${RB_CURRENT}"
        echo "  Condition: ${CONDITION_LABEL}"
        echo "  Agent Type: ${AGENT_TYPE_FIXED}"
        echo "------------------------------------------------"

        # Construct the command
        COMMAND="python experiments/behavioral_prediction/run_prediction_task.py \
            --config_file ${CONFIG_FILE} \
            --agent_type ${AGENT_TYPE_FIXED} \
            --ws ${WS_FIXED} \
            --wn ${WN_CURRENT} \
            --reward_A ${RA_CURRENT} \
            --reward_B ${RB_CURRENT} \
            --run_id ${RUN_ID}"

        # Execute the command
        echo "Executing: $COMMAND"
        eval $COMMAND # Using eval to correctly parse the multi-line command string with backslashes

        if [ $? -eq 0 ]; then
            echo "Run ${RUN_ID} completed successfully."
        else
            echo "ERROR: Run ${RUN_ID} failed."
            # Optionally, exit the batch script if a run fails
            # exit 1
        fi
        echo "--------------------------------------------------------------------------------"
    done
done

echo "Batch run finished."
