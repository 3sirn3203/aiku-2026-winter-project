import os
import argparse
import yaml
import pandas as pd
from autogluon.tabular import TabularPredictor

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='AutoGluon Baseline Script')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load config
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)

    # Define paths from config
    TRAIN_PATH = config['data']['train_path']
    TEST_PATH = config['data']['test_path']
    SUBMISSION_DIR = config['data']['submission_dir']
    OUTPUT_PATH = config['data']['output_path']
    
    LABEL = config['model']['label']
    EVAL_METRIC = config['model']['eval_metric']

    NUM_GPUS = config['training']['num_gpus']
    PRESETS = config['training']['presets']
    FOLD_STRATEGY = config['training']['fold_fitting_strategy']
    PROBLEM_TYPE = config['training']['problem_type']

    # Create submission directory if it doesn't exist
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Load data
    print("Loading data...")
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Train file not found at {TRAIN_PATH}")
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Test file not found at {TEST_PATH}")

    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    # Check if label exists in train data
    if LABEL not in train_data.columns:
        raise ValueError(f"Label column '{LABEL}' not found in train data")

    # Drop ID column from training data as it's not a feature
    if 'ID' in train_data.columns:
        train_data = train_data.drop(columns=['ID'])
    
    # Keep ID for submission
    if 'ID' in test_data.columns:
        test_ids = test_data['ID']
        test_data_features = test_data.drop(columns=['ID'])
    else:
        raise ValueError("ID column not found in test data")

    # Train
    print(f"Training AutoGluon predictor with presets='{PRESETS}'...")
    predictor = TabularPredictor(
        label=LABEL,
        eval_metric=EVAL_METRIC,
        problem_type=PROBLEM_TYPE,
    ).fit(
        train_data,
        num_gpus=NUM_GPUS,
        presets=PRESETS,
        ag_args_ensemble={'fold_fitting_strategy': FOLD_STRATEGY}
    )

    print(predictor.leaderboard(silent=True))

    # Predict
    print("Predicting on test data...")
    predictions = predictor.predict(test_data_features)

    # Create submission file
    submission = pd.DataFrame({
        'ID': test_ids,
        LABEL: predictions
    })

    # Save submission
    print(f"Saving submission to {OUTPUT_PATH}...")
    submission.to_csv(OUTPUT_PATH, index=False)

    print("Done!")

if __name__ == "__main__":
    main()
