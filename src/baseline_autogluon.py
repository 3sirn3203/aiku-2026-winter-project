import os
import pandas as pd
from autogluon.tabular import TabularPredictor

def main():
    # Define paths
    DATA_DIR = "./data"
    TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
    TEST_PATH = os.path.join(DATA_DIR, "test.csv")
    SUBMISSION_DIR = os.path.join(DATA_DIR, "submissions")
    OUTPUT_PATH = os.path.join(SUBMISSION_DIR, "autogluon_submission.csv")
    MODEL_PATH = os.path.join("generated", "autogluon_model")
    NUM_GPUS = 1

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

    # Define label
    label = 'completed'

    # Check if label exists in train data
    if label not in train_data.columns:
        raise ValueError(f"Label column '{label}' not found in train data")

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
    print("Training AutoGluon predictor...")
    # presets='best_quality' usually gives better results but takes longer. 
    # 'medium_quality' or default is faster. I'll stick to default for a baseline.
    predictor = TabularPredictor(label=label, path=MODEL_PATH).fit(
        train_data,
        num_gpus=NUM_GPUS,
    )

    print(predictor.leaderboard(silent=True))

    # Predict
    print("Predicting on test data...")
    predictions = predictor.predict(test_data_features)

    # Create submission file
    submission = pd.DataFrame({
        'ID': test_ids,
        'completed': predictions
    })

    # Save submission
    print(f"Saving submission to {OUTPUT_PATH}...")
    submission.to_csv(OUTPUT_PATH, index=False)

    print("Done!")

if __name__ == "__main__":
    main()
