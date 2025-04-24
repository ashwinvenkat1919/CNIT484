from data_processing import load_and_process_data
from model_training import train_model
from evaluate import evaluate_model

def main():
    print("Processing data...")
    train_dataset, test_dataset, test_df = load_and_process_data()
    print("Training model...")
    train_model(train_dataset, test_dataset)

if __name__ == "__main__":
    main()