from src.preprocessing import load_data, preprocess_data, feature_selection
from src.model_training import train_and_evaluate

def main():
    # Load and preprocess data
    data = load_data('data/raw/dataset.csv')  # Adjust the path to your dataset
    X, y = preprocess_data(data)

    # Optionally, perform feature selection
    selected_features = feature_selection(X, y)

    # Train and evaluate models
    performance_results = train_and_evaluate(X[selected_features], y)

    # Output or save results
    print(performance_results)

if __name__ == "__main__":
    main()
