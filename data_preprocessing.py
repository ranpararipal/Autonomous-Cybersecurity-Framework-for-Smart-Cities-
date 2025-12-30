import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)
    
    # Feature extraction: Selecting relevant features
    features = data[['Feature1', 'Feature2', 'Feature3']]  # Replace with actual features
    labels = data['Label']  # Replace with the label column
    
    # Data normalization
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(normalized_features, labels, test_size=0.2)
    
    return X_train, X_test, y_train, y_test

# Example usage
file_path = 'your_dataset.csv'
X_train, X_test, y_train, y_test = preprocess_data(file_path)
