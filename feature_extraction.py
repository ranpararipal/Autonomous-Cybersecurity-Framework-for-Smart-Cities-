import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv("data.csv")

# Feature extraction and normalization
features = data[['feature1', 'feature2', 'feature3']]
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Prepare train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(normalized_features, data['label'], test_size=0.2)
