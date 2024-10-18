import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('./data/traffic_data.csv')

# Features and target (only keep the 5 important features)
X = df[['is_holiday', 'air_pollution_index', 'temperature', 'rain_p_h', 'visibility_in_miles']]
y = df['traffic_condition']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# List of model files
model_files = {
    'Perceptron': './src/perceptron_model.pkl',
    'ID3': './src/id3_model.pkl',
    'Neural Network': './src/neural_network_model.pkl',
    'Ensemble Model': './src/ensemble_model.pkl'
}

# Evaluate each model
for model_name, model_file in model_files.items():
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{model_name} Accuracy: {accuracy:.2f}')
    
    # Print detailed classification report
    print(f'Classification Report for {model_name}:\n', classification_report(y_test, y_pred))




