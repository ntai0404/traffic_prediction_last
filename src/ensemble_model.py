import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

output_dir = 'D:/machine learning/traffic_prediction_streamlit/web/static/png'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv('./data/traffic_data.csv')

X = df[['is_holiday', 'air_pollution_index', 'temperature', 'rain_p_h', 'visibility_in_miles', 'time_of_day']]
y = df['traffic_condition']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

with open('./src/perceptron_model.pkl', 'rb') as file:
    perceptron_model = pickle.load(file)

with open('./src/id3_model.pkl', 'rb') as file:
    decision_tree_model = pickle.load(file)

with open('./src/neural_network_model.pkl', 'rb') as file:
    neural_network_model = pickle.load(file)

base_models = [
    ('perceptron', perceptron_model),
    ('decision_tree', decision_tree_model),
    ('neural_network', neural_network_model)
]

meta_model = decision_tree_model
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

stacking_model.fit(X_train, y_train)

y_train_pred = stacking_model.predict(X_train)
y_val_pred = stacking_model.predict(X_val)
y_test_pred = stacking_model.predict(X_test)

train_report = classification_report(y_train, y_train_pred, output_dict=False)
val_report = classification_report(y_val, y_val_pred, output_dict=False)
test_report = classification_report(y_test, y_test_pred, output_dict=False)

train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

with open('./src/ensemble_report.txt', 'w') as report_file:
    report_file.write("Training Report:\n")
    report_file.write(train_report)
    report_file.write(f"\nAccuracy: {train_accuracy:.2f}       Total samples: {len(y_train)}\n")
    
    report_file.write("Validation Report:\n")
    report_file.write(val_report)
    report_file.write(f"\nAccuracy: {val_accuracy:.2f}       Total samples: {len(y_val)}\n")
    
    report_file.write("Testing Report:\n")
    report_file.write(test_report)
    report_file.write(f"\nAccuracy: {test_accuracy:.2f}       Total samples: {len(y_test)}\n")

print(f'Stacking Model Training Accuracy: {train_accuracy:.2f}')
print(f'Stacking Model Validation Accuracy: {val_accuracy:.2f}')
print(f'Stacking Model Testing Accuracy: {test_accuracy:.2f}')

conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn - Ensemble Model')

confusion_matrix_image_path = os.path.join(output_dir, 'ensemble_confusion_matrix.png')
plt.savefig(confusion_matrix_image_path)
plt.close()
print(f"Confusion matrix saved at: {confusion_matrix_image_path}")

train_sizes, train_scores, val_scores = learning_curve(stacking_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training Accuracy', color='blue')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color='blue', alpha=0.2)
plt.plot(train_sizes, val_scores_mean, label='Validation Accuracy', color='green')
plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, color='green', alpha=0.2)

plt.title('Learning Curve - Ensemble Model')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend(loc='best')

learning_curve_image_path = os.path.join(output_dir, 'ensemble_learning_curve.png')
plt.savefig(learning_curve_image_path)
plt.close()
print(f"Learning curve saved at: {learning_curve_image_path}")

try:
    with open('./src/ensemble_model.pkl', 'wb') as file:
        pickle.dump(stacking_model, file)
    print("Ensemble model saved successfully!")
except Exception as e:
    print(f"Error saving model: {e}")















