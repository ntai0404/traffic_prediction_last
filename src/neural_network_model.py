import pandas as pd
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Đường dẫn lưu hình ảnh
output_dir = 'D:/machine learning/traffic_prediction_streamlit/web/static/png'
os.makedirs(output_dir, exist_ok=True)

# Đọc dữ liệu
df = pd.read_csv('./data/traffic_data.csv')
df['time_of_day'] = df['time_of_day'].astype(int)

# Xác định các đặc trưng và nhãn
X = df[['is_holiday', 'air_pollution_index', 'temperature', 'rain_p_h', 'visibility_in_miles', 'time_of_day']]
y = df['traffic_condition']

# Chia dữ liệu thành các tập huấn luyện, validation và kiểm tra
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.6, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Thiết lập các tham số cho GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],  # Giảm kích thước lớp ẩn
    'activation': ['relu'],
    'alpha': [0.0001, 0.001],  # Giảm alpha để tăng tốc độ huấn luyện
    'max_iter': [500],  # Giảm số lần lặp
    'learning_rate_init': [0.01],  # Sử dụng learning rate cao hơn
    'early_stopping': [True],
    'validation_fraction': [0.1],
    'n_iter_no_change': [10],  # Số lần lặp không thay đổi để dừng
    'batch_size': ['auto', 32],  # Batch size tự động
}

# Khởi tạo mô hình MLP
mlp = MLPClassifier(random_state=42, warm_start=True)  # Dùng warm_start để tiết kiệm thời gian huấn luyện
grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Lấy mô hình tốt nhất
model = grid_search.best_estimator_

# Dự đoán trên các tập huấn luyện, validation và kiểm tra
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Báo cáo kết quả
train_report = classification_report(y_train, y_train_pred, output_dict=False)
val_report = classification_report(y_val, y_val_pred, output_dict=False)
test_report = classification_report(y_test, y_test_pred, output_dict=False)

train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Lưu báo cáo vào tệp
with open('./src/neural_network.txt', 'w') as report_file:
    report_file.write("Training:\n")
    report_file.write(train_report)
    report_file.write(f"\nAccuracy: {train_accuracy:.2f}       Total samples: {len(y_train)}\n")
    
    report_file.write("Validation:\n")
    report_file.write(val_report)
    report_file.write(f"\nAccuracy: {val_accuracy:.2f}       Total samples: {len(y_val)}\n")
    
    report_file.write("Testing:\n")
    report_file.write(test_report)
    report_file.write(f"\nAccuracy: {test_accuracy:.2f}       Total samples: {len(y_test)}\n")

print(f'Neural Network Training Accuracy: {train_accuracy:.2f}')
print(f'Neural Network Validation Accuracy: {val_accuracy:.2f}')
print(f'Neural Network Testing Accuracy: {test_accuracy:.2f}')

# Tạo ma trận nhầm lẫn
conf_matrix = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn')

confusion_matrix_image_path = os.path.join(output_dir, 'neural_network_confusion_matrix.png')
plt.savefig(confusion_matrix_image_path)
plt.close()

# Tạo learning curve
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=3, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training score')  # Màu xanh dương
plt.plot(train_sizes, test_scores_mean, 'o-', color='green', label='Validation score')  # Màu xanh lá

# Vẽ vùng màu cho độ lệch chuẩn
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color='blue', alpha=0.1)  # Vùng màu xanh dương
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color='green', alpha=0.1)  # Vùng màu xanh lá

plt.xlabel('Số lượng mẫu huấn luyện')
plt.ylabel('Điểm số')
plt.title('Learning Curve')
plt.legend(loc='best')

learning_curve_image_path = os.path.join(output_dir, 'neural_network_learning_curve.png')
plt.savefig(learning_curve_image_path)
plt.close()

# Lưu mô hình vào file
with open('./src/neural_network_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Neural Network model saved successfully!")






