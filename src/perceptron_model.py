import pandas as pd
import pickle
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Đọc dataset
df = pd.read_csv('./data/traffic_data.csv')

# Chia dữ liệu thành X và y
X = df.drop('traffic_condition', axis=1)
y = df['traffic_condition']

# In ra các điều kiện giao thông duy nhất
print(f"Unique traffic conditions: {y.unique()}")

# Chia tập dữ liệu thành training, validation, và test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Khởi tạo và huấn luyện mô hình Perceptron
model = Perceptron(max_iter=1000, eta0=0.1, tol=1e-3, random_state=42)
model.fit(X_train, y_train)

# Dự đoán trên các tập dữ liệu
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Lấy tất cả các nhãn theo thứ tự
all_labels = sorted(y.unique())

# Tạo báo cáo classification cho training, validation, và test sets
train_report = classification_report(y_train, y_train_pred, labels=all_labels, output_dict=False)
val_report = classification_report(y_val, y_val_pred, labels=all_labels, output_dict=False)
test_report = classification_report(y_test, y_test_pred, labels=all_labels, output_dict=False)

# Tính toán độ chính xác (accuracy)
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Lưu báo cáo vào file
with open('./src/perceptron.txt', 'w') as report_file:
    report_file.write("Training Report:\n")
    report_file.write(train_report)
    report_file.write(f"\nAccuracy: {train_accuracy:.2f}       Total samples: {len(y_train)}\n")
    
    report_file.write("Validation Report:\n")
    report_file.write(val_report)
    report_file.write(f"\nAccuracy: {val_accuracy:.2f}       Total samples: {len(y_val)}\n")
    
    report_file.write("Testing Report:\n")
    report_file.write(test_report)
    report_file.write(f"\nAccuracy: {test_accuracy:.2f}       Total samples: {len(y_test)}\n")

# In ra kết quả độ chính xác
print(f'Perceptron Training Accuracy: {train_accuracy:.2f}')
print(f'Perceptron Validation Accuracy: {val_accuracy:.2f}')
print(f'Perceptron Testing Accuracy: {test_accuracy:.2f}')

# Tạo thư mục để lưu hình ảnh nếu chưa tồn tại
output_dir = './web/static/png'
os.makedirs(output_dir, exist_ok=True)

# Tạo và lưu ma trận nhầm lẫn (confusion matrix)
conf_matrix = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=all_labels, yticklabels=all_labels)
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn')

# Lưu ma trận nhầm lẫn dưới dạng hình ảnh
confusion_matrix_image_path = os.path.join(output_dir, 'perceptron_confusion_matrix.png')
plt.savefig(confusion_matrix_image_path)
plt.close()

print(f"Ma trận nhầm lẫn đã được lưu tại: {confusion_matrix_image_path}")

# Tạo và lưu đường cong học (learning curve)
train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

# Tính toán trung bình và độ lệch chuẩn của độ chính xác
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training Accuracy', color='blue')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color='blue', alpha=0.2)
plt.plot(train_sizes, val_scores_mean, label='Validation Accuracy', color='green')
plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, color='green', alpha=0.2)

plt.title('Learning Curve - Perceptron Model')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend(loc='best')

# Lưu đường cong học dưới dạng hình ảnh
learning_curve_image_path = os.path.join(output_dir, 'perceptron_learning_curve.png')
plt.savefig(learning_curve_image_path)
plt.close()

print(f"Learning curve đã được lưu tại: {learning_curve_image_path}")

# Lưu mô hình đã huấn luyện
with open('./src/perceptron_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Perceptron model saved successfully!")





















