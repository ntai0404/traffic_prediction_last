import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Đọc dữ liệu và chuyển đổi kiểu dữ liệu
df = pd.read_csv('./data/traffic_data.csv')
df['time_of_day'] = df['time_of_day'].astype(int)

# Tách dữ liệu thành các biến đầu vào và đầu ra
X = df[['is_holiday', 'air_pollution_index', 'temperature', 'rain_p_h', 'visibility_in_miles', 'time_of_day']]
y = df['traffic_condition']

# Chia dữ liệu thành tập huấn luyện, xác thực và kiểm tra (70% huấn luyện, 15% kiểm tra, 15% xác thực)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Khởi tạo mô hình ID3 (DecisionTreeClassifier)
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=10, random_state=42)
model.fit(X_train, y_train)

# Dự đoán cho tập huấn luyện, xác thực và kiểm tra
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Tạo báo cáo và tính độ chính xác
train_report = classification_report(y_train, y_train_pred)
val_report = classification_report(y_val, y_val_pred)
test_report = classification_report(y_test, y_test_pred)

train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Lưu báo cáo vào tệp
with open('./src/id3.txt', 'w') as report_file:
    report_file.write("Training Report:\n")
    report_file.write(train_report)
    report_file.write(f"\nAccuracy: {train_accuracy:.2f}       Total samples: {len(y_train)}\n")
    
    report_file.write("\nValidation Report:\n")
    report_file.write(val_report)
    report_file.write(f"\nAccuracy: {val_accuracy:.2f}       Total samples: {len(y_val)}\n")
    
    report_file.write("\nTesting Report:\n")
    report_file.write(test_report)
    report_file.write(f"\nAccuracy: {test_accuracy:.2f}       Total samples: {len(y_test)}\n")

print(f'ID3 Training Accuracy: {train_accuracy:.2f}')
print(f'ID3 Validation Accuracy: {val_accuracy:.2f}')
print(f'ID3 Testing Accuracy: {test_accuracy:.2f}')

# Tạo thư mục cho hình ảnh (nếu chưa tồn tại)
output_dir = './web/static/png'
os.makedirs(output_dir, exist_ok=True)

# Vẽ và lưu ma trận nhầm lẫn
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn')

confusion_matrix_image_path = os.path.join(output_dir, 'id3_confusion_matrix.png')
plt.savefig(confusion_matrix_image_path)
plt.close()

print(f"Ma trận nhầm lẫn đã được lưu tại: {confusion_matrix_image_path}")

# Vẽ và lưu biểu đồ learning curve
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
)

train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_mean = test_scores.mean(axis=1)
test_scores_std = test_scores.std(axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, label='Training Accuracy', color='blue')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color='blue', alpha=0.2)
plt.plot(train_sizes, test_scores_mean, label='Cross-validation Accuracy', color='green')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color='green', alpha=0.2)
plt.title('Learning Curve - ID3 Model')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.legend(loc='best')

learning_curve_image_path = os.path.join(output_dir, 'id3_learning_curve.png')
plt.savefig(learning_curve_image_path)
plt.close()

print(f"Learning curve đã được lưu tại: {learning_curve_image_path}")

# Lưu mô hình đã huấn luyện
with open('./src/id3_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("ID3 model saved successfully!")






