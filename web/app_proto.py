import streamlit as st
import pandas as pd
import pickle
import matplotlib
import numpy as np
from sklearn.model_selection import train_test_split  # Nhập khẩu cần thiết
import os

matplotlib.use('Agg')  # Để vẽ biểu đồ mà không cần GUI

# Load các mô hình
models = {
    'Perceptron': pickle.load(open('../src/perceptron_model.pkl', 'rb')),
    'ID3': pickle.load(open('../src/id3_model.pkl', 'rb')),
    'Neural Network': pickle.load(open('../src/neural_network_model.pkl', 'rb')),
    'Ensemble Model': pickle.load(open('../src/ensemble_model.pkl', 'rb'))
}

# Hàm đọc báo cáo
def read_report(model_name):
    try:
        if model_name.lower() == "ensemble model":
            with open('../src/ensemble.txt', 'r') as file:
                report = file.read()
        else:
            with open(f'../src/{model_name.lower().replace(" ", "_")}.txt', 'r') as file:
                report = file.read()
        return report
    except FileNotFoundError:
        return "Report not found."

# Tiêu đề ứng dụng
st.title("Dự đoán tình trạng giao thông")

# Chọn mô hình
selected_model = st.selectbox("Chọn mô hình để dự đoán:", list(models.keys()))

# Tải dữ liệu
df = pd.read_csv('../data/traffic_data.csv')

# Hiển thị thông tin dữ liệu (nếu cần)
st.write("Dữ liệu đầu vào:")
st.dataframe(df)

# Thông tin chi tiết về mô hình đã chọn
st.write(f"Thông tin về mô hình: {selected_model}")

# Dự đoán
if st.button('Dự đoán'):
    X = df[['is_holiday', 'air_pollution_index', 'temperature', 'rain_p_h', 'visibility_in_miles', 'time_of_day']]
    y = df['traffic_condition']

    # Chia dữ liệu
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Chọn mô hình
    model = models[selected_model]
    
    # Dự đoán
    predictions = model.predict(X_test)
    
    # Đọc báo cáo
    report = read_report(selected_model)

    # Đường dẫn tới hình ảnh ma trận nhầm lẫn và biểu đồ học
    confusion_matrix_image = f'static/png/{selected_model.lower().replace(" ", "_")}_confusion_matrix.png'
    learning_curve_image = f'static/png/{selected_model.lower().replace(" ", "_")}_learning_curve.png'
    
    # Hiển thị kết quả dự đoán
    st.write(f"Kết quả dự đoán: {predictions[0]}")
    
    # Hiển thị báo cáo
    st.write("Báo cáo kết quả:")
    st.text(report)

    # Hiển thị hình ảnh
    if os.path.exists(confusion_matrix_image):
        st.image(confusion_matrix_image, caption='Ma trận nhầm lẫn')
    else:
        st.write("Không tìm thấy hình ảnh ma trận nhầm lẫn.")

    if os.path.exists(learning_curve_image):
        st.image(learning_curve_image, caption='Đường học')
    else:
        st.write("Không tìm thấy hình ảnh đường học.")

# Chạy ứng dụng
if __name__ == '__main__':
    st.write("Chạy ứng dụng trên Streamlit.")





