import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split

# Định nghĩa đường dẫn tuyệt đối tới thư mục hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

# Sử dụng đường dẫn tuyệt đối để tải các mô hình
models = {
    'Perceptron': pickle.load(open(os.path.join(current_dir, '../src/perceptron_model.pkl'), 'rb')),
    'ID3': pickle.load(open(os.path.join(current_dir, '../src/id3_model.pkl'), 'rb')),
    'Neural Network': pickle.load(open(os.path.join(current_dir, '../src/neural_network_model.pkl'), 'rb')),
    'Ensemble Model': pickle.load(open(os.path.join(current_dir, '../src/ensemble_model.pkl'), 'rb'))
}

# Hàm đọc báo cáo
def read_report(model_name):
    try:
        report_path = os.path.join(current_dir, f"../src/{model_name.lower().replace(' ', '_')}.txt")
        if model_name.lower() == "ensemble model":
            report_path = os.path.join(current_dir, '../src/ensemble.txt')
        with open(report_path, 'r') as file:
            report = file.read()
        return report
    except FileNotFoundError:
        return "Report not found."

# Tiêu đề ứng dụng
st.set_page_config(page_title="Dự đoán giao thông", page_icon="./static/logo2.jpg")

# Thêm CSS
st.markdown(
    """
    <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f4f4f4;
        margin: 0;
        padding: 0;
    }

    .web {
        max-width: 800px;
        margin: 20px auto;
        padding: 20px;
        background-color: #fff;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .header {
        text-align: center;
        margin-bottom: 20px;
        background-image: url(https://img4.thuthuatphanmem.vn/uploads/2020/08/27/anh-nen-dep-ve-ha-noi_054022808.jpg);
        background-size: cover; 
        background-position: center;
        padding: 40px 20px; 
        color: #fff;
        border-radius: 8px;
    }

    .header h1 {
        font-size: 44px;
        margin: 0;
        color: #FFFAFA;
    }

    .container {
        display: flex;
        flex-direction: column;
    }
    
    select {
        color: #888;
    }
    
    label {
        margin-bottom: 5px;
        font-weight: bold;
    }

    input[type="text"], select {
        width: 100%;
        padding: 10px;
        margin-bottom: 15px;
        border: 1px solid #ddd;
        border-radius: 4px;
        box-sizing: border-box;
    }

    input[type="submit"] {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 15px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }

    input[type="submit"]:hover {
        background-color: #45a049;
    }

    h2 {
        margin-top: 20px;
        color: #333;
    }

    p {
        font-size: 16px;
        color: #555;
    }
    </style>
    """, unsafe_allow_html=True
)

# Tiêu đề ứng dụng
st.title("Dự đoán giao thông")
st.markdown('<div class="header"><h1>Dự đoán giao thông</h1></div>', unsafe_allow_html=True)

# Chọn mô hình
selected_model = st.selectbox("Chọn mô hình:", list(models.keys()))

# Nhập dữ liệu
is_holiday = st.text_input("Ngày lễ (1 nếu là ngày lễ, 0 nếu không):")
air_pollution_index = st.text_input("Chỉ số ô nhiễm không khí:")
temperature = st.text_input("Nhiệt độ (°C):")
rain_p_h = st.text_input("Lượng mưa (mm/giờ):")
visibility_in_miles = st.text_input("Tầm nhìn (dặm):")
time_of_day = st.text_input("Thời gian trong ngày (0-3):")

# Đường dẫn tuyệt đối tới dữ liệu CSV
csv_path = os.path.join(current_dir, '../data/traffic_data.csv')
df = pd.read_csv(csv_path)

# Dự đoán
if st.button('Dự đoán'):
    # Chuyển đổi dữ liệu đầu vào thành DataFrame
    input_data = pd.DataFrame({
        'is_holiday': [int(is_holiday)],
        'air_pollution_index': [float(air_pollution_index)],
        'temperature': [float(temperature)],
        'rain_p_h': [float(rain_p_h)],
        'visibility_in_miles': [float(visibility_in_miles)],
        'time_of_day': [int(time_of_day)]
    })

    # Chọn mô hình
    model = models[selected_model]
    
    # Dự đoán
    predictions = model.predict(input_data)
    
    # Đọc báo cáo
    report = read_report(selected_model)

    # Đường dẫn tới hình ảnh ma trận nhầm lẫn và biểu đồ học
    confusion_matrix_image = os.path.join(current_dir, f'static/png/{selected_model.lower().replace(" ", "_")}_confusion_matrix.png')
    learning_curve_image = os.path.join(current_dir, f'static/png/{selected_model.lower().replace(" ", "_")}_learning_curve.png')
    
    # Hiển thị kết quả dự đoán
    st.write("Kết quả dự đoán:")
    condition = {0: "Thông thoáng", 1: "Đông đúc", 2: "Ùn tắc"}
    st.write(f"Điều kiện giao thông: {condition.get(predictions[0], 'Không xác định')}")
    
    # Hiển thị báo cáo
    st.write("Báo cáo mô hình:")
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