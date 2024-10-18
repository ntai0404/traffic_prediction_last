
import streamlit as st
import pandas as pd
import pickle
import os

# Lấy đường dẫn đến thư mục hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

# Hàm tải mô hình
def load_model(model_name):
    try:
        return pickle.load(open(os.path.join(current_dir, f'../src/{model_name}.pkl'), 'rb'))
    except FileNotFoundError:
        st.error(f"Mô hình '{model_name}' không tìm thấy.")
        return None

# Load các mô hình
models = {
    'Perceptron': load_model('perceptron_model'),
    'ID3': load_model('id3_model'),
    'Neural Network': load_model('neural_network_model'),
    'Ensemble Model': load_model('ensemble_model')
}

# Hàm đọc báo cáo
def read_report(model_name):
    report_path = os.path.join(current_dir, f'../src/{model_name.lower().replace(" ", "_")}.txt')
    try:
        with open(report_path, 'r') as file:
            return file.read()
    except FileNotFoundError as e:
        st.error(f"Có lỗi xảy ra: {e}")
        return None  # Hoặc có thể trả về một chuỗi rỗng

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
    </style>
    """, unsafe_allow_html=True
)

# Tiêu đề ứng dụng
st.title("Dự đoán giao thông")
st.markdown('<div class="header"><h1>Dự đoán giao thông</h1></div>', unsafe_allow_html=True)

# Chọn mô hình
selected_model = st.selectbox("Chọn mô hình:", list(models.keys()))

# Nhập dữ liệu
is_holiday = st.text_input("Ngày lễ (1 nếu là ngày lễ, 0 nếu không):", "0")
air_pollution_index = st.text_input("Chỉ số ô nhiễm không khí:", "0.0")
temperature = st.text_input("Nhiệt độ (°C):", "0.0")
rain_p_h = st.text_input("Lượng mưa (mm/giờ):", "0.0")
visibility_in_miles = st.text_input("Tầm nhìn (dặm):", "0.0")
time_of_day = st.text_input("Thời gian trong ngày (0-3):", "0")

# Dự đoán
if st.button('Dự đoán'):
    try:
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
        if model:
            predictions = model.predict(input_data)
            condition = {0: "Thông thoáng", 1: "Đông đúc", 2: "Ùn tắc"}
            st.write(f"Kết quả dự đoán: {condition.get(predictions[0], 'Không xác định')}")

            # Đọc báo cáo
            report = read_report(selected_model)
            if report:
                st.write("Báo cáo mô hình:")
                st.text(report)

            # Hiển thị hình ảnh
            confusion_matrix_image = os.path.join(current_dir, f'static/png/{selected_model.lower().replace(" ", "_")}_confusion_matrix.png')
            learning_curve_image = os.path.join(current_dir, f'static/png/{selected_model.lower().replace(" ", "_")}_learning_curve.png')

            # Kiểm tra và hiển thị hình ảnh
            if os.path.exists(confusion_matrix_image):
                st.image(confusion_matrix_image, caption='Ma trận nhầm lẫn')
            else:
                st.write(f"Không tìm thấy hình ảnh ma trận nhầm lẫn tại: {confusion_matrix_image}")

            if os.path.exists(learning_curve_image):
                st.image(learning_curve_image, caption='Đường học')
            else:
                st.write(f"Không tìm thấy hình ảnh đường học tại: {learning_curve_image}")

    except ValueError as e:
        st.error(f"Có lỗi xảy ra: {e}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra: {e}")

# Chạy ứng dụng
if __name__ == '__main__':
    st.write("Chạy ứng dụng trên Streamlit.")

