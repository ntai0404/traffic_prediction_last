import pandas as pd

# Đọc hai file CSV
df1 = pd.read_csv('./data/Traffic.csv')
df2 = pd.read_csv('./data/TrafficTwoMonth.csv')

# Hợp nhất hai dataset bằng cách nối thêm các hàng từ df2 vào df1
df_merged = pd.concat([df1, df2], ignore_index=True)

# Lưu dataset đã hợp nhất vào file mới (nếu cần)
df_merged.to_csv('./data/dataset.csv', index=False)

import pandas as pd
# Đọc file CSV
#df = pd.read_csv('./data/TrafficTwoMonth.csv')
df = pd.read_csv('./data/dataset.csv')
# Đếm số lần xuất hiện của từng chuỗi trong cột 'traffic_condition'
count_series = df['Traffic Situation'].value_counts()
#count_series = df['Date'].value_counts()
# Hiển thị kết quả
print(count_series)


import pandas as pd
# Đọc file CSV
df = pd.read_csv('./data/TrafficTwoMonth.csv')

# Lọc bỏ các dòng có giá trị 'Traffic Situation' là "low"
df_filtered = df[df['Traffic Situation'] != 'normal']

# Lưu lại file CSV sau khi lọc (nếu muốn ghi đè tệp cũ)
df_filtered.to_csv('./data/TrafficTwoMonth02.csv', index=False)

print("Đã loại bỏ các dòng có 'Traffic Situation' là 'normal'.")

import pandas as pd

# Đọc file CSV
df = pd.read_csv('./data/dataset.csv')
# Xóa cột 'Time'
df = df.drop(columns=['Time'])

# Chuyển đổi cột 'Traffic Situation' sang dạng số
traffic_situation_mapping = {
    'low': 1,
    'normal': 2,
    'high': 3,
    'heavy': 4
}
df['Traffic Situation'] = df['Traffic Situation'].map(traffic_situation_mapping)

# Chuyển đổi cột 'Day of the week' sang dạng số từ 1 đến 7
day_of_week_mapping = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}
df['Day of the week'] = df['Day of the week'].map(day_of_week_mapping)

# Lưu lại file CSV sau khi chuyển đổi
df.to_csv('./data/dataset.csv', index=False)

# Hiển thị dữ liệu sau khi chuyển đổi
print(df.head())

