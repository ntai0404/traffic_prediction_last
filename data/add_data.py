import pandas as pd
import numpy as np

def create_traffic_data(n_samples, traffic_condition):
    data = []
    for _ in range(n_samples):  
        is_holiday = np.random.choice([0, 1])
        air_pollution_index = round(np.random.uniform(50, 300), 1)
        temperature = round(np.random.uniform(15, 40), 1)
        rain_p_h = round(np.random.uniform(0, 50), 1)
        visibility_in_miles = round(np.random.uniform(1, 10), 1)

        if traffic_condition == 0:
            time_of_day = np.random.choice([0, 1])
        elif traffic_condition == 1:
            time_of_day = np.random.choice([1, 2])
        elif traffic_condition == 2:
            time_of_day = np.random.choice([2, 3])

        if traffic_condition == 0:
            air_pollution_index = round(air_pollution_index * 0.5, 1)
            visibility_in_miles = max(5, round(visibility_in_miles, 1))
        elif traffic_condition == 1:
            air_pollution_index = round(air_pollution_index * 1.2, 1)
            visibility_in_miles = max(3, round(visibility_in_miles, 1))
        elif traffic_condition == 2:
            air_pollution_index = round(air_pollution_index * 1.5, 1)
            visibility_in_miles = max(1, round(visibility_in_miles, 1))

        data.append([is_holiday, air_pollution_index, temperature, rain_p_h, visibility_in_miles, time_of_day, traffic_condition])
    
    return data

samples_per_class = 7000
data_class_0 = create_traffic_data(4000, 0)
data_class_1 = create_traffic_data(samples_per_class, 1)
data_class_2 = create_traffic_data(samples_per_class, 2)

all_data = data_class_0 + data_class_1 + data_class_2
df = pd.DataFrame(all_data, columns=['is_holiday', 'air_pollution_index', 'temperature', 'rain_p_h', 'visibility_in_miles', 'time_of_day', 'traffic_condition'])

df.to_csv('./data/traffic_data.csv', index=False)
print("Dữ liệu đã được tạo và lưu vào traffic_data.csv!")



