import pandas as pd

df1 = pd.read_csv('./data/Traffic.csv')
df2 = pd.read_csv('./data/TrafficTwoMonth02.csv')

df_merged = pd.concat([df1, df2], ignore_index=True)

df_merged.to_csv('./data/dataset.csv', index=False)



df = pd.read_csv('./data/dataset.csv')

count_series = df['Traffic Situation'].value_counts()
#count_series = df['Date'].value_counts()

print(count_series)




df = pd.read_csv('./data/TrafficTwoMonth.csv')


df_filtered = df[df['Traffic Situation'] != 'normal']


df_filtered.to_csv('./data/TrafficTwoMonth02.csv', index=False)

print("Đã loại bỏ các dòng có 'Traffic Situation' là 'normal'.")




df = pd.read_csv('./data/dataset.csv')

df = df.drop(columns=['Time'])


traffic_situation_mapping = {
    'low': 0,
    'normal': 1,
    'high': 2,
    'heavy': 3
}
df['Traffic Situation'] = df['Traffic Situation'].map(traffic_situation_mapping)

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

df.to_csv('./data/dataset.csv', index=False)

print(df.head())

