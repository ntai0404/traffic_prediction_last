import pandas as pd

df = pd.read_csv('./data/dataset.csv')

count_series = df['Traffic Situation'].value_counts()
#count_series = df['Date'].value_counts()
print(count_series)