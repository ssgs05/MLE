import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/dataset+based+on+uwb+for+clinical+establishments/clean_unemployment.csv")

# Data exploration
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.isnull())
print(df.duplicated())

# Data manipulation
df.set_index('country_name', inplace=True)
df_reset = df.reset_index()
df['Average'] = df.mean(axis=1)

# Data visualization
plt.figure(figsize=(16, 9))
plt.barh(df_reset['country_name'], df_reset['Average'], color='skyblue')
plt.xlabel('Average Value Across Years')
plt.ylabel('Country')
plt.title('Average Value Across Years for Each Country')
plt.show()

# Machine learning data preparation
features = df_reset.drop(['country_code', 'country_name', 'continent'], axis=1)
target = df_reset['continent']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)
