import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/dataset+based+on+uwb+for+clinical+establishments/planes.csv")

# Explore data
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())

# Handling missing values
df.dropna(subset=['Dep_Time', 'Duration'], inplace=True)

# Data visualization
airline_counts = df['Airline'].value_counts()
labels = ['Jet Airways', 'IndiGo', 'Air Asia', 'Multiple carriers', 'SpiceJet', 'Vistara', 'Air India', 'GoAir']
sizes = airline_counts.values

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['blue', 'green', 'orange', 'red'])
plt.title('Flight proportion per airline')
plt.show()

# Machine learning data preparation
features = df.drop(['Airline', 'Duration', 'Price'], axis=1)
target = df['Price']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)
