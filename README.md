# CARDIOVASCULAR-DISEASE-PREDITION
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Load the dataset
# Replace 'your_dataset.csv' with the actual file path
data = pd.read_csv('your_dataset.csv')
# Data Pre-processing
# Drop 'id' column as it may not contribute to the prediction
data = data.drop('id', axis=1)
# Handle missing data if any
data.dropna(inplace=True)
# Encode categorical variables (if 'gender' is categorical)
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])
# Normalize numerical features
scaler = StandardScaler()
numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
data[numerical_features] = scaler.fit_transform(data[numerical_features])
# Split dataset into features (X) and target variable (y)
X = data.drop('cardio', axis=1)
y = data['cardio']
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Data Analysis and Visualizations
# Example: Histograms, Pairplots, etc.
sns.pairplot(data, hue='cardio')
plt.show()
# Correlation Matrix
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
# Machine Learning Techniques
models = {
 'SVM': SVC(),
 'KNN': KNeighborsClassifier(),
 'Decision Tree': DecisionTreeClassifier(),
 'Logistic Regression': LogisticRegression(),
 'Random Forest': RandomForestClassifier()
}
for name, model in models.items():
 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)
 accuracy = accuracy_score(y_test, y_pred)
 print(f'{name} Accuracy: {accuracy}')
# Build Machine Learning Model
# Example: Choose the best model, fine-tune hyperparameters, train the final model
final_model = RandomForestClassifier()
final_model.fit(X, y)
# Now you can use this 'final_model' for heart disease detection.
# For example, if you have new data 'new_data', you can use: final_model.predict(new_data)
