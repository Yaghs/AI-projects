import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Read the dataset from the CSV file
dataset = pd.read_csv("observationKNN.csv")

# Extract features (X) and target values (y) from the dataset
X = dataset[['X Value', 'Y Value']].values
y = dataset['Target Value'].values

# New observation to classify
new_observation = np.array([[3.0, 4.0]])

# Initializing the KNN classifier with K=5
knn = KNeighborsClassifier(n_neighbors=5)

# "Training" the classifier with the dataset
knn.fit(X, y)

# Predicting the class of the new observation
predicted_class = knn.predict(new_observation)

# Output the predicted class
print(f"The predicted class of the new observation is: {predicted_class[0]}")
