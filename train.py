import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv('Dataset.csv')

# Split features and target
X = df.drop('class', axis=1)  # Features
y = df['class']               # Target label

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# Define model pipelines with standard scaling
pipelines = {
    'logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    'ridge_classifier': make_pipeline(StandardScaler(), RidgeClassifier()),
    'random_forest': make_pipeline(StandardScaler(), RandomForestClassifier()),
    'svc': make_pipeline(StandardScaler(), SVC()),
    'knn': make_pipeline(StandardScaler(), KNeighborsClassifier()),
    'naive_bayes': make_pipeline(StandardScaler(), GaussianNB()),
    'decision_tree': make_pipeline(StandardScaler(), DecisionTreeClassifier()),
    'adaboost': make_pipeline(StandardScaler(), AdaBoostClassifier()),
    'gradient_boosting': make_pipeline(StandardScaler(), GradientBoostingClassifier())
}

# Train and save each model
for name, pipeline in pipelines.items():
    print(f"Training model: {name}")
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {name}: {acc:.4f}")
    
    # Save model as .h5 (although not HDF5 format, saved with .h5 extension)
    filename = f"{name}.h5"
    with open(filename, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Saved model to {filename}\n")
