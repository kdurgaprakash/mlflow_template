# Import necessary libraries
import mlflow
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Load Iris dataset
dataset = load_iris()

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, stratify=dataset.target)

# Checkout train and test datasets
print(f"Train set shape: {X_train.shape[0]} rows, {X_train.shape[1]} columns", )
print(f"Test set shape: {X_test.shape[0]} rows, {X_test.shape[1]} columns")
print(f"Columns names: {dataset.feature_names}")

# Output:
# Train set shape: 120 rows, 4 columns
# Test set shape: 30 rows, 4 columns
# Columns names: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']


# Set the experiment for MLflow
mlflow.set_experiment('Baseline Model')

# Start an MLflow run context
with mlflow.start_run():
    # Initialize a LogisticRegression model
    model = LogisticRegression()
    
    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate various evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    # Log the evaluation metrics to the MLflow run
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall', recall)

    # Log the trained model to the MLflow run
    mlflow.sklearn.log_model(model, 'model')

    # Set developer information as a tag
    mlflow.set_tag('developer', 'Durga')

    # Set preprocessing details as a tag
    mlflow.set_tag('preprocessing', 'None')

    # Set the model type as a tag
    mlflow.set_tag('model', 'Logistic Regression')
