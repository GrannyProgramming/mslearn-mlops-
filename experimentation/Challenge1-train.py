import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#loacal testing
mlflow.set_tracking_uri("http://127.0.0.1:5000")


def split_data(df):
    """
    Split data into training and testing sets.
    
    Args:
    df (DataFrame): The DataFrame containing the dataset.

    Returns:
    tuple: A tuple containing the split data (X_train, X_test, y_train, y_test).
    """
    logging.info("Splitting data into features and labels.")
    X = df[['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age']].values
    y = df['Diabetic'].values
    
    logging.info("Performing train-test split.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    
    logging.info("Data split into training and testing sets successfully.")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, C=1.0, solver='liblinear'):
    """
    Train a logistic regression model.

    Args:
    X_train (array): Training features.
    y_train (array): Training labels.
    C (float): Inverse of regularization strength.
    solver (str): Algorithm to use in the optimization problem.

    Returns:
    model: The trained model.
    """
    logging.info("Training the model.")
    with mlflow.start_run():
        model = LogisticRegression(C=C, solver=solver)
        model.fit(X_train, y_train)
        mlflow.log_param("C", C)
        mlflow.log_param("solver", solver)

        logging.info("Model training complete.")
        mlflow.sklearn.log_model(model, "model")
        return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using the test dataset.

    Args:
    model: The trained model.
    X_test (array): Testing features.
    y_test (array): Testing labels.

    Returns:
    dict: A dictionary containing the model's evaluation metrics.
    """
    logging.info("Evaluating the model.")
    with mlflow.start_run(nested=True):
        y_pred = model.predict(X_test)
        y_scores = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_scores)
        
        mlflow.log_metric("auc_score", auc_score)

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig("roc_curve.png")  # Save the figure
        plt.close()  # Close the figure context

        # Log the plot as an artifact
        mlflow.log_artifact("roc_curve.png")
        
        logging.info(f"Model evaluation complete with AUC: {auc_score:.2f}.")
        return {"auc_score": auc_score}

def load_data(file_path):
    """ Load the dataset from a CSV file. """
    return pd.read_csv(file_path)

def main():

    # Load data
    df = pd.read_csv("data/diabetes-dev.csv")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    mlflow.sklearn.autolog(disable=True)  # Control autologging according to your needs
    main()

