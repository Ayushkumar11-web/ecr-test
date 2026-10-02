import os
import mlflow
import mlflow.sklearn

from src.Heart.utils.utils import load_object
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class ModelEvaluation:

    def __init__(self):
        pass

    def eval_metrics(self, actual, pred):

        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, zero_division=0)
        recall = recall_score(actual, pred, zero_division=0)
        f1 = f1_score(actual, pred, zero_division=0)

        return accuracy, precision, recall, f1

    def initate_model_evaluation(self, train_array, test_array):

        try:
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            model_path = os.path.join("Artifacts", "Model.pkl")
            model = load_object(model_path)

            # Local MLflow tracking directory
            tracking_path = os.path.abspath("mlruns")

            mlflow.set_tracking_uri("file:" + tracking_path)

            mlflow.set_experiment("Heart-Disease-Local-Ayush")

            with mlflow.start_run():

                predicted_qualities = model.predict(X_test)

                accuracy, precision, recall, f1 = self.eval_metrics(
                    y_test,
                    predicted_qualities
                )

                mlflow.log_metric("Testing Accuracy", accuracy)
                mlflow.log_metric("Precision Score", precision)
                mlflow.log_metric("Recall Score", recall)
                mlflow.log_metric("F1 Score", f1)

                # Log model locally without DagsHub registry
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="Model"
                )

                print("MLflow evaluation completed successfully.")

        except Exception as e:
            raise e