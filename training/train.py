import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from training.dataset import get_data

def train():
    X_train, X_test, y_train, y_test = get_data()
    
    mlflow.set_experiment("iris-classifier")
    
    with mlflow.start_run():
        n_estimators = 100
        max_depth = 3
        
        # Train
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train,y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test,predictions)
        
        # Log MLflow
        mlflow.log_param("n_estimators",n_estimators)
        mlflow.log_param("max_depth",max_depth)
        mlflow.log_metric("accuracy",accuracy)
        mlflow.sklearn.log_model(model,"model") # saves the trained model as an artifact
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    train()

        