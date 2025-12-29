# train.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_model(X, y):
    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = RandomForestClassifier(n_estimators=200,max_depth=10,random_state=42)

        model.fit(X_train, y_train)

        return model, X_test, y_test
