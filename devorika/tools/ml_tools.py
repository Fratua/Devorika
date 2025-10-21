"""
Machine Learning and AI Tools for Devorika
Model training, evaluation, hyperparameter tuning, and deployment.
"""

import os
import json
import subprocess
from typing import Dict, Any, List, Optional
from .base import Tool


class MLModelTrainerTool(Tool):
    """
    Train machine learning models with various algorithms.
    """

    name = "ml_model_trainer"
    description = "Train ML models (sklearn, xgboost, etc.)"

    def execute(self, data_file: str, target_column: str, model_type: str = "classification",
                algorithm: str = "random_forest", output_model: str = "model.pkl", **params) -> Dict[str, Any]:
        """
        Train ML model.

        Args:
            data_file: Path to training data (CSV)
            target_column: Target column name
            model_type: Type (classification, regression)
            algorithm: Algorithm to use
            output_model: Output model file path
            **params: Additional training parameters

        Returns:
            Dict with training results
        """
        try:
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
            import joblib

            # Load data
            df = pd.read_csv(data_file)

            if target_column not in df.columns:
                return {"error": f"Target column '{target_column}' not found in data"}

            # Prepare features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Handle categorical variables
            X = pd.get_dummies(X, drop_first=True)

            # Split data
            test_size = params.get('test_size', 0.2)
            random_state = params.get('random_state', 42)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Select and train model
            model = self._get_model(algorithm, model_type, params)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)

            if model_type == "classification":
                score = accuracy_score(y_test, y_pred)
                metric_name = "accuracy"
                report = classification_report(y_test, y_pred)
            else:
                score = mean_squared_error(y_test, y_pred, squared=False)
                metric_name = "rmse"
                report = f"RMSE: {score:.4f}"

            # Save model
            joblib.dump(model, output_model)

            return {
                'success': True,
                'model_type': model_type,
                'algorithm': algorithm,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                metric_name: score,
                'report': report,
                'model_file': output_model,
                'feature_count': X.shape[1]
            }

        except ImportError as e:
            return {"error": f"Required library not installed: {str(e)}"}
        except Exception as e:
            return {"error": f"Model training failed: {str(e)}"}

    def _get_model(self, algorithm: str, model_type: str, params: Dict[str, Any]):
        """Get ML model based on algorithm."""
        if model_type == "classification":
            if algorithm == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', None),
                    random_state=params.get('random_state', 42)
                )
            elif algorithm == "logistic_regression":
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(random_state=params.get('random_state', 42))
            elif algorithm == "svm":
                from sklearn.svm import SVC
                return SVC(random_state=params.get('random_state', 42))
            elif algorithm == "xgboost":
                from xgboost import XGBClassifier
                return XGBClassifier(random_state=params.get('random_state', 42))
        else:  # regression
            if algorithm == "random_forest":
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(
                    n_estimators=params.get('n_estimators', 100),
                    random_state=params.get('random_state', 42)
                )
            elif algorithm == "linear_regression":
                from sklearn.linear_model import LinearRegression
                return LinearRegression()
            elif algorithm == "xgboost":
                from xgboost import XGBRegressor
                return XGBRegressor(random_state=params.get('random_state', 42))

        raise ValueError(f"Unknown algorithm: {algorithm}")


class HyperparameterTunerTool(Tool):
    """
    Hyperparameter tuning using GridSearch or RandomSearch.
    """

    name = "hyperparameter_tuner"
    description = "Tune model hyperparameters automatically"

    def execute(self, data_file: str, target_column: str, model_type: str = "classification",
                algorithm: str = "random_forest", search_type: str = "grid",
                param_grid: Dict[str, List] = None, **kwargs) -> Dict[str, Any]:
        """
        Tune hyperparameters.

        Args:
            data_file: Training data file
            target_column: Target column
            model_type: Model type
            algorithm: Algorithm
            search_type: Search type (grid, random)
            param_grid: Parameter grid to search
            **kwargs: Additional parameters

        Returns:
            Dict with tuning results
        """
        try:
            import pandas as pd
            from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
            from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error

            # Load data
            df = pd.read_csv(data_file)
            X = df.drop(columns=[target_column])
            y = df[target_column]
            X = pd.get_dummies(X, drop_first=True)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Get base model
            trainer = MLModelTrainerTool()
            model = trainer._get_model(algorithm, model_type, {})

            # Default parameter grids
            if param_grid is None:
                if algorithm == "random_forest":
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10]
                    }
                else:
                    param_grid = {}

            # Perform search
            if search_type == "grid":
                search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
            else:
                search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5,
                                          n_jobs=-1, verbose=1, random_state=42)

            search.fit(X_train, y_train)

            # Evaluate best model
            y_pred = search.best_estimator_.predict(X_test)

            if model_type == "classification":
                test_score = accuracy_score(y_test, y_pred)
                metric = "accuracy"
            else:
                test_score = mean_squared_error(y_test, y_pred, squared=False)
                metric = "rmse"

            return {
                'success': True,
                'algorithm': algorithm,
                'search_type': search_type,
                'best_params': search.best_params_,
                'best_cv_score': search.best_score_,
                f'test_{metric}': test_score,
                'total_fits': len(search.cv_results_['params'])
            }

        except ImportError as e:
            return {"error": f"Required library not installed: {str(e)}"}
        except Exception as e:
            return {"error": f"Hyperparameter tuning failed: {str(e)}"}


class ModelEvaluatorTool(Tool):
    """
    Comprehensive model evaluation with metrics and visualizations.
    """

    name = "model_evaluator"
    description = "Evaluate ML models with comprehensive metrics"

    def execute(self, model_file: str, test_data: str, target_column: str,
                model_type: str = "classification") -> Dict[str, Any]:
        """
        Evaluate trained model.

        Args:
            model_file: Path to saved model
            test_data: Test data file
            target_column: Target column
            model_type: Model type

        Returns:
            Dict with evaluation metrics
        """
        try:
            import pandas as pd
            import joblib
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
            )

            # Load model and data
            model = joblib.load(model_file)
            df = pd.read_csv(test_data)

            X = df.drop(columns=[target_column])
            y = df[target_column]
            X = pd.get_dummies(X, drop_first=True)

            # Predict
            y_pred = model.predict(X)

            if model_type == "classification":
                metrics = {
                    'accuracy': accuracy_score(y, y_pred),
                    'precision': precision_score(y, y_pred, average='weighted'),
                    'recall': recall_score(y, y_pred, average='weighted'),
                    'f1_score': f1_score(y, y_pred, average='weighted'),
                    'confusion_matrix': confusion_matrix(y, y_pred).tolist()
                }
            else:
                metrics = {
                    'mse': mean_squared_error(y, y_pred),
                    'rmse': mean_squared_error(y, y_pred, squared=False),
                    'mae': mean_absolute_error(y, y_pred),
                    'r2_score': r2_score(y, y_pred)
                }

            return {
                'success': True,
                'model_file': model_file,
                'model_type': model_type,
                'test_samples': len(y),
                'metrics': metrics
            }

        except ImportError:
            return {"error": "Required libraries not installed"}
        except Exception as e:
            return {"error": f"Model evaluation failed: {str(e)}"}


class FeatureEngineeringTool(Tool):
    """
    Automated feature engineering and selection.
    """

    name = "feature_engineering"
    description = "Perform feature engineering and selection"

    def execute(self, data_file: str, target_column: str = None,
                operations: List[str] = None, output_file: str = "features.csv") -> Dict[str, Any]:
        """
        Perform feature engineering.

        Args:
            data_file: Input data file
            target_column: Target column (optional)
            operations: List of operations (scale, encode, polynomial, pca)
            output_file: Output file

        Returns:
            Dict with feature engineering results
        """
        try:
            import pandas as pd
            from sklearn.preprocessing import StandardScaler, PolynomialFeatures
            from sklearn.decomposition import PCA

            # Load data
            df = pd.read_csv(data_file)
            original_shape = df.shape

            if operations is None:
                operations = ['scale', 'encode']

            features_added = []

            # Separate features and target
            if target_column and target_column in df.columns:
                X = df.drop(columns=[target_column])
                y = df[target_column]
            else:
                X = df
                y = None

            # Encode categorical variables
            if 'encode' in operations:
                categorical_cols = X.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
                    features_added.append(f"One-hot encoded {len(categorical_cols)} categorical features")

            # Scale numerical features
            if 'scale' in operations:
                numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
                if len(numerical_cols) > 0:
                    scaler = StandardScaler()
                    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
                    features_added.append(f"Scaled {len(numerical_cols)} numerical features")

            # Polynomial features
            if 'polynomial' in operations:
                poly = PolynomialFeatures(degree=2, include_bias=False)
                X_poly = poly.fit_transform(X)
                X = pd.DataFrame(X_poly, columns=[f"poly_{i}" for i in range(X_poly.shape[1])])
                features_added.append(f"Created polynomial features (degree 2)")

            # PCA
            if 'pca' in operations:
                n_components = min(10, X.shape[1])
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X)
                X = pd.DataFrame(X_pca, columns=[f"pc_{i+1}" for i in range(n_components)])
                features_added.append(f"Applied PCA ({n_components} components)")

            # Combine with target if exists
            if y is not None:
                X[target_column] = y.values

            # Save
            X.to_csv(output_file, index=False)

            return {
                'success': True,
                'original_shape': original_shape,
                'new_shape': X.shape,
                'operations': operations,
                'features_added': features_added,
                'output_file': output_file
            }

        except ImportError:
            return {"error": "Required libraries not installed"}
        except Exception as e:
            return {"error": f"Feature engineering failed: {str(e)}"}


class ModelDeploymentTool(Tool):
    """
    Deploy ML models as REST APIs.
    """

    name = "model_deployment"
    description = "Deploy ML model as REST API"

    def execute(self, model_file: str, output_dir: str = "deployment",
                framework: str = "fastapi") -> Dict[str, Any]:
        """
        Create deployment code for ML model.

        Args:
            model_file: Path to trained model
            output_dir: Output directory
            framework: Framework (fastapi, flask)

        Returns:
            Dict with deployment files
        """
        try:
            os.makedirs(output_dir, exist_ok=True)

            if framework == "fastapi":
                app_file = os.path.join(output_dir, 'app.py')
                requirements_file = os.path.join(output_dir, 'requirements.txt')

                app_content = f'''from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

app = FastAPI(title="ML Model API", description="Deployed by Devorika")

# Load model
model = joblib.load("{model_file}")

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: float
    probability: List[float] = None

@app.get("/")
async def root():
    return {{"message": "ML Model API is running"}}

@app.get("/health")
async def health():
    return {{"status": "healthy", "model": "{model_file}"}}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Reshape features
        features = np.array(request.features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]

        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features)[0].tolist()

        return {{
            "prediction": float(prediction),
            "probability": probability
        }}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

                requirements_content = '''fastapi>=0.104.0
uvicorn>=0.24.0
joblib>=1.3.0
scikit-learn>=1.3.0
numpy>=1.24.0
pydantic>=2.0.0
'''

                with open(app_file, 'w') as f:
                    f.write(app_content)

                with open(requirements_file, 'w') as f:
                    f.write(requirements_content)

                # Create Dockerfile
                dockerfile = os.path.join(output_dir, 'Dockerfile')
                dockerfile_content = '''FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
'''

                with open(dockerfile, 'w') as f:
                    f.write(dockerfile_content)

                return {
                    'success': True,
                    'framework': 'fastapi',
                    'files': [app_file, requirements_file, dockerfile],
                    'output_dir': output_dir,
                    'instructions': 'Run: cd deployment && pip install -r requirements.txt && python app.py'
                }

            else:
                return {"error": f"Unsupported framework: {framework}"}

        except Exception as e:
            return {"error": f"Model deployment failed: {str(e)}"}


class DataAnalysisTool(Tool):
    """
    Automated data analysis and profiling.
    """

    name = "data_analysis"
    description = "Perform automated data analysis and profiling"

    def execute(self, data_file: str, output_report: str = "analysis_report.json") -> Dict[str, Any]:
        """
        Analyze dataset.

        Args:
            data_file: Data file to analyze
            output_report: Output report file

        Returns:
            Dict with analysis results
        """
        try:
            import pandas as pd
            import numpy as np

            # Load data
            df = pd.read_csv(data_file)

            # Basic info
            analysis = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                'duplicates': int(df.duplicated().sum()),
                'numerical_stats': {},
                'categorical_stats': {}
            }

            # Numerical statistics
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numerical_cols:
                analysis['numerical_stats'][col] = {
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'q25': float(df[col].quantile(0.25)),
                    'q75': float(df[col].quantile(0.75))
                }

            # Categorical statistics
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                analysis['categorical_stats'][col] = {
                    'unique_values': int(df[col].nunique()),
                    'top_values': value_counts.head(5).to_dict(),
                    'missing': int(df[col].isnull().sum())
                }

            # Save report
            with open(output_report, 'w') as f:
                json.dump(analysis, f, indent=2)

            return {
                'success': True,
                'data_file': data_file,
                'analysis': analysis,
                'report_file': output_report
            }

        except ImportError:
            return {"error": "pandas not installed"}
        except Exception as e:
            return {"error": f"Data analysis failed: {str(e)}"}
