"""Training module"""

from typing import Callable, Union, Dict, Optional, List

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import log_loss, f1_score, roc_auc_score, recall_score, precision_score
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline

from settings.params import SEED


class Trainer:
    def __init__(
        self,
        data: pd.DataFrame,
        numerical_transformer: list,
        categorical_transformer: list,
        estimator: Callable,
        target: str,
        features: Optional[List[str]] = None,
        test_size: Optional[float] = 0.25,
        cv: Optional[int] = None,
    ):
        logger.info(f"Test size: {test_size} | cross validation: {cv}")
        self.test_size = test_size
        self.cv = cv
        self.numerical_transformer = numerical_transformer
        self.categorical_transformer = categorical_transformer
        self.estimator = estimator

        # Split the data into training and test sets.
        data_train, data_test = train_test_split(data, test_size=self.test_size, random_state=SEED)
        logger.info(f"Train size: {len(data_train)} | Test size: {len(data_test)}")

        # The predicted column is target
        self.y_train = data_train[target]
        self.y_test = data_test[target]

        # Get features data
        if not features:
            self.x_train = data_train.drop([target], axis=1)
            self.x_test = data_test.drop([target], axis=1)
        else:
            self.x_train = data_train.loc[:, features]
            self.x_test = data_test.loc[:, features]

    def define_pipeline(
        self,
        numerical_transformer: list,
        categorical_transformer: list,
        classifier: Callable,
    ) -> Pipeline:
        """Define pipeline for modeling

        Args:
            numerical_transformer: List of transformers for numerical features.
            categorical_transformer: List of transformers for categorical features.
            classifier: The classifier to be used.

        Returns:
            Pipeline: sklearn pipeline
        """
        numerical_pipeline = make_pipeline(*numerical_transformer)
        categorical_pipeline = make_pipeline(*categorical_transformer)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_pipeline, make_column_selector(dtype_include=["number"])),
                ("cat", categorical_pipeline, make_column_selector(dtype_include=["object", "bool"])),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        model_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

        return model_pipeline

    @staticmethod
    def eval_metrics(
        y_actual: Union[pd.DataFrame, pd.Series, np.ndarray],
        y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
        y_pred_proba: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> Dict[str, float]:
        """Compute evaluation metrics for classification models.

        Args:
            y_actual: Ground truth (correct) target values.
            y_pred: Estimated target values.
            y_pred_proba: Predicted probabilities for the positive class.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
                Expected keys are: "log_loss", "f1", "auc", "recall", "precision"
        """
        y_actual = np.array(y_actual)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)

        logloss = log_loss(y_actual, y_pred_proba)
        f1 = f1_score(y_actual, y_pred, zero_division=0)
        auc = roc_auc_score(y_actual, y_pred_proba)
        recall = recall_score(y_actual, y_pred, zero_division=0)
        precision = precision_score(y_actual, y_pred, zero_division=0)

        return {"log_loss": logloss, "f1": f1, "auc": auc, "recall": recall, "precision": precision}

    def tune_hyperparams(self, model: Pipeline, param_grid: dict):
        grid_search = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=2
        )

        grid_search.fit(self.x_train, self.y_train)

        print(f'Best parameters: {grid_search.best_params_}')
        print(f'Best score: {grid_search.best_score_}')

        return grid_search.best_params_, grid_search.best_score_

    def train(self):
        """Train the model."""
        with mlflow.start_run():
            sk_model = self.define_pipeline(
                numerical_transformer=self.numerical_transformer,
                categorical_transformer=self.categorical_transformer,
                classifier=self.estimator,
            )

            sk_model.fit(self.x_train, self.y_train)

            y_train_pred = sk_model.predict(self.x_train)
            y_train_pred_proba = sk_model.predict_proba(self.x_train)[:, 1]
            y_test_pred = sk_model.predict(self.x_test)
            y_test_pred_proba = sk_model.predict_proba(self.x_test)[:, 1]

            train_metrics = Trainer.eval_metrics(self.y_train, y_train_pred, y_train_pred_proba)
            test_metrics = Trainer.eval_metrics(self.y_test, y_test_pred, y_test_pred_proba)

            logger.info(f"Train metrics: {train_metrics}")
            logger.info(f"Test metrics: {test_metrics}")
