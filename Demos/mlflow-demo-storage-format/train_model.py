"""Modulo para entrenar un modelo de regresion para el dataset winequality-red.
Este codigo entrena un modelo de regresion ElasticNet para el dataset winequality-red.
"""

import warnings

warnings.filterwarnings("ignore")

import sys


import mlflow.sklearn
from sklearn.linear_model import ElasticNet
import mlflow
import pandas as pd
import common


def train_estimator(alpha=0.5, l1_ratio=0.5, verbose=1):

    x, y = common.load_data()
    x_train, x_test, y_train, y_test = common.make_train_test_split(x, y)

    print("Tracking directory:", mlflow.get_tracking_uri())

    with mlflow.start_run():

        estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=12345)
        estimator.fit(x_train, y_train)
        mse, mae, r2 = common.eval_metrics(y_test, y_pred=estimator.predict(x_test))
        if verbose > 0:
            common.report(estimator, mse, mae, r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(estimator, "model")

        # -------------------------------------------------------------------------------
        #
        # Guardado del modelo para posible ejecución.
        # Crea el directori
        #
        mlflow.sklearn.save_model(
            estimator,
            "/home/elicoubuntu/Producto_de_Datos/Demos/mlflow-demo-storage-format/new_model",
        )


if __name__ == "__main__":

    alpha = float(sys.argv[1])
    l1_ratio = float(sys.argv[2])
    verbose = int(sys.argv[3])
    train_estimator(alpha=alpha, l1_ratio=l1_ratio, verbose=verbose)
