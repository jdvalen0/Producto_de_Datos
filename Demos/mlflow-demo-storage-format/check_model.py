import common


def check_estimator():

    import mlflow

    x, y = common.load_data()
    x_train, x_test, y_train, y_test = common.make_train_test_split(x, y)

    # -------------------------------------------------------------------------
    # Se carga directamente de la carpeta en que se almacenó en el código
    # anterior
    estimator_path = (
        "/home/elicoubuntu/Producto_de_Datos/Demos/mlflow-demo-storage-format/new_model"
    )
    # -------------------------------------------------------------------------

    estimator = mlflow.pyfunc.load_model(estimator_path)
    mse, mae, r2 = common.eval_metrics(y_test, y_pred=estimator.predict(x_test))
    common.report(estimator, mse, mae, r2)


if __name__ == "__main__":
    #
    # Debe coincidir con el mejor modelo encontrado en la celdas anteriores
    #
    check_estimator()
