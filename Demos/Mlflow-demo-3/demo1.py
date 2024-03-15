def load_data():

    import pandas as pd

    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")

    y = df["quality"]
    x = df.copy()
    x.pop("quality")

    return x, y


def make_train_test_split(x, y):

    from sklearn.model_selection import train_test_split

    (x_train, x_test, y_train, y_test) = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=123456,
    )
    return x_train, x_test, y_train, y_test


def eval_metrics(y_true, y_pred):

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, mae, r2


def report(estimator, mse, mae, r2):

    print(estimator, ":", sep="")
    print(f"  MSE: {mse}")
    print(f"  MAE: {mae}")
    print(f"  R2: {r2}")


def make_experiment(experiment_name, alphas, l1_ratios, n_splits=5, verbose=1):

    import os

    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import GridSearchCV

    import mlflow
    import mlflow.sklearn

    x, y = load_data()
    x_train, x_test, y_train, y_test = make_train_test_split(x, y)

    param_grid = {
        "alpha": alphas,
        "l1_ratio": l1_ratios,
    }

    estimator = GridSearchCV(
        estimator=ElasticNet(
            random_state=12345,
        ),
        param_grid=param_grid,
        cv=n_splits,
        refit=True,
        verbose=0,
        return_train_score=False,
    )

    #
    # Establece el directorio de tracking. Esta es la dirección absoluta al
    # directorio actual en este ejemplo.
    #
    if not os.path.exists("corridas"):
        os.makedirs("corridas")
    mlflow.set_tracking_uri(
        "//home/elicoubuntu/Producto_de_Datos/Demos/Mlflow-demo-3/workspace/mlflow/corridas"
    )
    print("Tracking directory:", mlflow.get_tracking_uri())

    #
    # Autotracking
    #
    mlflow.sklearn.autolog(
        log_input_examples=False,
        log_model_signatures=True,
        log_models=True,
        disable=False,
        exclusive=False,
        disable_for_unsupported_versions=False,
        silent=False,
        max_tuning_runs=10,
        log_post_training_metrics=True,
        serialization_format="cloudpickle",
        registered_model_name=None,
    )

    #
    # Almancena las corridas  en el experimento indicado
    #
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:

        run = mlflow.active_run()
        print("Active run_id: {}".format(run.info.run_id))

        estimator.fit(x_train, y_train)

        #
        # Reporta el mejor modelo encontrado en la corrida
        #
        estimator = estimator.best_estimator_
        mse, mae, r2 = eval_metrics(y_test, y_pred=estimator.predict(x_test))
        if verbose > 0:
            report(estimator, mse, mae, r2)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)


if __name__ == "__main__":

    import numpy as np

    #
    # Se realizar el primer tanteo
    #
    make_experiment(
        experiment_name="red-wine",
        alphas=np.linspace(0.0001, 0.5, 10),
        l1_ratios=np.linspace(0.0001, 0.5, 10),
        n_splits=5,
        verbose=1,
    )

    #
    # Se realizar el segundo tanteo
    #
    make_experiment(
        experiment_name="red-wine",
        alphas=np.linspace(0.0000001, 0.0002, 10),
        l1_ratios=np.linspace(0.1, 0.2, 10),
        n_splits=5,
        verbose=1,
    )
