"""" Este es un ejemplo de la corrida desde consola de Mlflow"""
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

def train_estimator(alpha=0.5, l1_ratio=0.5, verbose=1):

    import mlflow.sklearn
    from sklearn.linear_model import ElasticNet

    import mlflow

    x, y = load_data()
    x_train, x_test, y_train, y_test = make_train_test_split(x, y)

    # Configurar la URI de artefactos
    mlflow.set_tracking_uri(mlflow.get_tracking_uri())

    print('Tracking directory:', mlflow.get_tracking_uri())

    with mlflow.start_run():

        estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=12345)
        estimator.fit(x_train, y_train)
        mse, mae, r2 = eval_metrics(y_test, y_pred=estimator.predict(x_test))
        if verbose > 0:
            report(estimator, mse, mae, r2)


        #
        # Tracking de parámetros
        #
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        #
        # Tracking de metricas
        #
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        #
        # Tracking del modelo
        #
        mlflow.sklearn.log_model(estimator, "model")

        
    # -------------------------------------------------------------------------
    # Ya no se requiere con MLflow
    # -------------------------------------------------------------------------
    #
    # best_estimator = load_best_estimator()
    # if best_estimator is None or estimator.score(x_test, y_test) > best_estimator.score(
    #     x_test, y_test
    # ):
    #     best_estimator = estimator
    #
    # save_best_estimator(best_estimator)

def main():
    import mlflow
# Configurar la URI de seguimiento local
    mlflow.set_tracking_uri("file:///home/elicoubuntu/Producto_de_Datos/Demos/Mlflow-demo-3/mlruns")

    train_estimator(0.2, 0.2)
    train_estimator(0.1, 0.1)
    train_estimator(0.1, 0.05)

if __name__== "__main__":
    main()