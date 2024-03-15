import mlflow
logged_model = 'runs:/b952de9cb7454f5c9842aa0f3b3c90f9/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

def load_data():

    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")

    y = df["quality"]
    x = df.copy()
    x.pop("quality")

    return x, y

data, _ = load_data()

prediccion = loaded_model.predict(data)

print(prediccion)
