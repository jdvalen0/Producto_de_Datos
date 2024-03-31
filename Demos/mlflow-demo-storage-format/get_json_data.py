import common


def get_json_test_data():

    x, y = common.load_data()
    x_train, x_test, y_train, y_test = common.make_train_test_split(x, y)

    data = x_test.iloc[0:10, :].to_json(orient="split")

    data = repr(data)
    return data


if __name__ == "__main__":

    data = get_json_test_data()
    print(data)
