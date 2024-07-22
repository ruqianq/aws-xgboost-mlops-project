if "transformer" not in globals():
    from mage_ai.data_preparation.decorators import transformer
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import mlflow


@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    x_train = data[0]
    x_test = data[1]
    y_train = data[2]
    y_test = data[3]
    params_rf = {
        "max_depth": 40,
        "min_samples_leaf": 2,
        "min_samples_split": 4,
        "n_estimators": 1000,
        "random_state": 40,
    }

    model_rf = RandomForestClassifier(**params_rf)
    mlflow.set_tracking_uri("[your mlflow tracking uri]")
    mlflow.set_experiment("customer-satisfaction_ml")
    with mlflow.start_run(run_name="mlops_project"):

        mlflow.set_tag("developer", "daisy_lin")

        mlflow.log_param("train-data-path", "data/restaurant_customer_satisfaction.csv")

        mlflow.log_params(params_rf)

        model_rf.fit(x_train, y_train)

        y_pred = model_rf.predict(x_test)
        roc_auc = roc_auc_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("roc", roc_auc)
        mlflow.log_metric("accurancy", acc)

    return model_rf


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, "The output is undefined"
