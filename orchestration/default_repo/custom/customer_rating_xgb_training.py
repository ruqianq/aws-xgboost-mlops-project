if "custom" not in globals():
    from mage_ai.data_preparation.decorators import custom
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score, accuracy_score

import mlflow

mlflow.set_tracking_uri("[your mlflow tracking uri]")
mlflow.set_experiment("customer-satisfaction_ml")


@custom
def transform_custom(data, *args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    mlflow.set_tracking_uri("[your mlflow tracking uri]")
    mlflow.set_experiment("customer-satisfaction_ml")
    x_train = data[0]
    x_test = data[1]
    y_train = data[2]
    y_test = data[3]

    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")

        def objective(space):
            xgb_model = xgb.XGBClassifier(
                n_estimators=space["n_estimators"],
                max_depth=int(space["max_depth"]),
                gamma=space["gamma"],
                reg_alpha=int(space["reg_alpha"]),
                min_child_weight=int(space["min_child_weight"]),
                colsample_bytree=int(space["colsample_bytree"]),
            )
            xgb_model.fit(x_train, y_train)
            y_pred = xgb_model.predict(x_test)
            roc_auc = roc_auc_score(y_test, y_pred)
            score = accuracy_score(y_test, y_pred)
            mlflow.log_metric("roc", roc_auc)
            mlflow.log_metric("accurancy", score)
            return {"loss": -score, "status": STATUS_OK}

        space = {
            "max_depth": hp.quniform("max_depth", 3, 18, 1),
            "gamma": hp.uniform("gamma", 1, 9),
            "reg_alpha": hp.quniform("reg_alpha", 40, 180, 1),
            "reg_lambda": hp.uniform("reg_lambda", 0, 1),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
            "min_child_weight": hp.quniform("min_child_weight", 0, 10, 1),
            "n_estimators": 180,
            "seed": 0,
        }
        trials = Trials()
        best_params = fmin(
            objective, space, algo=tpe.suggest, max_evals=100, trials=trials
        )

        mlflow.log_params(best_params)

    return best_params


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, "The output is undefined"
