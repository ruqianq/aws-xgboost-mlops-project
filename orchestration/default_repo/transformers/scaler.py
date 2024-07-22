if "transformer" not in globals():
    from mage_ai.data_preparation.decorators import transformer
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test
from sklearn.preprocessing import StandardScaler
import pandas as pd


@transformer
def transform(df, *args, **kwargs):
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
    # Specify your transformation logic here
    num_columns = [
        "Age",
        "Income",
        "AverageSpend",
        "GroupSize",
        "WaitTime",
        "ServiceRating",
        "FoodRating",
        "AmbianceRating",
    ]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df[num_columns])
    df[num_columns] = pd.DataFrame(x_scaled, columns=num_columns)

    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, "The output is undefined"
