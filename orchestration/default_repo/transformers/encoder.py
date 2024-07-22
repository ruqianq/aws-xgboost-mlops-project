if "transformer" not in globals():
    from mage_ai.data_preparation.decorators import transformer
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test

import pandas as pd
from sklearn.preprocessing import LabelEncoder


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
    columns_to_dummy = [
        "VisitFrequency",
        "PreferredCuisine",
        "TimeOfVisit",
        "DiningOccasion",
    ]
    columns_to_encode = ["Gender", "MealType"]
    df = pd.get_dummies(df, columns=columns_to_dummy, dtype=int, drop_first=True)
    labelencoder = LabelEncoder()

    for col in columns_to_encode:
        df[col] = labelencoder.fit_transform(df[col])

    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, "The output is undefined"
