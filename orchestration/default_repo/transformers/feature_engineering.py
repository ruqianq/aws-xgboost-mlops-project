if "transformer" not in globals():
    from mage_ai.data_preparation.decorators import transformer
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test


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
    df["Gender"] = df.Gender.astype("category")
    df["VisitFrequency"] = df.VisitFrequency.astype("category")
    df["PreferredCuisine"] = df.PreferredCuisine.astype("category")
    df["TimeOfVisit"] = df.TimeOfVisit.astype("category")
    df["DiningOccasion"] = df.DiningOccasion.astype("category")
    df["MealType"] = df.MealType.astype("category")
    df["DiningOccasion"] = df.DiningOccasion.astype("category")
    df["Income_per_AverageSpend"] = df["Income"] / df["AverageSpend"]
    df["AverageSpend_per_GroupSize"] = df["AverageSpend"] / df["GroupSize"]
    df["Income_per_GroupSize"] = df["Income"] / df["GroupSize"]

    for col in df.columns:
        if df[col].dtype != "category":
            df[col] = df[col].astype(float)

    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, "The output is undefined"
