#!/usr/bin/env python
# coding: utf-8
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import LabelEncoder

import pickle
import pandas as pd
from flask import Flask, request, jsonify
import boto3
import os
import awsgi

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")


def get_model(model_name):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'").latest_versions:
        s3_bucket = mv.tags["s3_bucket"]
        s3_key = mv.tags["s3_key"]

    return s3_bucket, s3_key


def read_data(s3_bucket, key):
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    response = s3_client.get_object(Bucket=s3_bucket, Key=key)
    df = pd.read_csv(response.get("Body"))
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

    columns_to_dummy = [
        "VisitFrequency",
        "PreferredCuisine",
        "TimeOfVisit",
        "DiningOccasion",
    ]
    df = pd.get_dummies(df, columns=columns_to_dummy, dtype=int, drop_first=True)

    return df


def predict(s3_bucket, key, df):
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    response = s3_client.get_object(Bucket=s3_bucket, Key=key)
    scaler, labelencoder, model = pickle.load(response.get("Body"))
    print(df)
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
    columns_to_encode = ["Gender", "MealType"]
    labelencoder = LabelEncoder()

    for col in columns_to_encode:
        df[col] = labelencoder.fit_transform(df[col])
    x_scaled = scaler.transform(df[num_columns])
    df[num_columns] = pd.DataFrame(x_scaled, columns=num_columns)
    return model.predict(df)


def save_results(df, y_pred):
    df_result = pd.DataFrame()
    df_result["CustomerID"] = df["CustomerID"]
    df_result["predicted_rating"] = y_pred
    return df_result.loc[:, "predicted_rating"].mean()


app = Flask("customer-rating-prediction")


@app.route("/", methods=["POST"])
def index(s3_key):
    try:
        df = read_data(
            AWS_S3_BUCKET,
            s3_key,
        )
        model = get_model("customer-satisfaction-classifier")
        y_pred = predict(s3_bucket=model[0], key=model[1], df=df)
        result = save_results(df, y_pred)
    except Exception as e:
        return jsonify({"error": str(e)}, status_code=500)

    return jsonify(result, stus_code=200)


def lambda_handler(event, context):
    return awsgi.response(app, event, context)
