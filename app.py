import os
import sys
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import logging

os.environ["MLFLOW_TRACKING_URI"] ="http://ec2-54-175-65-41.compute-1.amazonaws.com:5000/"


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def evaluate(actual,pred):
    mae = mean_absolute_error(actual,pred)
    mse = mean_squared_error(actual,pred)
    r2 = r2_score(actual,pred)
    return mae,mse,r2

if __name__ =="__main__":
    csv_url ="https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/refs/heads/master/winequality-red.csv"
    try:
        data = pd.read_csv(csv_url,sep=";")
    except Exception as e:
        logger.error("Error reading csv file")

train,test = train_test_split(data)

train_x =train.drop(["quality"],axis=1)
test_x = test.drop(["quality"], axis =1)
train_y = train["quality"]
test_y = test["quality"]

alpha =float(sys.argv[1]) if len(sys.argv)>1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv)>2 else 0.5

with mlflow.start_run():
    model =ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=42)
    model.fit(train_x,train_y)
    predict =model.predict(test_x)
    (mae,mse,r2) = evaluate(test_y,predict)

    print(f"Elastic model alpha : {alpha} and l1_ratio:{l1_ratio}")
    print(f"MAE: {mae}, MSE: {mse}, R2: {r2}")


    mlflow.log_param("alpha",alpha)
    mlflow.log_param("l1_ratio",l1_ratio) 
    mlflow.log_metric("MAE",mae)
    mlflow.log_metric("MSE",mse)
    mlflow.log_metric("R2",r2)


    remote_server = "http://ec2-54-175-65-41.compute-1.amazonaws.com:5000/"
    mlflow.set_tracking_uri(remote_server)
    tracking_url_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_store != "file":
        mlflow.sklearn.log_model(model , "model",registered_model_name ="elastic wine model")
    else:
        mlflow.sklearn.log_model(model,"model")