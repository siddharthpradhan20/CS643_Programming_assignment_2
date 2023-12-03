import findspark
findspark.find()
findspark.init()

from datetime import datetime
from io import StringIO
from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.tree import RandomForest, RandomForestModel, DecisionTree
from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.session import SparkSession	
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score , f1_score
import boto3
import numpy as np
import os
import pandas as pd
import pyspark
import shutil
import sys
from urllib.parse import urlparse

def main():

    conf = SparkConf().setAppName('WineQuality Training')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    trainPath = "s3a://dataset-programming-assignment-2/TrainingDataset.csv"
    print("Importing: " + trainPath)
    
    if not trainPath.startswith("s3://"):
        s3ModelPath = "s3a://dataset-programming-assignment-2/models"
    else: 
        s3ModelPath = os.path.join(os.path.dirname(trainPath), "models")
    print(">>>> Model Path set: " + s3ModelPath)


    df_train = spark.read.csv(trainPath, header=True, sep=";")
    df_train.printSchema()
    df_train.show()

    df_train = df_train.withColumnRenamed('""""quality"""""', "myLabel")
    for column in df_train.columns:
        df_train = df_train.withColumnRenamed(column, column.replace('"', ''))

    for idx, col_name in enumerate(df_train.columns):
        if idx not in [6 - 1, 7 - 1, len(df_train.columns) - 1]:
            df_train = df_train.withColumn(col_name, col(col_name).cast("double"))
        elif idx in [6 - 1, 7 - 1, len(df_train.columns) - 1]:
            df_train = df_train.withColumn(col_name, col(col_name).cast("integer"))

    df_train = df_train.rdd.map(lambda row: LabeledPoint(row[-1], row[:-1]))

    print("Training DecisionTree model...")
    model_dt = DecisionTree.trainClassifier(df_train, numClasses=10, categoricalFeaturesInfo={},
                                            impurity='gini', maxDepth=10, maxBins=32)
    print("Model - DecisionTree Created")

    model_path = s3ModelPath + "/model_dt.model"
    s3_deleteAndOverwrite(model_path, "model_dt.model")
    model_dt.save(sc, model_path)

    print(">>>>> DecisionTree model saved")

    print("Training RandomForest model...")
    model_rf = RandomForest.trainClassifier(df_train, numClasses=10, categoricalFeaturesInfo={},
                                    numTrees=10, featureSubsetStrategy="auto",
                                    impurity='gini', maxDepth=10, maxBins=32)
    print("Model - Randomforest Created")
    
    model_path = s3ModelPath + "/model_rf.model"
    s3_deleteAndOverwrite(model_path, "model_rf.model")
    model_rf.save(sc, model_path)

    print("Model for Random Forest algorithm")
    print("Data Training Completed")

def s3_deleteAndOverwrite(model_path, targetFolderName): 
    print(model_path)
    print(targetFolderName)
    pathTest = folder_exists(get_bucket_name(model_path), "models/"+ targetFolderName)
    print(pathTest)
    if (pathTest): 
        delete_directory(get_bucket_name(model_path), "models/"+ targetFolderName)
        


def folder_exists(bucket_name, path_to_folder):
    try:
        s3 = boto3.client('s3')
        res = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=path_to_folder
        )
        return 'Contents' in res
    except ClientError as e:
        raise e

def get_bucket_name(s3_path):
    try:
        return urlparse(s3_path).netloc
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def delete_directory(bucketName, folder_name):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucketName)
    bucket.objects.filter(Prefix=f"{folder_name}").delete()
    print(">>>> Prexisting Folder Deleted: " + folder_name)

if __name__ == "__main__": main()