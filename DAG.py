#!/usr/bin/env python
# -*- coding: utf-8 -*-

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.api.client.local_client import Client


from datetime import datetime
from scipy.stats import ttest_ind, norm
import pytz
import logging
import numpy as np
import pandas as pd
import random
import boto3
import json


BUCKET_NAME = "silvercloud.airflow.bucket"
PARAMS_FILE = "ab_test_params.json"
DATA_FILE = "ab_test_data.csv"
HISTORY_FILE = "ab_test_history.csv"
ALPHA = 0.05  
CHECKPOINT_INTERVAL = 2
TOTAL_VALUES = 2400 

def get_s3_client():
    session = boto3.session.Session()
    return session.client(service_name="s3", endpoint_url="https://storage.yandexcloud.net")

def obrien_fleming_alpha(t, alpha=ALPHA):

    z_alpha_over2 = norm.ppf(1 - alpha / 2)
    alpha_spent = 2 * (1 - norm.cdf(z_alpha_over2 / np.sqrt(t)))

    return alpha_spent

def generate_test_params():
    s3 = get_s3_client()
    
    existing_objects = s3.list_objects_v2(Bucket=BUCKET_NAME)
    if "Contents" in existing_objects:
        for obj in existing_objects["Contents"]:
            if obj["Key"] == PARAMS_FILE:
                return

    loc = 10
    scale = 3
    dif = np.random.uniform(-0.5, 0.5)
    
    params = {"loc": loc, "scale": scale, "dif": dif}
    
    s3.put_object(Bucket=BUCKET_NAME, Key=PARAMS_FILE, Body=json.dumps(params))
    print("Параметры эксперимента сохранены")

# --- Функция генерации случайных данных для теста ---
def generate_ab_test_data():
    s3 = get_s3_client()

    response = s3.get_object(Bucket=BUCKET_NAME, Key=PARAMS_FILE)
    params = json.loads(response["Body"].read().decode("utf-8"))
    
    loc = params["loc"]
    scale = params["scale"]
    dif = params["dif"]

    control = np.random.normal(loc, scale, 50)
    test = np.random.normal(loc, scale, 50)

    test = test + dif

    try:
        old_df = pd.read_csv(f"s3://{BUCKET_NAME}/{DATA_FILE}")
    except:
        print("Файл с данными не найден, создаем новый")
        old_df = pd.DataFrame(columns=["day", "group", "value"])


    current_day = old_df["day"].max() + 1 if len(old_df) != 0 else 1

    df = pd.DataFrame({
        "day": [current_day] * 100,
        "group": ["control"] * 50 + ["test"] * 50,
        "value": np.concatenate([control, test])
    })

    df = pd.concat([old_df, df])

    s3.put_object(Bucket=BUCKET_NAME, Key=DATA_FILE, Body=df.to_csv(index=False))
    print("Данные загружены в облако")

# --- Функция проверки статистики ---
def check_statistics():
    s3 = get_s3_client()

    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=DATA_FILE)
        df = pd.read_csv(obj["Body"])
    except:
        print("Нет данных для анализа.")
        return
    
    current_day = df["day"].max()
    
    if current_day % CHECKPOINT_INTERVAL != 0:
        print(f"Сегодня {current_day}-й день, но анализ проводится каждые {CHECKPOINT_INTERVAL} дней.")
        return

    group_a = df[df["group"] == "control"]["value"].values
    group_b = df[df["group"] == "test"]["value"].values     

    alpha_threshold = obrien_fleming_alpha(len(df) / TOTAL_VALUES)

    _, p_value = ttest_ind(group_a, group_b, equal_var=False)
    
    print(f"День {current_day}: p-value = {p_value:.5f}, порог α = {alpha_threshold:.5f}")

    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=HISTORY_FILE)
        history_df = pd.read_csv(obj["Body"])
    except:
        print("Файл истории не найден, создаем новый")
        history_df = pd.DataFrame(columns=["day", "p_value", "threshold"])


    new_entry = pd.DataFrame({"day": [current_day], "p_value": [p_value], "threshold": [alpha_threshold]})
    history_df = pd.concat([history_df, new_entry])

    s3.put_object(Bucket=BUCKET_NAME, Key=HISTORY_FILE, Body=history_df.to_csv(index=False))
    print(f"История обновлена и загружена в {HISTORY_FILE}")

    if p_value < alpha_threshold:
        print("Результат статистически значим! Можно остановить тест досрочно.")
    else:
        print("Пока нет статистической значимости, продолжаем тест.")

    if len(df) >= TOTAL_VALUES:
        print('Данных достаточно для завершения теста')
        
# --- Определяем DAG ---
default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 2, 25, 20, 32, tzinfo=pytz.timezone("Europe/Moscow")),
    "catchup": False
}

dag = DAG(
    "ab_test_pipeline",
    default_args=default_args,
    schedule_interval="*/1 * * * *",  # DAG запускается раз в "день"
)

# --- Определяем таски ---
task_generate_params = PythonOperator(
    task_id="generate_test_params",
    python_callable=generate_test_params,
    dag=dag
)

task_generate_data = PythonOperator(
    task_id="generate_ab_test_data",
    python_callable=generate_ab_test_data,
    dag=dag
)

task_check_statistics = PythonOperator(
    task_id="check_statistics",
    python_callable=check_statistics,
    dag=dag
)

# --- Определяем последовательность выполнения ---
task_generate_params >> task_generate_data >> task_check_statistics
