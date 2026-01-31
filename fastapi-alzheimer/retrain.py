import os
import pickle
import numpy as np
import mlflow
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# =========================
# CONFIGURACIÓN GENERAL
# =========================

BASE_DIR = Path(__file__).resolve().parent

BUFFER_PATH = BASE_DIR / "buffer" / "new_patients.pkl"

MODEL_NAME = "Alzheimer_XGBoost"
EXPERIMENT_NAME = "Alzheimer_Modelamiento"

PREPROCESSING_RUN_ID = "a9302cdf7df7439d8a59ea7c3fb148ff"

MIN_PATIENTS = 4
HISTORICAL_PERCENTAGE = 0.4


# =========================
# DATASET HISTÓRICO DESDE MLFLOW
# =========================

def load_dataset_base_from_mlflow():
    """
    Descarga el dataset ya transformado desde MLflow
    (fuente única de verdad).
    """
    dataset_path = mlflow.artifacts.download_artifacts(
        run_id=PREPROCESSING_RUN_ID,
        artifact_path="dataset/dataset_transformado.pkl"
    )

    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    return data


def load_historical_sample():
    """
    Toma un porcentaje del dataset histórico
    y devuelve también el set de test intacto.
    """
    data = load_dataset_base_from_mlflow()

    X_hist = data["X_train_prep"]
    y_hist = data["y_train"]
    X_test = data["X_test_prep"]
    y_test = data["y_test"]

    X_pct, _, y_pct, _ = train_test_split(
        X_hist,
        y_hist,
        train_size=HISTORICAL_PERCENTAGE,
        random_state=42,
        stratify=y_hist
    )

    return X_pct, y_pct, X_test, y_test


# =========================
# BUFFER (SOLO DATA TRANSFORMADA)
# =========================

def save_to_buffer(X_prep, y):
    """
    Guarda SOLO datos ya transformados. Asegura que cada muestra se guarde
    como una lista (2D) para evitar entradas escalar/1D que provoquen
    inconsistencia en el buffer.
    """
    BUFFER_PATH.parent.mkdir(exist_ok=True)

    if BUFFER_PATH.exists():
        with open(BUFFER_PATH, "rb") as f:
            buffer = pickle.load(f)
    else:
        buffer = {"X": [], "y": []}

    # Forzar formato 2D por fila y añadir
    X_list = np.atleast_2d(X_prep).tolist()
    buffer["X"].extend(X_list)
    buffer["y"].extend(y)

    with open(BUFFER_PATH, "wb") as f:
        pickle.dump(buffer, f)

    return len(buffer["X"])


# =========================
# REENTRENAMIENTO INCREMENTAL
# =========================

def retrain_incremental():
    """
    Reentrena el modelo usando:
    - porcentaje del histórico
    - nuevos pacientes confirmados
    """

    # cargar buffer
    with open(BUFFER_PATH, "rb") as f:
        buffer = pickle.load(f)

    X_new = np.array(buffer["X"])
    y_new = np.array(buffer["y"])

    # histórico
    X_hist, y_hist, X_test, y_test = load_historical_sample()

    # combinar
    X_train = np.vstack([X_hist, X_new])
    y_train = np.hstack([y_hist, y_new])

    # modelo base (latest)
    modelo_base = mlflow.xgboost.load_model(
        model_uri=f"models:/{MODEL_NAME}/latest"
    )

    # nuevo modelo incremental
    modelo = XGBClassifier(
        n_estimators=100,          # solo árboles nuevos
        learning_rate=0.05,
        max_depth=5,
        eval_metric="logloss",
        random_state=42
    )

    modelo.fit(
        X_train,
        y_train,
        xgb_model=modelo_base
    )

    # evaluación
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro")

    # MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="incremental_5_patients"):

        mlflow.log_param("historical_percentage", HISTORICAL_PERCENTAGE)
        mlflow.log_param("new_patients", len(y_new))
        mlflow.log_param("label_source", "human_confirmed")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_macro", prec)
        mlflow.log_metric("recall_macro", rec)
        mlflow.log_metric("f1_macro", f1)

        mlflow.xgboost.log_model(
            modelo,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

    # limpiar buffer
    BUFFER_PATH.unlink()

    return f"Modelo reentrenado con {len(y_new)} pacientes nuevos"


# =========================
# INTERFAZ LLAMADA DESDE FASTAPI
# =========================

def handle_new_patient(X_new_prep, y_new):
    """
    Recibe SOLO X ya transformado desde app.py
    """
    total = save_to_buffer(X_new_prep, y_new)

    if total < MIN_PATIENTS:
        return f"Paciente confirmado ({total}/{MIN_PATIENTS})"

    return retrain_incremental()
