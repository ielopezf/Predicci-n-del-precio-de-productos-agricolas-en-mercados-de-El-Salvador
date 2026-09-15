"""model_registry.py — Carga de datos y modelos + utilidades de predicción para la API.

Se ejecuta UNA vez al arrancar la API (módulo importado por app.py): carga
`dataset_modelado.csv`, reconstruye la misma matriz de features que usaron los
notebooks 03/04 (`construir_matriz_modelado`) y localiza el mismo tramo de
test ("prueba final", nunca usado para entrenar/ajustar ningún modelo) usando
`evaluation.indices_train_val_test` — la misma función que usó el
entrenamiento, para no duplicar la aritmética del split y garantizar que la
API muestre exactamente el tramo que cada modelo nunca vio.

Todos los modelos se sirven en modo "backtest": precio real vs. predicho
sobre datos históricos, NUNCA un pronóstico hacia el futuro — no existen
valores futuros reales de las variables externas (combustible, IPC, clima),
así que un forecast inventaría esos supuestos. Ver `fairness.LIMITACIONES_ETICAS`.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import joblib
import numpy as np
import pandas as pd

from prediccion_precios import config, features as ft, evaluation as ev
from prediccion_precios import models_advanced as ma, interpretability as it, fairness as fa


def slugify(texto: str) -> str:
    """'MAÍZ BLANCO' -> 'maiz_blanco' (slug usable en query params/URLs)."""
    t = unicodedata.normalize("NFKD", texto)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-zA-Z0-9]+", "_", t).strip("_").lower()
    return t


# --------------------------------------------------------------------------- #
# Datos: dataset limpio (sin NaN) + tramo de test, alineado con Fecha/Producto
# --------------------------------------------------------------------------- #
_data = pd.read_csv(config.DATASET_MODELADO, parse_dates=[config.COL_FECHA])
_df_clean = (_data.sort_values(config.COL_FECHA).reset_index(drop=True)
                   .dropna().reset_index(drop=True))
_meta = _df_clean[[config.COL_FECHA, config.COL_PRODUCTO, config.COL_PRECIO]].copy()
X, y = ft.construir_matriz_modelado(_df_clean, dropna=False)

_N_TRAIN, _N_VAL, _N_TEST = ev.indices_train_val_test(len(X))
X_TEST, Y_TEST = X.iloc[-_N_TEST:].reset_index(drop=True), y.iloc[-_N_TEST:].reset_index(drop=True)
META_TEST = _meta.iloc[-_N_TEST:].reset_index(drop=True)

_GRUPO_COLS = [c for c in X_TEST.columns if c.startswith("grupo_")]
GRUPO_TEST = X_TEST[_GRUPO_COLS].idxmax(axis=1).str.replace("grupo_", "", regex=False)

PRODUCTOS = [{"producto": p, "slug": slugify(p)} for p in config.PRODUCTOS_OBJETIVO]
SLUG_A_PRODUCTO = {p["slug"]: p["producto"] for p in PRODUCTOS}

# --------------------------------------------------------------------------- #
# Secuencias para la LSTM (ventana propia, distinta de X_TEST — ver notebook 04)
# --------------------------------------------------------------------------- #
Xs, ys, fechas_s, productos_s = ma.preparar_secuencias(_data, ventana=8, columnas=[config.COL_PRECIO])
_ns_train, _ns_val, _ns_test = ev.indices_train_val_test(len(Xs))
XS_TEST, YS_TEST = Xs[-_ns_test:], ys[-_ns_test:]
FECHAS_S_TEST, PRODUCTOS_S_TEST = fechas_s[-_ns_test:], productos_s[-_ns_test:]


# --------------------------------------------------------------------------- #
# Carga de modelos entrenados (tolera que alguno aún no exista)
# --------------------------------------------------------------------------- #
def _cargar(nombre, cargador):
    try:
        return cargador()
    except FileNotFoundError:
        return None


MODELOS = {
    "regresion_lineal": _cargar("regresion_lineal", lambda: joblib.load(config.MODELS_DIR / "regresion_lineal.pkl")),
    "random_forest": _cargar("random_forest", lambda: joblib.load(config.MODELS_DIR / "random_forest.pkl")),
    "xgboost": _cargar("xgboost", lambda: joblib.load(config.MODELS_DIR / "xgboost.pkl")),
    "xgboost_optimizado": _cargar("xgboost_optimizado", lambda: joblib.load(config.MODELS_DIR / "xgboost_optimizado.pkl")),
    "ensemble_stacking": _cargar("ensemble_stacking", lambda: joblib.load(config.MODELS_DIR / "ensemble_stacking.pkl")),
    "red_neuronal": _cargar("red_neuronal", lambda: ma.RedNeuronalRegresora.cargar(str(config.MODELS_DIR / "red_neuronal"))),
    "lstm": _cargar("lstm", lambda: ma.LSTMRegresor.cargar(str(config.MODELS_DIR / "lstm"))),
}

TIPO_MODELO = {
    "regresion_lineal": "baseline", "random_forest": "baseline", "xgboost": "baseline",
    "xgboost_optimizado": "avanzado", "ensemble_stacking": "avanzado",
    "red_neuronal": "avanzado", "lstm": "avanzado",
}

MODELOS_DISPONIBLES = [n for n, m in MODELOS.items() if m is not None]


# --------------------------------------------------------------------------- #
# Predicciones / métricas (todas en modo backtest sobre el tramo de test)
# --------------------------------------------------------------------------- #
def _es_lstm(nombre):
    return nombre == "lstm"


def predicciones(nombre_modelo, productos=None):
    """DataFrame [Fecha, Producto, precio_real, precio_predicho] en el tramo de test.

    `productos`: lista de nombres reales de producto para filtrar (None = todos).
    """
    modelo = MODELOS.get(nombre_modelo)
    if modelo is None:
        raise KeyError(f"Modelo no disponible: {nombre_modelo}")

    if _es_lstm(nombre_modelo):
        pred = modelo.predict(XS_TEST)
        df = pd.DataFrame({
            config.COL_FECHA: FECHAS_S_TEST, config.COL_PRODUCTO: PRODUCTOS_S_TEST,
            "precio_real": YS_TEST, "precio_predicho": pred,
        })
    else:
        pred = modelo.predict(X_TEST)
        df = META_TEST.rename(columns={config.COL_PRECIO: "precio_real"}).copy()
        df["precio_predicho"] = pred

    if productos:
        df = df[df[config.COL_PRODUCTO].isin(productos)]
    return df.sort_values([config.COL_PRODUCTO, config.COL_FECHA]).reset_index(drop=True)


def metricas_modelo(nombre_modelo):
    """Métricas globales + por producto + por temporada, sobre el tramo de test COMPLETO.

    Se calculan directamente sobre los arrays de test (no sobre el resultado
    de `predicciones`, que puede venir filtrado/reordenado por producto) para
    no depender de alinear índices de un DataFrame ya reordenado.
    """
    modelo = MODELOS.get(nombre_modelo)
    if modelo is None:
        raise KeyError(f"Modelo no disponible: {nombre_modelo}")

    if _es_lstm(nombre_modelo):
        pred = modelo.predict(XS_TEST)
        por_producto = fa.metricas_por_grupo(YS_TEST, pred, PRODUCTOS_S_TEST)
        return {"globales": ev.calcular_metricas(YS_TEST, pred),
                "por_producto": por_producto.reset_index().to_dict(orient="records"),
                "por_temporada": None}

    pred = modelo.predict(X_TEST)
    por_producto = fa.metricas_por_grupo(Y_TEST, pred, GRUPO_TEST)
    por_temporada = fa.metricas_por_temporada(Y_TEST, pred, X_TEST["es_cosecha"])
    return {
        "globales": ev.calcular_metricas(Y_TEST, pred),
        "por_producto": por_producto.reset_index().to_dict(orient="records"),
        "por_temporada": por_temporada.reset_index().to_dict(orient="records"),
    }


def importancia_modelo(nombre_modelo, top=15):
    """Importancia de variables (solo modelos con feature_importances_/coef_)."""
    modelo = MODELOS.get(nombre_modelo)
    if modelo is None or not (hasattr(modelo, "feature_importances_") or hasattr(modelo, "coef_")):
        return None
    tabla = it.importancia_variables(modelo, X_TEST.columns).head(top)
    return tabla.to_dict(orient="records")


def equidad_modelo(nombre_modelo):
    """Tablas de equidad (por producto / por temporada) + reporte textual.

    Igual que `metricas_modelo`, calcula sobre el tramo de test completo
    directamente (no sobre un DataFrame ya filtrado/reordenado).
    """
    metricas = metricas_modelo(nombre_modelo)
    if metricas["por_temporada"] is None:
        return {"por_producto": metricas["por_producto"], "por_temporada": None, "reporte": None}
    tabla_grupo = pd.DataFrame(metricas["por_producto"]).set_index("grupo")
    tabla_temp = pd.DataFrame(metricas["por_temporada"]).set_index("grupo")
    return {
        "por_producto": metricas["por_producto"],
        "por_temporada": metricas["por_temporada"],
        "reporte": fa.reporte_texto(tabla_grupo, tabla_temp),
    }
