"""
models_baseline.py — Modelos baseline (CRISP-DM: Modelado - Etapa 1).

Cubre el requisito de Etapa 1: "Baseline models (al menos 3 algoritmos)".
Tres modelos de complejidad creciente como línea base antes de los modelos
avanzados de la Etapa 2:

    1. Regresión Lineal  -> referencia simple e interpretable (perfil, ZauZau 2023).
    2. Random Forest      -> no linealidades sin mucho ajuste.
    3. XGBoost            -> modelo fuerte del perfil (Krishna et al., 2025).

Las dependencias (scikit-learn, xgboost) se importan de forma DIFERIDA dentro de
cada función, para que el módulo se importe aunque aún no estén instaladas
(instalar con `pip install -r requirements.txt`).

Cada modelo se expone con:
  * una fábrica `nuevo_*()` (modelo sin entrenar) — útil para la validación cruzada;
  * un entrenador `entrenar_*(X, y)` que ajusta y devuelve el modelo.
"""

from __future__ import annotations

from . import config


# --------------------------------------------------------------------------- #
# Fábricas (modelo sin entrenar)
# --------------------------------------------------------------------------- #
def nuevo_regresion_lineal():
    """Crea una Regresión Lineal (baseline 1)."""
    from sklearn.linear_model import LinearRegression
    return LinearRegression()


def nuevo_random_forest(**kwargs):
    """Crea un Random Forest Regressor (baseline 2)."""
    from sklearn.ensemble import RandomForestRegressor
    params = dict(n_estimators=300, random_state=config.RANDOM_STATE, n_jobs=-1)
    params.update(kwargs)
    return RandomForestRegressor(**params)


def nuevo_xgboost(**kwargs):
    """Crea un XGBoost Regressor (baseline 3)."""
    from xgboost import XGBRegressor
    params = dict(n_estimators=400, learning_rate=0.05, max_depth=5,
                  subsample=0.9, colsample_bytree=0.9,
                  random_state=config.RANDOM_STATE, n_jobs=-1)
    params.update(kwargs)
    return XGBRegressor(**params)


# Registro nombre -> fábrica (para bucles de entrenamiento y CV).
FABRICAS = {
    "regresion_lineal": nuevo_regresion_lineal,
    "random_forest": nuevo_random_forest,
    "xgboost": nuevo_xgboost,
}


# --------------------------------------------------------------------------- #
# Entrenadores
# --------------------------------------------------------------------------- #
def entrenar_regresion_lineal(X_train, y_train):
    """Entrena y devuelve una Regresión Lineal ajustada."""
    return nuevo_regresion_lineal().fit(X_train, y_train)


def entrenar_random_forest(X_train, y_train, **kwargs):
    """Entrena y devuelve un Random Forest ajustado."""
    return nuevo_random_forest(**kwargs).fit(X_train, y_train)


def entrenar_xgboost(X_train, y_train, **kwargs):
    """Entrena y devuelve un XGBoost ajustado."""
    return nuevo_xgboost(**kwargs).fit(X_train, y_train)


def entrenar_todos(X_train, y_train):
    """Entrena los tres baselines y los devuelve en un dict {nombre: modelo}."""
    return {nombre: fabrica().fit(X_train, y_train)
            for nombre, fabrica in FABRICAS.items()}


# --------------------------------------------------------------------------- #
# Persistencia
# --------------------------------------------------------------------------- #
def guardar_modelo(modelo, nombre):
    """Serializa un modelo entrenado en config.MODELS_DIR (joblib)."""
    import joblib
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = config.MODELS_DIR / f"{nombre}.pkl"
    joblib.dump(modelo, ruta)
    return ruta


def cargar_modelo(nombre):
    """Carga un modelo serializado desde config.MODELS_DIR."""
    import joblib
    return joblib.load(config.MODELS_DIR / f"{nombre}.pkl")
