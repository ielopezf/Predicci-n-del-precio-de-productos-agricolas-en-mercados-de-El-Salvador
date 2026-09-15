"""
interpretability.py — Interpretabilidad del modelo (Etapa 1).

Cubre el requisito de Etapa 1: "Interpretabilidad básica (SHAP o LIME)".
Explica QUÉ variables influyen más en la predicción del precio, clave para el
criterio "Aporte local claro" (¿pesa más el combustible?, ¿la estacionalidad?,
¿la producción?).

SHAP y LIME se importan de forma DIFERIDA para no exigirlos si solo se usa la
importancia nativa del modelo. Las figuras se guardan en config.FIGURES_DIR.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


def importancia_variables(modelo, nombres_features):
    """Importancia de variables nativa del modelo (feature_importances_ / coef_).

    Sirve para Random Forest y XGBoost (feature_importances_) y para modelos
    lineales (coef_). Devuelve un DataFrame ordenado de mayor a menor.
    """
    if hasattr(modelo, "feature_importances_"):
        imp = np.asarray(modelo.feature_importances_, dtype=float)
    elif hasattr(modelo, "coef_"):
        imp = np.abs(np.asarray(modelo.coef_, dtype=float)).ravel()
    else:
        raise AttributeError("El modelo no expone feature_importances_ ni coef_.")

    tabla = (pd.DataFrame({"feature": list(nombres_features), "importancia": imp})
               .sort_values("importancia", ascending=False)
               .reset_index(drop=True))
    return tabla


def graficar_importancia(modelo, nombres_features, top=15, guardar=True):
    """Gráfico de barras con las variables más importantes."""
    tabla = importancia_variables(modelo, nombres_features).head(top)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(tabla) + 1))
    ax.barh(tabla["feature"][::-1], tabla["importancia"][::-1])
    ax.set_title("Importancia de variables")
    ax.set_xlabel("Importancia")
    if guardar:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        ruta = config.FIGURES_DIR / "importancia_variables.png"
        fig.savefig(ruta, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return ruta
    plt.close(fig)
    return None


def explicar_shap(modelo, X, guardar=True, max_muestras=200):
    """Calcula valores SHAP y genera el summary plot (requiere `shap`).

    Se usa TreeExplainer para modelos de árboles (RF/XGBoost). Para acelerar,
    limita el número de muestras.
    """
    import shap
    X_muestra = X.iloc[:max_muestras] if hasattr(X, "iloc") else X[:max_muestras]

    # Se explica siempre vía la función predict() (en vez de TreeExplainer):
    # algunas combinaciones de versión xgboost/shap fallan al parsear el
    # `base_score` interno del booster con TreeExplainer. Usar el callable
    # predict es más lento (permutación) pero funciona para cualquier modelo.
    explainer = shap.Explainer(modelo.predict, X_muestra)
    shap_values = explainer(X_muestra)

    plt.figure()
    shap.summary_plot(shap_values, X_muestra, show=False)
    if guardar:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        ruta = config.FIGURES_DIR / "shap_summary.png"
        plt.savefig(ruta, dpi=120, bbox_inches="tight")
        plt.close()
        return ruta
    plt.close()
    return shap_values


def explicar_lime(modelo, X, indice=0):
    """Explicación local LIME para una predicción individual (requiere `lime`)."""
    from lime.lime_tabular import LimeTabularExplainer
    Xv = X.values if hasattr(X, "values") else np.asarray(X)
    nombres = list(X.columns) if hasattr(X, "columns") else None
    explainer = LimeTabularExplainer(Xv, feature_names=nombres, mode="regression")
    return explainer.explain_instance(Xv[indice], modelo.predict)
