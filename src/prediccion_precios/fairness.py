"""fairness.py — Evaluación ética y de sesgos (CRISP-DM: Evaluación - Etapa 2).

No basta con un error promedio bajo: un modelo puede predecir muy bien un
producto y muy mal otro, o ser sistemáticamente peor en cierta temporada. Este
módulo mide esa disparidad de desempeño (equidad) entre grupos y documenta las
limitaciones éticas del dataset para su uso en la app de despliegue.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluation as ev


def metricas_por_grupo(y_true, y_pred, grupo):
    """Métricas de error (MAE/RMSE/MAPE/R²) desglosadas por grupo/categoría.

    `grupo` es un array paralelo a y_true/y_pred (p. ej. Producto o Grupo de
    cultivo). Detecta si el modelo predice sistemáticamente peor para ciertos
    productos: si fuera mucho peor en maíz/frijol (canasta básica) que en
    naranja, cualquier decisión basada en el modelo perjudicaría de forma
    desigual el análisis de alimentos esenciales.
    """
    df = pd.DataFrame({
        "grupo": grupo,
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
    })
    filas = []
    for g, sub in df.groupby("grupo"):
        m = ev.calcular_metricas(sub["y_true"], sub["y_pred"])
        m["grupo"] = g
        m["n"] = len(sub)
        filas.append(m)
    tabla = pd.DataFrame(filas).set_index("grupo")[["n", "MAE", "RMSE", "MAPE_%", "R2"]]
    return tabla.sort_values("MAPE_%")


def metricas_por_temporada(y_true, y_pred, es_cosecha):
    """Compara el error dentro y fuera de temporada de cosecha (sesgo estacional)."""
    etiqueta = np.where(np.asarray(es_cosecha) == 1, "cosecha", "no_cosecha")
    return metricas_por_grupo(y_true, y_pred, etiqueta)


def brecha_equidad(tabla_metricas_por_grupo, columna="MAPE_%"):
    """Razón (peor/mejor) del error entre grupos: cuantifica la disparidad de desempeño.

    Cercano a 1 = desempeño parejo entre grupos. Valores altos señalan que el
    modelo es mucho menos confiable para algunos productos/temporadas que para
    otros, lo que debe advertirse antes de usar la predicción para decisiones.
    """
    peor = tabla_metricas_por_grupo[columna].max()
    mejor = tabla_metricas_por_grupo[columna].min()
    return float(peor / mejor) if mejor > 0 else float("inf")


LIMITACIONES_ETICAS = """
- Los precios son de mercados MAYORISTAS del MAG: no reflejan directamente lo
  que paga el consumidor final ni lo que recibe el pequeño productor (los
  márgenes de intermediación no están capturados). No usar la predicción como
  proxy del ingreso del productor ni del precio al consumidor sin ajuste.
- Cobertura acotada (5 productos, 2021-2024, mercados formales): el modelo no
  generaliza a otros cultivos, regiones o choques atípicos fuera del
  histórico (sequías extremas, crisis de combustible, cierres de frontera).
- Uso previsto: apoyo informativo (planificación de compras institucionales,
  alerta temprana de variabilidad de precios). No debe usarse para
  especulación, acaparamiento o fijación de precios que perjudique a
  productores o consumidores.
- Desempeño desigual entre productos y temporadas (ver `metricas_por_grupo` /
  `metricas_por_temporada`): comunicar el error de CADA producto, no solo el
  promedio general, para no sobre-representar la confiabilidad del modelo en
  los casos donde es peor.
""".strip()


def reporte_texto(tabla_grupo, tabla_temporada):
    """Resumen textual (para notebook/app) con hallazgos de equidad + limitaciones."""
    brecha_g = brecha_equidad(tabla_grupo)
    brecha_t = brecha_equidad(tabla_temporada)
    return (
        f"Disparidad de error entre productos (peor/mejor MAPE): {brecha_g:.2f}x\n"
        f"Disparidad de error cosecha vs. no cosecha (peor/mejor MAPE): {brecha_t:.2f}x\n\n"
        f"Limitaciones y consideraciones éticas:\n{LIMITACIONES_ETICAS}"
    )
