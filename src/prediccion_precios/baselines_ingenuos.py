"""baselines_ingenuos.py — Baselines ingenuos y de series de tiempo clásicas (A1).

Por qué existe: un MAPE solo es "bueno" respecto a una referencia. Este módulo
da las referencias mínimas que un pronóstico de precios debe superar:

    persistencia      y_{t+h} = y_t                      (el precio no cambia)
    media_movil_4     promedio de las últimas 4 semanas
    estacional_52     y_{t+h} = y_{t+h-52}               (mismo valor del año anterior)
    deriva            y_t + h * (y_t - y_1) / (t - 1)
    ets_amortiguado   Holt con tendencia amortiguada (suavizado exponencial)
    arima_p10         ARIMA(p,1,0): AR(p) sobre la serie diferenciada, p por AIC

ETS y ARIMA están implementados con NumPy/SciPy (sin statsmodels) para que el
módulo corra con las dependencias que ya tiene el proyecto. Si prefieren la
implementación de statsmodels (ETSModel / ARIMA), la interfaz `pronosticar`
no cambia: basta reemplazar `_pronosticar_ets` y `_pronosticar_arima`.

Convención de pronóstico: dado el historial `hist` (precios hasta el origen,
inclusive) devuelve un arreglo de `h_max` valores: pronóstico a 1..h_max semanas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config

METODOS = ("persistencia", "media_movil_4", "estacional_52", "deriva",
           "ets_amortiguado", "arima_p10")
HORIZONTES = (1, 2, 4, 8)


# --------------------------------------------------------------------------- #
# Pronosticadores simples
# --------------------------------------------------------------------------- #
def _persistencia(hist, h_max):
    return np.full(h_max, hist[-1], dtype=float)


def _media_movil(hist, h_max, k=4):
    return np.full(h_max, float(np.mean(hist[-k:])))


def _estacional(hist, h_max, m=52):
    if len(hist) < m:
        return np.full(h_max, np.nan)
    base = hist[len(hist) - m:]
    return np.array([base[(i) % m] for i in range(h_max)], dtype=float)


def _deriva(hist, h_max):
    if len(hist) < 2:
        return _persistencia(hist, h_max)
    pendiente = (hist[-1] - hist[0]) / (len(hist) - 1)
    return hist[-1] + pendiente * np.arange(1, h_max + 1)


# --------------------------------------------------------------------------- #
# ETS con tendencia amortiguada (Holt-amortiguado), ajuste por mínimos cuadrados
# --------------------------------------------------------------------------- #
def _sse_ets(params, y):
    alpha, beta, phi = params
    nivel, tend = y[0], (y[1] - y[0]) if len(y) > 1 else 0.0
    sse = 0.0
    for t in range(1, len(y)):
        pred = nivel + phi * tend
        err = y[t] - pred
        sse += err * err
        nivel_nuevo = alpha * y[t] + (1 - alpha) * (nivel + phi * tend)
        tend = beta * (nivel_nuevo - nivel) + (1 - beta) * phi * tend
        nivel = nivel_nuevo
    return sse


def _pronosticar_ets(hist, h_max):
    y = np.asarray(hist, dtype=float)
    if len(y) < 4:
        return _persistencia(y, h_max)
    mejor = None
    for x0 in ((0.8, 0.1, 0.9), (0.5, 0.05, 0.95)):
        r = minimize(_sse_ets, x0=x0, args=(y,), method="L-BFGS-B",
                     bounds=[(0.01, 0.99), (0.001, 0.99), (0.8, 0.98)])
        if mejor is None or r.fun < mejor.fun:
            mejor = r
    alpha, beta, phi = mejor.x
    nivel, tend = y[0], y[1] - y[0]
    for t in range(1, len(y)):
        nivel_nuevo = alpha * y[t] + (1 - alpha) * (nivel + phi * tend)
        tend = beta * (nivel_nuevo - nivel) + (1 - beta) * phi * tend
        nivel = nivel_nuevo
    acum = np.cumsum(phi ** np.arange(1, h_max + 1))
    return nivel + acum * tend


# --------------------------------------------------------------------------- #
# ARIMA(p,1,0): AR(p) con intercepto sobre la serie diferenciada
# --------------------------------------------------------------------------- #
def _ajustar_ar(dy, p):
    """OLS de dy_t sobre (1, dy_{t-1..t-p}). Devuelve (coef, rss, n)."""
    n = len(dy) - p
    X = np.column_stack([np.ones(n)] + [dy[p - j - 1: len(dy) - j - 1] for j in range(p)])
    z = dy[p:]
    coef, *_ = np.linalg.lstsq(X, z, rcond=None)
    rss = float(np.sum((z - X @ coef) ** 2))
    return coef, rss, n


def elegir_orden_arima(hist, p_max=4):
    """p que minimiza el AIC del AR(p) sobre la serie diferenciada."""
    dy = np.diff(np.asarray(hist, dtype=float))
    mejor_p, mejor_aic = 1, np.inf
    for p in range(1, p_max + 1):
        if len(dy) <= p + 5:
            break
        _, rss, n = _ajustar_ar(dy, p)
        aic = n * np.log(max(rss, 1e-12) / n) + 2 * (p + 1)
        if aic < mejor_aic:
            mejor_p, mejor_aic = p, aic
    return mejor_p


def _pronosticar_arima(hist, h_max, p=1):
    y = np.asarray(hist, dtype=float)
    dy = np.diff(y)
    if len(dy) <= p + 5:
        return _persistencia(y, h_max)
    coef, _, _ = _ajustar_ar(dy, p)
    pasado = list(dy[-p:][::-1])           # dy_t, dy_{t-1}, ...
    nivel, salida = y[-1], []
    for _ in range(h_max):
        d = coef[0] + float(np.dot(coef[1:], pasado[:p]))
        nivel += d
        salida.append(nivel)
        pasado.insert(0, d)
    return np.array(salida)


def pronosticar(metodo, hist, h_max, p_arima=1):
    """Pronóstico a 1..h_max semanas desde el final de `hist`."""
    hist = np.asarray(hist, dtype=float)
    if metodo == "persistencia":
        return _persistencia(hist, h_max)
    if metodo == "media_movil_4":
        return _media_movil(hist, h_max, 4)
    if metodo == "estacional_52":
        return _estacional(hist, h_max, 52)
    if metodo == "deriva":
        return _deriva(hist, h_max)
    if metodo == "ets_amortiguado":
        return _pronosticar_ets(hist, h_max)
    if metodo == "arima_p10":
        return _pronosticar_arima(hist, h_max, p_arima)
    raise ValueError(f"Método desconocido: {metodo}")


# --------------------------------------------------------------------------- #
# Evaluación sobre el mismo tramo de prueba que usan los demás modelos
# --------------------------------------------------------------------------- #
def marcar_imputadas(precios):
    """Heurística: punto central de un tramo EXACTAMENTE lineal = semana interpolada.

    Solo es una aproximación para el dataset procesado; la marca definitiva debe
    salir del pipeline de datos (columna `precio_imputado`, tarea A3).
    """
    p = np.asarray(precios, dtype=float)
    marca = np.zeros(len(p), dtype=bool)
    if len(p) >= 3:
        d1 = np.abs(np.diff(p))
        d2 = np.abs(np.diff(p, 2))
        marca[1:-1] = (d2 < 1e-9) & (d1[1:] > 1e-9)
    return marca


def filas_de_prueba(df, test_size=config.TEST_SIZE):
    """Ids (producto, posición) de las filas de prueba, idénticos a los de los notebooks.

    Replica `features.construir_matriz_modelado` (orden por fecha, sin NaN) y
    `evaluation.indices_train_val_test`: la prueba son las últimas ceil(0.2*n) filas.
    """
    d = df.sort_values(config.COL_FECHA).reset_index(drop=True).dropna().reset_index(drop=True)
    n_test = int(np.ceil(len(d) * test_size))
    return d.iloc[len(d) - n_test:][[config.COL_FECHA, "Grupo"]].reset_index(drop=True)


def evaluar(df, metodos=METODOS, horizontes=HORIZONTES, test_size=config.TEST_SIZE,
            verbose=False):
    """Evalúa cada método a cada horizonte sobre las filas de prueba.

    Devuelve un DataFrame largo con una fila por (metodo, horizonte, producto, fecha):
    real, predicho, origen_imputado/objetivo_imputado y la escala MASE del producto.
    """
    h_max = max(horizontes)
    prueba = filas_de_prueba(df, test_size)
    registros = []

    for grupo, ssub in df.groupby("Grupo"):
        serie = ssub.sort_values(config.COL_FECHA).reset_index(drop=True)
        fechas = serie[config.COL_FECHA].to_numpy()
        y = serie[config.COL_PRECIO].to_numpy(dtype=float)
        imputada = marcar_imputadas(y)
        objetivos = prueba[prueba["Grupo"] == grupo][config.COL_FECHA].to_numpy()
        pos_obj = np.array([int(np.where(fechas == f)[0][0]) for f in objetivos])
        inicio_prueba = pos_obj.min()
        # Escala MASE: MAE del pronóstico ingenuo a 1 semana, SOLO con datos previos a la prueba.
        escala = float(np.mean(np.abs(np.diff(y[:inicio_prueba]))))
        p_arima = elegir_orden_arima(y[:inicio_prueba])

        orígenes = sorted({int(p - h) for p in pos_obj for h in horizontes})
        for metodo in metodos:
            cache = {o: pronosticar(metodo, y[:o + 1], h_max, p_arima) for o in orígenes}
            for pos in pos_obj:
                for h in horizontes:
                    o = int(pos - h)
                    registros.append({
                        "metodo": metodo, "horizonte": h, "producto": grupo,
                        "fecha": fechas[pos], "real": y[pos], "predicho": cache[o][h - 1],
                        "imputada": bool(imputada[pos] or imputada[o]),
                        "escala_mase": escala,
                    })
        if verbose:
            print(f"{grupo}: p_arima={p_arima}, escala MASE={escala:.4f}, "
                  f"{len(pos_obj)} filas de prueba")
    return pd.DataFrame(registros)


def resumir(detalle):
    """Métricas por (metodo, horizonte, producto) y globales (producto='GLOBAL')."""
    def _m(g):
        ok = g["predicho"].notna()
        g = g[ok]
        if g.empty:
            return pd.Series({"n": 0, "MAE": np.nan, "MAPE_%": np.nan, "MASE": np.nan,
                              "MAPE_sin_imputadas_%": np.nan})
        err = np.abs(g["real"] - g["predicho"])
        sin = g[~g["imputada"]]
        return pd.Series({
            "n": len(g),
            "MAE": err.mean(),
            "MAPE_%": (err / g["real"]).mean() * 100,
            "MASE": (err / g["escala_mase"]).mean(),
            "MAPE_sin_imputadas_%": ((np.abs(sin["real"] - sin["predicho"]) / sin["real"]).mean() * 100
                                     if len(sin) else np.nan),
        })

    por_producto = (detalle.groupby(["metodo", "horizonte", "producto"])
                    .apply(_m, include_groups=False).reset_index())
    glob = (detalle.groupby(["metodo", "horizonte"]).apply(_m, include_groups=False)
            .reset_index().assign(producto="GLOBAL"))
    tabla = pd.concat([por_producto, glob], ignore_index=True)
    # Razón de MAE frente a la persistencia (mismo horizonte y producto): < 1 = la mejora.
    ref = (tabla[tabla["metodo"] == "persistencia"]
           .set_index(["horizonte", "producto"])["MAE"].rename("MAE_persistencia"))
    tabla = tabla.join(ref, on=["horizonte", "producto"])
    tabla["razon_MAE_vs_persistencia"] = tabla["MAE"] / tabla["MAE_persistencia"]
    cols = ["metodo", "horizonte", "producto", "n", "MAE", "MAPE_%", "MASE",
            "razon_MAE_vs_persistencia", "MAPE_sin_imputadas_%"]
    return tabla[cols]
