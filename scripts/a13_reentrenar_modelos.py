"""a13_reentrenar_modelos.py — Reentrena los 7 modelos de la Etapa 2 con el dataset CAUSAL.

Qué hace (replica los notebooks 03 y 04, cambiando SOLO los datos):
  1. Respalda models/ en models_etapa2/ (si no existe).
  2. Construye data/processed/dataset_modelado_causal.csv (sin recorte IQR, sin interpolar: ffill)
     y dataset_causal_flags.csv (precio_imputado, n_dias, semanas_sin_cambio).
  3. Notebook 03: regresion_lineal, random_forest, xgboost  (split 80/20).
  4. Notebook 04: xgboost_optimizado (Optuna, 40 pruebas), red_neuronal, lstm, ensemble_stacking
     (split 65/15/20). Guarda todo en models/ con los mismos nombres.
  5. Escribe reports/etapa3/reentrenamiento_tabla.csv: Etapa 2 vs dataset causal, más la
     persistencia (último precio) y el MAPE solo en semanas con precio real publicado.

Requiere: xgboost, optuna y tensorflow/keras (los mismos del requirements.txt).
Uso:  python scripts/a13_reentrenar_modelos.py --precios data/interim/precios_consolidados.xlsx
      python scripts/a13_reentrenar_modelos.py --solo-dataset --precios ...   (solo arma el dataset)
"""
import argparse, shutil, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from prediccion_precios import config, features as ft, evaluation as ev  # noqa: E402
from prediccion_precios import dataset_causal as dc, models_baseline as mb  # noqa: E402
warnings.filterwarnings("ignore")


def mape(y, p, mask=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if mask is not None:
        y, p = y[mask], p[mask]
    return round(float(np.mean(np.abs(y - p) / y) * 100), 4) if len(y) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precios", required=True)
    ap.add_argument("--solo-dataset", action="store_true")
    a = ap.parse_args()

    # 1) respaldo
    resp = RAIZ / "models_etapa2"
    if config.MODELS_DIR.exists() and not resp.exists():
        shutil.copytree(config.MODELS_DIR, resp)
        print(f"Respaldo de los modelos de la Etapa 2 en {resp}")

    # 2) dataset causal
    data, flags = dc.construir(a.precios)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(config.DATASET_CAUSAL, index=False)
    flags.to_csv(config.FLAGS_CAUSAL, index=False)
    print(f"Dataset causal: {data.shape} -> {config.DATASET_CAUSAL}")
    print(f"Semanas arrastradas (precio_imputado=1): {int(flags['precio_imputado'].sum())} de {len(flags)}")
    if a.solo_dataset:
        return

    X, y, meta = ft.construir_matriz_modelado(data, return_meta=True)
    X = X.astype(float)
    grupo_de = data.drop_duplicates("Grupo").set_index(config.COL_PRODUCTO)["Grupo"].to_dict()
    meta["Grupo"] = meta[config.COL_PRODUCTO].map(grupo_de)
    meta = meta.merge(flags, on=[config.COL_FECHA, "Grupo"], how="left")
    print("X", X.shape)

    resultados, resultados_real = {}, {}

    # 3) Notebook 03 — baselines (split 80/20)
    X_tr, X_te, y_tr, y_te = ev.split_temporal(X, y)
    n_te = len(y_te)
    real_te = (meta["precio_imputado"].iloc[-n_te:].to_numpy() == 0)
    base = mb.entrenar_todos(X_tr, y_tr)
    for n, m in base.items():
        mb.guardar_modelo(m, n)
    nombre_baseline = {"regresion_lineal": "regresion_lineal", "random_forest": "random_forest", "xgboost": "xgboost_baseline"}
    for n, m in base.items():
        p = m.predict(X_te)
        resultados[nombre_baseline.get(n, n)] = ev.calcular_metricas(y_te, p)
        resultados_real[nombre_baseline.get(n, n)] = mape(y_te, p, real_te)

    # 4) Notebook 04 — avanzados (split 65/15/20)
    from prediccion_precios import models_advanced as ma
    X_train, X_val, X_test, y_train, y_val, y_test = ev.split_train_val_test(X, y)
    t0 = time.time()
    xgb_opt, info = ma.optimizar_xgboost(X_train, y_train, metodo="optuna", n_trials=40)
    print(f"HPO listo en {time.time() - t0:.0f}s", info.get("mejores_parametros"))
    nn = ma.RedNeuronalRegresora(epochs=300, batch_size=16, paciencia=30, verbose=0)
    nn.fit(X_train, y_train, X_val, y_val)
    lin = mb.entrenar_regresion_lineal(X_train, y_train)
    rf = mb.entrenar_random_forest(X_train, y_train)
    ens = ma.construir_ensemble({"regresion_lineal": lin, "random_forest": rf, "xgboost_optimizado": xgb_opt},
                                X_val, y_val, metodo="stacking")
    p_opt, p_nn, p_ens = xgb_opt.predict(X_test), nn.predict(X_test), ens.predict(X_test)
    real_te4 = (meta["precio_imputado"].iloc[-len(y_test):].to_numpy() == 0)
    for n, p in (("xgboost_optimizado", p_opt), ("red_neuronal", p_nn), ("ensemble_stacking", p_ens)):
        resultados[n] = ev.calcular_metricas(y_test, p)
        resultados_real[n] = mape(y_test, p, real_te4)
    pers = X_test["precio_lag1"].to_numpy()
    resultados["persistencia (último precio)"] = ev.calcular_metricas(y_test, pers)
    resultados_real["persistencia (último precio)"] = mape(y_test, pers, real_te4)

    Xs, ys, _, _ = ma.preparar_secuencias(data, ventana=8, columnas=[config.COL_PRECIO])
    n = len(Xs); n_te_s = int(np.ceil(n * config.TEST_SIZE)); n_va_s = int(np.ceil(n * 0.15)); n_tr_s = n - n_te_s - n_va_s
    lstm = ma.LSTMRegresor(unidades=64, epochs=300, batch_size=16, paciencia=30, verbose=0)
    lstm.fit(Xs[:n_tr_s], ys[:n_tr_s], Xs[n_tr_s:n_tr_s + n_va_s], ys[n_tr_s:n_tr_s + n_va_s])
    p_lstm = lstm.predict(Xs[n_tr_s + n_va_s:])
    resultados["lstm (tramo propio)"] = ev.calcular_metricas(ys[n_tr_s + n_va_s:], p_lstm)
    resultados_real["lstm (tramo propio)"] = float("nan")

    mb.guardar_modelo(xgb_opt, "xgboost_optimizado")
    mb.guardar_modelo(ens, "ensemble_stacking")
    nn.guardar(str(config.MODELS_DIR / "red_neuronal"))
    lstm.guardar(str(config.MODELS_DIR / "lstm"))
    print("Modelos guardados en", config.MODELS_DIR)

    # 5) tabla Etapa 2 vs causal
    tabla = ev.tabla_comparativa(resultados)
    tabla["MAPE_%_solo_semanas_reales"] = pd.Series(resultados_real)
    viejo = pd.read_csv(RAIZ / "reports" / "etapa2" / "tabla_comparativa_etapa2.csv").set_index("Modelo")
    tabla["MAPE_%_etapa2"] = viejo["MAPE_%"]
    out = RAIZ / "reports" / "etapa3"; out.mkdir(parents=True, exist_ok=True)
    tabla.round(4).to_csv(out / "reentrenamiento_tabla.csv")
    pd.set_option("display.width", 160)
    print("\n== Etapa 2 (interpolado) vs dataset causal ==")
    print(tabla[["MAPE_%_etapa2", "MAPE_%", "MAPE_%_solo_semanas_reales", "MAE", "R2"]].round(3))
    print("\nNota: la LSTM se evalúa en un tramo de secuencias propio; no es comparable fila a fila.")


if __name__ == "__main__":
    main()
