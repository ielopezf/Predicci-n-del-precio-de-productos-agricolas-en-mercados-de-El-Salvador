"""a11_lstm.py — LSTM con el MISMO protocolo que a6 (variación relativa, origen móvil, misma prueba).

Entrada: ventana de las últimas L=8 semanas de las variables del origen (las de a4), por producto.
Salida: variación relativa y_{t+h}/p_t - 1 (igual que Ridge/RF/GBM), reentreno cada 8 semanas.
Hiperparámetros: 4 configuraciones aleatorias elegidas con los mismos 3 pliegues previos a la prueba y
el mismo criterio (promedio entre productos de MAE modelo / MAE persistencia).

IMPORTANTE: requiere tensorflow (o keras). Este script se verificó en modo --stub, que reemplaza la red
por Ridge sobre la ventana aplanada solo para comprobar el flujo de datos; la parte de Keras no se pudo
ejecutar donde se escribió. Si falla, revisen primero la función `ajustar_lstm`.

Uso: python scripts/a11_lstm.py --precios data/interim/precios_consolidados.xlsx [--stub]
Salida: reports/etapa3/lstm_detalle.csv y lstm_resumen.csv
"""
import argparse, sys, importlib.util, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from prediccion_precios import config, pipeline_causal as pc  # noqa: E402
spec = importlib.util.spec_from_file_location("a6", RAIZ / "scripts" / "a6_afinacion.py")
a6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a6)
a4 = a6.a4
warnings.filterwarnings("ignore")
SEED, VENTANA, N_CONFIG = 42, 8, 4
ESPACIO = {"unidades": [8, 16, 32], "dropout": [0.0, 0.2], "lr": [0.003, 0.01]}


def ajustar_lstm(Xtr, ytr, Xte, cfg, stub=False):
    """Entrena y predice. Xtr: (n, VENTANA, F) ya escalado; ytr: (n,) escalado."""
    if stub:   # solo para verificar el flujo de datos sin tensorflow
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=100.0).fit(Xtr.reshape(len(Xtr), -1), ytr)
        return m.predict(Xte.reshape(len(Xte), -1))
    try:
        import keras
    except ImportError:
        from tensorflow import keras
    keras.utils.set_random_seed(SEED)
    n_val = max(10, int(len(Xtr) * 0.15))                  # validación = el tramo más reciente del entrenamiento
    xt, yt, xv, yv = Xtr[:-n_val], ytr[:-n_val], Xtr[-n_val:], ytr[-n_val:]
    modelo = keras.Sequential([
        keras.layers.Input(shape=Xtr.shape[1:]),
        keras.layers.LSTM(cfg["unidades"]),
        keras.layers.Dropout(cfg["dropout"]),
        keras.layers.Dense(1),
    ])
    modelo.compile(optimizer=keras.optimizers.Adam(cfg["lr"]), loss="huber")
    modelo.fit(xt, yt, validation_data=(xv, yv), epochs=80, batch_size=32, verbose=0,
               callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    return modelo.predict(Xte, verbose=0).ravel()


def preparar_secuencias(d, feats, train_mask):
    """Z escalado con el entrenamiento; ventanas por producto (relleno con ceros al inicio)."""
    X = d[feats].astype(float)
    med = X[train_mask].median()
    X = X.fillna(med)
    esc = StandardScaler().fit(X[train_mask])
    Z = esc.transform(X)
    Z = np.nan_to_num(Z, nan=0.0)
    seq = np.zeros((len(d), VENTANA, Z.shape[1]), dtype=np.float32)
    grupos = d["Grupo"].to_numpy()
    inicio = {}
    for i, g in enumerate(grupos):
        inicio.setdefault(g, i)
    for i, g in enumerate(grupos):
        j = i - inicio[g]
        k = min(VENTANA, j + 1)
        seq[i, VENTANA - k:, :] = Z[i - k + 1:i + 1, :]
    return seq


def predecir_bloque(d, feats, tr_mask, te_mask, cfg, stub):
    seq = preparar_secuencias(d, feats, tr_mask.to_numpy())
    y = (d["y"] / d["p0"] - 1).to_numpy()
    ytr = y[tr_mask.to_numpy()]
    s = ytr.std() or 1.0
    p = ajustar_lstm(seq[tr_mask.to_numpy()], ytr / s, seq[te_mask.to_numpy()], cfg, stub) * s
    return d.loc[te_mask, "p0"].to_numpy() * (1 + p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precios", required=True)
    ap.add_argument("--stub", action="store_true", help="solo verifica el flujo de datos (sin tensorflow)")
    a = ap.parse_args()
    base = pd.read_csv(config.DATASET_MODELADO, parse_dates=[config.COL_FECHA])
    df = a4.features_origen(pc.con_externas(pc.semanal_causal(a.precios), base))
    dums = pd.get_dummies(df["Grupo"], prefix="g", dtype=float)
    df = pd.concat([df, dums], axis=1)
    feats = a4.lista_features(df) + list(dums.columns)
    INI = a4.INI_TEST
    configs = list(ParameterSampler(ESPACIO, n_iter=N_CONFIG, random_state=SEED))

    filas = []
    for h in a4.HORIZ:
        d = a6.preparar(df, h, feats)
        previa = d[d["ok"] & (d["fecha_obj"] < INI)]
        fechas = np.array(sorted(previa["fecha_obj"].unique()))
        cortes = [fechas[int(len(fechas) * q)] for q in (0.55, 0.70, 0.85)] + [fechas[-1] + pd.Timedelta(days=1)]
        res = []
        for cfg in configs:
            sc = []
            for i in range(3):
                tr = d["ok"] & (d["fecha_obj"] < cortes[i])
                te = d["ok"] & (d["fecha_obj"] >= cortes[i]) & (d["fecha_obj"] < cortes[i + 1])
                pred = predecir_bloque(d, feats, tr, te, cfg, a.stub)
                sc.append(a6.puntaje(d[te], pred))
            res.append((float(np.mean(sc)), cfg))
        mejor = min(res, key=lambda r: r[0])[1]
        print(f"h={h}: mejor configuración {mejor}", flush=True)
        test = d["ok"] & (d["fecha_obj"] >= INI)
        bloques = np.array_split(sorted(d.loc[test, "fecha_obj"].unique()), int(np.ceil(32 / a4.REFIT)))
        for bl in bloques:
            tr = d["ok"] & (d["fecha_obj"] < bl[0])
            te = test & d["fecha_obj"].isin(bl)
            if te.sum() == 0:
                continue
            pred = predecir_bloque(d, feats, tr, te, mejor, a.stub)
            t = d[te]
            filas.append(pd.DataFrame({"horizonte": h, "modelo": "LSTM_afinado", "producto": t["Grupo"].to_numpy(),
                                       "fecha": t["fecha_obj"].to_numpy(), "real": t["y"].to_numpy(),
                                       "predicho": pred, "persistencia": t["p0"].to_numpy()}))
    det = pd.concat(filas, ignore_index=True)
    det["err"] = (det["real"] - det["predicho"]).abs(); det["ape"] = det["err"] / det["real"] * 100
    det["err_pers"] = (det["real"] - det["persistencia"]).abs()
    sufijo = "_stub" if a.stub else ""
    out = RAIZ / "reports" / "etapa3"
    det.to_csv(out / f"lstm_detalle{sufijo}.csv", index=False)
    t = (det.groupby(["horizonte", "producto"]).agg(n=("err", "size"), MAE=("err", "mean"), MAPE=("ape", "mean"),
         MAE_pers=("err_pers", "mean")).reset_index())
    t["razon_vs_persistencia"] = t["MAE"] / t["MAE_pers"]
    t.round(4).to_csv(out / f"lstm_resumen{sufijo}.csv", index=False)
    print("\n== Razón MAE LSTM / persistencia por producto ==")
    print(t.pivot(index="producto", columns="horizonte", values="razon_vs_persistencia").round(2))
    sa = t[t.producto != "ARROZ"].groupby("horizonte").razon_vs_persistencia.mean().round(3)
    print("\nPromedio sin arroz:", sa.to_dict())


if __name__ == "__main__":
    main()
