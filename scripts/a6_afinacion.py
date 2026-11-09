"""a6_afinacion.py — Afinación equivalente de todos los modelos y tabla por producto (A6).

Misma validación para todos: 3 pliegues de origen móvil (ventana creciente) en las semanas
ANTERIORES a la prueba. Criterio: promedio, entre productos, de MAE_modelo / MAE_persistencia
(así el tomate no domina). Búsqueda aleatoria con el mismo presupuesto por modelo.
Luego se evalúa en la prueba con el protocolo de a4 (reentreno cada 8 semanas).
Uso: python scripts/a6_afinacion.py --precios precios_consolidados.xlsx
"""
import argparse, sys, warnings, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from prediccion_precios import config, pipeline_causal as pc  # noqa: E402
spec = importlib.util.spec_from_file_location("a4", RAIZ / "scripts" / "a4_horizontes.py")
a4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a4)
warnings.filterwarnings("ignore")
SEED, N_ITER, HORIZ, INI = 42, 10, a4.HORIZ, a4.INI_TEST

ESPACIO = {
    "Ridge_var": ("variacion", {"alpha": [1, 10, 30, 100, 300, 1000, 3000]},
                  lambda p: make_pipeline(StandardScaler(), Ridge(**p))),
    "RF_var": ("variacion", {"min_samples_leaf": [3, 5, 10, 20], "max_features": [0.3, 0.5, 1.0], "max_depth": [3, 5, None]},
               lambda p: RandomForestRegressor(200, n_jobs=2, random_state=SEED, **p)),
    "HGB_var": ("variacion", {"max_depth": [2, 3, 4], "learning_rate": [0.02, 0.05, 0.1], "max_iter": [100, 200, 400],
                              "l2_regularization": [0, 1, 10], "min_samples_leaf": [10, 20, 40]},
                lambda p: HistGradientBoostingRegressor(random_state=SEED, **p)),
    "RL_nivel": ("nivel", {"_": [0]}, lambda p: make_pipeline(StandardScaler(), LinearRegression())),
}


def preparar(df, h, feats_m):
    d = df.copy()
    d["y"] = d.groupby("Grupo")["precio_obs"].shift(-h)
    d["fecha_obj"] = d.groupby("Grupo")[config.COL_FECHA].shift(-h)
    d["p0"] = d["precio_vigente"]
    d["ok"] = d["y"].notna() & d["p0"].notna() & d["precio_imputado"].eq(0)
    return d


def ajustar_predecir(nombre, p, tr, te, feats_m):
    obj, _, mk = ESPACIO[nombre]
    cols = feats_m + (["precio_vigente"] if obj == "nivel" else [])
    Xtr, Xte = tr[cols], te[cols]
    if nombre != "HGB_var":
        med = Xtr.median(); Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    ytr = tr["y"] if obj == "nivel" else tr["y"] / tr["p0"] - 1
    pr = mk(p).fit(Xtr, ytr).predict(Xte)
    return pr if obj == "nivel" else te["p0"].to_numpy() * (1 + pr)


def puntaje(te, pred):
    e = pd.DataFrame({"g": te["Grupo"].to_numpy(), "m": np.abs(te["y"].to_numpy() - pred),
                      "b": np.abs(te["y"].to_numpy() - te["p0"].to_numpy())})
    s = e.groupby("g")[["m", "b"]].mean()
    return float((s["m"] / s["b"]).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--precios", required=True); a = ap.parse_args()
    base = pd.read_csv(config.DATASET_MODELADO, parse_dates=[config.COL_FECHA])
    df = a4.features_origen(pc.con_externas(pc.semanal_causal(a.precios), base))
    feats = a4.lista_features(df)
    dums = pd.get_dummies(df["Grupo"], prefix="g", dtype=float)
    df = pd.concat([df, dums], axis=1); feats_m = feats + list(dums.columns)

    mejores, filas_cv = {}, []
    for h in HORIZ:
        d = preparar(df, h, feats_m)
        previa = d[d["ok"] & (d["fecha_obj"] < INI)]
        fechas = np.array(sorted(previa["fecha_obj"].unique()))
        cortes = [fechas[int(len(fechas) * q)] for q in (0.55, 0.70, 0.85)] + [fechas[-1] + pd.Timedelta(days=1)]
        for nombre, (obj, grid, _) in ESPACIO.items():
            cands = list(ParameterSampler(grid, n_iter=min(N_ITER, int(np.prod([len(v) for v in grid.values()]))), random_state=SEED))
            res = []
            for p in cands:
                sc = []
                for i in range(3):
                    # no hay fuga: el entrenamiento solo usa objetivos conocidos antes del inicio del pliegue
                    tr = previa[previa["fecha_obj"] < cortes[i]]
                    te = previa[(previa["fecha_obj"] >= cortes[i]) & (previa["fecha_obj"] < cortes[i + 1])]
                    sc.append(puntaje(te, ajustar_predecir(nombre, p, tr, te, feats_m)))
                res.append((np.mean(sc), p))
                filas_cv.append({"horizonte": h, "modelo": nombre, "params": str(p), "puntaje_cv": np.mean(sc)})
            mejores[(h, nombre)] = min(res, key=lambda r: r[0])[1]
            print(f"h={h} {nombre}: {mejores[(h, nombre)]}")
    pd.DataFrame(filas_cv).to_csv(RAIZ / "reports" / "etapa3" / "afinacion_cv.csv", index=False)

    filas = []
    for h in HORIZ:
        d = preparar(df, h, feats_m)
        test = d[d["ok"] & (d["fecha_obj"] >= INI)]
        bloques = np.array_split(sorted(test["fecha_obj"].unique()), int(np.ceil(32 / a4.REFIT)))
        for nombre in ESPACIO:
            for bl in bloques:
                tr = d[d["ok"] & (d["fecha_obj"] < bl[0])]; te = test[test["fecha_obj"].isin(bl)]
                if len(te) == 0: continue
                pr = ajustar_predecir(nombre, mejores[(h, nombre)], tr, te, feats_m)
                filas.append(pd.DataFrame({"horizonte": h, "modelo": nombre + "_afinado" if nombre != "RL_nivel" else nombre,
                                           "producto": te["Grupo"].to_numpy(), "fecha": te["fecha_obj"].to_numpy(),
                                           "real": te["y"].to_numpy(), "predicho": pr, "persistencia": te["p0"].to_numpy()}))
    det = pd.concat(filas, ignore_index=True)
    per = det.drop_duplicates(["horizonte", "producto", "fecha"]).assign(modelo="persistencia", predicho=lambda x: x["persistencia"])
    det = pd.concat([det, per], ignore_index=True)
    det["err"] = (det["real"] - det["predicho"]).abs(); det["ape"] = det["err"] / det["real"] * 100
    det["err_pers"] = (det["real"] - det["persistencia"]).abs()
    det.to_csv(RAIZ / "reports" / "etapa3" / "afinado_detalle.csv", index=False)
    t = (det.groupby(["horizonte", "producto", "modelo"])
         .agg(n=("err", "size"), MAE=("err", "mean"), MAPE=("ape", "mean"), MAE_pers=("err_pers", "mean")).reset_index())
    t["razon_vs_persistencia"] = t["MAE"] / t["MAE_pers"]
    t.round(4).to_csv(RAIZ / "reports" / "etapa3" / "tabla_por_producto.csv", index=False)
    for h in HORIZ:
        print(f"\n== MAPE % por producto, h={h} ==")
        print(t[t.horizonte == h].pivot(index="modelo", columns="producto", values="MAPE").round(2))
        print(f"-- razón MAE vs persistencia, h={h} --")
        print(t[t.horizonte == h].pivot(index="modelo", columns="producto", values="razon_vs_persistencia").round(2))
    prom = t.groupby(["horizonte", "modelo"])["razon_vs_persistencia"].mean().unstack(0).round(3)
    print("\n== Promedio entre productos de la razón MAE (peso igual) =="); print(prom)


if __name__ == "__main__":
    main()
