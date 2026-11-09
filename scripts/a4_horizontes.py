"""a4_horizontes.py — Horizontes 1/2/4/8 semanas, objetivo en variación y origen móvil (A4+A5).

Para cada horizonte h se pronostica el precio de la semana t+h con información hasta t.
Objetivo "nivel": y_{t+h}.  Objetivo "variacion": y_{t+h}/p_t - 1 (p_t = precio vigente).
Prueba: las semanas del tramo de prueba del proyecto (2023-11-26 a 2024-06-30). Reentreno
cada REFIT semanas (origen móvil, ventana expansiva). Solo se evalúan objetivos observados
(no interpolados) y orígenes con precio vigente. Persistencia = p_t.
Uso: python scripts/a4_horizontes.py --precios precios_consolidados.xlsx
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from prediccion_precios import config, pipeline_causal as pc  # noqa: E402
warnings.filterwarnings("ignore")

HORIZ = (1, 2, 4, 8); REFIT = 8; SEED = config.RANDOM_STATE if hasattr(config, "RANDOM_STATE") else 42
INI_TEST = pd.Timestamp("2023-11-26")
CAL = ["mes_sin", "mes_cos", "semana_sin", "semana_cos", "es_cosecha"]


def features_origen(df):
    df = df.sort_values(["Grupo", config.COL_FECHA]).reset_index(drop=True)
    g = df.groupby("Grupo")["precio_vigente"]
    for k in (1, 3, 7):
        df[f"ret_{k}"] = df["precio_vigente"] / g.shift(k) - 1
    for v in (4, 8, 12):
        df[f"mm_{v}"] = df["precio_vigente"] / g.transform(lambda s: s.rolling(v, min_periods=2).mean()) - 1
        df[f"dv_{v}"] = g.transform(lambda s: s.pct_change().rolling(v, min_periods=2).std())
    sem = df[config.COL_FECHA].dt.isocalendar().week.astype(float)
    mes = df[config.COL_FECHA].dt.month.astype(float)
    df["mes_sin"], df["mes_cos"] = np.sin(2*np.pi*mes/12), np.cos(2*np.pi*mes/12)
    df["semana_sin"], df["semana_cos"] = np.sin(2*np.pi*sem/52), np.cos(2*np.pi*sem/52)
    from prediccion_precios import features as ft
    df = ft.marcar_temporada_cosecha(df)
    for c in pc.EXTERNAS:
        df[c] = df[c].astype(float)
    return df


def lista_features(df):
    base = [c for c in df.columns if c.startswith(("ret_", "mm_", "dv_"))]
    X = base + CAL + ["n_dias", "semanas_sin_cambio", "precio_imputado"] + pc.EXTERNAS
    return X


def modelos():
    return {
        "RL_nivel": ("nivel", lambda: make_pipeline(StandardScaler(), LinearRegression())),
        "Ridge_var": ("variacion", lambda: make_pipeline(StandardScaler(), Ridge(alpha=30.0))),
        "RF_var": ("variacion", lambda: RandomForestRegressor(200, min_samples_leaf=5, max_features=0.5, n_jobs=2, random_state=SEED)),
        "HGB_var": ("variacion", lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=SEED)),
    }


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--precios", required=True); a = ap.parse_args()
    base = pd.read_csv(config.DATASET_MODELADO, parse_dates=[config.COL_FECHA])
    df = pc.con_externas(pc.semanal_causal(a.precios), base)
    df = features_origen(df)
    feats = lista_features(df)
    fechas_test = sorted(df.loc[df[config.COL_FECHA] >= INI_TEST, config.COL_FECHA].unique())
    prods = pd.get_dummies(df["Grupo"], prefix="g", dtype=float)
    df = pd.concat([df, prods], axis=1); feats_m = feats + list(prods.columns)
    filas = []
    for h in HORIZ:
        d = df.copy()
        d["y"] = d.groupby("Grupo")["precio_obs"].shift(-h)
        d["fecha_obj"] = d.groupby("Grupo")[config.COL_FECHA].shift(-h)
        d["p0"] = d["precio_vigente"]
        valido = d["y"].notna() & d["p0"].notna() & d["precio_imputado"].eq(0)
        d = d[d[feats_m].notna().all(axis=1) | True]  # HGB admite NaN; los demás se imputan abajo
        test = d[valido & (d["fecha_obj"] >= INI_TEST)].copy()
        bloques = np.array_split(sorted(test["fecha_obj"].unique()), max(1, int(np.ceil(len(fechas_test) / REFIT))))
        for nombre, (obj, mk) in modelos().items():
            for bl in bloques:
                ini = bl[0]
                # entrenamiento: solo objetivos conocidos ANTES del primer objetivo del bloque
                tr = d[valido & (d["fecha_obj"] < ini)]
                te = test[test["fecha_obj"].isin(bl)]
                if len(te) == 0 or len(tr) < 50: continue
                cols = feats_m + (['precio_vigente'] if obj == 'nivel' else [])
                Xtr, Xte = tr[cols], te[cols]
                if nombre != "HGB_var":
                    med = Xtr.median(); Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
                ytr = tr["y"] if obj == "nivel" else tr["y"] / tr["p0"] - 1
                m = mk().fit(Xtr, ytr); p = m.predict(Xte)
                pred = p if obj == "nivel" else te["p0"].to_numpy() * (1 + p)
                filas.append(pd.DataFrame({"horizonte": h, "modelo": nombre, "producto": te["Grupo"].to_numpy(),
                                           "fecha": te["fecha_obj"].to_numpy(), "real": te["y"].to_numpy(),
                                           "predicho": pred, "persistencia": te["p0"].to_numpy()}))
    det = pd.concat(filas, ignore_index=True)
    per = det.drop_duplicates(["horizonte", "producto", "fecha"]).assign(modelo="persistencia", predicho=lambda x: x["persistencia"])
    det = pd.concat([det, per], ignore_index=True)
    det["err"] = (det["real"] - det["predicho"]).abs(); det["ape"] = det["err"] / det["real"] * 100
    det["err_pers"] = (det["real"] - det["persistencia"]).abs()
    out = RAIZ / "reports" / "etapa3"; det.to_csv(out / "horizontes_detalle.csv", index=False)

    rng = np.random.default_rng(SEED)
    def resumen(g):
        sem = g["fecha"].astype("datetime64[ns]").dt.to_period("W").astype(str)
        u = np.array(sorted(sem.unique())); idx = {s: np.where(sem.to_numpy() == s)[0] for s in u}
        r = []
        for _ in range(500):  # bootstrap por bloques de 4 semanas
            ini = rng.integers(0, len(u), size=int(np.ceil(len(u) / 4)))
            sel = np.concatenate([idx[s] for i in ini for s in u[i:i+4]])
            r.append(g["err"].to_numpy()[sel].mean() / g["err_pers"].to_numpy()[sel].mean())
        return pd.Series({"n": len(g), "MAE": g["err"].mean(), "MAPE_%": g["ape"].mean(),
                          "razon_MAE_vs_pers": g["err"].mean() / g["err_pers"].mean(),
                          "IC95_inf": np.percentile(r, 2.5), "IC95_sup": np.percentile(r, 97.5)})
    glob = det.groupby(["horizonte", "modelo"]).apply(resumen, include_groups=False).reset_index().assign(producto="GLOBAL")
    prod = det.groupby(["horizonte", "modelo", "producto"]).apply(resumen, include_groups=False).reset_index()
    res = pd.concat([glob, prod], ignore_index=True); res.to_csv(out / "horizontes_resumen.csv", index=False)
    for col in ("MAPE_%", "razon_MAE_vs_pers"):
        print(f"\n== {col} GLOBAL =="); print(glob.pivot(index="modelo", columns="horizonte", values=col).round(3))
    print("\n== IC95 razon MAE vs persistencia, GLOBAL =="); print(glob.assign(ic=lambda x: x.IC95_inf.round(2).astype(str)+"-"+x.IC95_sup.round(2).astype(str)).pivot(index="modelo", columns="horizonte", values="ic"))
    print("\n== razon MAE por producto, h=4 =="); print(prod[prod.horizonte==4].pivot(index="modelo", columns="producto", values="razon_MAE_vs_pers").round(2))

if __name__ == "__main__":
    main()
