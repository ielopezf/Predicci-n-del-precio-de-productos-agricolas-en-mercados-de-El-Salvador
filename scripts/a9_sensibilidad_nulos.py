"""a9_sensibilidad_nulos.py — ¿Cambia la conclusión según cómo se traten nulos y outliers?

Variantes del tratamiento de la serie de precios (las columnas externas se toman del
dataset actual, sin cambios):
  V0  actual: recorte IQR con toda la muestra + interpolación lineal bidireccional
  V1  sin recorte, interpolación lineal bidireccional
  V2  sin recorte, causal: sin interpolar el objetivo, lags con ffill, filas sin objetivo fuera
  V3  recorte con límites calculados SOLO con entrenamiento, causal como V2
Todas se evalúan contra el precio semanal CRUDO (sin recorte ni interpolar), en las
mismas filas de prueba (las 157 del proyecto) donde existe dato real.
Modelos: persistencia y regresión lineal (StandardScaler + LinearRegression).
Uso: python scripts/a9_sensibilidad_nulos.py --precios precios_consolidados.xlsx
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from prediccion_precios import config, features as ft, auditoria_datos as au  # noqa: E402
from prediccion_precios import baselines_ingenuos as bl  # noqa: E402

N_TRAIN = 510


def semanal_crudo(raw):
    s = raw.dropna(subset=[config.COL_PRECIO])
    diario = s.groupby(["Grupo", config.COL_FECHA])[config.COL_PRECIO].mean()
    out = {}
    for g, x in diario.groupby(level=0):
        out[g] = x.droplevel(0).resample(config.FRECUENCIA).mean()
    return out


def serie_variante(v, w, fechas, lim_train):
    """w: semanal crudo (NaN donde no hay dato). Devuelve la serie usada por la variante."""
    w = w.reindex(fechas)
    if v == "V1":
        return w.interpolate(limit_direction="both")
    if v == "V2":
        return w
    if v == "V3":
        lo, hi = lim_train
        return w.clip(lo, hi)
    raise ValueError(v)


def construir(v, base, crudo, dd_raw, fecha_corte):
    partes = []
    for g, sub in base.groupby("Grupo"):
        sub = sub.sort_values(config.COL_FECHA)
        fechas = pd.DatetimeIndex(sub[config.COL_FECHA])
        if v == "V0":
            serie = pd.Series(sub[config.COL_PRECIO].to_numpy(), index=fechas)
        else:
            lim = None
            if v == "V3":
                d = dd_raw[(dd_raw["Grupo"] == g) & (dd_raw[config.COL_FECHA] < fecha_corte)]
                d = d[config.COL_PRECIO].dropna()
                q1, q3 = d.quantile([.25, .75]); i = q3 - q1
                lim = (q1 - 1.5 * i, q3 + 1.5 * i)
            serie = serie_variante(v, crudo[g], fechas, lim)
        t = sub.copy()
        if v in ("V2", "V3"):
            t[config.COL_PRECIO] = serie.to_numpy()
            ff = serie.ffill()          # causal: solo pasado
            t["_ff"] = ff.to_numpy()
            tmp = t.copy(); tmp[config.COL_PRECIO] = tmp["_ff"]
            tmp = ft.agregar_lags(tmp, lags=(1, 2, 4, 8))
            tmp = ft.agregar_medias_moviles(tmp, ventanas=(4, 8, 12))
            tmp[config.COL_PRECIO] = serie.to_numpy()
            t = tmp.drop(columns=["_ff"])
        else:
            t[config.COL_PRECIO] = serie.to_numpy()
            t = ft.agregar_lags(t, lags=(1, 2, 4, 8))
            t = ft.agregar_medias_moviles(t, ventanas=(4, 8, 12))
        partes.append(t)
    return pd.concat(partes, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precios", required=True)
    a = ap.parse_args()

    base = pd.read_csv(config.DATASET_MODELADO, parse_dates=[config.COL_FECHA])
    raw = au.cargar_precios_crudos(a.precios)
    crudo = semanal_crudo(raw)
    dd_raw = raw.copy()

    # Claves (idénticas al proyecto): ordenar por fecha, quitar NaN, últimos 20 % = prueba
    d = base.sort_values(config.COL_FECHA).reset_index(drop=True).dropna().reset_index(drop=True)
    n_test = int(np.ceil(len(d) * config.TEST_SIZE))
    claves = d[[config.COL_FECHA, "Grupo"]]
    train_k, test_k = claves.iloc[:N_TRAIN], claves.iloc[len(d) - n_test:]
    fecha_corte = train_k[config.COL_FECHA].max() + pd.Timedelta(days=1)
    print(f"train={len(train_k)} test={len(test_k)} corte={fecha_corte.date()}")

    # Verdad: crudo semanal en claves de prueba; subconjunto "limpio" = verdad y semana previa observadas
    verdad = test_k.copy()
    verdad["real"] = [crudo[g].get(f, np.nan) for f, g in zip(verdad[config.COL_FECHA], verdad["Grupo"])]
    verdad["prev"] = [crudo[g].get(f - pd.Timedelta(weeks=1), np.nan)
                      for f, g in zip(verdad[config.COL_FECHA], verdad["Grupo"])]
    verdad["limpia"] = verdad["real"].notna() & verdad["prev"].notna()
    print(f"prueba con dato real: {int(verdad['real'].notna().sum())}; limpia (real y previa): {int(verdad['limpia'].sum())}")

    filas = []
    for v in ("V0", "V1", "V2", "V3"):
        df = construir(v, base, crudo, dd_raw, fecha_corte)
        feats = [c for c in df.columns if c not in (config.COL_FECHA, config.COL_PRODUCTO, config.COL_PRECIO)]
        df = pd.get_dummies(df, columns=["Grupo"], prefix="grupo", dtype=float).merge(
            base[[config.COL_FECHA, "Grupo"]], on=config.COL_FECHA, how="left") if False else df
        X = pd.get_dummies(df[feats], columns=["Grupo"], prefix="grupo", dtype=float)
        X.index = pd.MultiIndex.from_frame(df[[config.COL_FECHA]].assign(Grupo=df["Grupo"]))
        y = pd.Series(df[config.COL_PRECIO].to_numpy(), index=X.index)
        ktr = pd.MultiIndex.from_frame(train_k)
        kte = pd.MultiIndex.from_frame(test_k)
        Xtr, ytr = X.loc[ktr], y.loc[ktr]
        ok = ytr.notna() & Xtr.notna().all(axis=1)
        Xtr, ytr = Xtr[ok], ytr[ok]
        mdl = make_pipeline(StandardScaler(), LinearRegression()).fit(Xtr, ytr)
        Xte = X.loc[kte].fillna(0)
        pred_lr = mdl.predict(Xte)
        lag1 = df.set_index(pd.MultiIndex.from_frame(df[[config.COL_FECHA]].assign(Grupo=df["Grupo"])))["precio_lag1"].loc[kte].to_numpy()
        for nombre, pred in (("persistencia", lag1), ("regresion_lineal", pred_lr)):
            t = verdad.copy(); t["pred"] = pred; t["variante"] = v; t["modelo"] = nombre
            filas.append(t)
    det = pd.concat(filas, ignore_index=True)

    def met(g):
        e = (g["real"] - g["pred"]).abs()
        return pd.Series({"n": len(g), "MAE": e.mean(), "MAPE_%": (e / g["real"]).mean() * 100})

    res = []
    for etiqueta, mask in (("real_observado", det["real"].notna() & det["pred"].notna()),
                           ("limpia", det["limpia"] & det["pred"].notna())):
        s = det[mask]
        g = s.groupby(["variante", "modelo"]).apply(met, include_groups=False).reset_index()
        g["subconjunto"] = etiqueta; g["producto"] = "GLOBAL"
        p = s.groupby(["variante", "modelo", "Grupo"]).apply(met, include_groups=False).reset_index()
        p = p.rename(columns={"Grupo": "producto"}); p["subconjunto"] = etiqueta
        res += [g, p]
    res = pd.concat(res, ignore_index=True)
    out = RAIZ / "reports" / "etapa3"
    res.to_csv(out / "sensibilidad_nulos.csv", index=False)
    det.to_csv(out / "sensibilidad_nulos_detalle.csv", index=False)
    for sub in ("real_observado", "limpia"):
        print(f"\n== MAPE % global, subconjunto {sub} ==")
        print(res[(res.subconjunto == sub) & (res.producto == "GLOBAL")]
              .pivot(index="variante", columns="modelo", values="MAPE_%").round(2))
        print(f"== MAPE % TOMATE, {sub} ==")
        print(res[(res.subconjunto == sub) & (res.producto == "TOMATE")]
              .pivot(index="variante", columns="modelo", values="MAPE_%").round(2))


if __name__ == "__main__":
    main()
