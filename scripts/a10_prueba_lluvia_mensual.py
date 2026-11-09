"""a10_prueba_lluvia_mensual.py — ¿Aporta algo la lluvia mensual al modelo?

Mismo protocolo de a6 (hiperparámetros fijos de afinacion_cv.csv, reentreno cada 8 semanas).
Variantes de variables:
  base          : variables actuales (incluye lluvia diaria semanal)
  sin_lluvia    : base SIN precipitacion_mm ni lluvia_max_mm (¿aporta algo la lluvia diaria?)
  +mensual_prev : base + lluvia del mes ANTERIOR (conocida en el origen; causal)
  +mensual_act  : base + lluvia del mes del origen (incluye días futuros del mes: cota optimista, NO causal)
Salida: reports/etapa3/prueba_lluvia_mensual.csv
Uso: python scripts/a10_prueba_lluvia_mensual.py --precios <xlsx> --lluvia-mensual <xlsx>
"""
import argparse, ast, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from prediccion_precios import config, pipeline_causal as pc  # noqa: E402
spec = importlib.util.spec_from_file_location("a6", RAIZ / "scripts" / "a6_afinacion.py")
a6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a6)
a4 = a6.a4
MESES = {"Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6, "Julio": 7,
         "Agosto": 8, "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12}
MODELOS = ("RL_nivel", "Ridge_var", "RF_var", "HGB_var")


def cargar_mensual(ruta):
    m = pd.read_excel(ruta)
    m = m[m["Fecha"].astype(str).str.contains(r"\d{4}$")].copy()
    m["Mes"] = m["Fecha"].str.split().str[0].map(MESES)
    m["Anio"] = m["Fecha"].str.split().str[1].astype(int)
    return m.set_index(["Anio", "Mes"])["Lluvia Promedio Mensual (mm)"].astype(float)


def params(h, modelo):
    if modelo == "RL_nivel": return {"_": 0}
    cv = pd.read_csv(RAIZ / "reports" / "etapa3" / "afinacion_cv.csv")
    s = cv[(cv.horizonte == h) & (cv.modelo == modelo)].sort_values("puntaje_cv").iloc[0]
    return ast.literal_eval(s["params"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precios", required=True); ap.add_argument("--lluvia-mensual", required=True)
    a = ap.parse_args()
    base = pd.read_csv(config.DATASET_MODELADO, parse_dates=[config.COL_FECHA])
    df = a4.features_origen(pc.con_externas(pc.semanal_causal(a.precios), base))
    mens = cargar_mensual(a.lluvia_mensual)
    f = df[config.COL_FECHA]
    df["lluvia_mes_act"] = [mens.get((y, m), np.nan) for y, m in zip(f.dt.year, f.dt.month)]
    prev = f - pd.DateOffset(months=1)
    df["lluvia_mes_prev"] = [mens.get((y, m), np.nan) for y, m in zip(prev.dt.year, prev.dt.month)]
    dums = pd.get_dummies(df["Grupo"], prefix="g", dtype=float)
    df = pd.concat([df, dums], axis=1)
    feats = a4.lista_features(df) + list(dums.columns)
    variantes = {
        "base": feats,
        "sin_lluvia": [c for c in feats if c not in ("precipitacion_mm", "lluvia_max_mm")],
        "+mensual_prev": feats + ["lluvia_mes_prev"],
        "+mensual_act": feats + ["lluvia_mes_act"],
    }
    filas = []
    for h in a4.HORIZ:
        d = a6.preparar(df, h, feats)
        test = d[d["ok"] & (d["fecha_obj"] >= a4.INI_TEST)]
        bloques = np.array_split(sorted(test["fecha_obj"].unique()), int(np.ceil(32 / a4.REFIT)))
        for var, fs in variantes.items():
            for mod in MODELOS:
                P = params(h, mod)
                for bl in bloques:
                    tr = d[d["ok"] & (d["fecha_obj"] < bl[0])]; te = test[test["fecha_obj"].isin(bl)]
                    if len(te) == 0: continue
                    pr = a6.ajustar_predecir(mod, P, tr, te, fs)
                    filas.append(pd.DataFrame({"horizonte": h, "variante": var, "modelo": mod, "producto": te["Grupo"].to_numpy(),
                                               "err": np.abs(te["y"].to_numpy() - pr),
                                               "err_pers": np.abs(te["y"].to_numpy() - te["p0"].to_numpy())}))
    det = pd.concat(filas, ignore_index=True)
    t = det.groupby(["horizonte", "variante", "modelo", "producto"]).agg(MAE=("err", "mean"), MAE_pers=("err_pers", "mean")).reset_index()
    t["razon"] = t["MAE"] / t["MAE_pers"]
    t.round(4).to_csv(RAIZ / "reports" / "etapa3" / "prueba_lluvia_mensual.csv", index=False)
    sin_arroz = t[t.producto != "ARROZ"].groupby(["horizonte", "variante", "modelo"]).razon.mean().reset_index()
    print("== Razón MAE vs persistencia (promedio de 4 productos, sin arroz) ==")
    for h in a4.HORIZ:
        print(f"\n h={h}"); print(sin_arroz[sin_arroz.horizonte == h].pivot(index="modelo", columns="variante", values="razon")[list(variantes)].round(3))
    g = sin_arroz.groupby(["variante", "modelo"]).razon.mean().unstack(0)[list(variantes)]
    print("\n== Promedio sobre los 4 horizontes =="); print(g.round(3))
    print("\nPromedio de todos los modelos:", g.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
