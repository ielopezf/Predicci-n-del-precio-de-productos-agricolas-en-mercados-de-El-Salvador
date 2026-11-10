import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prediccion_precios import config, dataset_causal as dc, pipeline_causal as pc


def _seudo_causal(base):
    c = base[[config.COL_FECHA, "Grupo"] + pc.EXTERNAS].copy()
    c["precio_vigente"] = base[config.COL_PRECIO].to_numpy()
    c["precio_imputado"], c["n_dias"], c["semanas_sin_cambio"] = 0, 1, 0
    return c


def test_receta_reproduce_dataset_etapa2():
    base = pd.read_csv(config.DATASET_MODELADO, parse_dates=[config.COL_FECHA])
    prod = base.drop_duplicates("Grupo").set_index("Grupo")[config.COL_PRODUCTO].to_dict()
    d, _ = dc.armar_dataset_causal(_seudo_causal(base), prod, list(base.columns))
    a = base.sort_values(["Grupo", config.COL_FECHA]).reset_index(drop=True)
    assert list(d.columns) == list(base.columns)
    for col in base.columns:
        if col == config.COL_PRODUCTO:
            continue
        x, y = a[col].to_numpy(), d[col].to_numpy()
        if np.issubdtype(x.dtype, np.number):
            assert np.allclose(x, y, rtol=1e-4, atol=1e-5, equal_nan=True), col
        else:
            assert (x == y).all(), col


def test_ffill_sin_futuro_ni_relleno_hacia_atras():
    base = pd.read_csv(config.DATASET_MODELADO, parse_dates=[config.COL_FECHA])
    prod = base.drop_duplicates("Grupo").set_index("Grupo")[config.COL_PRODUCTO].to_dict()
    c = _seudo_causal(base).sort_values(["Grupo", config.COL_FECHA]).reset_index(drop=True)
    g = c["Grupo"] == "MAÍZ"
    idx = c.index[g]
    c.loc[idx[:2], "precio_vigente"] = np.nan          # arranque sin precio: se descarta
    c.loc[idx[10], "precio_vigente"] = c.loc[idx[9], "precio_vigente"]   # semana arrastrada
    d, flags = dc.armar_dataset_causal(c, prod, list(base.columns))
    m = d[d["Grupo"] == "MAÍZ"]
    assert len(m) == g.sum() - 2
    assert m[config.COL_PRECIO].notna().all()
