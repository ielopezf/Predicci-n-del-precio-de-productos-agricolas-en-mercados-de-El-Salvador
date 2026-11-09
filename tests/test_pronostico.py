import sys, io
from pathlib import Path
import numpy as np, pandas as pd, pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prediccion_precios import pronostico as pr

def hist(n=80, g="TOMATE", base=10.0):
    f = pd.date_range("2022-01-02", periods=n, freq="7D")
    rng = np.random.default_rng(0)
    p = base * np.cumprod(1 + rng.normal(0, 0.05, n))
    return pd.DataFrame({"Fecha": f, "Grupo": g, "Producto": g + " X", "Precio": p})

def test_punto_es_persistencia_e_intervalo_ordenado():
    h = hist(); r = pr.pronosticar(h, "TOMATE", hoy=pd.Timestamp("2022-12-01")).tabla()
    assert (r["pronostico"] == r["precio_base"]).all()
    assert (r["lim_inf_90"] <= r["lim_inf_80"]).all() and (r["lim_sup_80"] <= r["lim_sup_90"]).all()
    assert (r["lim_inf_80"] <= r["pronostico"]).all() and (r["pronostico"] <= r["lim_sup_80"]).all()

def test_precio_manual_y_validacion():
    h = hist(); r = pr.pronosticar(h, "TOMATE", precio_base=20.0).tabla()
    assert (r["precio_base"] == 20.0).all()
    with pytest.raises(ValueError): pr.pronosticar(h, "TOMATE", precio_base=-1)
    with pytest.raises(ValueError): pr.pronosticar(h, "NOEXISTE")

def test_calibracion_no_mira_el_futuro():
    h = hist(120); p, f = h["Precio"], h["Fecha"]
    rel = pr.variaciones_relativas(p, 1, f, pr.FIN_CALIBRACION)
    assert len(rel) == int((f.iloc[1:] < pr.FIN_CALIBRACION).sum())

def test_semaforo():
    assert pr.semaforo(2) == "verde" and pr.semaforo(10) == "amarillo"
    assert pr.semaforo(30) == "rojo" and pr.semaforo(float("nan")) == "gris"

def test_avisos_tomate_cosecha_y_dato_viejo():
    a = pr.avisos("TOMATE", pd.Timestamp("2024-12-01"), pd.Timestamp("2024-06-30"), pd.Timestamp("2026-01-01"))
    assert len(a) == 3
    assert pr.avisos("ARROZ", pd.Timestamp("2024-07-10"), pd.Timestamp("2024-06-30"), pd.Timestamp("2024-07-05")) == []

def _csv(txt, nombre="r.csv"):
    f = io.StringIO(txt); f.name = nombre; return f

def test_leer_reporte_mag_valido():
    out = pr.leer_reporte_mag(_csv("Fecha,Producto,Precio\n2025-01-05,TOMATE DE PASTA GRANDE,12.5\n2025-01-05,MAIZ BLANCO,0.5\n"))
    assert out["TOMATE"][1] == 12.5 and out["MAÍZ"][1] == 0.5

@pytest.mark.parametrize("txt,nombre", [
    ("a,b\n1,2\n", "r.csv"),
    ("Fecha,Producto,Precio,Extra\n2025-01-05,TOMATE,1,x\n", "r.csv"),
    ("Fecha,Producto,Precio\nayer,TOMATE,1\n", "r.csv"),
    ("Fecha,Producto,Precio\n2025-01-05,TOMATE,-3\n", "r.csv"),
    ("Fecha,Producto,Precio\n2099-01-05,TOMATE,3\n", "r.csv"),
    ("Fecha,Producto,Precio\n2025-01-05,PAPAS,3\n", "r.csv"),
    ("Fecha,Producto,Precio\n2025-01-05,TOMATE,3\n", "r.txt"),
])
def test_leer_reporte_mag_rechaza(txt, nombre):
    with pytest.raises(ValueError):
        pr.leer_reporte_mag(_csv(txt, nombre))

def test_rango_plausible_y_plantilla():
    h = pr.cargar_historia()
    with pytest.raises(ValueError, match="plausible"):
        pr.leer_reporte_mag(_csv("Fecha,Producto,Precio\n2025-01-05,TOMATE,3000\n"), h)
    ok = pr.leer_reporte_mag(_csv(pr.plantilla_csv(h)), h)   # la plantilla pasa su propia validación
    assert set(ok) == set(pr.PRODUCTOS_ESPERADOS)

def test_explicacion_con_datos_reales():
    h = pr.cargar_historia()
    t = pr.pronosticar_todos(h, hoy=pd.Timestamp("2024-07-01"))
    assert len(t) == 5 * 4 and t["semaforo"].isin(["verde", "amarillo", "rojo"]).all()
    assert "semana" in pr.explicacion(t.iloc[0])