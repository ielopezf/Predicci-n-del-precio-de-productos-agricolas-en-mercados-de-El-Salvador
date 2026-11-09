"""
Extractor de Precios de Combustibles - SINAPP DGEHM El Salvador
URL: https://sinapp.dgehm.gob.sv/drhm/estadisticas.aspx?uid=3

ESTRATEGIA: Parsear el HTML del DOM directamente usando BeautifulSoup.
Los datos están en <div> con clases CSS únicas dentro del ReportViewer.
No hay iframes. Los datos están en el DOM principal.

Resultado: precios_combustibles_sv.xlsx con columnas:
  Año | Mes | Día | Diesel C | Diesel O | Diesel Or |
  Diesel Azufre C | Diesel Azufre O | Diesel Azufre Or |
  Gas Regular C | Gas Regular O | Gas Regular Or |
  Gas Especial C | Gas Especial O | Gas Especial Or

Requisitos:
    pip install selenium pandas openpyxl webdriver-manager beautifulsoup4
"""

import re, time
import pandas as pd
from datetime import date, timedelta
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
URL            = "https://sinapp.dgehm.gob.sv/drhm/estadisticas.aspx?uid=3"
ARCHIVO_SALIDA = "precios_combustibles_sv.xlsx"
HEADLESS       = True   # Cambiar a False para ver el navegador

MESES = {
    "ene":1,"feb":2,"mar":3,"abr":4,"may":5,"jun":6,
    "jul":7,"ago":8,"sep":9,"oct":10,"nov":11,"dic":12
}

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def to_float(txt):
    if not txt: return 0.0
    t = txt.strip().replace(",",".")
    try:    return float(t)
    except: return 0.0

def parse_fecha(txt, anio):
    """'28 dic' → date(anio, 12, 28)"""
    if not txt: return None
    p = txt.strip().lower().split()
    if len(p) != 2: return None
    try:
        d = int(p[0])
        m = MESES.get(p[1][:3])
        if not m: return None
        return date(anio, m, d)
    except: return None

def rango_dias(d_ini, d_fin):
    dias, d = [], d_ini
    while d <= d_fin:
        dias.append(d)
        d += timedelta(days=1)
    return dias

def iniciar_driver():
    opts = Options()
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1600,1000")
    svc = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=svc, options=opts)

def esperar_reporte(driver, timeout=20):
    """Espera a que aparezca el div con el contenido del reporte."""
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div[id*='VisibleReportContent']")
            )
        )
        time.sleep(2)
        return True
    except TimeoutException:
        # Fallback: esperar cualquier tabla con datos
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "td"))
            )
            time.sleep(2)
            return True
        except:
            return False

# ─────────────────────────────────────────
# PARSEO DEL HTML CON BEAUTIFULSOUP
# ─────────────────────────────────────────

def parsear_html(html, anio):
    """
    Parsea el HTML del reporte y extrae todas las filas de datos.
    Cada fila tiene 15 <td>:
      [0] vacío (separador visual)
      [1] fecha inicio  [2] fecha fin
      [3][4][5]   = Diesel        C / O / Or
      [6][7][8]   = Diesel Azufre C / O / Or
      [9][10][11] = Gas Regular   C / O / Or
      [12][13][14]= Gas Especial  C / O / Or
    """
    soup = BeautifulSoup(html, "html.parser")
    registros = []

    # Buscar todas las filas <tr> con 15 <td>
    for tr in soup.find_all("tr", valign="top"):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 13:
            continue

        # Extraer texto de cada celda (texto dentro de <div> o directo)
        def cell(i):
            if i >= len(tds): return ""
            div = tds[i].find("div")
            return (div.get_text(strip=True) if div else tds[i].get_text(strip=True))

        # La celda [1] y [2] deben ser fechas tipo "28 dic"
        txt_ini = cell(1)
        txt_fin = cell(2)

        if not re.match(r"^\d{1,2}\s+[a-záéíóúü]{3}$", txt_ini, re.IGNORECASE):
            continue

        # Determinar año de la fecha de inicio
        mes_ini = MESES.get(txt_ini.strip().lower().split()[-1][:3], 0)
        mes_fin_txt = txt_fin.strip().lower().split()[-1][:3] if txt_fin else ""
        mes_fin = MESES.get(mes_fin_txt, 0)

        # Manejar cruce de año: inicio dic → fin ene del siguiente año
        anio_ini = anio
        anio_fin = anio
        if mes_ini == 12 and mes_fin == 1:
            anio_ini = anio - 1  # dic pertenece al año anterior en el reporte del año siguiente
            # Pero para el año 2011 (primer año), dic 28 → ene 3 es dic 2010
            # La lógica: si el reporte es del año X, el primer período dic→ene
            # corresponde a dic(X-1) → ene(X)
            anio_fin = anio

        d_ini = parse_fecha(txt_ini, anio_ini)
        d_fin = parse_fecha(txt_fin, anio_fin)

        if not d_ini or not d_fin:
            continue

        # Si por algún motivo d_ini > d_fin (cruce no detectado), ajustar
        if d_ini > d_fin:
            d_fin = date(d_fin.year + 1, d_fin.month, d_fin.day)

        def g(i): return to_float(cell(i))

        for dia in rango_dias(d_ini, d_fin):
            registros.append({
                "Año":                      dia.year,
                "Mes":                      dia.month,
                "Día":                      dia.day,
                "Diesel Central":           g(3),
                "Diesel Occidental":        g(4),
                "Diesel Oriental":          g(5),
                "Diesel Azufre Central":    g(6),
                "Diesel Azufre Occidental": g(7),
                "Diesel Azufre Oriental":   g(8),
                "Gas Regular Central":      g(9),
                "Gas Regular Occidental":   g(10),
                "Gas Regular Oriental":     g(11),
                "Gas Especial Central":     g(12),
                "Gas Especial Occidental":  g(13),
                "Gas Especial Oriental":    g(14),
            })

    return registros

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Extractor Combustibles SINAPP-DGEHM — El Salvador")
    print("=" * 65)

    driver = iniciar_driver()
    todos  = []

    try:
        print(f"\n🌐 Cargando página...")
        driver.get(URL)
        esperar_reporte(driver)

        # Obtener años disponibles
        anios = []
        for sel in driver.find_elements(By.TAG_NAME, "select"):
            try:
                s = Select(sel)
                opts = [o.text.strip() for o in s.options if o.text.strip().isdigit()]
                if opts:
                    anios = opts
                    print(f"  📅 Años disponibles: {anios}")
                    break
            except: pass

        if not anios:
            print("❌ No se encontró selector de años.")
            return

        for anio_txt in anios:
            anio = int(anio_txt)
            print(f"\n📅 Procesando año {anio}...")

            # Seleccionar año
            for sel in driver.find_elements(By.TAG_NAME, "select"):
                try:
                    s = Select(sel)
                    if anio_txt in [o.text.strip() for o in s.options]:
                        s.select_by_visible_text(anio_txt)
                        break
                except: pass

            # Click en View Report
            try:
                btn = driver.find_element(By.XPATH,
                    "//*[contains(@value,'View Report') or contains(text(),'View Report')]")
                btn.click()
            except: pass

            esperar_reporte(driver)

            # Parsear HTML actual
            html = driver.page_source
            registros = parsear_html(html, anio)
            print(f"   → {len(registros)} días extraídos.")
            todos.extend(registros)

    except Exception as e:
        import traceback
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
    finally:
        driver.quit()

    if not todos:
        print("\n❌ No se obtuvieron datos.")
        return

    cols = [
        "Año","Mes","Día",
        "Diesel Central","Diesel Occidental","Diesel Oriental",
        "Diesel Azufre Central","Diesel Azufre Occidental","Diesel Azufre Oriental",
        "Gas Regular Central","Gas Regular Occidental","Gas Regular Oriental",
        "Gas Especial Central","Gas Especial Occidental","Gas Especial Oriental",
    ]
    df = pd.DataFrame(todos, columns=cols)
    df.sort_values(["Año","Mes","Día"], inplace=True, ignore_index=True)
    df.to_excel(ARCHIVO_SALIDA, index=False)

    print("\n" + "=" * 65)
    print(f"  ✅ Registros totales  : {len(df)}")
    print(f"  📅 Rango              : {df['Año'].min()} – {df['Año'].max()}")
    print(f"  📄 Guardado en        : {ARCHIVO_SALIDA}")
    print("=" * 65)

if __name__ == "__main__":
    main()