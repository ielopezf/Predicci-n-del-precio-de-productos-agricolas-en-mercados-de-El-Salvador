from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import requests
import os
import time

# CONFIGURAR CHROME
options = webdriver.ChromeOptions()

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

# ABRIR PAGINA
url = "https://www.mag.gob.sv/informe-diario-de-precios-de-productos-agropecuarios/"
driver.get(url)

# ESPERAR CARGA
time.sleep(5)

# HACER SCROLL HASTA ABAJO
last_height = driver.execute_script("return document.body.scrollHeight")

while True:

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    time.sleep(3)

    new_height = driver.execute_script("return document.body.scrollHeight")

    if new_height == last_height:
        break

    last_height = new_height

print("Scroll completado")

# EXTRAER LINKS
links = []

elementos = driver.find_elements(By.TAG_NAME, "a")

for e in elementos:

    href = e.get_attribute("href")

    if href:

        # BUSCAR ARCHIVOS EXCEL
        if ".xlsx" in href.lower() or ".xls" in href.lower():

            links.append(href)

# ELIMINAR DUPLICADOS
links = list(set(links))

print(f"\nExcel encontrados: {len(links)}")

# CREAR CARPETA
os.makedirs("excels_mag", exist_ok=True)

# DESCARGAR
headers = {
    "User-Agent": "Mozilla/5.0"
}

for i, link in enumerate(links):

    try:

        print(f"Descargando {i+1}/{len(links)}")

        archivo = requests.get(link, headers=headers)

        nombre = link.split("/")[-1]

        ruta = os.path.join("excels_mag", nombre)

        with open(ruta, "wb") as f:
            f.write(archivo.content)

    except Exception as e:
        print("Error:", e)

driver.quit()

print("Proceso terminado")
