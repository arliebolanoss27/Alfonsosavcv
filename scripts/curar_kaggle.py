"""
Script para filtrar, limpiar y organizar las imágenes útiles del dataset de Kaggle.
Detecta y elimina duplicados exactos mediante hash MD5, descarta archivos no válidos
o corruptos, y selecciona únicamente las 6 clases permitidas por la documentación.
"""

import sys
import hashlib
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from PIL import Image
from tqdm import tqdm

import config

KAGGLE_RAW_DIR = config.DATASETS_RAW_DIR / "kaggle_recognizer"
OUTPUT_DIR = config.BASE_DIR / "dataset_intermedio" / "kaggle"

# Mapeo oficial documentado. NO se incluye capacitor ni otras clases no permitidas.
MAPEO_KAGGLE = {
    "resistor": "resistencia",
    "diode": "diodo",
    "led": "led",
    "potentiometer": "potenciometro",
    "jumper wire": "jumper",
    "breadboard": "protoboard",
}

# Dimensiones mínimas para descartar íconos o miniaturas corruptas
MIN_WIDTH = 50
MIN_HEIGHT = 50


def calcular_hash_archivo(filepath: Path) -> str:
    """Calcula el hash MD5 del contenido de la imagen."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def curar_kaggle():
    if not KAGGLE_RAW_DIR.exists():
        raise FileNotFoundError(f"No se encontró el directorio de Kaggle en {KAGGLE_RAW_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hashes_vistos = set()

    conteo_utiles = {c: 0 for c in config.CLASES}
    conteo_duplicados = 0
    conteo_invalidos = 0

    print("[Kaggle Recognizer] Iniciando curación de clases permitidas...")

    for carpeta_origen, clase_destino in MAPEO_KAGGLE.items():
        dir_origen = KAGGLE_RAW_DIR / carpeta_origen
        if not dir_origen.exists():
            print(f"  [Aviso] Carpeta '{carpeta_origen}' no encontrada. Omitiendo.")
            continue

        dir_destino = OUTPUT_DIR / clase_destino
        dir_destino.mkdir(parents=True, exist_ok=True)

        # Buscar imágenes tanto directamente como en subcarpetas anidadas
        archivos = [
            f for f in dir_origen.rglob("*")
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        ]

        print(f"\nProcesando '{carpeta_origen}' -> '{clase_destino}' ({len(archivos)} imágenes)...")

        for img_path in tqdm(archivos, desc=f"Curando {clase_destino}"):
            # 1. Comprobar duplicado exacto por hash
            file_hash = calcular_hash_archivo(img_path)
            if file_hash in hashes_vistos:
                conteo_duplicados += 1
                continue

            # 2. Validar que la imagen se pueda abrir y no sea miniatura corrupta
            try:
                with Image.open(img_path) as img:
                    img_rgb = img.convert("RGB")
                    w, h = img_rgb.size
                    if w < MIN_WIDTH or h < MIN_HEIGHT:
                        conteo_invalidos += 1
                        continue

                    hashes_vistos.add(file_hash)
                    dest_file = dir_destino / f"kg_{clase_destino}_{file_hash[:10]}.jpg"
                    img_rgb.save(dest_file, "JPEG", quality=95)
                    conteo_utiles[clase_destino] += 1

            except Exception:
                conteo_invalidos += 1
                continue

    print("\n--- Resumen de curación de Kaggle ---")
    print(f"Total imágenes duplicadas descartadas: {conteo_duplicados}")
    print(f"Total imágenes corruptas / muy pequeñas descartadas: {conteo_invalidos}")
    print("Imágenes útiles guardadas por clase:")
    for cls_name in config.CLASES:
        cant = conteo_utiles.get(cls_name, 0)
        if cant > 0:
            print(f"  - {cls_name}: {cant}")


if __name__ == "__main__":
    curar_kaggle()
