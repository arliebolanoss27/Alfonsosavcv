"""
Script para procesar y recortar las anotaciones de detección de ElectroCom-61.
Recorta cada bounding box con ~10% de padding, filtra recortes diminutos
y selecciona únicamente las clases mapeadas a nuestro proyecto ToolLighthouse.
"""

import sys
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from PIL import Image
import yaml
from tqdm import tqdm

import config

PADDING_RATIO = 0.10  # 10% de margen alrededor del objeto
MIN_BOX_SIZE = 20     # Descartar recortes con ancho o alto menor a 20 px

ELECTROCOM_ROOT = config.DATASETS_RAW_DIR / "electrocom61" / "ElectroCom61 A Multiclass Dataset for Detection of Electronic Components" / "ElectroCom-61_v2"
OUTPUT_DIR = config.BASE_DIR / "dataset_intermedio" / "electrocom61"

MAPEO_ELECTROCOM = {
    "Resistor": "resistencia",
    "Diode": "diodo",  # Se ignora Zener-Diode deliberadamente
    "LED-Light": "led",
    "Taper-Potentiometer": "potenciometro",  # Se ignora Trimmer-Potentiometer
    "Servo-Motor": "servo",
    "Breadboard": "protoboard",
    "Arduino-Nano": "arduino_nano",
    "Capacitor-10mf": "capacitor_electrolitico",
    "Capacitor-470mf": "capacitor_electrolitico",
}


def cargar_nombres_electrocom(yaml_path: Path) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return {}


def procesar_electrocom():
    yaml_path = ELECTROCOM_ROOT / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No se encontró data.yaml en {yaml_path}")

    id_to_name = cargar_nombres_electrocom(yaml_path)
    clases_utiles_ids = {
        cls_id: MAPEO_ELECTROCOM[orig_name]
        for cls_id, orig_name in id_to_name.items()
        if orig_name in MAPEO_ELECTROCOM
    }

    print(f"[ElectroCom-61] Clases objetivo identificadas: {len(clases_utiles_ids)}")
    for cid, cname in clases_utiles_ids.items():
        print(f"  - ID {cid}: {id_to_name[cid]} -> {cname}")

    particiones = ["train", "valid", "test"]
    conteo_recortes = {c: 0 for c in config.CLASES}

    for particion in particiones:
        img_dir = ELECTROCOM_ROOT / particion / "images"
        lbl_dir = ELECTROCOM_ROOT / particion / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        archivos_img = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        print(f"\nProcesando partición original '{particion}': {len(archivos_img)} imágenes...")

        for img_path in tqdm(archivos_img, desc=f"Recortando {particion}"):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size

                    with open(lbl_path, "r", encoding="utf-8") as lf:
                        lines = [line.strip() for line in lf if line.strip()]

                    for idx, line in enumerate(lines):
                        parts = line.split()
                        if len(parts) < 5:
                            continue

                        cls_id = int(parts[0])
                        if cls_id not in clases_utiles_ids:
                            continue

                        clase_destino = clases_utiles_ids[cls_id]

                        # Formato YOLO: x_center, y_center, width, height (normalizados)
                        xc, yc, w, h = map(float, parts[1:5])
                        box_w = w * img_w
                        box_h = h * img_h

                        if box_w < MIN_BOX_SIZE or box_h < MIN_BOX_SIZE:
                            continue

                        # Coordenadas absolutas
                        x1 = (xc - w / 2) * img_w
                        y1 = (yc - h / 2) * img_h
                        x2 = (xc + w / 2) * img_w
                        y2 = (yc + h / 2) * img_h

                        # Aplicar padding del 10%
                        pad_x = box_w * PADDING_RATIO
                        pad_y = box_h * PADDING_RATIO
                        crop_x1 = max(0, int(x1 - pad_x))
                        crop_y1 = max(0, int(y1 - pad_y))
                        crop_x2 = min(img_w, int(x2 + pad_x))
                        crop_y2 = min(img_h, int(y2 + pad_y))

                        if (crop_x2 - crop_x1) < MIN_BOX_SIZE or (crop_y2 - crop_y1) < MIN_BOX_SIZE:
                            continue

                        crop = img.crop((crop_x1, crop_y1, crop_x2, crop_y2)).convert("RGB")

                        # Carpeta de destino
                        dest_dir = OUTPUT_DIR / clase_destino
                        dest_dir.mkdir(parents=True, exist_ok=True)

                        crop_name = f"e61_{particion}_{img_path.stem}_box{idx}.jpg"
                        crop.save(dest_dir / crop_name, quality=95)
                        conteo_recortes[clase_destino] += 1

            except Exception as e:
                # Omitir imágenes corruptas
                continue

    print("\n--- Resumen de recortes extraídos de ElectroCom-61 ---")
    for cls_name in config.CLASES:
        cant = conteo_recortes.get(cls_name, 0)
        if cant > 0:
            print(f"  - {cls_name}: {cant} recortes")


if __name__ == "__main__":
    procesar_electrocom()
