"""
Convierte los datasets extra de Kaggle (datasets/extra/) al formato de detección YOLO.

  - IoTKITs (Mrpaul0007/iotkits): cajas del Arduino-Nano (COCO -> YOLO). Las fotos de otras placas
    (Uno, Mega, Raspberry Pi...) se usan como NEGATIVOS (sin cajas) para que no se confundan con un Nano.
  - pictures-of-simple-electronic-components: carpeta Capacitors (clasificación). La caja se calcula
    con FastSAM (el objeto contrasta con el fondo liso).

Salida: dataset_deteccion/extra/{images,labels} y una hoja de revisión en scratch/extra_capacitores.jpg
Uso:    python scripts/preparar_extra.py
"""

import json
import random
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageOps

import config

EXTRA = config.DATASETS_RAW_DIR / "extra"
SALIDA = config.BASE_DIR / "dataset_deteccion" / "extra"
CLASES_DET = [c for c in config.CLASES if c != "nada"]
NEGATIVOS = 150


def iotkits():
    raiz = EXTRA / "iotkits"
    nano_id = CLASES_DET.index("arduino_nano")
    con, sin = [], []
    for split in ("train", "valid", "test"):
        d = json.loads((raiz / split / "_annotations.coco.json").read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        por_img = {}
        for a in d["annotations"]:
            por_img.setdefault(a["image_id"], []).append(a)
        for im in d["images"]:
            anns = por_img.get(im["id"], [])
            nanos = [a for a in anns if cats[a["category_id"]] == "Arduino-Nano"]
            ruta = raiz / split / im["file_name"]
            if nanos:
                lineas = []
                for a in nanos:
                    x, y, w, h = a["bbox"]
                    lineas.append(f"{nano_id} {(x + w / 2) / im['width']:.6f} {(y + h / 2) / im['height']:.6f} "
                                  f"{w / im['width']:.6f} {h / im['height']:.6f}")
                con.append((ruta, lineas))
            elif anns:  # otra placa: negativo
                sin.append((ruta, []))
    random.Random(0).shuffle(sin)
    return con, sin[:NEGATIVOS]


def capacitores():
    from ultralytics import FastSAM
    pesos = config.BASE_DIR / "FastSAM-s.pt"
    modelo = FastSAM(str(pesos) if pesos.exists() else "FastSAM-s.pt")
    cid = CLASES_DET.index("capacitor_electrolitico")
    salida, vistas = [], []
    for k, f in enumerate(sorted((EXTRA / "pictures-of-simple-electronic-components" / "dataset" / "Capacitors").glob("*"))):
        try:
            img = cv2.cvtColor(np.array(ImageOps.exif_transpose(Image.open(f).convert("RGB"))), cv2.COLOR_RGB2BGR)
        except Exception:
            continue
        H, W = img.shape[:2]
        r = modelo(img, device=0, retina_masks=True, imgsz=640, conf=0.3, iou=0.7, verbose=False)[0]
        if r.masks is None:
            continue
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        borde = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        dist = np.linalg.norm(lab - np.median(borde, axis=0), axis=2)  # color de fondo = el de los bordes
        union = np.zeros((H, W), np.uint8)
        for mk in r.masks.data.cpu().numpy():
            mk = cv2.resize(mk.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            if mk.mean() < 0.01 or mk.mean() > 0.85 or dist[mk].mean() < 25:
                continue
            union |= mk.astype(np.uint8)
        if union.sum() == 0:
            continue
        ys, xs = np.where(union > 0)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        if (x1 - x0) * (y1 - y0) < 0.04 * W * H:
            continue
        linea = f"{cid} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}"
        salida.append((f, [linea]))
        if len(vistas) < 24:
            v = img.copy()
            cv2.rectangle(v, (x0, y0), (x1, y1), (0, 0, 255), 4)
            vistas.append(cv2.resize(v, (260, 220)))
    if vistas:
        while len(vistas) % 6:
            vistas.append(np.zeros_like(vistas[0]))
        cv2.imwrite(str(config.BASE_DIR / "scratch" / "extra_capacitores.jpg"),
                    np.vstack([np.hstack(vistas[i:i + 6]) for i in range(0, len(vistas), 6)]))
    return salida


def guardar(items, prefijo):
    (SALIDA / "images").mkdir(parents=True, exist_ok=True)
    (SALIDA / "labels").mkdir(parents=True, exist_ok=True)
    for i, (ruta, lineas) in enumerate(items):
        nombre = f"{prefijo}_{i:04d}"
        im = Image.open(ruta).convert("RGB")
        im.save(SALIDA / "images" / f"{nombre}.jpg", quality=95)
        (SALIDA / "labels" / f"{nombre}.txt").write_text("\n".join(lineas) + ("\n" if lineas else ""))


def main():
    shutil.rmtree(SALIDA, ignore_errors=True)
    con, sin = iotkits()
    guardar(con, "nano_iot")
    guardar(sin, "neg_iot")
    print(f"IoTKITs: {len(con)} imágenes con Nano, {len(sin)} negativos")
    caps = capacitores()
    guardar(caps, "cap_kaggle")
    print(f"Capacitores de Kaggle con caja: {len(caps)} de 100")


if __name__ == "__main__":
    main()
