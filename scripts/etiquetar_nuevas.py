"""
Etiqueta las 113 fotos de "Nuevas fotos de los obejetos" (cajas en formato YOLO) con FastSAM.

La clase de cada foto se decidió mirándolas una por una (rangos por orden de hora/número):
    0-15 jumper | 16-24 diodo | 25-32 capacitor | 33-36, 56, 60-85 resistencia
    37-55, 57-59 led | 86-94 arduino_nano | 95-108 potenciometro | 109-112 protoboard

Cada objeto separado de la foto recibe su propia caja. Los jumpers (aros, madejas) llevan UNA caja
alrededor de todo el grupo. Salida: dataset_deteccion/nuevas/{images,labels} + hoja de revisión.

Uso:  python scripts/etiquetar_nuevas.py
"""

import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageOps

import config

CARPETA = config.BASE_DIR.parent / "Nuevas fotos de los obejetos"
SALIDA = config.BASE_DIR / "dataset_deteccion" / "nuevas"
CLASES_DET = [c for c in config.CLASES if c != "nada"]

RANGOS = (
    [(i, "jumper") for i in range(0, 16)] + [(i, "diodo") for i in range(16, 25)]
    + [(i, "capacitor_electrolitico") for i in range(25, 33)]
    + [(i, "resistencia") for i in [*range(33, 37), 56, *range(60, 86)]]
    + [(i, "led") for i in [*range(37, 56), *range(57, 60)]]
    + [(i, "arduino_nano") for i in range(86, 95)] + [(i, "potenciometro") for i in range(95, 109)]
    + [(i, "protoboard") for i in range(109, 113)]
)
CLASE_DE = dict(RANGOS)


def orden():
    def clave(f):
        n = f.name
        h = re.search(r"at (\d+)\.(\d+)\.(\d+)", n)
        k = re.search(r"\((\d+)\)", n)
        return (int(h[1]), int(h[2]), int(h[3]), int(k[1]) if k else 0)
    return sorted(CARPETA.glob("*.jpeg"), key=clave)


def cargar(ruta):
    return cv2.cvtColor(np.array(ImageOps.exif_transpose(Image.open(ruta).convert("RGB"))), cv2.COLOR_RGB2BGR)


UNICOS = {"arduino_nano", "potenciometro", "protoboard"}  # un solo objeto por foto: una caja, la del más grande


def segmentar(modelo, img, area_max=0.10):
    """Máscara de todo lo que contrasta con la madera, según FastSAM."""
    H, W = img.shape[:2]
    r = modelo(img, device=0, retina_masks=True, imgsz=1280, conf=0.2, iou=0.7, verbose=False)[0]
    union = np.zeros((H, W), np.uint8)
    if r.masks is None:
        return union
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    dist = np.linalg.norm(lab - np.median(lab.reshape(-1, 3), axis=0), axis=2)
    for mk in r.masks.data.cpu().numpy():
        mk = cv2.resize(mk.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        a = mk.mean()
        if a < 0.00003 or a > area_max or dist[mk].mean() < 22:
            continue
        ys, xs = np.where(mk)
        if (xs.min() <= 2 or ys.min() <= 2 or xs.max() >= W - 3 or ys.max() >= H - 3) and a > 0.004:
            continue  # sombra o borde de la mesa
        union |= mk.astype(np.uint8)
    return union


def cajas(clase, union):
    H, W = union.shape
    if union.sum() == 0:
        return []
    radio = max(7, int((0.022 if clase in UNICOS or clase == 'jumper' else 0.011) * W))  # une cuerpo y patas; separa objetos cercanos
    juntos = cv2.dilate(union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radio + 1, 2 * radio + 1)))
    if clase == "jumper":  # un solo grupo
        ys, xs = np.where(juntos > 0)
        return [[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]]
    n, _, st, _ = cv2.connectedComponentsWithStats(juntos)
    if clase in UNICOS:  # solo el componente más grande
        i = 1 + int(np.argmax(st[1:, 4]))
        x, y, w, h = st[i, :4]
        ys, xs = np.where(union[y:y + h, x:x + w] > 0)
        return [[int(x + xs.min()), int(y + ys.min()), int(x + xs.max()), int(y + ys.max())]]
    salida = []
    for i in range(1, n):
        x, y, w, h = st[i, :4]
        sub = union[y:y + h, x:x + w]
        if sub.sum() < 40:
            continue
        ys, xs = np.where(sub > 0)
        x0, y0, x1, y1 = int(x + xs.min()), int(y + ys.min()), int(x + xs.max()), int(y + ys.max())
        if max(x1 - x0, y1 - y0) < 0.04 * W:  # cajita suelta: ruido, no un componente
            continue
        if (x0 <= 3 or y0 <= 3 or x1 >= W - 4 or y1 >= H - 4) and max(x1 - x0, y1 - y0) < 0.12 * W:
            continue  # trozo pegado al borde
        salida.append([x0, y0, x1, y1])
    if len(salida) > 1:  # varias cajas: las pegadas al borde suelen ser el borde de la mesa, no un componente
        sin_borde = [c for c in salida if not (c[0] <= 3 or c[1] <= 3 or c[2] >= W - 4 or c[3] >= H - 4)]
        salida = sin_borde or salida
    return salida


def main():
    from ultralytics import FastSAM
    pesos = config.BASE_DIR / "FastSAM-s.pt"
    modelo = FastSAM(str(pesos) if pesos.exists() else "FastSAM-s.pt")
    (SALIDA / "images").mkdir(parents=True, exist_ok=True)
    (SALIDA / "labels").mkdir(parents=True, exist_ok=True)
    fotos = orden()
    print(len(fotos), "fotos")
    hojas = []
    for i, f in enumerate(fotos):
        clase = CLASE_DE[i]
        img = cargar(f)
        H, W = img.shape[:2]
        cs = cajas(clase, segmentar(modelo, img, 0.6 if clase == "protoboard" else 0.10))
        if i == 112:  # la protoboard queda cortada por el borde derecho de la foto: la caja debe llegar hasta ahí
            cs = [[cs[0][0], cs[0][1], W - 1, cs[0][3]]]
        nombre = f"{clase}_nueva{i:03d}"
        cv2.imwrite(str(SALIDA / "images" / f"{nombre}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cid = CLASES_DET.index(clase)
        lineas = []
        for x0, y0, x1, y1 in cs:
            p = 6  # un poco de margen
            x0, y0, x1, y1 = max(0, x0 - p), max(0, y0 - p), min(W - 1, x1 + p), min(H - 1, y1 + p)
            lineas.append(f"{cid} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
        (SALIDA / "labels" / f"{nombre}.txt").write_text("\n".join(lineas) + ("\n" if lineas else ""))
        vis = img.copy()
        for l in lineas:
            _, cx, cy, w, h = (float(v) for v in l.split())
            cv2.rectangle(vis, (int((cx - w / 2) * W), int((cy - h / 2) * H)), (int((cx + w / 2) * W), int((cy + h / 2) * H)), (0, 0, 255), 6)
        t = cv2.resize(vis, (300, 300 * H // W))
        t = cv2.copyMakeBorder(t, 0, max(0, 400 - t.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:400]
        cv2.putText(t, f"{i} {len(cs)}", (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        hojas.append(t)
        print(f"{i:3d} {clase:24s} {len(cs)} caja(s)")
    carpeta = config.BASE_DIR / "scratch"
    for k in range(0, len(hojas), 15):
        grupo = hojas[k:k + 15]
        while len(grupo) % 5:
            grupo.append(np.zeros_like(hojas[0]))
        cv2.imwrite(str(carpeta / f"etiq_{k // 15}.jpg"), np.vstack([np.hstack(grupo[j:j + 5]) for j in range(0, len(grupo), 5)]))


if __name__ == "__main__":
    main()
