"""
Anota cajas (bounding boxes) en las fotos propias de dataset_intermedio/propios/.

FastSAM propone una caja por foto; tú la revisas:
    Enter / espacio : aceptar la caja propuesta
    r               : redibujar con el mouse: arrastra la caja y pulsa Enter. Luego Enter = listo,
                      a = agregar otra caja (varios objetos; todas llevan la clase de la carpeta).
                      Si te equivocas al arrastrar, pulsa c para cancelar y vuelve a empezar.
    s               : saltar la foto (no se usa para entrenar)
    b               : volver a la foto anterior
    q               : guardar y salir (al volver a correr, continúa donde te quedaste)

Salida (formato YOLO de detección): dataset_deteccion/propios/{images,labels}/
La clase la da la carpeta de la foto. Si una foto tiene objetos de otra clase (p. ej. un LED
en una foto de diodo), usa 's' para saltarla o dibuja solo la caja del objeto correcto.

Uso:  python scripts/anotar_cajas.py
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageOps

import config

ENTRADA = config.BASE_DIR / "dataset_intermedio" / "propios"
SALIDA = config.BASE_DIR / "dataset_deteccion" / "propios"
ESTADO = SALIDA / "estado.json"
CLASES_DET = [c for c in config.CLASES if c != "nada"]  # en detección, "nada" = sin cajas
ANCHO_VENTANA = 1100


def cargar(ruta: Path) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(ruta).convert("RGB"))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def proponer_caja(modelo, img):
    """Caja de la unión de los segmentos de FastSAM que contrastan con la madera. None si no hay."""
    H, W = img.shape[:2]
    r = modelo(img, device="cpu", retina_masks=True, imgsz=1024, conf=0.25, iou=0.7, verbose=False)[0]
    if r.masks is None:
        return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    dist = np.linalg.norm(lab - np.median(lab.reshape(-1, 3), axis=0), axis=2)
    union = np.zeros((H, W), np.uint8)
    for mk in r.masks.data.cpu().numpy():
        mk = cv2.resize(mk.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        a = mk.mean()
        if a < 0.0002 or a > 0.25 or dist[mk].mean() < 28:
            continue
        union |= mk.astype(np.uint8)
    if union.sum() == 0:
        return None
    union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    ys, xs = np.where(union > 0)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def guardar(ruta: Path, clase: str, cajas, img_shape):
    H, W = img_shape[:2]
    (SALIDA / "images").mkdir(parents=True, exist_ok=True)
    (SALIDA / "labels").mkdir(parents=True, exist_ok=True)
    destino = SALIDA / "images" / f"{clase}_{ruta.name}"
    cv2.imwrite(str(destino), cargar(ruta), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cid = CLASES_DET.index(clase)
    lineas = []
    for x0, y0, x1, y1 in cajas:
        cx, cy, w, h = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H
        lineas.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    (SALIDA / "labels" / f"{clase}_{ruta.stem}.txt").write_text("\n".join(lineas) + "\n")


def quitar(ruta: Path, clase: str):
    for p in ((SALIDA / "images" / f"{clase}_{ruta.name}"), (SALIDA / "labels" / f"{clase}_{ruta.stem}.txt")):
        p.unlink(missing_ok=True)


def main():
    from ultralytics import FastSAM

    fotos = sorted(p for p in ENTRADA.glob("*/*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    SALIDA.mkdir(parents=True, exist_ok=True)
    estado = json.loads(ESTADO.read_text()) if ESTADO.exists() else {}
    modelo = FastSAM("FastSAM-s.pt")
    i = 0
    while i < len(fotos):
        ruta = fotos[i]
        clave = f"{ruta.parent.name}/{ruta.name}"
        clase = ruta.parent.name
        if clase not in CLASES_DET:
            i += 1
            continue
        if clave in estado:
            i += 1
            continue
        img = cargar(ruta)
        H, W = img.shape[:2]
        esc = ANCHO_VENTANA / W
        caja = proponer_caja(modelo, img)
        vis = cv2.resize(img, None, fx=esc, fy=esc)
        if caja:
            cv2.rectangle(vis, (int(caja[0] * esc), int(caja[1] * esc)), (int(caja[2] * esc), int(caja[3] * esc)), (0, 0, 255), 3)
        cv2.putText(vis, f"[{i + 1}/{len(fotos)}] {clase}   Enter=ok  r=redibujar  s=saltar  b=atras  q=salir",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.imshow("anotar", vis)
        k = cv2.waitKey(0) & 0xFF
        if k in (13, 32) and caja:
            guardar(ruta, clase, [caja], img.shape)
            estado[clave] = "ok"
            i += 1
        elif k == ord("r"):
            cajas = []
            base = vis.copy()
            while True:
                x, y, w, h = cv2.selectROI("anotar", base, showCrosshair=False)  # arrastra y pulsa Enter
                if w == 0 or h == 0:  # cancelado con c
                    break
                cajas.append([int(x / esc), int(y / esc), int((x + w) / esc), int((y + h) / esc)])
                cv2.rectangle(base, (x, y), (x + w, y + h), (0, 255, 0), 3)
                msg = base.copy()
                cv2.putText(msg, "Enter=listo   a=agregar otra caja", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                cv2.imshow("anotar", msg)
                if cv2.waitKey(0) & 0xFF != ord("a"):
                    break
            if cajas:
                guardar(ruta, clase, cajas, img.shape)
                estado[clave] = "ok"
                i += 1
        elif k == ord("s"):
            quitar(ruta, clase)
            estado[clave] = "saltada"
            i += 1
        elif k == ord("b") and i > 0:
            i -= 1
            estado.pop(f"{fotos[i].parent.name}/{fotos[i].name}", None)
        elif k == ord("q"):
            break
        ESTADO.write_text(json.dumps(estado, indent=1))
    ESTADO.write_text(json.dumps(estado, indent=1))
    cv2.destroyAllWindows()
    print(f"Anotadas: {sum(v == 'ok' for v in estado.values())}, saltadas: {sum(v == 'saltada' for v in estado.values())}")


if __name__ == "__main__":
    main()
