"""
Prepara el dataset de DETECCIÓN (cajas) para YOLO26 a partir de:
  - ElectroCom-61 (datasets/electrocom61): se queda solo con las cajas de nuestras clases.
  - Fotos propias con cajas (dataset_deteccion/propios, hecho con scripts/anotar_cajas.py).

Salida (dataset_deteccion/, no se sube a git salvo propios/):
  electrocom/{train,valid,test}/{images,labels}   copias filtradas a nuestras clases
  listas/train.txt, valid.txt, valid_propios.txt  listas de imágenes (las propias se repiten en train)
  data.yaml, data_propios.yaml                    configuración para entrenar y para medir solo con propias

Clase "nada": en detección no existe como clase; es una imagen sin cajas. Se incluye una fracción de
imágenes de ElectroCom-61 sin ninguna de nuestras clases como ejemplos de fondo.

Uso:  python scripts/preparar_deteccion.py
"""

import random
import re
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config

DET = config.BASE_DIR / "dataset_deteccion"
PROPIOS = DET / "propios"
CLASES_DET = [c for c in config.CLASES if c != "nada"]

# clase de ElectroCom-61 -> nuestra clase
EQUIVALENCIA = {
    "Resistor": "resistencia",
    "Diode": "diodo",
    "LED-Light": "led",
    "Taper-Potentiometer": "potenciometro",
    "Servo-Motor": "servo",
    "Breadboard": "protoboard",
    "Arduino-Nano": "arduino_nano",
    "Capacitor-10mf": "capacitor_electrolitico",
    "Capacitor-470mf": "capacitor_electrolitico",
}
FRACCION_FONDO = 0.10   # imágenes sin nuestras clases, respecto a las que sí las tienen
REPETIR_PROPIAS = 3     # veces que se repiten las 38 fotos propias antiguas en train
REPETIR_NUEVAS = 5      # veces que se repiten las 113 fotos nuevas (misma cámara y mesa que la demo)
# Fotos nuevas que NO se usan para entrenar, para medir (índice de la foto por orden de hora)
VAL_NUEVAS = {23, 24, 31, 32, 14, 15, 83, 84, 85, 57, 58, 59, 93, 94, 107, 108, 112}
VAL_PROPIAS_POR_CLASE = {"diodo": 2, "jumper": 2, "potenciometro": 2, "servo": 2, "led": 1, "resistencia": 1}
SEMILLA = 0


def leer_nombres(yaml_path: Path) -> list[str]:
    txt = yaml_path.read_text(encoding="utf-8")
    return eval(re.search(r"names:\s*(\[.*?\])", txt, re.S).group(1))


def convertir_electrocom():
    yaml_path = next((config.DATASETS_RAW_DIR / "electrocom61").rglob("data.yaml"))
    raiz = yaml_path.parent
    nombres = leer_nombres(yaml_path)
    mapa = {i: CLASES_DET.index(EQUIVALENCIA[n]) for i, n in enumerate(nombres) if n in EQUIVALENCIA}
    rnd = random.Random(SEMILLA)
    listas = {}
    for split in ("train", "valid", "test"):
        con, sin = [], []
        for lab in sorted((raiz / split / "labels").glob("*.txt")):
            lineas = []
            for l in lab.read_text().splitlines():
                p = l.split()
                if p and int(p[0]) in mapa:
                    lineas.append(" ".join([str(mapa[int(p[0])])] + p[1:]))
            img = next((raiz / split / "images").glob(lab.stem + ".*"), None)
            if img is None:
                continue
            (con if lineas else sin).append((img, lineas))
        rnd.shuffle(sin)
        elegidas = con + sin[: int(len(con) * FRACCION_FONDO)]
        out_i, out_l = DET / "electrocom" / split / "images", DET / "electrocom" / split / "labels"
        for d in (out_i, out_l):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
        for img, lineas in elegidas:
            shutil.copy(img, out_i / img.name)
            (out_l / (img.stem + ".txt")).write_text("\n".join(lineas) + ("\n" if lineas else ""))
        listas[split] = sorted(out_i.glob("*.jpg"))
        print(f"electrocom/{split}: {len(con)} con cajas nuestras + {len(elegidas) - len(con)} de fondo")
    return listas


def separar_propias():
    train, val = [], []
    por_clase = {}
    for img in sorted((PROPIOS / "images").glob("*.jpg")):
        por_clase.setdefault(img.name.split("_")[0], []).append(img)
    for clase, imgs in por_clase.items():
        k = VAL_PROPIAS_POR_CLASE.get(clase, 1)
        val += imgs[-k:]
        train += imgs[:-k]
    print(f"propias: {len(train)} train, {len(val)} val")
    return train, val


def separar_nuevas():
    train, val = [], []
    for img in sorted((DET / "nuevas" / "images").glob("*.jpg")):
        idx = int(img.stem.split("nueva")[-1])
        (val if idx in VAL_NUEVAS else train).append(img)
    print(f"nuevas: {len(train)} train, {len(val)} val")
    return train, val


def escribir_lista(ruta: Path, imgs):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(str(p.resolve()) for p in imgs) + "\n")


def escribir_yaml(ruta: Path, train: str, val: str):
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASES_DET))
    ruta.write_text(f"path: {DET.resolve()}\ntrain: {train}\nval: {val}\nnames:\n{names}\n")


def main():
    ec = convertir_electrocom()
    ptrain, pval = separar_propias()
    ntrain, nval = separar_nuevas()
    extra = sorted((DET / "extra" / "images").glob("*.jpg"))
    print(f"extra (Kaggle): {len(extra)}")
    listas = DET / "listas"
    escribir_lista(listas / "train.txt", ec["train"] + ptrain * REPETIR_PROPIAS + ntrain * REPETIR_NUEVAS + extra)
    escribir_lista(listas / "valid.txt", ec["valid"] + pval + nval)
    escribir_lista(listas / "valid_propios.txt", pval + nval)
    escribir_yaml(DET / "data.yaml", "listas/train.txt", "listas/valid.txt")
    escribir_yaml(DET / "data_propios.yaml", "listas/train.txt", "listas/valid_propios.txt")
    print("Listo. Clases:", CLASES_DET)


if __name__ == "__main__":
    main()
