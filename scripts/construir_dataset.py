"""
Script integrador del dataset para YOLO26-cls.
Combina las fuentes intermedias (ElectroCom-61 recortado, Kaggle curado y Datos Propios),
aplica el split en entrenamiento (train), validación (val) y prueba (test) respetando
los grupos de origen para evitar fuga de datos (data leakage).
Genera la estructura final estándar de clasificación de Ultralytics.
"""

import os
import sys
import shutil
import random
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from collections import defaultdict
from tqdm import tqdm

import config

INTERMEDIO_DIR = config.BASE_DIR / "dataset_intermedio"
FINAL_DATASET_DIR = config.DATASET_PROCESADO_DIR

# Proporción de particiones
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42


def agrupar_por_origen(archivos: list[Path]) -> dict[str, list[Path]]:
    """
    Agrupa archivos por su sesión, video o imagen base original
    para garantizar que todas las variaciones o frames de un mismo origen
    vayan a la misma partición y nunca se dispersen entre train y val/test.
    """
    grupos = defaultdict(list)
    for f in archivos:
        nombre = f.stem
        # Prefijos conocidos
        if nombre.startswith("e61_"):
            # e61_train_IMG20240228125654_jpg.rf.9e9dbf556d80e21645e808b3edd44214_box0
            # Extraer el identificador de sesión (timestamp IMG202402281256)
            partes = nombre.split("_")
            id_sesion = partes[2][:15] if len(partes) > 2 else nombre
            grupos[f"e61_{id_sesion}"].append(f)
        elif nombre.startswith("propio_"):
            # propio_clase_video01_f00025 o propio_clase_foto01_hash
            partes = nombre.split("_")
            id_origen = "_".join(partes[:4]) if len(partes) >= 4 else nombre
            grupos[id_origen].append(f)
        else:
            # En Kaggle cada imagen curada representa un origen único e independiente
            grupos[nombre].append(f)
    return grupos


def construir_dataset_final():
    random.seed(RANDOM_SEED)

    if not INTERMEDIO_DIR.exists():
        print(f"Directorio intermedio no encontrado: {INTERMEDIO_DIR}")
        return

    # Limpiar o crear la estructura final
    for particion in ["train", "val", "test"]:
        for clase in config.CLASES:
            (FINAL_DATASET_DIR / particion / clase).mkdir(parents=True, exist_ok=True)

    print("[Constructor Dataset Final] Integrando fuentes y realizando split agrupado...")

    resumen = defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "fuentes": set()})

    for clase in config.CLASES:
        # Recolectar todas las imágenes procesadas de todas las fuentes para esta clase
        archivos_clase = []
        for fuente in ["electrocom61", "kaggle", "propios"]:
            fuente_dir = INTERMEDIO_DIR / fuente / clase
            if fuente_dir.exists():
                imgs = list(fuente_dir.glob("*.jpg"))
                if imgs:
                    archivos_clase.extend(imgs)
                    resumen[clase]["fuentes"].add(fuente)

        if not archivos_clase:
            print(f"  [Aviso] No hay imágenes para la clase '{clase}'.")
            continue

        # Agrupar por origen para evitar fuga
        grupos_dict = agrupar_por_origen(archivos_clase)
        claves_grupos = list(grupos_dict.keys())
        random.shuffle(claves_grupos)

        total_grupos = len(claves_grupos)
        n_train = max(1, int(total_grupos * TRAIN_RATIO))
        n_val = max(1, int(total_grupos * VAL_RATIO)) if total_grupos > 3 else 0

        grupos_train = set(claves_grupos[:n_train])
        grupos_val = set(claves_grupos[n_train:n_train + n_val])
        grupos_test = set(claves_grupos[n_train + n_val:])

        # Si había muy pocos grupos, asegurar al menos val
        if not grupos_val and len(grupos_train) > 1:
            g_movido = list(grupos_train)[-1]
            grupos_train.remove(g_movido)
            grupos_val.add(g_movido)

        for g_clave, archivos in grupos_dict.items():
            if g_clave in grupos_train:
                particion = "train"
            elif g_clave in grupos_val:
                particion = "val"
            else:
                particion = "test"

            dest_folder = FINAL_DATASET_DIR / particion / clase
            for src_file in archivos:
                shutil.copy2(src_file, dest_folder / src_file.name)
                resumen[clase][particion] += 1

    print("\n================ RESUMEN DEL DATASET FINAL ================")
    print(f"{'Clase':<25} | {'Train':<7} | {'Val':<7} | {'Test':<7} | {'Total':<7} | {'Fuentes'}")
    print("-" * 80)
    for clase in config.CLASES:
        d = resumen[clase]
        total = d["train"] + d["val"] + d["test"]
        fuentes_str = ", ".join(sorted(d["fuentes"])) if d["fuentes"] else "Ninguna"
        print(f"{clase:<25} | {d['train']:<7} | {d['val']:<7} | {d['test']:<7} | {total:<7} | {fuentes_str}")
    print("===========================================================")
    print(f"Dataset estructurado en: {FINAL_DATASET_DIR}")


if __name__ == "__main__":
    construir_dataset_final()
