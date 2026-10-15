"""
Captura fotogramas de la cámara real (iPhone) para entrenar con ELLA y no con datasets de otros laboratorios.

Uso:  python scripts/capturar.py [--camara 1]
(cierra main_det.py antes: la cámara virtual de Iriun solo la puede usar un programa a la vez)

Teclas:
    1-9   elegir la clase (ver abajo)            0 = mesa vacía / mano sin objeto (fondo)
    espacio  guarda UN fotograma de la clase elegida
    b        ráfaga: guarda uno cada ~0.5 s mientras esté activa (vuelve a pulsar b para parar)
    q        salir

    1 resistencia  2 diodo  3 led  4 potenciometro  5 jumper
    6 servo  7 protoboard  8 arduino_nano  9 capacitor_electrolitico  0 vacio

Cómo sacar buenos datos (lo más importante):
  - Cámara y mesa EXACTAMENTE como en la demo (misma altura, misma luz).
  - Mueve y gira el objeto entre fotos; cambia la distancia, la posición y el ángulo.
  - 40-60 fotogramas por clase. Para 'vacio': mesa sola, mano sola, y otras cosas que puedan aparecer.
  - Un solo objeto de la clase por cuadro (las cajas salen más fáciles); algunos cuadros con varios objetos.

Guarda en dataset_captura/<clase>/. Después anota las cajas con:
    python scripts/anotar_cajas.py --origen dataset_captura --destino dataset_deteccion/capturas
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2

import config
from main import abrir_fuente

CLASES = [c for c in config.CLASES if c != "nada"] + ["vacio"]
TECLAS = {ord(str(i + 1)): c for i, c in enumerate(CLASES[:-1])} | {ord("0"): "vacio"}
SALIDA = config.BASE_DIR / "dataset_captura"
INTERVALO_RAFAGA = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camara", type=int)
    args = ap.parse_args()
    args.video = None

    cap = abrir_fuente(args)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara. ¿Está cerrado main_det.py? ¿Iriun muestra el video del teléfono?")

    clase = CLASES[0]
    rafaga, ultimo = False, 0.0
    cuenta = {c: len(list((SALIDA / c).glob("*.jpg"))) for c in CLASES}

    def guardar(frame):
        (SALIDA / clase).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(SALIDA / clase / f"{clase}_{time.strftime('%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cuenta[clase] += 1

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        if rafaga and time.time() - ultimo >= INTERVALO_RAFAGA:
            guardar(frame)
            ultimo = time.time()

        vis = frame.copy()
        color = (0, 0, 255) if rafaga else (0, 200, 0)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 70), (30, 30, 30), -1)
        cv2.putText(vis, f"Clase: {clase}   guardadas: {cuenta[clase]}   {'RAFAGA ACTIVA' if rafaga else ''}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(vis, "1-9 clase | 0 vacio | espacio=foto | b=rafaga | q=salir", (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
        resumen = "  ".join(f"{i + 1}:{cuenta[c]}" for i, c in enumerate(CLASES[:-1])) + f"  0:{cuenta['vacio']}"
        cv2.putText(vis, resumen, (12, vis.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
        cv2.imshow("capturar", cv2.resize(vis, None, fx=0.8, fy=0.8))

        k = cv2.waitKey(1) & 0xFF
        if k in TECLAS:
            clase, rafaga = TECLAS[k], False
        elif k == ord(" "):
            guardar(frame)
        elif k == ord("b"):
            rafaga, ultimo = not rafaga, 0.0
        elif k == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Capturas guardadas:", {c: n for c, n in cuenta.items() if n})


if __name__ == "__main__":
    main()
