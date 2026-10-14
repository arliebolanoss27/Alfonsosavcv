"""
ToolLighthouse: cámara -> recorte (ROI) -> modelo -> estabilidad -> serial, con dashboard.

Uso:
    python main.py                      # cámara de config.CAMARA_INDEX, serial simulado si no hay Nano
    python main.py --camara 1           # elegir el índice de la cámara (iPhone con Iriun/DroidCam)
    python main.py --puerto COM5        # usar el Nano en ese puerto
    python main.py --sin-serial         # nunca tocar el puerto serial
    python main.py --video prueba.mp4   # probar con un video o imagen en vez de cámara

Teclas: q o Esc = salir | espacio = pausar/reanudar el envío al Nano | r = reiniciar estabilidad
"""

import argparse
import time
import unicodedata

import cv2
import numpy as np

import config
from clasificador import predecir_probs
from enlace_serial import EnlaceSerial
from estabilidad import Estabilizador

ALTO_VIDEO = 540
ANCHO_PANEL = 360
VERDE, ROJO, GRIS, BLANCO, AMARILLO = (80, 200, 80), (60, 60, 220), (90, 90, 90), (240, 240, 240), (0, 200, 255)


def sin_acentos(texto: str) -> str:
    # cv2.putText no dibuja tildes ni ñ
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()


def nombre(clase: str) -> str:
    return sin_acentos(config.CLASES_DISPLAY.get(clase, clase))


def abrir_fuente(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        indice = args.camara if args.camara is not None else config.CAMARA_INDEX
        cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMARA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMARA_HEIGHT)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # el iPhone puede ignorarlo; se bloquea desde la app
        for _ in range(15):  # calentamiento: los primeros frames salen rotos o negros
            cap.read()
    return cap


def recortar_roi(frame):
    h, w = frame.shape[:2]
    lado = min(config.ROI_SIZE, h, w)
    x0, y0 = (w - lado) // 2, (h - lado) // 2
    return frame[y0:y0 + lado, x0:x0 + lado], (x0, y0, lado)


def dibujar_dashboard(frame, roi_box, promedio, estable, enviado, enlace, fps, envio_activo):
    h, w = frame.shape[:2]
    ancho_video = int(w * ALTO_VIDEO / h)
    esc = ALTO_VIDEO / h
    video = cv2.resize(frame, (ancho_video, ALTO_VIDEO))

    x0, y0, lado = roi_box
    color_roi = VERDE if estable and estable != "nada" else BLANCO
    cv2.rectangle(video, (int(x0 * esc), int(y0 * esc)), (int((x0 + lado) * esc), int((y0 + lado) * esc)), color_roi, 2)

    panel = np.full((ALTO_VIDEO, ANCHO_PANEL, 3), 30, np.uint8)
    mejor = max(promedio, key=promedio.get) if promedio else None
    conf = promedio[mejor] if mejor else 0.0

    cv2.putText(panel, "ToolLighthouse", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLANCO, 2)
    if estable:
        titulo, color = nombre(estable), VERDE
        ubic = config.MAPA_UBICACION.get(estable)
        subt = "Sin LED" if ubic in (None, "x") else f"Ubicacion {ubic}  (LED D{int(ubic) + 1})"
    else:
        titulo, color = ("Analizando..." if mejor else "Sin imagen"), AMARILLO
        subt = f"Mejor: {nombre(mejor)}" if mejor else ""
    cv2.putText(panel, titulo, (16, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    cv2.putText(panel, subt, (16, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLANCO, 1)
    cv2.putText(panel, f"Confianza {conf * 100:5.1f}%  (umbral {config.CONFIDENCE_THRESHOLD * 100:.0f}%)",
                (16, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLANCO, 1)

    # barras de probabilidad de las 10 clases
    y = 152
    for clase in sorted(promedio, key=promedio.get, reverse=True):
        p = promedio[clase]
        cv2.putText(panel, nombre(clase)[:18], (16, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, BLANCO, 1)
        cv2.rectangle(panel, (190, y), (190 + 110, y + 14), GRIS, 1)
        relleno = VERDE if clase == mejor and p >= config.CONFIDENCE_THRESHOLD else AMARILLO if clase == mejor else GRIS
        cv2.rectangle(panel, (190, y), (190 + int(110 * p), y + 14), relleno, -1)
        cv2.putText(panel, f"{p * 100:3.0f}", (308, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, BLANCO, 1)
        y += 24

    # estado del enlace con el Nano
    y = ALTO_VIDEO - 80
    cv2.putText(panel, f"Serial: {enlace.estado}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLANCO, 1)
    ultimo = enlace.ultimo_enviado if enlace.ultimo_enviado else "-"
    cv2.putText(panel, f"Ultimo enviado: {ultimo}", (16, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLANCO, 1)
    estado_envio = "ACTIVO" if envio_activo else "PAUSADO (espacio)"
    cv2.putText(panel, f"Envio: {estado_envio}", (16, y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                VERDE if envio_activo else ROJO, 1)
    cv2.putText(panel, f"{fps:4.1f} FPS", (ANCHO_PANEL - 90, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRIS, 1)

    return np.hstack([video, panel])


def main():
    ap = argparse.ArgumentParser(description="ToolLighthouse: cámara -> modelo -> Arduino")
    ap.add_argument("--camara", type=int, help="índice de la cámara (ver scripts/listar_camaras.py)")
    ap.add_argument("--video", help="ruta de un video o imagen para probar sin cámara")
    ap.add_argument("--puerto", help=f"puerto serial del Nano (por defecto {config.SERIAL_PORT})")
    ap.add_argument("--sin-serial", action="store_true", help="no abrir ningún puerto serial")
    args = ap.parse_args()

    cap = abrir_fuente(args)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara/video. Corre scripts/listar_camaras.py para ver los índices.")

    enlace = EnlaceSerial(args.puerto, simulado=args.sin_serial)
    estabilizador = Estabilizador()
    envio_activo = True
    ultimo_t, fps = time.time(), 0.0
    promedio = {c: 0.0 for c in config.CLASES}
    estable = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.video:  # el video terminó: volver a empezar
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("[camara] No llegó imagen; reintentando...")
                time.sleep(0.2)
                continue

            roi, roi_box = recortar_roi(frame)
            promedio, estable = estabilizador.agregar(predecir_probs(roi))
            if estable and envio_activo:
                enlace.enviar_clase(estable)

            ahora = time.time()
            fps = 0.9 * fps + 0.1 / max(ahora - ultimo_t, 1e-6)
            ultimo_t = ahora

            cv2.imshow("ToolLighthouse", dibujar_dashboard(frame, roi_box, promedio, estable, None, enlace, fps, envio_activo))
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == ord(" "):
                envio_activo = not envio_activo
            if tecla == ord("r"):
                estabilizador.reiniciar()
    finally:
        cap.release()
        enlace.cerrar()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
