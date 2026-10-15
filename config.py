"""
Configuraciones centralizadas del proyecto ToolLighthouse / Alfonsosavcv.
Define clases, hiperparámetros de visión, configuración de cámara e interfaz serial.
"""

from pathlib import Path

# Directorios principales
BASE_DIR = Path(__file__).resolve().parent
DATASETS_RAW_DIR = BASE_DIR / "datasets"
DATASET_PROCESADO_DIR = BASE_DIR / "dataset"
DATOS_PROPIOS_DIR = BASE_DIR / "datos_propios"
MODELOS_DIR = BASE_DIR / "modelos"

# Lista canónica de las 10 clases (orden exacto según documentación)
CLASES = [
    "resistencia",
    "diodo",
    "led",
    "potenciometro",
    "jumper",
    "servo",
    "protoboard",
    "arduino_nano",
    "capacitor_electrolitico",
    "nada",
]

# Nombres legibles para display visual en OpenCV / Interfaz
CLASES_DISPLAY = {
    "resistencia": "Resistencia",
    "diodo": "Diodo",
    "led": "LED",
    "potenciometro": "Potenciómetro",
    "jumper": "Cables Jumper",
    "servo": "Servomotor",
    "protoboard": "Protoboard",
    "arduino_nano": "Arduino Nano",
    "capacitor_electrolitico": "Capacitor Electrolítico",
    "nada": "Sin Componente",
}

# Mapeo a ubicaciones físicas en el estante (pines D2 a D10 en Arduino Nano)
# Caracteres '1' a '9' para cada ubicación individual.
# 'x' propuesto para apagar LEDs cuando detecta 'nada' (o None para no enviar nada).
MAPA_UBICACION = {
    "resistencia": "1",
    "diodo": "2",
    "led": "3",
    "potenciometro": "4",
    "jumper": "5",
    "servo": "6",
    "protoboard": "7",
    "arduino_nano": "8",
    "capacitor_electrolitico": "9",
    "nada": "x",  # Pendiente de confirmación con electrónica; si no se admite, se ignora
}

# Parámetros del modelo y visión por computadora
MODELO_IMGSZ = 224
MODELO_DEFAULT_WEIGHTS = "yolo26s-cls.pt"
CONFIDENCE_THRESHOLD = 0.80  # Umbral de activación para la electrónica
ESTABILIDAD_FRAMES = 8  # Tamaño de la ventana para promedio móvil de probabilidades

# Configuración de captura de video (iPhone 16 / Cámara web USB)
CAMARA_INDEX = 1  # en esta laptop el iPhone (Iriun) es el 1 y la webcam integrada el 0; ver scripts/listar_camaras.py
CAMARA_WIDTH = 1280
CAMARA_HEIGHT = 720
ROI_SIZE = 400  # Tamaño del cuadrado central para recortar y clasificar (en píxeles)

# Configuración de comunicación Serial con el microcontrolador
SERIAL_PORT = "COM3"
SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT = 1.0

# Mientras electrónica no confirme que el firmware apaga los LEDs con 'x',
# la clase "nada" no envía nada por serial.
SOPORTA_APAGADO = False
SERIAL_ESPERA_RESET = 2.0  # segundos que tarda el Nano en reiniciarse al abrir el puerto

# Detección (varios objetos, una caja por objeto)
MODELO_DET_PESOS = "best_det.pt"
DET_IMGSZ = 640
DET_CONF_MIN = 0.35   # por debajo de esto ni se dibuja la caja
DET_CONF_ACTIVAR = 0.65  # confianza promedio mínima para que un objeto cuente como "estable"
DET_VOTOS = 5         # en cuántos de los últimos ESTABILIDAD_FRAMES frames debe aparecer la clase

# Voz y láser
MICROFONO_INDEX = None   # índice fijo del micrófono; None = buscarlo por nombre (abajo)
MICROFONO_NOMBRE = "Microphone Array"  # el del laptop. El de Iriun (iPhone) llega en silencio
VOZ_NEURAL = "es-MX-DaliaNeural"  # voz de respuesta; otras: es-MX-JorgeNeural, es-SV-LorenaNeural, es-SV-RodrigoNeural
TECLA_VOZ = " "        # tecla para preguntar por voz (" " = espacio); cualquier carácter: "v", "m"...
TECLA_PAUSA = "p"      # pausa/reanuda el envío al Nano
VOZ_SEGUNDOS = 4.0      # cuánto se graba cada vez que se pulsa la tecla v
# Protocolo del láser: PENDIENTE de acordar con electrónica. Mientras LASER_ACTIVO sea False,
# solo se imprime lo que se enviaría; no se escribe nada en el puerto serial.
LASER_ACTIVO = False
LASER_COMANDO = "L{n}\n"   # apunta a la caja n (1 a 9)
LASER_APAGAR = "L0\n"      # apaga el láser
