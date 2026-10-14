"""
Asistente de voz de ToolLighthouse: la persona pregunta por un componente y el sistema
responde por voz y apunta el láser a su caja.

    "¿Dónde está la resistencia?"  ->  "La resistencia está en la caja 1."  +  láser a la caja 1

Todo corre local (sin internet):
    - voz a texto: faster-whisper (español)
    - entender qué pidió: tabla de sinónimos (no hace falta un LLM para 9 clases)
    - respuesta hablada: voces de Windows (pyttsx3); conviene tener instalada una voz en español

Piezas:
    interpretar(texto)  -> clase o None
    Escucha(...)        -> graba y transcribe
    hablar(texto)       -> dice el texto en voz alta
    Asistente           -> junta todo; se usa desde main_det.py con la tecla config.TECLA_VOZ
"""

import threading
import unicodedata

import numpy as np

import config

FRECUENCIA = 16000  # lo que espera Whisper

# Cómo puede nombrar la gente cada componente (sin tildes, en minúsculas)
SINONIMOS = {
    "resistencia": ["resistencia", "resistencias", "resistor", "resistores", "resis"],
    "diodo": ["diodo", "diodos"],
    "led": ["led", "leds", "ledes", "diodo led", "luz led", "foco led"],
    "potenciometro": ["potenciometro", "potenciometros", "potenciómetro", "pot", "perilla"],
    "jumper": ["jumper", "jumpers", "yumper", "yumpers", "cable", "cables", "cable jumper", "cables jumper", "puente", "puentes"],
    "servo": ["servo", "servos", "servomotor", "servomotores", "servo motor", "motor servo"],
    "protoboard": ["protoboard", "protoboards", "breadboard", "proto board", "placa de pruebas", "tablero de pruebas"],
    "arduino_nano": ["arduino nano", "nano", "arduino", "arduinos"],
    "capacitor_electrolitico": ["capacitor", "capacitores", "condensador", "condensadores", "capacitor electrolitico", "electrolitico"],
}
# Para frases largas, ganan primero las más específicas ("diodo led" antes que "diodo")
_FRASES = sorted(((f, c) for c, fs in SINONIMOS.items() for f in fs), key=lambda x: -len(x[0]))

ARTICULO = {
    "resistencia": "la resistencia",
    "diodo": "el diodo",
    "led": "el LED",
    "potenciometro": "el potenciómetro",
    "jumper": "los cables jumper",
    "servo": "el servomotor",
    "protoboard": "la protoboard",
    "arduino_nano": "el Arduino Nano",
    "capacitor_electrolitico": "el capacitor electrolítico",
}

PALABRAS_APAGAR = ("apaga", "apagar", "apagalo", "quita", "cancela", "detente", "para el laser")

# Ayuda a Whisper a acertar los nombres técnicos
_PISTA = ("Dónde está la resistencia, el diodo, el LED, el potenciómetro, los jumper, el servo, "
          "la protoboard, el Arduino Nano, el capacitor electrolítico.")


def _normalizar(texto: str) -> str:
    sin = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
    return "".join(c if c.isalnum() or c == " " else " " for c in sin)


def interpretar(texto: str) -> str | None:
    """Devuelve la clase que se pidió ('resistencia', ...), 'apagar', o None si no se entendió."""
    t = " " + " ".join(_normalizar(texto).split()) + " "
    if any(f" {p} " in t or t.strip().startswith(p) for p in PALABRAS_APAGAR):
        return "apagar"
    for frase, clase in _FRASES:
        if f" {_normalizar(frase).strip()} " in t:
            return clase
    return None


def respuesta_para(clase: str | None) -> str:
    if clase == "apagar":
        return "Listo, apagué el láser."
    if clase is None:
        return "No te entendí. Repite qué componente buscas."
    ubic = config.MAPA_UBICACION.get(clase)
    if ubic in (None, "x"):
        return f"No sé dónde está {ARTICULO[clase]}."
    sujeto = ARTICULO[clase]
    verbo = "están" if sujeto.startswith("los ") else "está"
    return f"{sujeto[0].upper() + sujeto[1:]} {verbo} en la caja {ubic}."


def _nombre_tecla() -> str:
    return "espacio" if config.TECLA_VOZ == " " else f"tecla {config.TECLA_VOZ}"


# ---------- voz a texto ----------

class Escucha:
    def __init__(self, modelo: str = "small", dispositivo: int | None = None):
        self._nombre_modelo = modelo
        self._modelo = None
        self.dispositivo = dispositivo if dispositivo is not None else config.MICROFONO_INDEX

    def _cargar(self):
        if self._modelo is None:
            from faster_whisper import WhisperModel
            self._modelo = WhisperModel(self._nombre_modelo, device="cpu", compute_type="int8")
        return self._modelo

    def calentar(self):
        """Carga el modelo antes de usarlo (la primera vez descarga ~460 MB)."""
        self._cargar()

    def grabar(self, segundos: float = config.VOZ_SEGUNDOS) -> np.ndarray:
        import sounddevice as sd
        audio = sd.rec(int(segundos * FRECUENCIA), samplerate=FRECUENCIA, channels=1, dtype="float32",
                       device=self.dispositivo)
        sd.wait()
        return audio[:, 0]

    def transcribir(self, audio: np.ndarray) -> str:
        if audio.size == 0 or float(np.abs(audio).max()) < 0.01:
            return ""  # silencio: no dejar que Whisper invente texto
        segmentos, _ = self._cargar().transcribe(audio, language="es", beam_size=1, vad_filter=True,
                                                 initial_prompt=_PISTA, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segmentos).strip()


# ---------- texto a voz ----------

def _voz_en_espanol(engine):
    for v in engine.getProperty("voices"):
        idiomas = " ".join(str(i).lower() for i in getattr(v, "languages", []))
        nombre = v.name.lower()
        if idiomas.startswith("es") or "spanish" in nombre or "espa" in nombre                 or any(n in nombre for n in ("sabina", "helena", "laura", "pablo", "raul", "zira")) and "english" not in nombre:
            return v
    return None


def hablar(texto: str):
    """Dice el texto. Bloquea hasta terminar; llámalo desde un hilo si no quieres congelar la ventana."""
    import pyttsx3
    engine = pyttsx3.init()
    voz = _voz_en_espanol(engine)
    if voz is not None:
        engine.setProperty("voice", voz.id)
    else:
        print("[voz] No hay voz en español instalada en Windows; se usa la voz por defecto (acento inglés).")
    engine.setProperty("rate", 165)
    engine.say(texto)
    engine.runAndWait()


# ---------- todo junto ----------

class Asistente:
    """
    Escucha una pregunta, apunta el láser y responde. Corre en un hilo para no congelar el dashboard.
    `estado` es un texto corto que el dashboard puede mostrar.
    """

    def __init__(self, enlace, escucha: Escucha | None = None):
        self.enlace = enlace
        self.escucha = escucha or Escucha()
        self.estado = f"Voz: lista ({_nombre_tecla()})"
        self._ocupado = threading.Lock()

    def precargar(self):
        threading.Thread(target=self._precargar, daemon=True).start()

    def _precargar(self):
        self.estado = "Voz: cargando modelo..."
        try:
            self.escucha.calentar()
            self.estado = f"Voz: lista ({_nombre_tecla()})"
        except Exception as e:
            self.estado = f"Voz: error al cargar ({type(e).__name__})"

    def preguntar(self):
        """Inicia una pregunta por voz. Si ya hay una en curso, la ignora."""
        if self._ocupado.locked():
            return
        threading.Thread(target=self._ciclo, daemon=True).start()

    def _ciclo(self):
        with self._ocupado:
            try:
                self.estado = "Escuchando... habla ahora"
                audio = self.escucha.grabar()
                self.estado = "Entendiendo..."
                texto = self.escucha.transcribir(audio)
                clase = interpretar(texto)
                self.estado = f'Oí: "{texto}"' if texto else "No oí nada"
                if clase == "apagar":
                    self.enlace.apagar_laser()
                elif clase:
                    self.enlace.apuntar(clase)
                hablar(respuesta_para(clase))
                self.estado = f'Oí: "{texto}"' if texto else "No oí nada"
            except Exception as e:
                self.estado = f"Voz: error ({type(e).__name__}: {e})"
