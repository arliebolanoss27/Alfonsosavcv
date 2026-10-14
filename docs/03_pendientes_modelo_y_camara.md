# Pendientes del modelo y conexión de la cámara (traspaso)

Documento para quien tome el trabajo del modelo. Dice qué está hecho, qué falta y cómo conectar la cámara con el modelo y con el Arduino. Contexto: [`00_plan_y_decisiones.md`](00_plan_y_decisiones.md). Lo que no esté aquí o allá, se pregunta al equipo antes de decidirlo.

## 1. Estado real

| Parte | Estado |
|---|---|
| Selección de modelo (YOLO26-cls) | Documentada ([`01`](01_seleccion_modelo.md)); **sin entrenar ni medir nada** |
| Datos | Descargados en `datasets/` (ver sección 2); **sin curar ni procesar** |
| Código | **No existe ninguno**: ni preparación de datos, ni entrenamiento, ni cámara, ni serial |
| Fotos/videos propios | **No se han tomado/compartido** |
| Mapa clase -> ubicación | **No definido** |

## 2. Datos incluidos en el repo

| Carpeta | Contenido | Licencia | Notas |
|---|---|---|---|
| `datasets/electrocom61/` | ElectroCom-61 v2 (Mendeley): fotos con cajas en formato YOLO (`train/valid/test`, `data.yaml`) | CC BY 4.0 | Es de **detección**: hay que **recortar cada caja** para clasificar |
| `datasets/kaggle_recognizer/` | Electronic Component Recognizer (Kaggle), una carpeta por clase, 20 clases | MIT | **Muy ruidoso** (diagramas, logos, personas, suéteres en `jumper`). No usar sin filtrar |

Solo se usan algunas clases de cada uno (tablas de equivalencia en [`contexto_original_del_reto.md`](contexto_original_del_reto.md), sección 4a). Las demás carpetas se ignoran, no hace falta borrarlas.

## 3. Lo que falta hacer, en orden

### A. Estructura y configuración
1. Crear `config.py` con: lista de clases, umbral (0.80), mapa clase -> ubicación (1 a 9), puerto serial y baudrate (115200).
2. Crear `requirements.txt` (`ultralytics`, `opencv-python`, `pyserial`, `torch` con CUDA).
3. Confirmar que la laptop (RTX 4070) ve la GPU: `python -c "import torch; print(torch.cuda.is_available())"`. Si no, usar Colab.

### B. Preparar datos
4. Script que **recorte las cajas de ElectroCom-61** a las clases propias y las guarde como `dataset/<clase>/*.jpg`:
   resistencia <- Resistor, diodo <- Diode (no Zener), led <- LED-Light, potenciometro <- Taper-Potentiometer (sin Trimmer), servo <- Servo-Motor, protoboard <- Breadboard, arduino_nano <- Arduino-Nano, capacitor_electrolitico <- Capacitor-10mf y Capacitor-470mf.
   Los IDs de clase están en `data.yaml`. Cuidado con los *tiles* y recortes muy chicos: descartar cajas diminutas.
5. Script que tome de Kaggle resistor, diode, led, potentiometer, jumper wire y breadboard (**no** capacitores). Quitar duplicados exactos y revisar visualmente; descartar lo que no sea el componente.
6. Script que **extraiga frames de los videos propios** (5 videos de ~15 s por clase con el iPhone 16, más 5 de `nada`). Son la fuente más importante porque se parecen a la demo.
7. **Split 80/20 por grupo de origen** (video o sesión de fotos), nunca por imagen suelta. Si no, la validación sale inflada por fuga de datos.
8. Revisar a ojo una muestra de cada clase antes de entrenar.
9. Balancear: cantidad parecida de imágenes por clase. Resistencia, diodo, led y jumper dependen de los videos propios (ElectroCom-61 trae muy pocas).

### C. Entrenar y evaluar
10. Entrenar YOLO26s-cls y YOLO26m-cls (transfer learning, 224 px). Augmentation **al vuelo** (voltear, rotar, brillo), **sin cambiar el tono de color** (el color identifica resistencias, LEDs, etc.).

    ```python
    from ultralytics import YOLO
    model = YOLO("yolo26s-cls.pt")
    model.train(data="dataset", imgsz=224, epochs=30, hsv_h=0.0)
    ```
11. Evaluar: exactitud en validación, **matriz de confusión**, y exactitud **solo con fotos propias** (la que importa para la demo). Vigilar resistencia vs. diodo y jumper vs. otros cables.
12. Elegir entre s y m con esos resultados y anotarlo al final de [`01_seleccion_modelo.md`](01_seleccion_modelo.md).
13. Guardar el mejor peso (`best.pt`) fuera de git si pesa mucho, o en una carpeta `modelos/` si es pequeño.

### D. Código de integración (Día 3)
14. `clasificador.py` con una sola función:
    ```python
    def predecir(frame) -> tuple[str, float]:   # (clase, confianza)
    ```
15. `main.py` (ver sección 4).
16. Pruebas de punta a punta. Si una clase falla, grabar más videos de esa clase y reentrenar.

### E. Día 4
17. Ajustar umbral y estabilidad, ensayar la demo, documentación final.

## 4. Cómo conectar la cámara -> modelo -> Arduino

```
iPhone 16 (USB) -> OpenCV -> recorte de la zona -> predecir() -> promedio ~8 frames -> serial -> Nano -> LED
```

**Cámara**
- iPhone 16 como webcam por **cable USB** con DroidCam o Iriun. Probar primero que Windows lo detecta como cámara.
- En OpenCV: `cv2.VideoCapture(indice)`; probar índices 0, 1, 2 hasta dar con el iPhone. Llevar una **webcam USB de respaldo**.
- Bloquear enfoque y exposición, cámara fija, fondo gris mate y luz constante. Si la luz cambia, el modelo falla.
- Recortar la zona marcada donde se muestra el objeto antes de clasificar (los objetos pequeños se pierden a 224 px).

**Estabilidad**
- Promediar las probabilidades de los últimos ~8 frames y decidir sobre el promedio (en vez de exigir N frames seguidos).
- Solo actuar si la confianza promedio es **>= 0.80**.

**Serial al Nano**
- `pyserial`, **115200 baudios**. Esperar ~2 s tras abrir el puerto (el Nano se reinicia).
- Enviar un solo carácter: `'1'` a `'9'` enciende esa ubicación; `'0'` activa la secuencia de prueba; cualquier otro (como `\n`) se ignora.
- Enviar **solo cuando la clase estable cambia**, no en cada frame.
- La clase `nada` debe apagar todos los LEDs. El firmware actual **no tiene** ese comando: se propuso que `'x'` apague los 9 LEDs, **pendiente de confirmar con quien hace la electrónica**. Mientras tanto, no enviar nada para `nada`.

```python
import serial, time
ser = serial.Serial("COM3", 115200, timeout=1)   # ajustar el puerto
time.sleep(2)
ser.write(b"3")   # enciende la ubicación 3
```

Pines: D2 a D10 = ubicaciones 1 a 9. D0 y D1 están reservados para el serial.

## 5. Decisiones que siguen abiertas

| Pendiente | Quién decide |
|---|---|
| Mapa clase -> ubicación (1 a 9) | Equipo + electrónica |
| Comando `'x'` para apagar LEDs en el firmware | Electrónica |
| Transformaciones y cantidad exactas de augmentation | Quien entrene, midiendo en validación |
| Probar si el iPhone se detecta por USB en Windows | Equipo |
| Fotos de referencia y videos propios | Equipo |
| Si el Nano da problemas de reconocimiento, cambiar a Arduino Uno | Equipo |

## 6. Riesgos

- **Domain gap:** los datos públicos vienen de otras cámaras, mesas y luces. Medir siempre también con fotos propias.
- **Poca diversidad** en resistencia, diodo y LED.
- **Objetos delgados:** sostenerlos cerca de la cámara.
- La validación sobre datos públicos puede dar mucho más alto que el rendimiento real en la demo.
