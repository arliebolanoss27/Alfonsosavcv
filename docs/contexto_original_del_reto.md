# Contexto para el modelo de visión · Reto ToolLighthouse

Este documento resume **todo lo que el equipo ha decidido** sobre el modelo de visión artificial. Lo que todavía no está decidido aparece marcado como **PENDIENTE** y no se debe tomar como definitivo. Si algo no está aquí, pregunta antes de decidirlo.

---

## 1. El proyecto

Es un reto universitario de 4 días en un laboratorio de electrónica, trabajado por un equipo de 5 personas.

**Objetivo:** que una persona muestre un componente frente a una cámara y el sistema le indique automáticamente dónde guardarlo, encendiendo el LED de su ubicación en un estante.

**Flujo del sistema (según la presentación del reto):**
```
Objeto → Cámara → Visión artificial → Software (Python) → Microcontrolador → LED de la ubicación
```
"No buscamos solamente un modelo que reconozca imágenes. Buscamos un sistema que vea, decida y actúe."

**Entregable final:** un sistema funcional capaz de reconocer los objetos y señalizar automáticamente su ubicación. Tiene cuatro funciones:
1. **Reconocer:** cámara + modelo identifican qué objeto está frente al sistema.
2. **Procesar:** determinar la ubicación que le corresponde.
3. **Comunicar:** enviar el resultado al microcontrolador por serial.
4. **Señalizar:** encender la ubicación correspondiente.

### Calendario e hitos

| Día | Tema | Contenido | Hito |
|---|---|---|---|
| 1 | VER | Dataset, reconocimiento, entrenamiento, evaluación | La computadora reconoce un objeto |
| 2 (hoy) | ACTUAR | Microcontrolador, LEDs, conexiones, comunicación, impresión 3D | El sistema activa cada ubicación individualmente |
| 3 | INTEGRAR | Modelo → software, software → micro, pruebas de punta a punta | Mostrar un objeto ilumina su ubicación |
| 4 | DEMOSTRAR | Ajustes y estabilidad, ensayo de demo, pitch | Prototipo estable y presentación preparada |

**El modelo tiene que estar listo para integrarse el Día 3.**

### Rúbrica de evaluación

| Criterio | Peso | "Excelente" significa |
|---|---|---|
| Funcionalidad del prototipo | 25 % | Reconoce y señaliza de forma estable durante la demo en vivo |
| Integración IA + electrónica | 20 % | Flujo automático completo: cámara → micro → LED |
| Precisión del reconocimiento | 15 % | Identifica correctamente todos los objetos |
| Creatividad e innovación | 15 % | Propuesta original con valor agregado claro |
| Pitch de venta | 15 % | Mensaje claro y convincente |
| Calidad técnica y diseño | 10 % | Montaje ordenado, robusto y bien documentado |

La presentación final dura 10 minutos de pitch con demo en vivo, más 5 minutos de preguntas. Según los organizadores, "un modelo muy preciso que no enciende ninguna luz no puede ganar".

### División del trabajo
- **Modelo de visión:** tú (este documento).
- **Electrónica y CAD/impresión 3D del estante:** otro integrante.

---

## 2. Tarea del modelo

**Clasificación de imagen**, como pide la presentación. La detección es opcional y no la hemos elegido.

### Las 10 clases

| # | Clase | Nota |
|---|---|---|
| 1 | Resistencias | |
| 2 | Diodos | Diodo normal, sin incluir Zener |
| 3 | LEDs | |
| 4 | Potenciómetro | |
| 5 | Jumpers | |
| 6 | Servo | |
| 7 | Protoboard | |
| 8 | Arduino Nano | Si el Nano da problemas, se cambia por Arduino Uno |
| 9 | Capacitor electrolítico | |
| 10 | Nada | Clase vacía: sin ella, el modelo siempre elige uno de los 9 objetos aunque no haya nada frente a la cámara |

Las primeras 9 reemplazan a una lista original que incluía estaño, pasta para soldar y caimanes. Esas tres se descartaron porque no existen datasets públicos de ellas.

**PENDIENTE:** los nombres exactos de las clases en el código.

---

## 3. Modelo

- **Decidido: no se usará el modelo previo de 130+ clases** de un integrante. Se evaluó y no aporta ninguna ventaja.
- **Decidido: se usará un modelo nuevo preentrenado y se le hará fine-tuning** (transfer learning) a nuestras 10 clases.
- **PENDIENTE: qué arquitectura usar.** Todavía no está decidido si será YOLO. Las dos opciones que menciona el material del reto son:
  - MobileNetV2 (Keras).
  - YOLOv8n-cls (Ultralytics).

  Ambas se entrenan con transfer learning en Google Colab con GPU.

---

## 4. Datos

Combinamos dos fuentes.

### 4a. Datasets públicos

**[Electronic Component Recognizer](https://www.kaggle.com/datasets/mdfaisalahmed025/electronic-component-recognizer)**
- Kaggle, licencia MIT.
- 3,661 imágenes en 20 clases, con una carpeta por clase (formato de clasificación).
- Las imágenes salieron de búsquedas en DuckDuckGo.
- De aquí se usan solo estas clases:

| Nuestra clase | Clase en el dataset |
|---|---|
| Resistencias | Resistor |
| Diodos | Diode |
| LEDs | LED |
| Potenciómetro | Potentiometer |
| Jumpers | Jumper Wire |
| Protoboard | Breadboard |

**[ElectroCom-61](https://universe.roboflow.com/richard-w3clx/electrocom-61-m3u2e)**
- Roboflow, licencia CC BY 4.0. Versión original también en [Mendeley Data](https://data.mendeley.com/datasets/6scy6h8sjz/2).
- 2,121 fotos de un laboratorio universitario, 61 clases, 12,937 objetos etiquetados.
- Es un dataset de **detección** (cajas por objeto). Para clasificación **hay que recortar cada caja** y quedarse solo con las de nuestras clases.
- Para bajarlo de Roboflow se necesita una API key gratuita de roboflow.com.
- La versión de Roboflow (workspace `richard-w3clx`, proyecto `electrocom-61-m3u2e`, versión 1) parte cada foto en 4 *tiles* (2×2) y trae variantes aumentadas en train. Los objetos grandes pueden quedar cortados. La versión de Mendeley tiene las fotos sin partir.
- De aquí se usan:

| Nuestra clase | Clase en el dataset |
|---|---|
| Resistencias | Resistor |
| Diodos | Diode (no Zener-Diode) |
| LEDs | LED-Light |
| Potenciómetro | Taper-Potentiometer / Trimmer-Potentiometer |
| Servo | Servo-Motor |
| Protoboard | Breadboard |
| Arduino Nano | Arduino-Nano (Arduino-Uno si se cambia) |
| Capacitor electrolítico | Capacitor-10mf, Capacitor-470mf |

**Decidido:** los capacitores salen **solo** de ElectroCom-61, no del Recognizer, porque en el Recognizer pueden venir mezclados con cerámicos.

**PENDIENTE:** si el potenciómetro incluye *Trimmer-Potentiometer* o solo *Taper-Potentiometer*.

**No hay que borrar nada a mano** de los datasets: el script toma solo las carpetas y cajas de nuestras clases e ignora el resto.

### 4b. Fotos de referencia propias
- El equipo **tomó fotos de referencia de los objetos reales del laboratorio** para las 10 clases. Todavía no se han compartido.
- **Se van a generar variaciones** de esas fotos: alterar un poco los píxeles **sin perder las características que definen cada clase**.
- **PENDIENTE:** qué transformaciones exactas usar, cuántas variaciones por foto y si se harán offline o al vuelo durante el entrenamiento.

### 4c. Reglas de datos (del material del reto y de lo hablado)
- **Fuga de datos:** frames de un mismo video, variantes aumentadas de una misma foto y variaciones de una misma foto base **nunca** pueden quedar a la vez en train y en validación. El material del reto advierte que eso infla falsamente la validación.
- **Split:** el material del reto usa 80 % train y 20 % validación.
- **Data augmentation al entrenar:** el material del reto menciona voltear, rotar y variar brillo.
- **Balance:** cantidad parecida de imágenes por clase.
- **Revisión visual:** antes de entrenar, mirar unas cuantas imágenes por clase, porque los datasets públicos pueden traer etiquetas equivocadas.

---

## 5. Evaluación (entregable del Día 1)

Según el material del reto:
- Exactitud en **validación**, con imágenes que el modelo no vio.
- **Matriz de confusión** (filas = clase real, columnas = predicha).
- **Prueba en el mundo real**, cambiando luz, ángulo, mano y fondo.
- Grad-CAM aparece en el material como opcional, para ver en qué zonas se fijó el modelo.

**Riesgos que ya identificamos:**
- **Domain gap:** las imágenes públicas vienen de internet y de otro laboratorio, pero la demo será con nuestra cámara, mesa y luz. La validación con datasets públicos puede salir más alta que el desempeño real. Por eso las fotos propias son importantes.
- **Pares que se pueden confundir:** resistencia ↔ diodo (los dos son axiales con bandas) y jumper ↔ otros cables.
- **Objetos pequeños:** a 224×224 píxeles, un componente pequeño ocupa muy pocos píxeles si la cámara está lejos.

---

## 6. Inferencia e integración (Día 3)

**Lo que define el material del reto:**
- Un script de Python con OpenCV captura la cámara (`cv2.VideoCapture`).
- El modelo clasifica la imagen.
- **Si la confianza supera el umbral de 0.80**, Python envía la ubicación al microcontrolador **por serial (USB)** y este enciende el LED de esa ubicación.
- La clase "Nada" no debe encender ningún LED.

**Lo que ya existe del lado de la electrónica:**
- El microcontrolador del estante es un **Arduino Nano** con **9 LEDs**, uno por ubicación, en los pines **D2 a D10** (ubicación 1 = D2, …, ubicación 9 = D10).
- El firmware actual lee el serial a **115200 baudios**:
  - Un carácter de `'1'` a `'9'` enciende solo esa ubicación.
  - `'0'` vuelve a la secuencia de prueba.
  - Cualquier otro carácter, como el salto de línea, se ignora.
- Los pines D0 y D1 del Nano están reservados para esa comunicación serial.

**PENDIENTE:**
- Qué clase va en qué ubicación (1–9) del estante.
- La forma exacta de la función o interfaz entre el modelo y el script de cámara. Depende de la arquitectura que se elija.
- Si el sistema exige que la misma clase se mantenga varios frames seguidos antes de enviar la orden, y cuántos. Se propuso para dar estabilidad, pero no está confirmado.

---

## 7. Material existente (hecho antes de varias decisiones; revisar antes de usar)

- **Notebook de Colab `ToolLighthouse_Dataset_y_Entrenamiento.ipynb`:**
  - La parte de datos se probó con datos sintéticos: descarga, filtrado de clases, recorte de ElectroCom-61, quitar duplicados, balance y split.
  - La parte de entrenamiento está escrita para YOLOv8n-cls, que **no está decidido**, y no se ha probado.
  - Usa nombres de clase provisionales en minúsculas (`resistencia`, `diodo`, `led`, `potenciometro`, `jumper`, `servo`, `protoboard`, `arduino_nano`, `capacitor_electrolitico`, `nada`).
  - Del potenciómetro solo toma *Taper-Potentiometer*.
  - **No incluye** las fotos de referencia propias.
- **Script `probar_camara.py`:** prueba en vivo con la webcam. Está atado a Ultralytics/YOLO, que **no está decidido**. Su envío por serial está escrito para otro formato y **no coincide** con el firmware actual del Nano.
- **Repositorio del equipo:** https://github.com/arliebolanoss27/Alfonsosavcv

---

## 8. Resumen de decisiones pendientes que te tocan

1. Arquitectura del modelo preentrenado.
2. Nombres exactos de las clases en el código.
3. Si el potenciómetro incluye *Trimmer-Potentiometer*.
4. Cómo generar las variaciones de las fotos propias: transformaciones, cantidad y offline o en línea.
5. Interfaz entre el modelo y el script de cámara.
6. Si se exigen varios frames seguidos antes de enviar la orden, y cuántos.

El mapa de clase → ubicación del estante (1–9) también está pendiente, pero se decide junto con quien hace la electrónica y el estante.
