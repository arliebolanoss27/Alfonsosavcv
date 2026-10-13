# Descripción del dataset

> **Estado:** borrador con los datos públicos ya revisados. Las cifras marcadas con ⏳ se completan cuando se terminen la curación de Kaggle y las fotos y videos propios.

## 1. Clases

Son 10 clases de clasificación. Las primeras 9 son componentes que el estante señaliza con un LED; la décima evita que el modelo elija un objeto cuando no hay ninguno.

| # | Clase (nombre en el código) | Notas |
|---|---|---|
| 1 | `resistencia` | |
| 2 | `diodo` | Diodo normal, sin Zener |
| 3 | `led` | |
| 4 | `potenciometro` | Solo el de perilla (taper); el trimmer queda fuera |
| 5 | `jumper` | Cables jumper de protoboard |
| 6 | `servo` | |
| 7 | `protoboard` | |
| 8 | `arduino_nano` | Si da problemas, se cambia por Uno |
| 9 | `capacitor_electrolitico` | |
| 10 | `nada` | Base vacía y mano sin objeto |

La lista original incluía estaño, pasta para soldar y caimanes; se descartaron porque no existen datasets públicos de esos objetos.

## 2. Fuentes

Se combinan tres fuentes.

### 2.1 ElectroCom-61 (fotos reales de laboratorio)
- 2,121 fotos de un laboratorio universitario, 61 clases, 12,937 objetos etiquetados con cajas. Licencia CC BY 4.0. Versión original de [Mendeley Data](https://data.mendeley.com/datasets/6scy6h8sjz/2).
- Es un dataset de **detección**: para clasificar se **recorta cada caja** (con ~10 % de margen) y se conserva solo lo de nuestras clases.
- Se usó la versión de Mendeley porque trae las fotos completas; la de Roboflow las parte en cuatro y corta los objetos grandes.
- Las imágenes vienen redimensionadas a 640×640 con estiramiento, por lo que los recortes quedan algo deformados.
- Las fotos se tomaron en ráfaga: varias del mismo arreglo de objetos con segundos de diferencia.

| Clase | Recortes | Fotos distintas | Origen en ElectroCom-61 |
|---|---|---|---|
| `resistencia` | 169 | 54 | Resistor |
| `diodo` | 265 | 200 | Diode |
| `led` | 293 | 184 | LED-Light |
| `potenciometro` | 172 | 171 | Taper-Potentiometer |
| `servo` | 174 | 174 | Servo-Motor |
| `protoboard` | 181 | 181 | Breadboard |
| `arduino_nano` | 194 | 192 | Arduino-Nano |
| `capacitor_electrolitico` | 437 | 354 | Capacitor-10mf + Capacitor-470mf |

Los capacitores salen solo de ElectroCom-61 porque en otros datasets vienen mezclados con cerámicos. `jumper` y `nada` no existen en esta fuente.

### 2.2 Electronic Component Recognizer (Kaggle)
- Imágenes obtenidas de búsquedas en internet, 20 clases en carpetas por clase, licencia MIT.
- Se usan solo 6 clases: resistor, diode, led, potentiometer, jumper wire y breadboard.

| Clase | Imágenes en la carpeta | Sin duplicados exactos |
|---|---|---|
| `resistencia` | 171 | 155 |
| `diodo` | 151 | 148 |
| `led` | 174 | 125 |
| `potenciometro` | 161 | 147 |
| `jumper` | 250 | 164 |
| `protoboard` | 141 | 135 |

**Calidad.** Al revisar a ojo una muestra de 16 imágenes por clase, **una parte importante no son fotos del componente**: diagramas y símbolos de circuitos, logos de tiendas, capturas de videos con personas, tablas de colores de resistencias. En `led` hay tiras de luces, focos y lámparas; en `jumper` aparecen suéteres (por la palabra *jumper* en inglés) y pinzas de batería de auto. En `diodo` hay además puentes rectificadores y diagramas de Zener. Las imágenes de `potenciometro` y `protoboard` son bastante mejores. Por eso **estas imágenes no se usan sin filtrar**.

### 2.3 Fotos y videos propios ⏳
- Fotos de referencia de los objetos reales del laboratorio, para las 10 clases (cantidad ⏳).
- Videos cortos con el teléfono de la demo (iPhone 16), sobre la base gris mate y con la iluminación del montaje: **5 videos de ~15 s por clase**, moviendo y rotando el objeto en la mano, más 5 de `nada` (base vacía y mano sola). De cada video se extraen frames (cantidad ⏳).
- Son la fuente más importante: se parecen exactamente a las condiciones de la demo.

## 3. Preparación

1. **Recorte** de las cajas de ElectroCom-61 hacia las clases propias.
2. **Filtro de Kaggle:** se eliminan duplicados exactos y las imágenes que no son fotos reales del componente (revisión visual, apoyada por un modelo inicial que marque las de baja confianza).
3. **Revisión visual** de una muestra por clase antes de entrenar.
4. **Balance** de cantidades parecidas por clase, limitando las clases con más recortes.
5. **Split 80 % train / 20 % validación por grupo de origen:** por sesión de fotos en ElectroCom-61 (se agrupan por la hora en el nombre del archivo), por video en los propios y por imagen en Kaggle. Nunca queda una foto de un grupo en train y otra del mismo grupo en validación.
6. **Augmentation al vuelo** durante el entrenamiento (voltear, rotar, brillo y contraste), sin alterar el tono de color.

## 4. Conteo final ⏳

| Clase | ElectroCom-61 | Kaggle (filtrado) | Propias | Total | Train | Val |
|---|---|---|---|---|---|---|
| `resistencia` | 169 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| `diodo` | 265 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| `led` | 293 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| `potenciometro` | 172 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| `jumper` | 0 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| `servo` | 174 | 0 | ⏳ | ⏳ | ⏳ | ⏳ |
| `protoboard` | 181 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| `arduino_nano` | 194 | 0 | ⏳ | ⏳ | ⏳ | ⏳ |
| `capacitor_electrolitico` | 437 | 0 | ⏳ | ⏳ | ⏳ | ⏳ |
| `nada` | 0 | 0 | ⏳ | ⏳ | ⏳ | ⏳ |

## 5. Riesgos y limitaciones

- **Domain gap.** Los datasets públicos vienen de internet y de otro laboratorio; la demo es con nuestra cámara, mesa y luz. La exactitud con datos públicos puede salir más alta que la real. Por eso la validación se mide también **solo con fotos propias**.
- **Poca diversidad en algunas clases.** Las 169 resistencias de ElectroCom-61 salen de solo 54 fotos, y las de Kaggle son en buena parte diagramas. Resistencia, diodo y LED dependen mucho de los videos propios.
- **`jumper` y `nada` no tienen fuente de calidad.** Dependen de Kaggle (ruidoso) y de material propio.
- **Pares confundibles:** resistencia↔diodo (ambos axiales con bandas) y jumper↔otros cables.
- **Objetos delgados.** Resistencias, diodos y LEDs ocupan poca área del recorte; en la demo hay que sostenerlos cerca de la cámara.
- **Fuga de datos** controlada con el split por grupo (ver sección 3).

## 6. Licencias

- ElectroCom-61: CC BY 4.0 (requiere atribución).
- Electronic Component Recognizer: MIT.
- Fotos y videos propios: del equipo.
- Los datasets públicos **no se suben al repositorio**; se descargan desde sus páginas.
