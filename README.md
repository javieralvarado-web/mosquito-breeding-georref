# Georreferenciación de Criaderos de Mosquitos a partir de Videos de Dron

## Descripción

Este proyecto implementa un sistema de detección automática y georreferenciación de criaderos potenciales de mosquitos en secuencias de video aéreo capturadas mediante drones, utilizando modelos de detección de objetos basados en YOLO.

El objetivo principal es construir un pipeline completo que permita, a partir de un video de dron en resolución 4K, detectar criaderos del mosquito *Aedes aegypti* en cada frame del video, asignarles coordenadas geográficas precisas y exportarlos como capas vectoriales visualizables en QGIS sobre mapas satelitales.

El proyecto fue desarrollado como parte de la **Consultoría de Estancia 2026** en el **Centro Regional de Investigación en Salud Pública (CRISP)**, dentro de la Maestría en Cómputo Estadístico del Centro de Investigación en Matemáticas (CIMAT).

---

## Objetivo del Proyecto

Desarrollar una metodología para georreferenciar criaderos potenciales de mosquitos \textit{Aedes aegypti} detectados en videos capturados mediante vehículos aéreos no tripulados, integrando información visual y datos de telemetría del vuelo para estimar la ubicación geográfica de los objetos detectados y generar mapas espaciales que apoyen actividades de vigilancia entomológica y control vectorial, que integre:

- Detección de objetos con YOLOv12-nano sobre video 4K.
- Sincronización de la telemetría GPS del dron con los frames del video.
- Proyección nadiral de las detecciones al espacio geográfico.
- Corrección geométrica del error por inclinación del dron mediante homografía DLT.
- Exportación de los resultados como GeoPackage para su visualización en QGIS.

La evaluación del modelo se realizó utilizando métricas estándar de detección de objetos:

- Precision
- Recall
- mAP@50
- mAP@50:0.95

---

## Tecnologías Utilizadas

- Python
- Ultralytics YOLO
- OpenCV
- NumPy
- Pandas
- GeoPandas
- Shapely
- Google Colab
- QGIS

---

## Estructura del Repositorio

```
mosquito-breeding-georref/
│
├── README.md
├── requirements.txt
│
├── 00_entrenamiento/
│   ├── train_yolov12_v5.yaml          # Configuración del dataset MBG-V2
│   ├── train_yolov12n_v5.py           # Script de entrenamiento
│   └── valid_yolov12n_v52.py          # Script de validación y métricas
│
├── 01_deteccion/
│   └── detectar_video_completo.py     # Inferencia frame por frame sobre video completo
│
├── 02_sincronizacion/
│   ├── geo_trajectory_v17.py          # Sincronización telemetría-video (Vuelo 17)
│   └── geo_trajectory_v12.py          # Sincronización telemetría-video (Vuelo 12)
│
├── 03_georreferenciacion/
    ├── georreferenciacion_v17.py      # Proyección nadiral simple (Vuelo 17)
│   ├── georreferenciacion_v12.py      # Proyección nadiral simple (Vuelo 12)
│   └── homografia_px2px.py            # Corrección geométrica por homografía DLT
│
└── 04_qgis/
    ├── qgis_vuelo17.ipynb             # Generación de GeoPackage Vuelo 17 (Colab)
    └── qgis_vuelo12.ipynb             # Generación de GeoPackage Vuelo 12 (Colab)
```

---

## Pipeline del Proyecto

### 1. Entrenamiento del Modelo

**Script:** `00_entrenamiento/train_yolov12n_v5.py`  
**Configuración:** `00_entrenamiento/train_yolov12_v5.yaml`

Entrenamiento del modelo YOLOv12-nano mediante transferencia de conocimiento
(*transfer learning*) sobre el dataset MBG-V2, con una división 60/20/20
para entrenamiento, validación y prueba respectivamente.

El modelo fue entrenado con los siguientes parámetros:

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `epochs`  | 200   | Iteraciones completas sobre el dataset |
| `imgsz`   | 1024  | Resolución de entrada cuadrada |
| `batch`   | 8     | Tamaño de lote |
| `conf`    | 0.5   | Umbral de confianza mínima |
| `iou`     | 0.7   | Umbral IoU para NMS |
| `device`  | 1     | GPU NVIDIA Quadro RTX 8000 (48 GB VRAM) |

### 2. Validación del Modelo

**Script:** `00_entrenamiento/valid_yolov12n_v52.py`

Evaluación del modelo entrenado sobre el conjunto de validación.
Genera curvas de Precision-Recall por clase y matriz de confusión.

**Resultados sobre el conjunto de validación:**

| Clase  | Imágenes | Instancias | Precisión | Recall | mAP@0.5 | mAP@0.5:0.95 |
|--------|----------|------------|-----------|--------|---------|--------------|
| Charco | 320      | 618        | 0.838     | 0.822  | 0.853   | 0.690        |
| Cubeta | 642      | 1,126      | 0.788     | 0.225  | 0.511   | 0.314        |
| Llanta | 298      | 804        | 0.870     | 0.425  | 0.655   | 0.408        |
| Tinaco | 1,812    | 6,798      | 0.973     | 0.937  | 0.965   | 0.824        |
| Maceta | 539      | 1,968      | 0.874     | 0.328  | 0.607   | 0.355        |
| **Global** | **2,187** | **11,314** | **0.869** | **0.548** | **0.718** | **0.518** |

### 3. Inferencia sobre Video Completo

**Script:** `01_deteccion/detectar_video_completo.py`

Inferencia frame por frame sobre los videos completos de ambos vuelos.
Genera un CSV con las coordenadas normalizadas (formato YOLO) de cada
detección junto con la clase y la confianza del modelo.

| Vuelo | Frames | Detecciones | Promedio/frame |
|-------|--------|-------------|----------------|
| Vuelo 17 — Región I | 10,099 | 46,154 | 4.57 |
| Vuelo 12 — Región F |  4,169 | 19,331 | 4.64 |

### 4. Sincronización Telemetría-Video

**Scripts:** `02_sincronizacion/geo_trajectory_v17.py` y `02_sincronizacion/geo_trajectory_v12.py`

La telemetría del dron (~9.9–17.2 Hz) se sincroniza con los frames del
video (29.97 fps) mediante interpolación lineal con `numpy.interp`,
asignando coordenadas GPS al 100% de los frames de ambos vuelos.

| Parámetro | Vuelo 17 | Vuelo 12 |
|-----------|----------|----------|
| Registros telemetría | 2,995 (~9.9 Hz) | 2,397 (~17.2 Hz) |
| Frames con GPS | 10,099 (100%) | 4,169 (100%) |
| Error máx. interpolación | ~1.02 m | ~1.02 m |
| Saltos > 200 ms | 179 (6%) | 49 (2%) |

### 5. Georreferenciación y Corrección por Homografía

**Scripts:** `03_georreferenciacion/georreferenciacion_v17.py`, `03_georreferenciacion/georreferenciacion_v12.py` y `homografia_px2px.py`

La proyección nadiral simple introduce un error sistemático de ~14 m por
la inclinación del dron (*pitch* máximo: -20.7°). La corrección por
homografía DLT estimada con 4 puntos de control (esquinas de un techo
plano identificados en el mapa satelital de Google) reduce este error:

| Métrica | Valor |
|---------|-------|
| Error de reproyección (4 puntos de control) | 0.00 px |
| Outliers fuera del rango válido | 0 / 46,154 (0%) |
| Corrección máxima vs proyección simple | 24.3 m |
| Corrección media zona central | 7.2–8.6 m |

### 6. Exportación a QGIS

**Notebooks:** `04_qgis/qgis_vuelo17.ipynb` y `04_qgis/qgis_vuelo12.ipynb`

Las detecciones georreferenciadas se exportan como GeoPackage con una
capa de puntos por clase de criadero y capas adicionales de trayectoria
del dron y buffer de cobertura, visualizables en QGIS sobre Google Satellite.

| Vuelo | Detecciones totales | Puntos en QGIS (filtro 300 frames) |
|-------|--------------------|------------------------------------|
| Vuelo 17 | 46,154 | 168 |
| Vuelo 12 | 19,331 | 47  |

---

## Dataset

El proyecto utiliza el dataset público **MBG-V2** (Mosquito Breeding Grounds V2):

- **Origen:** Laboratório Nacional de Computação Científica (LNCC), Brasil
- **Vuelos:** 26 videos sobre 13 regiones urbanas de Rio de Janeiro
- **Características:** Video 4K (3840×2160) a 29.97 fps
- **Altitud:** ~40 m
- **Clases:** 5 (charco, cubeta, llanta, tinaco, maceta)
- **DOI:** https://doi.org/10.5281/zenodo.7504421

Los vuelos analizados en este trabajo son:

| Vuelo | Región | Ubicación | Duración | Distancia |
|-------|--------|-----------|----------|-----------|
| Vuelo 17 | Región I | Jacarepaguá, RJ | 337 s (5:37 min) | 1,453 m |
| Vuelo 12 | Región F | Rio de Janeiro, RJ | 139 s (2:19 min) | 287 m |

---

## Instalación

Clonar el repositorio:

```bash
git clone https://github.com/javieralvarado-web/mosquito-breeding-georref.git
cd mosquito-breeding-georref
```

Crear el entorno virtual e instalar dependencias:

```bash
conda create -n mosquitos python=3.10
conda activate mosquitos
pip install -r requirements.txt
```

### Instalación de PyTorch

El archivo `requirements.txt` no incluye PyTorch debido a que la
instalación depende del hardware disponible y de la versión de CUDA.

**CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Sin GPU (CPU):**
```bash
pip install torch torchvision
```

### Software externo

- **FFmpeg** — para codificación del video anotado con detecciones:  
  Descargar desde https://www.gyan.dev/ffmpeg/builds/ (Windows) o `sudo apt install ffmpeg` (Linux)
- **QGIS 3.x** — para visualización del GeoPackage:  
  Descargar desde https://qgis.org

---

## Ejecución del Pipeline

```bash
# 0. Entrenamiento del modelo
python 00_entrenamiento/train_yolov12n_v5.py

# 0b. Validación y métricas
python 00_entrenamiento/valid_yolov12n_v52.py

# 1. Inferencia sobre video completo
python 01_deteccion/detectar_video_completo.py --vuelo 17
python 01_deteccion/detectar_video_completo.py --vuelo 12

# 2. Sincronización telemetría-video
python 02_sincronizacion/geo_trajectory_v17.py        # Vuelo 17
python 02_sincronizacion/geo_trajectory_v12.py    # Vuelo 12

# 3. Proyección nadiral simple
python 03_georreferenciacion/georreferenciacion_v17.py   # Vuelo 17
python 03_georreferenciacion/georreferenciacion_v12.py   # Vuelo 12

# 4. Corrección por homografía DLT
python 03_georreferenciacion/homografia_px2px.py

# 5. Generar GeoPackage (ejecutar en Google Colab)
#    Abrir 04_qgis/qgis_vuelo17.ipynb o qgis_vuelo12.ipynb
#    Subir los CSV generados y ejecutar todas las celdas
```

---

## Visualización en QGIS

1. Abrir QGIS 3.x
2. Agregar mapa base Google Satellite:
   ```
   XYZ Tiles → Nueva conexión
   URL: https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}
   ```
3. Cargar el GeoPackage generado:
   `Layer → Add Layer → Add Vector Layer → vuelo17_regionI.gpkg`
4. CRS: EPSG:4326 (WGS84)

---

## Trabajo Futuro

- Implementación de un módulo de seguimiento de objetos (*tracking*) para el conteo de criaderos únicos a lo largo del video.
- Corrección explícita del *pitch* y *roll* del dron usando el modelo de cámara del DJI Phantom 4 Pro.
- Calibración de la distorsión radial de la lente mediante patrón de tablero de ajedrez.
- Validación con mediciones GPS de campo (GPS RTK, precisión centimétrica).
- Extensión del análisis a los 26 vuelos del dataset MBG-V2.

---

## Autor

**Javier Eulalio Alvarado Acosta**  
Maestría en Cómputo Estadístico  
Centro de Investigación en Matemáticas (CIMAT)

**Asesor CIMAT:** Dr. Francisco Javier Hernández López  
**Responsable CRISP:** Dra. Kenia Mayela Valdez Delgado  
Centro Regional de Investigación en Salud Pública (CRISP) — 2026

---

## Referencia del Dataset

Passos, W. L., Araujo, G. M., Haque, U., Cruz-Roldán, F., & Netto, S. L. (2023).
*IEEE ICIP 2023 Challenge on the Automatic Detection of Mosquito Breeding Grounds.*
2023 IEEE International Conference on Image Processing Challenges and Workshops (ICIPCW),
pp. 3624–3628. https://doi.org/10.1109/ICIPC59416.2023.10328377

**Dataset MBG-V2:**
https://www02.smt.ufrj.br/~tvdigital/database/mosquito/page_02.html
