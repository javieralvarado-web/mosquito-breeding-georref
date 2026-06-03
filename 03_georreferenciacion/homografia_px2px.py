"""
homografia_px2px.py
===================
Georreferenciación corregida por homografía trabajando completamente
en el espacio de píxeles — sin dependencia del frame 0.

Pipeline:
  1. Convierte GPS de los 5 puntos de control → píxeles (u_qgis, v_qgis)
     usando la posición del dron en frame 0 como origen
  2. Estima H: (u_yolo, v_yolo) → (u_qgis, v_qgis)
     → ambos espacios en píxeles, misma escala, más estable numéricamente
  3. Para cada detección en cualquier frame:
       a. Aplica H directamente: (u_yolo, v_yolo) → (u_corr, v_corr)
          H corrige la distorsión geométrica por pitch dentro del frame
       b. Convierte (u_corr, v_corr) → GPS
          usando la posición real del dron en ESE frame como origen

Uso:
    python homografia_px2px.py

Entradas:
    detecciones_georref_completo.csv

Salida:
    detecciones_georref_px2px.csv
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path

# ── Rutas ─────────────────────────────────────────────────────────────────────
PATH_IN  = Path(r"C:\Users\javie\OneDrive\Documents\MCE Javier Alvarado\Cuarto Semestre\Consultoría\Georreferenciacion\detecciones_georref_completo.csv")
PATH_OUT = Path(r"C:\Users\javie\detecciones_georref_px2px_google.csv")

# ── Dimensiones del frame ─────────────────────────────────────────────────────
W = 3840
H = 2160

# ── Escala metros/píxel ───────────────────────────────────────────────────────
S = 0.015203   # m/px — calibrado con auto (D=4.3m, d_px=282.85px)

# ── Radio de la Tierra ────────────────────────────────────────────────────────
R = 6_378_137.0   # metros (WGS84)

# ── Posición del dron en frame 0 — solo se usa para estimar H ─────────────────
LAT_DRON_0 = -22.935684
LON_DRON_0 = -43.688063

# ── Puntos de control (frame 0) ───────────────────────────────────────────────
# (u_yolo, v_yolo, lat_qgis, lon_qgis)
PUNTOS = [
    #(3635.5,  941.9,  -22.935832, -43.687856),  # charco
    #(2215.9, 1901.0,  -22.935758, -43.688125),  # tinaco
    #(2516.8, 1981.9,  -22.935784, -43.688104),  # tinaco
    #(3452.6,  719.2,  -22.935790, -43.687873),  # charco
    #(2767.3,  344.3,  -22.935691, -43.687926),  # tinaco
    (1385.0,  227.0,  -22.9355755, -43.6880986),
    (2389.0,  581.0,  -22.9356744, -43.6880297),
    (1193.0,  792.0,  -22.9356024, -43.6881516),
    (2186.0, 1152.0,  -22.9357030, -43.6880813),
]


# =============================================================================
def gps_to_px(lat, lon, lat_origen, lon_origen):
    """
    Convierte coordenadas GPS a píxeles del frame.
    lat_origen, lon_origen es la posición del dron (= centro del frame).

    GPS → metros → píxeles
    """
    delta_lat = lat - lat_origen
    delta_lon = lon - lon_origen

    delta_Y_m = np.radians(delta_lat) * R
    delta_X_m = np.radians(delta_lon) * R * np.cos(np.radians(lat_origen))

    delta_u =  delta_X_m / S    # eje X imagen (→ derecha)
    delta_v = -delta_Y_m / S    # eje Y imagen (↓ abajo, invertido)

    u = W / 2 + delta_u
    v = H / 2 + delta_v
    return u, v


def px_to_gps(u, v, lat_origen, lon_origen):
    """
    Convierte píxeles del frame a coordenadas GPS.
    lat_origen, lon_origen es la posición del dron (= centro del frame).

    píxeles → metros → GPS
    """
    delta_u = u - W / 2
    delta_v = v - H / 2

    delta_X_m =  delta_u * S
    delta_Y_m = -delta_v * S    # invertir eje Y

    delta_lat = np.degrees(delta_Y_m / R)
    delta_lon = np.degrees(delta_X_m / (R * np.cos(np.radians(lat_origen))))

    lat = lat_origen + delta_lat
    lon = lon_origen + delta_lon
    return lat, lon


# =============================================================================
print("=" * 60)
print("  Georreferenciación por Homografía Píxel → Píxel")
print("=" * 60)

# ── Paso 1: Convertir GPS de los puntos de control → píxeles ─────────────────
print(f"\n[1/5] Convirtiendo GPS de QGIS → píxeles...")
print(f"  Origen para estimar H: lat={LAT_DRON_0}  lon={LON_DRON_0}  (dron frame 0)")
print()

pts_yolo = []
pts_qgis = []

for i, (u_y, v_y, lat_q, lon_q) in enumerate(PUNTOS):
    # Convertir GPS de QGIS a píxeles usando dron frame 0 como origen
    # (porque los puntos de control son del frame 0)
    u_q, v_q = gps_to_px(lat_q, lon_q, LAT_DRON_0, LON_DRON_0)
    pts_yolo.append([u_y, v_y])
    pts_qgis.append([u_q, v_q])

    dist_px = np.sqrt((u_q - u_y)**2 + (v_q - v_y)**2)
    dist_m  = dist_px * S
    print(f"  P{i+1}: YOLO=({u_y:.1f}, {v_y:.1f})  "
          f"QGIS_px=({u_q:.1f}, {v_q:.1f})  "
          f"desplazamiento={dist_px:.1f}px ({dist_m:.1f}m)")

pts_yolo = np.array(pts_yolo, dtype=np.float32)
pts_qgis = np.array(pts_qgis, dtype=np.float32)

desp_medio = np.mean([
    np.sqrt((q[0]-y[0])**2 + (q[1]-y[1])**2)
    for y, q in zip(pts_yolo, pts_qgis)
])
print(f"\n  Desplazamiento medio YOLO→QGIS: {desp_medio:.1f}px ({desp_medio*S:.1f}m)")

# ── Paso 2: Estimar homografía píxel → píxel ─────────────────────────────────
print(f"\n[2/5] Estimando homografía píxel → píxel con RANSAC...")

H_matrix, mask = cv2.findHomography(pts_yolo, pts_qgis, cv2.RANSAC, 5.0)
n_inliers = int(mask.sum()) if mask is not None else len(PUNTOS)
print(f"  Inliers: {n_inliers}/{len(PUNTOS)}")
print(f"  Matriz H:")
for row in H_matrix:
    print(f"    [{row[0]:10.6f}  {row[1]:10.6f}  {row[2]:10.4f}]")

# ── Paso 3: Verificar reproyección ───────────────────────────────────────────
print(f"\n[3/5] Verificando reproyección en puntos de control...")

errores_px = []
errores_m  = []

for i, (u_y, v_y) in enumerate(pts_yolo):
    pt  = np.array([[[u_y, v_y]]], dtype=np.float32)
    res = cv2.perspectiveTransform(pt, H_matrix)
    u_c = float(res[0][0][0])
    v_c = float(res[0][0][1])
    u_q, v_q = pts_qgis[i]

    err_px = np.sqrt((u_c - u_q)**2 + (v_c - v_q)**2)
    err_m  = err_px * S
    errores_px.append(err_px)
    errores_m.append(err_m)

    print(f"  P{i+1}: QGIS_px=({u_q:.1f},{v_q:.1f})  "
          f"H_corr=({u_c:.1f},{v_c:.1f})  "
          f"error={err_px:.2f}px ({err_m:.2f}m)")

print(f"\n  Error medio: {np.mean(errores_px):.2f}px ({np.mean(errores_m):.2f}m)")
print(f"  Error máx:   {np.max(errores_px):.2f}px ({np.max(errores_m):.2f}m)")

# ── Paso 4: Aplicar H frame por frame ────────────────────────────────────────
print(f"\n[4/5] Aplicando homografía frame por frame...")

df = pd.read_csv(PATH_IN)
df = df.dropna(subset=["lat_criadero", "lon_criadero"]).copy()
print(f"  Detecciones: {len(df):,}  |  Frames únicos: {df['frame'].nunique():,}")

lat_px2px = np.zeros(len(df))
lon_px2px = np.zeros(len(df))

for frame_id in sorted(df["frame"].unique()):
    mask_frame = (df["frame"] == frame_id).values

    # Posición real del dron en este frame
    lat_dron = df.loc[mask_frame, "lat"].values[0]
    lon_dron = df.loc[mask_frame, "lon"].values[0]

    # Píxeles de YOLO para este frame
    u_vals = df.loc[mask_frame, "u"].values
    v_vals = df.loc[mask_frame, "v"].values

    # ── 4a. Aplicar H directamente: (u_yolo, v_yolo) → (u_corr, v_corr) ────
    # H corrige la distorsión geométrica por pitch DENTRO del frame.
    # No necesita saber en qué frame estamos — solo corrige la geometría
    # interna de la imagen que es la misma para todos los frames
    # (altitud constante, misma cámara, misma distorsión).
    pts = np.array(
        [[[ui, vi]] for ui, vi in zip(u_vals, v_vals)],
        dtype=np.float32
    )
    res = cv2.perspectiveTransform(pts, H_matrix)

    u_corr = res[:, 0, 0]   # píxel corregido X
    v_corr = res[:, 0, 1]   # píxel corregido Y

    # ── 4b. Convertir (u_corr, v_corr) → GPS ────────────────────────────────
    # Usa la posición real del dron en ESTE frame como origen.
    # El centro del frame (W/2, H/2) = posición del dron.
    # u_corr, v_corr son las coordenadas corregidas dentro del frame.
    lat_c, lon_c = px_to_gps(u_corr, v_corr, lat_dron, lon_dron)

    lat_px2px[mask_frame] = lat_c
    lon_px2px[mask_frame] = lon_c

    if frame_id % 1000 == 0:
        print(f"  Frame {frame_id:>6}  "
              f"dron=({lat_dron:.6f},{lon_dron:.6f})  "
              f"dets={mask_frame.sum()}")

df["lat_px2px"] = lat_px2px
df["lon_px2px"] = lon_px2px

# Corrección en metros respecto a proyección simple
dlat = (df["lat_px2px"] - df["lat_criadero"]) * 111000
dlon = (df["lon_px2px"] - df["lon_criadero"]) * 111000 * np.cos(
    np.radians(df["lat_px2px"]))
df["correccion_m"] = np.sqrt(dlat**2 + dlon**2)

print(f"\n  Corrección media vs proyección simple: {df['correccion_m'].mean():.2f}m")
print(f"  Corrección máxima:                     {df['correccion_m'].max():.2f}m")

# ── Paso 5: Guardar ──────────────────────────────────────────────────────────
print(f"\n[5/5] Guardando resultado...")

cols_order = [c for c in [
    "frame", "t_video_s",
    "class_id", "class_name", "confidence",
    "u", "v", "w_px", "h_px",
    "delta_u", "delta_v", "delta_X_m", "delta_Y_m", "dist_dron_m",
    "lat", "lon",
    "alt_m", "yaw_deg", "pitch_deg", "roll_deg", "speed_mps",
    "lat_criadero", "lon_criadero",
    "lat_px2px", "lon_px2px",
    "correccion_m",
] if c in df.columns]

df[cols_order].to_csv(PATH_OUT, index=False, float_format="%.8f")

print(f"\n{'='*60}")
print(f"  Guardado: {PATH_OUT}")
print(f"  Detecciones: {len(df):,}")
print(f"{'='*60}")

# ── Ejemplos y comparación ────────────────────────────────────────────────────
cols_show = ["frame", "class_name", "u", "v",
             "lat_criadero", "lon_criadero",
             "lat_px2px", "lon_px2px", "correccion_m"]

print("\nEjemplo frame 0:")
print(df[df["frame"] == 0][cols_show].head(5).to_string(
    index=False, float_format=lambda x: f"{x:.6f}"))

print("\nEjemplo frame 3000:")
print(df[df["frame"] == 3000][cols_show].head(3).to_string(
    index=False, float_format=lambda x: f"{x:.6f}"))

print("\nEjemplo frame 9000:")
print(df[df["frame"] == 9000][cols_show].head(3).to_string(
    index=False, float_format=lambda x: f"{x:.6f}"))

print("\n" + "="*60)
print("  Comparación de métodos en puntos de control (frame 0):")
print("  proyección simple vs px2px")
print("="*60)
frame0 = df[df["frame"] == 0].head(5)
for _, row in frame0.iterrows():
    d_simple = np.sqrt(
        ((row["lat_criadero"] - row["lat"]) * 111000)**2 +
        ((row["lon_criadero"] - row["lon"]) * 111000 *
         np.cos(np.radians(row["lat"])))**2
    )
    d_px2px = np.sqrt(
        ((row["lat_px2px"] - row["lat"]) * 111000)**2 +
        ((row["lon_px2px"] - row["lon"]) * 111000 *
         np.cos(np.radians(row["lat"])))**2
    )
    print(f"  {row['class_name']:<8}  simple={d_simple:.1f}m  px2px={d_px2px:.1f}m")
