"""
georreferenciacion_v17.py
=========================
Proyección nadiral simple para el Vuelo 17 Región I.
Convierte coordenadas YOLO normalizadas a lat/lon usando la escala
calibrada s=0.015203 m/px y la posición GPS del dron por frame.

Uso:
    python 03_georreferenciacion/georreferenciacion_v17.py

Entradas:
    detecciones_video_completo.csv
    sync_table.csv

Salida:
    detecciones_georref_completo.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE = Path(r"C:\Users\javie\OneDrive\Documents\MCE Javier Alvarado\Cuarto Semestre\Consultoría\Georreferenciacion")

PATH_DET  = BASE / "detecciones_video_completo.csv"
PATH_SYNC = BASE / "sync_table.csv"
PATH_OUT  = BASE / "detecciones_georref_completo.csv"

# ── Parámetros ────────────────────────────────────────────────────────────────
W = 3840
H = 2160
S = 0.015203       # m/px — calibrado con vehículo (D=4.3m, d_px=282.85px)
R = 6_378_137.0    # metros (WGS84)

print("=" * 60)
print("  Georreferenciación — Proyección Nadiral — Vuelo 17 Región I")
print("=" * 60)

# ── Cargar datos ──────────────────────────────────────────────────────────────
print(f"\n[1/4] Cargando datos...")
df_det  = pd.read_csv(PATH_DET)
df_sync = pd.read_csv(PATH_SYNC)

print(f"  Detecciones:  {len(df_det):,}")
print(f"  Frames sync:  {len(df_sync):,}")
print(f"  Clases:       {sorted(df_det['class_name'].unique())}")

# ── Merge detecciones + telemetría ────────────────────────────────────────────
print(f"\n[2/4] Merge detecciones + telemetría por frame_id...")

if "frame" in df_det.columns and "frame_id" not in df_det.columns:
    df_det = df_det.rename(columns={"frame": "frame_id"})

df = df_det.merge(df_sync, on="frame_id", how="left")
sin_gps = df["lat"].isna().sum()
print(f"  Detecciones con GPS:    {(~df['lat'].isna()).sum():,}")
print(f"  Detecciones sin GPS:    {sin_gps:,}")
df = df.dropna(subset=["lat", "lon"]).copy()

# ── Proyección nadiral ────────────────────────────────────────────────────────
print(f"\n[3/4] Proyección nadiral simple...")

df["u"]    = df["x_center_norm"] * W
df["v"]    = df["y_center_norm"] * H
df["w_px"] = df["width_norm"]    * W
df["h_px"] = df["height_norm"]   * H

df["delta_u"] = df["u"] - W / 2
df["delta_v"] = df["v"] - H / 2

df["delta_X_m"] =  df["delta_u"] * S
df["delta_Y_m"] = -df["delta_v"] * S

df["dist_dron_m"] = np.sqrt(df["delta_X_m"]**2 + df["delta_Y_m"]**2)

phi0_rad = np.radians(df["lat"])
df["delta_lat"] = df["delta_Y_m"] / R
df["delta_lon"] = df["delta_X_m"] / (R * np.cos(phi0_rad))

df["lat_criadero"] = df["lat"] + np.degrees(df["delta_lat"])
df["lon_criadero"] = df["lon"] + np.degrees(df["delta_lon"])

print(f"  lat_criadero rango: {df['lat_criadero'].min():.6f} → {df['lat_criadero'].max():.6f}")
print(f"  lon_criadero rango: {df['lon_criadero'].min():.6f} → {df['lon_criadero'].max():.6f}")
print(f"  Distancia media al dron: {df['dist_dron_m'].mean():.1f}m")

# ── Guardar ───────────────────────────────────────────────────────────────────
print(f"\n[4/4] Guardando resultado...")

cols_order = [c for c in [
    "frame_id", "t_video_s",
    "class_id", "class_name", "confidence",
    "u", "v", "w_px", "h_px",
    "delta_u", "delta_v", "delta_X_m", "delta_Y_m", "dist_dron_m",
    "lat", "lon", "alt_m", "yaw_deg", "pitch_deg", "roll_deg", "speed_mps",
    "lat_criadero", "lon_criadero",
] if c in df.columns]

df[cols_order].to_csv(PATH_OUT, index=False, float_format="%.8f")

print(f"\n{'='*60}")
print(f"  Guardado: {PATH_OUT}")
print(f"  Detecciones: {len(df):,}")
print(f"{'='*60}")

print("\nDistribución por clase:")
print(df["class_name"].value_counts().to_string())

print(f"\nPitch rango: {df['pitch_deg'].min():.1f}° → {df['pitch_deg'].max():.1f}°")
print(f"Error estimado por pitch: {40 * np.tan(np.radians(abs(df['pitch_deg']).max())):.1f}m máx")

input("\nPresiona Enter para cerrar...")
