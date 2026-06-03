"""
geo_trajectory.py
=================
Sincronización telemetría ↔ frames para el Vuelo 17 Región I.
Interpolación lineal con np.interp para resamplear ~9.9 Hz → 29.97 fps.

Uso:
    python 02_sincronizacion/geo_trajectory.py

Entradas:
    video17_regionI.csv   — telemetría del dron

Salida:
    sync_table.csv        — 10,099 filas con GPS interpolado por frame
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Rutas ─────────────────────────────────────────────────────────────────────
PATH_TELEM = Path(r"C:\Users\javie\OneDrive\Documents\MCE Javier Alvarado\Cuarto Semestre\Consultoría\Georreferenciacion\video17_regionI.csv")
PATH_OUT   = Path(r"C:\Users\javie\OneDrive\Documents\MCE Javier Alvarado\Cuarto Semestre\Consultoría\Georreferenciacion\sync_table_v17.csv")

# ── Parámetros del video ───────────────────────────────────────────────────────
FPS          = 29.97
TOTAL_FRAMES = 10099
DURACION_S   = 337.0

print("=" * 60)
print("  Sincronización telemetría ↔ frames — Vuelo 17 Región I")
print("=" * 60)

# ── Cargar telemetría ─────────────────────────────────────────────────────────
print(f"\n[1/4] Cargando telemetría...")
df = pd.read_csv(PATH_TELEM)

df_video = df[df["isTakingVideo"] == 1].copy().reset_index(drop=True)
print(f"  Registros totales:          {len(df):,}")
print(f"  Registros durante video:    {len(df_video):,}")

# ── Tiempo relativo ───────────────────────────────────────────────────────────
print(f"\n[2/4] Calculando tiempo relativo...")
t0 = df_video["time(millisecond)"].iloc[0]
df_video["t_rel_s"] = (df_video["time(millisecond)"] - t0) / 1000.0

dt = df_video["t_rel_s"].diff().dropna()
print(f"  Intervalo mediana: {dt.median()*1000:.1f} ms")
print(f"  Intervalo máx:     {dt.max()*1000:.1f} ms")
saltos = (dt > 0.200).sum()
print(f"  Saltos >200ms:     {saltos} ({saltos/len(dt)*100:.1f}%)")

# ── Grilla de frames ──────────────────────────────────────────────────────────
print(f"\n[3/4] Generando grilla de {TOTAL_FRAMES:,} frames a {FPS} fps...")
frame_times = np.arange(TOTAL_FRAMES) / FPS

# ── Interpolación ─────────────────────────────────────────────────────────────
print(f"\n[4/4] Interpolando variables de telemetría...")

t_telem = df_video["t_rel_s"].values

cols = {
    "latitude":    "lat",
    "longitude":   "lon",
    "altitude(m)": "alt_m",
    "yaw(deg)":    "yaw_deg",
    "pitch(deg)":  "pitch_deg",
    "roll(deg)":   "roll_deg",
    "speed(mps)":  "speed_mps",
}

sync = {"frame_id": np.arange(TOTAL_FRAMES), "t_video_s": frame_times}

for col_telem, col_out in cols.items():
    vals = np.interp(frame_times, t_telem, df_video[col_telem].values)
    sync[col_out] = vals
    print(f"  {col_out:<12} rango: [{vals.min():.4f}, {vals.max():.4f}]")

sync["cum_dist_m"] = np.interp(frame_times, t_telem,
                                df_video["distance(m)"].values)

# ── Guardar ───────────────────────────────────────────────────────────────────
df_sync = pd.DataFrame(sync)
df_sync.to_csv(PATH_OUT, index=False, float_format="%.8f")

print(f"\n{'='*60}")
print(f"  Guardado: {PATH_OUT}")
print(f"  Filas:    {len(df_sync):,}")
print(f"{'='*60}")
print(f"\nPitch rango: {df_sync['pitch_deg'].min():.1f}° → {df_sync['pitch_deg'].max():.1f}°")
print(f"Vel. media:  {df_sync['speed_mps'].mean():.2f} m/s")

input("\nPresiona Enter para cerrar...")
