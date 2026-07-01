"""
Independent Validation — 2025 Data
====================================
Inputs:
  1. data/sample_table_2021_2024.csv   — training set (2021-2024)
  2. data/Landsat_Shanghai_2025.csv    — Landsat 2025 features
  3. data/MODIS_Shanghai_2025.csv      — MODIS 2025 LST labels
  4. data/station_2025.csv             — observed station air temperature

Workflow:
  Step 1  Re-train Physics-XGB on full 2021-2024 data
  Step 2  Build 2025 feature table (Landsat + MODIS spatial match)
  Step 3  Physics-XGB predicts LST for 2025 Landsat pixels
  Step 4  Aggregate to daily mean → compare with MODIS reference
  Step 5  Match with station air temperature (Ta) for cross-validation
  Step 6  Plot 6 figures

Output figures:
  fig_val1_timeseries.png   — daily LST time series (Pred vs MODIS vs Ta)
  fig_val2_scatter.png      — scatter: Pred vs MODIS
  fig_val3_scatter_ta.png   — scatter: Pred LST vs station Ta
  fig_val4_monthly.png      — monthly mean comparison
  fig_val5_residual.png     — residual analysis
  fig_val6_seasonal.png     — seasonal boxplot
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy import stats
from scipy.stats import norm as scipy_norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import json, re
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family']        = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi']        = 150

# ══════════════════════════════════════════════
# 0. Paths  ← EDIT THESE
# ══════════════════════════════════════════════
TRAIN_CSV   = Path(r"data\sample_table_2021_2024.csv")
L25_CSV     = Path(r"data\Landsat_Shanghai_2025.csv")
M25_CSV     = Path(r"data\MODIS_Shanghai_2025.csv")
STA_CSV     = Path(r"data\station_2025.csv")          # 上海.csv renamed
OUTPUT_DIR  = Path(r"outputs_validation")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ══════════════════════════════════════════════
# 1. Utility functions
# ══════════════════════════════════════════════
def parse_geo_fast(df):
    coords = df['.geo'].str.extract(r'\[([\d.]+),([\d.]+)\]')
    df = df.copy()
    df['lon'] = coords[0].astype(float)
    df['lat'] = coords[1].astype(float)
    return df

def compute_lse(ndvi, mndwi):
    fvc = ((ndvi - 0.2) / 0.3).clip(0, 1)
    lse = np.where(mndwi > 0, 0.991, 0.985*fvc + 0.970*(1-fvc))
    return lse, fvc

def compute_mwa(bt_K, lse, tau=0.60):
    a, b = -67.355351, 0.458606
    C  = lse * tau
    D  = (1-lse) * (1+(1-lse)) * tau
    Ta = 16.011 + 0.926 * bt_K
    return (a*(1-C-D) + (b*(1-C-D)+C+D)*bt_K - D*Ta) / C

def calc_metrics(y_true, y_pred, name=''):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    if name:
        print(f"  [{name}]  RMSE={rmse:.3f}  MAE={mae:.3f}"
              f"  R2={r2:.4f}  Bias={bias:+.3f}")
    return dict(name=name, RMSE=rmse, MAE=mae, R2=r2, Bias=bias)

FEATURES_PHYS = ['BT_B10','BT_diff','NDVI','NDBI','MNDWI',
                 'FVC','LSE','WVC','sin_month','cos_month',
                 'lat','lon','NDVI_sq','NDBI_sq','BT_NDVI','BT_NDBI',
                 'LST_MWA']
LABEL = 'label_K'

# ══════════════════════════════════════════════
# 2. Train Physics-XGB on full 2021-2024 data
# ══════════════════════════════════════════════
print("="*65)
print("Step 1: Training Physics-XGB on 2021-2024 ...")
df = pd.read_csv(TRAIN_CSV, parse_dates=['date'])
df['LSE'], df['FVC'] = compute_lse(df['NDVI'].values, df['MNDWI'].values)
df['LST_MWA']  = compute_mwa(df['BT_B10'].values, df['LSE'].values)
df['BT_diff']  = df['BT_B10'] - df.get('BT_B11', df['BT_B10'])
df['WVC']      = 2.0
df['NDVI_sq']  = df['NDVI']**2
df['NDBI_sq']  = df['NDBI']**2
df['BT_NDVI']  = df['BT_B10'] * df['NDVI']
df['BT_NDBI']  = df['BT_B10'] * df['NDBI']
df = df.dropna(subset=['BT_B10','NDVI','NDBI','MNDWI','label_K'])
df = df[(df['label_K'] > 260) & (df['label_K'] < 340)]
df = df[(df['LST_MWA'] > 255) & (df['LST_MWA'] < 345)]

n_tr = int(len(df)*0.90)
df_tr = df.iloc[:n_tr]; df_va = df.iloc[n_tr:]

model = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.03, max_depth=8,
    subsample=0.85, colsample_bytree=0.80,
    min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
    early_stopping_rounds=50, eval_metric='rmse',
    random_state=SEED, n_jobs=-1, verbosity=0)
model.fit(df_tr[FEATURES_PHYS], df_tr[LABEL],
          eval_set=[(df_va[FEATURES_PHYS], df_va[LABEL])], verbose=False)
print(f"  Training done. Best iter: {model.best_iteration}  "
      f"Samples used: {len(df):,}")

# ══════════════════════════════════════════════
# 3. Process Landsat 2025
# ══════════════════════════════════════════════
print("\nStep 2: Processing Landsat 2025 ...")
dl = pd.read_csv(L25_CSV, parse_dates=['date'])
dl = parse_geo_fast(dl)
dl['LSE'], dl['FVC'] = compute_lse(dl['NDVI'].values, dl['MNDWI'].values)
dl['LST_MWA']  = compute_mwa(dl['BT_B10'].values, dl['LSE'].values)
dl['BT_diff']  = 0.0          # Landsat 9 single band
dl['WVC']      = 2.0
dl['NDVI_sq']  = dl['NDVI']**2
dl['NDBI_sq']  = dl['NDBI']**2
dl['BT_NDVI']  = dl['BT_B10'] * dl['NDVI']
dl['BT_NDBI']  = dl['BT_B10'] * dl['NDBI']
dl['month']    = dl['date'].dt.month
dl['sin_month']= np.sin(2*np.pi*dl['month']/12)
dl['cos_month']= np.cos(2*np.pi*dl['month']/12)

# Quality filter
dl = dl.dropna(subset=['BT_B10','NDVI','NDBI','MNDWI'])
dl = dl[(dl['BT_B10'] > 260) & (dl['BT_B10'] < 340)]
dl = dl[(dl['LST_MWA'] > 255) & (dl['LST_MWA'] < 345)]
print(f"  Landsat 2025 valid pixels: {len(dl):,}  "
      f"Dates: {dl['date'].nunique()}")

# ── Physics-XGB prediction ────────────────────
dl['LST_pred_K'] = model.predict(dl[FEATURES_PHYS])
dl['LST_pred_C'] = dl['LST_pred_K'] - 273.15
dl['LST_mwa_C']  = dl['LST_MWA']   - 273.15

# ══════════════════════════════════════════════
# 4. Match MODIS 2025 as reference label
# ══════════════════════════════════════════════
print("\nStep 3: Matching MODIS 2025 as reference ...")
from scipy.spatial import cKDTree

dm = pd.read_csv(M25_CSV, parse_dates=['date'])
dm = parse_geo_fast(dm)
dm = dm.dropna(subset=['LST_MODIS_K'])
dm = dm[(dm['LST_MODIS_K'] > 260) & (dm['LST_MODIS_K'] < 340)]

modis_by_date = {d: g.reset_index(drop=True)
                 for d, g in dm.groupby('date')}

results = []
for date, lgrp in dl.groupby('date'):
    mgrp = modis_by_date.get(date)
    if mgrp is None or len(mgrp) == 0:
        continue
    tree = cKDTree(mgrp[['lat','lon']].values)
    dists, idxs = tree.query(lgrp[['lat','lon']].values, k=1)
    matched = lgrp.copy().reset_index(drop=True)
    matched['LST_modis_K'] = mgrp['LST_MODIS_K'].values[idxs]
    matched['LST_modis_C'] = mgrp['LST_MODIS_C'].values[idxs]
    matched['dist_deg']    = dists
    results.append(matched)

df25 = pd.concat(results, ignore_index=True)
df25 = df25[df25['dist_deg'] < 0.05]
df25 = df25.dropna(subset=['LST_modis_K','LST_pred_K'])
print(f"  Matched pixels: {len(df25):,}  Dates: {df25['date'].nunique()}")

# ══════════════════════════════════════════════
# 5. Daily aggregation
# ══════════════════════════════════════════════
print("\nStep 4: Aggregating to daily means ...")
daily = df25.groupby('date').agg(
    LST_pred_C  = ('LST_pred_C',  'mean'),
    LST_modis_C = ('LST_modis_C', 'mean'),
    LST_mwa_C   = ('LST_mwa_C',   'mean'),
    n_pixels    = ('LST_pred_C',  'count')
).reset_index()
daily['month'] = daily['date'].dt.month

# ── Load station data ─────────────────────────
sta = pd.read_csv(STA_CSV, encoding='gbk')
sta.columns = sta.columns.str.strip()
# find temp column
ta_col = [c for c in sta.columns if '气温' in c or 'Ta' in c][0]
sta['date'] = pd.to_datetime(dict(year=sta['年'], month=sta['月'], day=sta['日']))
sta['Ta_C']  = pd.to_numeric(sta[ta_col], errors='coerce')
sta = sta[['date','Ta_C']].dropna()

# Merge station Ta into daily
daily = daily.merge(sta, on='date', how='left')
daily_with_ta = daily.dropna(subset=['Ta_C'])
print(f"  Daily records total: {len(daily)}")
print(f"  Daily records with Ta: {len(daily_with_ta)}")

# ══════════════════════════════════════════════
# 6. Pixel-level metrics (Pred vs MODIS)
# ══════════════════════════════════════════════
print("\nStep 5: Accuracy metrics ...")
m_pred_vs_modis = calc_metrics(df25['LST_modis_K'].values,
                               df25['LST_pred_K'].values,
                               "Physics-XGB vs MODIS (pixel)")
m_mwa_vs_modis  = calc_metrics(df25['LST_modis_K'].values,
                               df25['LST_MWA'].values,
                               "MWA         vs MODIS (pixel)")

# Daily metrics
if len(daily) >= 5:
    m_daily = calc_metrics(daily['LST_modis_C'].values,
                           daily['LST_pred_C'].values,
                           "Physics-XGB vs MODIS (daily)")

# ══════════════════════════════════════════════
# FIGURE 1 — Time series: Pred vs MODIS vs Ta
# ══════════════════════════════════════════════
print("\nPlotting figures ...")

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=False)

# ── Panel A: full year Ta ──
ax = axes[0]
ax.plot(sta['date'], sta['Ta_C'], color='#1565C0', lw=1.3,
        label='Station Air Temp (Ta, °C)', alpha=0.85)
ax.fill_between(sta['date'], sta['Ta_C'], alpha=0.15, color='#1565C0')
ax.set_ylabel('Air Temp (°C)', fontsize=11)
ax.set_title('(a)  Station Air Temperature — 2025 (Daily, Station 58367)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3, linestyle='--')
ax.set_xlim(sta['date'].min(), sta['date'].max())

# ── Panel B: Landsat overpass days — Pred vs MODIS ──
ax = axes[1]
ax.scatter(daily['date'], daily['LST_modis_C'], c='#78909C', s=50,
           label='MODIS LST (reference)', zorder=3, alpha=0.85)
ax.scatter(daily['date'], daily['LST_pred_C'], c='#EF5350', s=50,
           marker='^', label='Physics-XGB LST (predicted)', zorder=4, alpha=0.85)
ax.scatter(daily['date'], daily['LST_mwa_C'], c='#FFA726', s=30,
           marker='s', label='MWA LST (physical)', zorder=2, alpha=0.65)
# connect Pred and MODIS with vertical lines
for _, row in daily.iterrows():
    ax.plot([row['date'], row['date']],
            [row['LST_pred_C'], row['LST_modis_C']],
            color='gray', lw=0.6, alpha=0.5)
ax.set_ylabel('LST (°C)', fontsize=11)
ax.set_title('(b)  LST on Landsat Overpass Days — Physics-XGB vs MODIS',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, ncol=3); ax.grid(alpha=0.3, linestyle='--')

# ── Panel C: Ta vs Pred LST on matched days ──
ax = axes[2]
if len(daily_with_ta) > 0:
    ax.plot(daily_with_ta['date'], daily_with_ta['Ta_C'],
            'o-', color='#1565C0', lw=1.5, ms=5,
            label='Station Ta (°C)', alpha=0.85)
    ax.plot(daily_with_ta['date'], daily_with_ta['LST_pred_C'],
            '^-', color='#EF5350', lw=1.5, ms=5,
            label='Physics-XGB LST (°C)', alpha=0.85)
    ax.plot(daily_with_ta['date'], daily_with_ta['LST_modis_C'],
            's--', color='#78909C', lw=1.2, ms=4,
            label='MODIS LST (°C)', alpha=0.70)
ax.set_ylabel('Temperature (°C)', fontsize=11)
ax.set_title('(c)  Ta vs Predicted LST on Landsat Overpass Days',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, ncol=3); ax.grid(alpha=0.3, linestyle='--')

fig.suptitle('Independent Validation — 2025 Station & Remote Sensing Data',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/'fig_val1_timeseries.png', dpi=300, bbox_inches='tight')
plt.close(); print("  fig_val1 saved")

# ══════════════════════════════════════════════
# FIGURE 2 — Scatter: Physics-XGB vs MODIS (pixel & daily)
# ══════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (x, y, m, title, color) in zip(axes, [
    (df25['LST_modis_C'].values, df25['LST_pred_C'].values,
     m_pred_vs_modis,
     'Physics-XGB vs MODIS (pixel-level)', '#EF5350'),
    (daily['LST_modis_C'].values, daily['LST_pred_C'].values,
     m_daily if len(daily)>=5 else m_pred_vs_modis,
     'Physics-XGB vs MODIS (daily mean)', '#EF5350'),
]):
    n_plot = min(3000, len(x))
    idx = np.random.choice(len(x), n_plot, replace=False)
    ax.scatter(x[idx], y[idx], c=color, alpha=0.35, s=12,
               edgecolors='none', rasterized=True)
    lo = min(x.min(), y.min()) - 1
    hi = max(x.max(), y.max()) + 1
    ax.plot([lo,hi],[lo,hi],'k--',lw=1.5,label='1:1 line')
    sl, ic, *_ = stats.linregress(x, y)
    xf = np.linspace(lo, hi, 200)
    ax.plot(xf, sl*xf+ic, color=color, lw=2,
            label=f'y={sl:.3f}x+{ic:.2f}')
    txt = (f"n = {len(x):,}\n"
           f"RMSE = {m['RMSE']:.3f} K\n"
           f"MAE  = {m['MAE']:.3f} K\n"
           f"R\u00b2   = {m['R2']:.4f}\n"
           f"Bias = {m['Bias']:+.3f} K")
    ax.text(0.04,0.97,txt,transform=ax.transAxes,
            va='top',ha='left',fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4',facecolor='white',
                      edgecolor=color,alpha=0.9))
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_aspect('equal')
    ax.set_xlabel('MODIS LST Reference (°C)', fontsize=11)
    ax.set_ylabel('Physics-XGB Predicted LST (°C)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.2, linestyle='--')

fig.suptitle('Validation Scatter — Physics-XGB vs MODIS (2025)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR/'fig_val2_scatter_modis.png', dpi=300, bbox_inches='tight')
plt.close(); print("  fig_val2 saved")

# ══════════════════════════════════════════════
# FIGURE 3 — Scatter: Pred LST vs Station Ta
# ══════════════════════════════════════════════
if len(daily_with_ta) >= 5:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Pred vs Ta
    ax = axes[0]
    x2 = daily_with_ta['Ta_C'].values
    y2 = daily_with_ta['LST_pred_C'].values
    ax.scatter(x2, y2, c='#EF5350', s=60, alpha=0.8,
               edgecolors='white', linewidth=0.5, zorder=3)
    sl2,ic2,r2_v,*_ = stats.linregress(x2, y2)
    lo2 = min(x2.min(),y2.min())-1; hi2 = max(x2.max(),y2.max())+1
    xf2 = np.linspace(lo2, hi2, 200)
    ax.plot([lo2,hi2],[lo2,hi2],'k--',lw=1.5,label='1:1 line')
    ax.plot(xf2, sl2*xf2+ic2,'r-',lw=2,
            label=f'Fit: y={sl2:.3f}x+{ic2:.2f}')
    txt2 = (f"n = {len(x2)}\n"
            f"R\u00b2 = {r2_v**2:.4f}\n"
            f"Slope = {sl2:.3f}\n"
            f"Note: LST > Ta\n(surface heating)")
    ax.text(0.04,0.97,txt2,transform=ax.transAxes,
            va='top',ha='left',fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4',facecolor='white',
                      edgecolor='#EF5350',alpha=0.9))
    ax.set_xlabel('Station Air Temperature Ta (°C)', fontsize=11)
    ax.set_ylabel('Physics-XGB Predicted LST (°C)', fontsize=11)
    ax.set_title('Predicted LST vs Station Ta\n(Landsat Overpass Days)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.2,linestyle='--')

    # Right: MODIS vs Ta
    ax = axes[1]
    x3 = daily_with_ta['Ta_C'].values
    y3 = daily_with_ta['LST_modis_C'].values
    ax.scatter(x3, y3, c='#78909C', s=60, alpha=0.8,
               edgecolors='white', linewidth=0.5, zorder=3)
    sl3,ic3,r3_v,*_ = stats.linregress(x3, y3)
    lo3 = min(x3.min(),y3.min())-1; hi3 = max(x3.max(),y3.max())+1
    xf3 = np.linspace(lo3, hi3, 200)
    ax.plot([lo3,hi3],[lo3,hi3],'k--',lw=1.5,label='1:1 line')
    ax.plot(xf3, sl3*xf3+ic3,'gray',lw=2,
            label=f'Fit: y={sl3:.3f}x+{ic3:.2f}')
    txt3 = (f"n = {len(x3)}\n"
            f"R\u00b2 = {r3_v**2:.4f}\n"
            f"Slope = {sl3:.3f}")
    ax.text(0.04,0.97,txt3,transform=ax.transAxes,
            va='top',ha='left',fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4',facecolor='white',
                      edgecolor='#78909C',alpha=0.9))
    ax.set_xlabel('Station Air Temperature Ta (°C)', fontsize=11)
    ax.set_ylabel('MODIS LST (°C)', fontsize=11)
    ax.set_title('MODIS LST vs Station Ta\n(Landsat Overpass Days)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.2,linestyle='--')

    fig.suptitle('LST vs Station Air Temperature Cross-Validation — 2025',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/'fig_val3_scatter_ta.png', dpi=300, bbox_inches='tight')
    plt.close(); print("  fig_val3 saved")

# ══════════════════════════════════════════════
# FIGURE 4 — Monthly mean comparison
# ══════════════════════════════════════════════
monthly_sta = sta.groupby(sta['date'].dt.month)['Ta_C'].mean()

daily['month'] = daily['date'].dt.month
monthly_pred  = daily.groupby('month')['LST_pred_C'].mean()
monthly_modis = daily.groupby('month')['LST_modis_C'].mean()

fig, ax = plt.subplots(figsize=(13, 5.5))
months_all = range(1, 13)
mlbl = ['Jan','Feb','Mar','Apr','May','Jun',
        'Jul','Aug','Sep','Oct','Nov','Dec']

ax.plot(mlbl, [monthly_sta.get(m, np.nan) for m in months_all],
        'o-', color='#1565C0', lw=2, ms=7, label='Station Ta (°C)', zorder=3)
ax.plot(mlbl, [monthly_pred.get(m, np.nan) for m in months_all],
        '^-', color='#EF5350', lw=2, ms=7,
        label='Physics-XGB LST (°C)', zorder=4)
ax.plot(mlbl, [monthly_modis.get(m, np.nan) for m in months_all],
        's--', color='#78909C', lw=1.8, ms=6,
        label='MODIS LST (°C)', zorder=2)

ax.fill_between(mlbl,
    [monthly_pred.get(m, np.nan) for m in months_all],
    [monthly_sta.get(m, np.nan) for m in months_all],
    alpha=0.12, color='#EF5350',
    label='LST–Ta offset (surface heating)')

ax.set_ylabel('Temperature (°C)', fontsize=11)
ax.set_xlabel('Month', fontsize=11)
ax.set_title('Monthly Mean Temperature Comparison — 2025',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(OUTPUT_DIR/'fig_val4_monthly.png', dpi=300, bbox_inches='tight')
plt.close(); print("  fig_val4 saved")

# ══════════════════════════════════════════════
# FIGURE 5 — Residual analysis (Pred vs MODIS, pixel)
# ══════════════════════════════════════════════
residual = df25['LST_pred_K'].values - df25['LST_modis_K'].values
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.scatter(df25['LST_pred_C'].values, residual,
           c='#EF5350', alpha=0.2, s=8, edgecolors='none', rasterized=True)
ax.axhline(0, color='black', lw=1.5, linestyle='--')
ax.set_xlabel('Predicted LST (°C)', fontsize=11)
ax.set_ylabel('Residual (K)', fontsize=11)
ax.set_title('Residual vs. Predicted — 2025 Validation',
             fontsize=12, fontweight='bold')
ax.grid(alpha=0.25, linestyle='--')

ax = axes[1]
ax.hist(residual, bins=60, color='#EF5350', edgecolor='white',
        alpha=0.85, density=True)
mu, std = residual.mean(), residual.std()
xr = np.linspace(residual.min(), residual.max(), 200)
ax.plot(xr, scipy_norm.pdf(xr, mu, std), 'k-', lw=2,
        label=f'Normal fit\n\u03bc={mu:+.3f} K\n\u03c3={std:.3f} K')
ax.axvline(0, color='navy', lw=1.5, linestyle='--')
ax.set_xlabel('Residual (K)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Residual Distribution — 2025 Validation',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.25, linestyle='--')

plt.tight_layout()
plt.savefig(OUTPUT_DIR/'fig_val5_residual.png', dpi=300, bbox_inches='tight')
plt.close(); print("  fig_val5 saved")

# ══════════════════════════════════════════════
# FIGURE 6 — Seasonal boxplot
# ══════════════════════════════════════════════
seasons = {
    'Spring\n(MAM)': [3,4,5],
    'Summer\n(JJA)': [6,7,8],
    'Autumn\n(SON)': [9,10,11],
    'Winter\n(DJF)': [12,1,2],
}
df25['month_num'] = df25['date'].dt.month

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
titles = ['Physics-XGB LST','MODIS LST','Prediction Error (Pred−MODIS)']
cols   = ['LST_pred_C','LST_modis_C','error']
df25['error'] = df25['LST_pred_K'] - df25['LST_modis_K']
colors = ['#EF5350','#78909C','#FFA726']

for ax, col, title, color in zip(axes, cols, titles, colors):
    boxes, xlbls = [], []
    for sn, mlist in seasons.items():
        idx = df25['month_num'].isin(mlist)
        if idx.sum() == 0: continue
        boxes.append(df25.loc[idx, col].values)
        xlbls.append(sn)
    if not boxes: continue
    bp = ax.boxplot(boxes, labels=xlbls, patch_artist=True,
                    medianprops=dict(color='black', lw=2),
                    flierprops=dict(marker='o', markersize=2,
                                    alpha=0.3, linestyle='none'),
                    whis=1.5)
    for patch in bp['boxes']:
        patch.set_facecolor(color); patch.set_alpha(0.75)
    if col == 'error':
        ax.axhline(0, color='black', lw=1.3, linestyle='--', alpha=0.7)
    ax.set_title(title, fontsize=12, fontweight='bold', color=color)
    ax.set_ylabel('Temperature (°C)' if col != 'error' else 'Error (K)',
                  fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

fig.suptitle('Seasonal Distribution — 2025 Independent Validation',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR/'fig_val6_seasonal.png', dpi=300, bbox_inches='tight')
plt.close(); print("  fig_val6 saved")

# ══════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════
print("\n" + "="*65)
print("2025 Independent Validation Summary")
print("="*65)
print(f"  Landsat pixels validated : {len(df25):,}")
print(f"  Overpass dates           : {df25['date'].nunique()}")
print(f"  Station days matched     : {len(daily_with_ta)}")
print()
print(f"  Physics-XGB vs MODIS (pixel) :")
print(f"    RMSE = {m_pred_vs_modis['RMSE']:.3f} K")
print(f"    MAE  = {m_pred_vs_modis['MAE']:.3f} K")
print(f"    R2   = {m_pred_vs_modis['R2']:.4f}")
print(f"    Bias = {m_pred_vs_modis['Bias']:+.3f} K")
print()
print(f"  MWA vs MODIS (pixel) :")
print(f"    RMSE = {m_mwa_vs_modis['RMSE']:.3f} K")
print(f"    MAE  = {m_mwa_vs_modis['MAE']:.3f} K")
print(f"    R2   = {m_mwa_vs_modis['R2']:.4f}")
print(f"    Bias = {m_mwa_vs_modis['Bias']:+.3f} K")
print("="*65)
print(f"\nAll figures saved to: {OUTPUT_DIR.resolve()}")