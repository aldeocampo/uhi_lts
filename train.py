"""
LST Retrieval - Training, Validation & Comparison Experiment
Methods compared:
  1. Mono-Window Algorithm (MWA)        - Traditional physical method
  2. Random Forest (RF)                 - Ensemble ML
  3. Gradient Boosting (GBDT)           - Ensemble ML
  4. Support Vector Regression (SVR)    - Kernel-based ML
  5. Pure XGBoost (Pure-XGB)            - Data-driven, no physics
  6. Physics-Constrained XGBoost        - Proposed method (OURS)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ── Font settings (English only, no Chinese font needed) ──
plt.rcParams['font.family']        = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi']        = 300

# ══════════════════════════════════════════════
# 0. Path Configuration
# ══════════════════════════════════════════════
DATA_PATH  = Path(r"data\sample_table_2021_2024.csv")
OUTPUT_DIR = Path(r"outputs2")
OUTPUT_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════
# 1. Load Data
# ══════════════════════════════════════════════
print("=" * 60)
print("Loading sample data...")
df = pd.read_csv(DATA_PATH, parse_dates=['date'])
print(f"Total samples: {len(df):,}")
print(f"Year distribution:\n{df['date'].dt.year.value_counts().sort_index()}")

# ══════════════════════════════════════════════
# 2. Feature Engineering
# ══════════════════════════════════════════════
print("\nComputing physical prior feature LST_MWA...")

def compute_lse(ndvi, mndwi):
    """Three-component NDVI threshold method for LSE estimation"""
    ndvi_s, ndvi_v = 0.2, 0.5
    fvc = ((ndvi - ndvi_s) / (ndvi_v - ndvi_s)).clip(0, 1)
    eps_v, eps_s, eps_w = 0.985, 0.970, 0.991
    lse = np.where(mndwi > 0, eps_w,
                   eps_v * fvc + eps_s * (1 - fvc))
    return lse, fvc

def compute_mwa(bt, lse, tau=0.70):
    """Mono-Window Algorithm (Qin et al., 2001)"""
    a, b = -67.355351, 0.458606
    C  = lse * tau
    D  = (1 - lse) * (1 + (1 - lse)) * tau
    Ta = 16.011 + 0.926 * bt
    return (a * (1 - C - D) + (b * (1 - C - D) + C + D) * bt - D * Ta) / C

df['LSE'], df['FVC'] = compute_lse(df['NDVI'].values, df['MNDWI'].values)
df['LST_MWA'] = compute_mwa(df['BT_B10'].values, df['LSE'].values)
df['BT_diff'] = df['BT_B10'] - df.get('BT_B11', df['BT_B10'])
df['WVC']     = 2.0   # Typical WVC for Shanghai; replace with actual if available

# Quality filtering
df = df.dropna(subset=['BT_B10', 'NDVI', 'NDBI', 'MNDWI', 'label_K'])
df = df[(df['label_K']  > 260) & (df['label_K']  < 340)]
df = df[(df['LST_MWA']  > 260) & (df['LST_MWA']  < 340)]
print(f"Samples after quality filtering: {len(df):,}")

# ══════════════════════════════════════════════
# 3. Feature Sets
# ══════════════════════════════════════════════
# Full feature set with physical prior (Physics-XGB)
FEATURES_PHYS = ['BT_B10', 'BT_diff', 'NDVI', 'NDBI', 'MNDWI',
                 'FVC', 'LSE', 'WVC', 'sin_month', 'cos_month',
                 'lat', 'lon', 'LST_MWA']

# Feature set without physical prior (RF, GBDT, SVR, Pure-XGB)
FEATURES_PURE = ['BT_B10', 'BT_diff', 'NDVI', 'NDBI', 'MNDWI',
                 'FVC', 'LSE', 'WVC', 'sin_month', 'cos_month',
                 'lat', 'lon']

LABEL = 'label_K'

# ══════════════════════════════════════════════
# 4. Chronological Train/Val/Test Split
# ══════════════════════════════════════════════
df_sorted = df.sort_values('date').reset_index(drop=True)
n       = len(df_sorted)
n_train = int(n * 0.70)
n_val   = int(n * 0.85)

df_train = df_sorted.iloc[:n_train]
df_val   = df_sorted.iloc[n_train:n_val]
df_test  = df_sorted.iloc[n_val:]

print(f"\nData split | Train: {len(df_train):,}  "
      f"Val: {len(df_val):,}  Test: {len(df_test):,}")
print(f"Train period: {df_train['date'].min().date()} "
      f"~ {df_train['date'].max().date()}")
print(f"Test  period: {df_test['date'].min().date()} "
      f"~ {df_test['date'].max().date()}")

X_tr_p = df_train[FEATURES_PHYS];  X_va_p = df_val[FEATURES_PHYS]
X_te_p = df_test[FEATURES_PHYS]
X_tr_u = df_train[FEATURES_PURE];  X_va_u = df_val[FEATURES_PURE]
X_te_u = df_test[FEATURES_PURE]
y_tr   = df_train[LABEL]
y_va   = df_val[LABEL]
y_te   = df_test[LABEL]

# ══════════════════════════════════════════════
# 5. Metrics Helper
# ══════════════════════════════════════════════
def compute_metrics(y_true, y_pred, name=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    if name:
        print(f"  [{name:<28s}]  RMSE={rmse:.3f}K  "
              f"MAE={mae:.3f}K  R2={r2:.4f}  Bias={bias:+.3f}K")
    return {'name': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Bias': bias}

# ══════════════════════════════════════════════
# 6. Method 1 — Mono-Window Algorithm (MWA)
# ══════════════════════════════════════════════
print("\n" + "─" * 60)
print("Training all models...")
print()
y_mwa  = df_test['LST_MWA'].values
m_mwa  = compute_metrics(y_te.values, y_mwa, "Mono-Window Alg. (MWA)")

# ══════════════════════════════════════════════
# 7. Method 2 — Random Forest (RF)
# ══════════════════════════════════════════════
print("  Fitting Random Forest...", end='', flush=True)
rf = RandomForestRegressor(
    n_estimators=300, max_depth=12, min_samples_leaf=5,
    random_state=42, n_jobs=-1)
rf.fit(X_tr_u, y_tr)
y_rf  = rf.predict(X_te_u)
m_rf  = compute_metrics(y_te.values, y_rf, "Random Forest (RF)")

# ══════════════════════════════════════════════
# 8. Method 3 — Gradient Boosting (GBDT)
# ══════════════════════════════════════════════
print("  Fitting Gradient Boosting...", end='', flush=True)
gbdt = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, min_samples_leaf=5, random_state=42)
gbdt.fit(X_tr_u, y_tr)
y_gbdt = gbdt.predict(X_te_u)
m_gbdt = compute_metrics(y_te.values, y_gbdt, "Gradient Boosting (GBDT)")

# ══════════════════════════════════════════════
# 9. Method 4 — Support Vector Regression (SVR)
# ══════════════════════════════════════════════
print("  Fitting SVR (may take ~2 min)...", end='', flush=True)
# Use Pipeline with StandardScaler (SVR is scale-sensitive)
svr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svr',    SVR(kernel='rbf', C=10, epsilon=0.1, gamma='scale'))
])
# SVR is slow on large data; subsample training set to 8000
np.random.seed(42)
idx_svr  = np.random.choice(len(X_tr_u), size=min(8000, len(X_tr_u)),
                             replace=False)
svr_pipe.fit(X_tr_u.iloc[idx_svr], y_tr.iloc[idx_svr])
y_svr  = svr_pipe.predict(X_te_u)
m_svr  = compute_metrics(y_te.values, y_svr, "Support Vector Reg. (SVR)")

# ══════════════════════════════════════════════
# 10. Method 5 — Pure XGBoost (no physics prior)
# ══════════════════════════════════════════════
print("  Fitting Pure-XGBoost...", end='', flush=True)
xgb_pure = xgb.XGBRegressor(
    n_estimators=800, learning_rate=0.04, max_depth=7,
    subsample=0.8, colsample_bytree=0.75,
    min_child_weight=5, reg_alpha=0.2, reg_lambda=1.5,
    early_stopping_rounds=40, eval_metric='rmse',
    random_state=42, n_jobs=-1, verbosity=0)
xgb_pure.fit(X_tr_u, y_tr,
             eval_set=[(X_va_u, y_va)], verbose=False)
y_pure = xgb_pure.predict(X_te_u)
m_pure = compute_metrics(y_te.values, y_pure, "Pure XGBoost (Pure-XGB)")

# ══════════════════════════════════════════════
# 11. Method 6 — Physics-Constrained XGBoost (OURS)
# ══════════════════════════════════════════════
print("  Fitting Physics-XGB (OURS)...", end='', flush=True)
xgb_phys = xgb.XGBRegressor(
    n_estimators=800, learning_rate=0.04, max_depth=7,
    subsample=0.8, colsample_bytree=0.75,
    min_child_weight=5, reg_alpha=0.2, reg_lambda=1.5,
    early_stopping_rounds=40, eval_metric='rmse',
    random_state=42, n_jobs=-1, verbosity=0)
xgb_phys.fit(X_tr_p, y_tr,
             eval_set=[(X_va_p, y_va)], verbose=False)
y_phys = xgb_phys.predict(X_te_p)
m_phys = compute_metrics(y_te.values, y_phys, "Physics-XGB (OURS) *")

all_metrics = [m_mwa, m_rf, m_gbdt, m_svr, m_pure, m_phys]

# ══════════════════════════════════════════════
# 12. Plot 1 — Scatter Plots (2 x 3 grid)
# ══════════════════════════════════════════════
METHODS = [
    ("MWA",           '#78909C', y_mwa),
    ("RF",            '#42A5F5', y_rf),
    ("GBDT",          '#26C6DA', y_gbdt),
    ("SVR",           '#AB47BC', y_svr),
    ("Pure-XGB",      '#FFA726', y_pure),
    ("Physics-XGB\n(OURS)", '#EF5350', y_phys),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for ax, (label, color, y_pred), m in zip(axes, METHODS, all_metrics):
    y_true = y_te.values

    # Random subsample for plotting speed
    n_plot = min(2500, len(y_true))
    rng    = np.random.default_rng(0)
    idx    = rng.choice(len(y_true), size=n_plot, replace=False)
    yt, yp = y_true[idx], y_pred[idx]

    ax.scatter(yt, yp, c=color, alpha=0.35, s=14,
               edgecolors='none', rasterized=True)

    lim_lo = min(yt.min(), yp.min()) - 1
    lim_hi = max(yt.max(), yp.max()) + 1

    # 1:1 reference line
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            'k--', lw=1.5, alpha=0.8, label='1:1 line')

    # Regression fit line
    sl, ic, *_ = stats.linregress(y_true, y_pred)
    xf = np.linspace(lim_lo, lim_hi, 200)
    ax.plot(xf, sl * xf + ic, color=color, lw=2.0,
            label=f'y = {sl:.3f}x + {ic:.2f}')

    # Stats text box
    txt = (f"RMSE = {m['RMSE']:.3f} K\n"
           f"MAE  = {m['MAE']:.3f} K\n"
           f"R\u00b2   = {m['R2']:.4f}\n"
           f"Bias = {m['Bias']:+.3f} K")
    ax.text(0.04, 0.97, txt, transform=ax.transAxes,
            va='top', ha='left', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='white', edgecolor=color, alpha=0.9))

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect('equal')
    ax.set_xlabel('MODIS LST Reference (K)', fontsize=11)
    ax.set_ylabel('Predicted LST (K)', fontsize=11)

    title_color = '#B71C1C' if 'OURS' in label else '#333333'
    ax.set_title(label, fontsize=13, fontweight='bold', color=title_color)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.2, linestyle='--')

fig.suptitle('LST Retrieval Accuracy Comparison — Test Set  (Shanghai, 2021-2024)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig1_scatter_comparison.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("\nFig 1 saved -> fig1_scatter_comparison.png")

# ══════════════════════════════════════════════
# 13. Plot 2 — Accuracy Metrics Bar Chart
# ══════════════════════════════════════════════
short_labels  = ['MWA', 'RF', 'GBDT', 'SVR', 'Pure-XGB', 'Physics-XGB\n(OURS)']
bar_colors    = ['#78909C','#42A5F5','#26C6DA','#AB47BC','#FFA726','#EF5350']
metric_keys   = ['RMSE', 'MAE', 'R2', 'Bias']
metric_titles = ['RMSE (K)', 'MAE (K)', 'R\u00b2', 'Bias (K)']

fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))

for ax, mkey, mtitle in zip(axes, metric_keys, metric_titles):
    vals = [m[mkey] for m in all_metrics]
    bars = ax.bar(short_labels, vals, color=bar_colors,
                  width=0.6, edgecolor='white', linewidth=0.8)

    # Highlight Physics-XGB
    bars[-1].set_edgecolor('#7B1FA2' if mkey == 'R2' else '#B71C1C')
    bars[-1].set_linewidth(2.5)

    # Value labels
    for bar, v in zip(bars, vals):
        ypos = bar.get_height() + abs(max(vals) - min(vals)) * 0.02
        if mkey == 'Bias' and v < 0:
            ypos = bar.get_height() - abs(max(vals) - min(vals)) * 0.08
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f'{v:.3f}', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold')

    ax.set_title(mtitle, fontsize=12, fontweight='bold')
    ax.set_ylabel(mtitle, fontsize=10)
    ax.tick_params(axis='x', labelsize=8.5, rotation=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    if mkey == 'R2':
        vmin = min(vals)
        ax.set_ylim(max(0, vmin - 0.1), 1.05)
    elif mkey == 'Bias':
        yabs = max(abs(v) for v in vals)
        ax.set_ylim(-yabs * 1.5, yabs * 1.5)
        ax.axhline(0, color='black', lw=1.2, linestyle='-')

fig.suptitle('Accuracy Metrics Comparison — Test Set',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig2_metrics_bar.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Fig 2 saved -> fig2_metrics_bar.png")

# ══════════════════════════════════════════════
# 14. Plot 3 — Feature Importance (Physics-XGB)
# ══════════════════════════════════════════════
importance  = xgb_phys.feature_importances_
idx_sorted  = np.argsort(importance)
feat_labels = FEATURES_PHYS

fi_colors = ['#EF5350' if 'LST_MWA' in feat_labels[i]
             else '#90CAF9' for i in idx_sorted]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh([feat_labels[i] for i in idx_sorted],
               importance[idx_sorted],
               color=fi_colors, edgecolor='white', height=0.65)

# Annotate physical prior bar
for bar, fname in zip(bars, [feat_labels[j] for j in idx_sorted]):
    if fname == 'LST_MWA':
        ax.text(bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                '<-- Physical Prior (Innovation)',
                va='center', fontsize=9.5,
                color='#B71C1C', fontweight='bold')

from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor='#EF5350', label='Physical prior (LST_MWA)'),
    Patch(facecolor='#90CAF9', label='Other features'),
]
ax.legend(handles=legend_els, fontsize=10, loc='lower right')
ax.set_xlabel('Feature Importance (Gain)', fontsize=11)
ax.set_title('Physics-XGB Feature Importance\n'
             '(Red = Physics-constrained prior)',
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig3_feature_importance.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Fig 3 saved -> fig3_feature_importance.png")

# ══════════════════════════════════════════════
# 15. Plot 4 — Error Distribution Boxplot
# ══════════════════════════════════════════════
errors = {
    'MWA':              y_mwa   - y_te.values,
    'RF':               y_rf    - y_te.values,
    'GBDT':             y_gbdt  - y_te.values,
    'SVR':              y_svr   - y_te.values,
    'Pure-XGB':         y_pure  - y_te.values,
    'Physics-XGB\n(OURS)': y_phys - y_te.values,
}

fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot(list(errors.values()),
                labels=list(errors.keys()),
                patch_artist=True,
                medianprops=dict(color='black', lw=2.0),
                flierprops=dict(marker='o', markersize=2,
                                alpha=0.3, linestyle='none'),
                whis=1.5)

for patch, color in zip(bp['boxes'], bar_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.78)

ax.axhline(0, color='black', lw=1.3, linestyle='--', alpha=0.7)
ax.set_ylabel('Prediction Error (K)', fontsize=11)
ax.set_xlabel('Method', fontsize=11)
ax.set_title('Error Distribution Comparison — Test Set',
             fontsize=12, fontweight='bold')
ax.tick_params(axis='x', labelsize=9.5)
ax.grid(axis='y', alpha=0.25, linestyle='--')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig4_error_boxplot.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Fig 4 saved -> fig4_error_boxplot.png")

# ══════════════════════════════════════════════
# 16. Plot 5 — Radar Chart (综合对比)
# ══════════════════════════════════════════════
from matplotlib.patches import FancyArrowPatch

# Normalize metrics for radar (higher = better for all axes)
rmse_vals = np.array([m['RMSE'] for m in all_metrics])
mae_vals  = np.array([m['MAE']  for m in all_metrics])
r2_vals   = np.array([m['R2']   for m in all_metrics])
bias_vals = np.abs([m['Bias']   for m in all_metrics])

def normalize_inv(arr):   # lower is better -> invert
    mn, mx = arr.min(), arr.max()
    return 1 - (arr - mn) / (mx - mn + 1e-9)

def normalize(arr):       # higher is better
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)

radar_data = np.column_stack([
    normalize_inv(rmse_vals),   # RMSE: lower better
    normalize_inv(mae_vals),    # MAE:  lower better
    normalize(r2_vals),         # R2:   higher better
    normalize_inv(bias_vals),   # |Bias|: lower better
])

categories = ['RMSE\n(lower better)', 'MAE\n(lower better)',
              'R2\n(higher better)', '|Bias|\n(lower better)']
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 8),
                       subplot_kw=dict(polar=True))
method_names = ['MWA', 'RF', 'GBDT', 'SVR', 'Pure-XGB', 'Physics-XGB (OURS)']

for i, (name, color, data_row) in enumerate(
        zip(method_names, bar_colors, radar_data)):
    vals = data_row.tolist() + data_row[:1].tolist()
    lw   = 2.8 if 'OURS' in name else 1.4
    ls   = '-' if 'OURS' in name else '--'
    a    = 0.20 if 'OURS' in name else 0.08
    ax.plot(angles, vals, color=color, lw=lw, ls=ls, label=name)
    ax.fill(angles, vals, color=color, alpha=a)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_yticks([0.25, 0.50, 0.75, 1.00])
ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=8)
ax.set_ylim(0, 1)
ax.set_title('Comprehensive Performance Comparison\n(Normalized Score, higher = better)',
             fontsize=12, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.38, 1.12),
          fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig5_radar_comparison.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Fig 5 saved -> fig5_radar_comparison.png")

# ══════════════════════════════════════════════
# 17. Summary Table
# ══════════════════════════════════════════════
print("\n" + "=" * 72)
print("Accuracy Summary Table (Test Set)")
print("=" * 72)
print(f"{'Method':<32} {'RMSE(K)':>8} {'MAE(K)':>8} {'R2':>8} {'Bias(K)':>10}")
print("-" * 72)
for m in all_metrics:
    star = "  <-- OURS" if "OURS" in m['name'] else ""
    print(f"{m['name']+star:<32} {m['RMSE']:>8.3f} {m['MAE']:>8.3f} "
          f"{m['R2']:>8.4f} {m['Bias']:>+10.3f}")
print("=" * 72)
print(f"\nAll figures saved to: {OUTPUT_DIR.resolve()}")