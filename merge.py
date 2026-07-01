import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

DATA_DIR = Path("data")  # ← 改成你的文件夹路径

# ── 1. 快速解析.geo列（比json.loads快10倍）──
def parse_geo_fast(df):
    coords = df['.geo'].str.extract(r'\[([\d.]+),([\d.]+)\]')
    df['lon'] = coords[0].astype(float)
    df['lat'] = coords[1].astype(float)
    return df

# ── 2. 处理单个Landsat文件 ──────────────────
def process_landsat(csv_path):
    print(f"  读取 {csv_path.name}...", end='', flush=True)
    df = pd.read_csv(csv_path)
    df = parse_geo_fast(df)
    df['LST_C']     = df['BT_B10'] - 273.15
    df['date']      = pd.to_datetime(df['date'])
    df['year']      = df['date'].dt.year
    df['month']     = df['date'].dt.month
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df = df.drop(columns=['system:index', '.geo'], errors='ignore')
    df = df.dropna(subset=['lon', 'lat', 'BT_B10', 'NDVI'])
    df = df[(df['LST_C'] >= -5) & (df['LST_C'] <= 65)]
    print(f" {len(df):,} 条")
    return df

# ── 3. 处理单个MODIS文件 ──────────────────
def process_modis(csv_path):
    print(f"  读取 {csv_path.name}...", end='', flush=True)
    df = pd.read_csv(csv_path)
    df = parse_geo_fast(df)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop(columns=['system:index', '.geo'], errors='ignore')
    df = df.dropna(subset=['lon', 'lat', 'LST_MODIS_K'])
    df = df[(df['LST_MODIS_C'] >= -5) & (df['LST_MODIS_C'] <= 65)]
    print(f" {len(df):,} 条")
    return df

# ── 4. 合并Landsat ──────────────────────────
print("=== 合并Landsat数据 ===")
landsat_files = sorted(DATA_DIR.glob("Landsat_Shanghai_202*.csv"))
if not landsat_files:
    raise FileNotFoundError(f"在 {DATA_DIR} 下找不到 Landsat_Shanghai_202*.csv，请检查路径")

landsat_list = [process_landsat(f) for f in landsat_files]
landsat_all  = pd.concat(landsat_list, ignore_index=True)
print(f"Landsat总样本数: {len(landsat_all):,}")
print(f"年份分布:\n{landsat_all['year'].value_counts().sort_index()}\n")

# ── 5. 合并MODIS ────────────────────────────
print("=== 合并MODIS数据 ===")
modis_files = sorted(DATA_DIR.glob("MODIS_Shanghai_202*.csv"))
if not modis_files:
    raise FileNotFoundError(f"在 {DATA_DIR} 下找不到 MODIS_Shanghai_202*.csv，请检查路径")

modis_list = [process_modis(f) for f in modis_files]
modis_all  = pd.concat(modis_list, ignore_index=True)
print(f"MODIS总样本数: {len(modis_all):,}\n")

# ── 6. 时空匹配（KD树，带进度）──────────────
print("=== 开始时空匹配 ===")

# 按日期建立MODIS索引，避免每次重复过滤
modis_by_date = {date: grp.reset_index(drop=True)
                 for date, grp in modis_all.groupby('date')}

landsat_dates = landsat_all['date'].unique()
total = len(landsat_dates)
print(f"Landsat共 {total} 个日期需要匹配")

results = []
for i, date in enumerate(sorted(landsat_dates), 1):
    if i % 20 == 0 or i == total:
        print(f"  进度: {i}/{total} 个日期", flush=True)

    lgroup = landsat_all[landsat_all['date'] == date]
    mgroup = modis_by_date.get(date)
    if mgroup is None or mgroup.empty:
        continue

    tree = cKDTree(mgroup[['lat', 'lon']].values)
    dists, idxs = tree.query(lgroup[['lat', 'lon']].values, k=1)

    matched = lgroup.copy().reset_index(drop=True)
    matched['label_C']  = mgroup['LST_MODIS_C'].values[idxs]
    matched['label_K']  = mgroup['LST_MODIS_K'].values[idxs]
    matched['dist_deg'] = dists
    results.append(matched)

print("匹配完成，正在合并结果...")
sample_table = pd.concat(results, ignore_index=True)

# ── 7. 质量过滤 ──────────────────────────────
sample_table = sample_table[sample_table['dist_deg'] < 0.05]
sample_table = sample_table.dropna(subset=['label_C'])

# ── 8. 打印统计 ──────────────────────────────
print(f"\n=== 最终样本表统计 ===")
print(f"总样本数:   {len(sample_table):,}")
print(f"日期数:     {sample_table['date'].nunique()}")
print(f"年份分布:\n{sample_table['year'].value_counts().sort_index()}")
print(f"月份分布:\n{sample_table['month'].value_counts().sort_index()}")
print(f"Landsat亮温: {sample_table['LST_C'].min():.1f}℃ ~ {sample_table['LST_C'].max():.1f}℃")
print(f"MODIS标签:   {sample_table['label_C'].min():.1f}℃ ~ {sample_table['label_C'].max():.1f}℃")
print(f"列名: {sample_table.columns.tolist()}")

# ── 9. 保存 ──────────────────────────────────
out_path = DATA_DIR / "sample_table_2021_2024.csv"
sample_table.to_csv(out_path, index=False)
print(f"\n✓ 已保存: {out_path}")