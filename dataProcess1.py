import pandas as pd
import json

# ── 1. 读取并清洗 Landsat ──────────────────────
landsat = pd.read_csv('data/Landsat_Shanghai_2023.csv')

def parse_geo(geo_str):
    try:
        geo = json.loads(geo_str)
        return pd.Series({'lon': geo['coordinates'][0],
                          'lat': geo['coordinates'][1]})
    except:
        return pd.Series({'lon': None, 'lat': None})

landsat[['lon','lat']] = landsat['.geo'].apply(parse_geo)
landsat['LST_C'] = landsat['BT_B10'] - 273.15
landsat['date']  = pd.to_datetime(landsat['date'])
landsat = landsat.drop(columns=['system:index', '.geo'])
landsat = landsat.dropna(subset=['lon','lat','BT_B10'])
print(f'Landsat样本数: {len(landsat):,}')

# ── 2. 读取并清洗 MODIS ───────────────────────
modis = pd.read_csv('data/MODIS_Shanghai_2023.csv')

modis[['lon','lat']] = modis['.geo'].apply(parse_geo)
modis['date'] = pd.to_datetime(modis['date'])
modis = modis.drop(columns=['system:index', '.geo'])
modis = modis.dropna(subset=['lon','lat','LST_MODIS_K'])
modis = modis.rename(columns={'LST_MODIS_C': 'label_C',
                               'LST_MODIS_K': 'label_K'})
print(f'MODIS样本数: {len(modis):,}')

# ── 3. 按日期 + 最近邻空间匹配 ────────────────
from scipy.spatial import cKDTree

results = []
for date, lgroup in landsat.groupby('date'):
    mgroup = modis[modis['date'] == date]
    if mgroup.empty:
        continue

    # 用KD树找每个Landsat像元最近的MODIS像元（1km格网）
    tree = cKDTree(mgroup[['lat','lon']].values)
    dists, idxs = tree.query(lgroup[['lat','lon']].values, k=1)

    matched = lgroup.copy().reset_index(drop=True)
    matched['label_C'] = mgroup['label_C'].values[idxs]
    matched['label_K'] = mgroup['label_K'].values[idxs]
    matched['dist_deg'] = dists  # 匹配距离（度），>0.05约5km则质量差
    results.append(matched)

sample_table = pd.concat(results, ignore_index=True)

# ── 4. 质量过滤 ───────────────────────────────
# 剔除匹配距离过大的点（超过0.05°≈5km说明附近无MODIS有效像元）
sample_table = sample_table[sample_table['dist_deg'] < 0.05]

print(f'\n匹配后样本数: {len(sample_table):,}')
print(f'日期数: {sample_table["date"].nunique()}')
print(f'Landsat亮温范围: {sample_table["LST_C"].min():.1f}℃ ~ {sample_table["LST_C"].max():.1f}℃')
print(f'MODIS标签范围:   {sample_table["label_C"].min():.1f}℃ ~ {sample_table["label_C"].max():.1f}℃')

sample_table.to_csv('sample_table_2023.csv', index=False)
print('\n✓ 样本表已保存: sample_table_2023.csv')
print(f'  列名: {sample_table.columns.tolist()}')