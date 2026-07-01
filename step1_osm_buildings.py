"""
Step 1: OpenStreetMap 建筑数据获取 + 温度映射
=================================================
功能：
  1. 从OpenStreetMap下载上海中心城区建筑轮廓（约60 km²）
  2. 估算每栋建筑高度（OSM building:levels字段，缺失时按区域均值）
  3. 与Physics-XGB预测的LST栅格做空间叠加，给每栋建筑赋温度
  4. 保存为GeoJSON（CesiumJS / Deck.gl / Mapbox 都能直接用）
  5. 生成2D俯视图预览

依赖安装：
  pip install osmnx geopandas rasterio shapely matplotlib contextily
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon, box
import osmnx as ox

# ══════════════════════════════════════════════
# 0. 配置
# ══════════════════════════════════════════════
OUTPUT_DIR = Path("outputs_3d")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 上海中心城区范围（内环+部分中环，约60 km²）──
# 西界121.43°E、东界121.55°E、南界31.18°N、北界31.27°N
BBOX_WEST  = 121.43
BBOX_SOUTH = 31.18
BBOX_EAST  = 121.55
BBOX_NORTH = 31.27

# OSMnx要求(north, south, east, west)顺序
BBOX = (BBOX_NORTH, BBOX_SOUTH, BBOX_EAST, BBOX_WEST)

# 配置OSMnx
ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.cache_folder = str(OUTPUT_DIR / "osmnx_cache")

# ══════════════════════════════════════════════
# 1. 下载OSM建筑数据
# ══════════════════════════════════════════════
print("="*65)
print("Step 1: Downloading building footprints from OpenStreetMap")
print("="*65)
print(f"  Bounding box: W={BBOX_WEST}, S={BBOX_SOUTH}, "
      f"E={BBOX_EAST}, N={BBOX_NORTH}")
print(f"  Area ≈ {(BBOX_EAST-BBOX_WEST)*(BBOX_NORTH-BBOX_SOUTH)*111*97:.1f} km²")

tags = {'building': True}

try:
    # osmnx >= 2.0 用 features_from_bbox
    buildings = ox.features_from_bbox(
        bbox=(BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH),
        tags=tags
    )
except (AttributeError, TypeError):
    # 兼容旧版本 osmnx
    buildings = ox.geometries_from_bbox(
        north=BBOX_NORTH, south=BBOX_SOUTH,
        east=BBOX_EAST, west=BBOX_WEST,
        tags=tags
    )

print(f"  Raw buildings downloaded: {len(buildings):,}")

# 只保留多边形几何
buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
buildings = buildings.reset_index(drop=True)
print(f"  Valid polygons: {len(buildings):,}")

# ══════════════════════════════════════════════
# 2. 提取关键字段 + 高度估算
# ══════════════════════════════════════════════
print("\n[2] Extracting building attributes and estimating heights...")

def estimate_height(row):
    """根据OSM字段估算建筑高度（米）"""
    # 优先用 height 字段
    if 'height' in row.index and pd.notna(row.get('height')):
        try:
            return float(str(row['height']).replace('m', '').strip())
        except Exception:
            pass
    # 其次用 building:levels（楼层数 × 3m）
    if 'building:levels' in row.index and pd.notna(row.get('building:levels')):
        try:
            return float(row['building:levels']) * 3.0
        except Exception:
            pass
    # 最后用建筑类型推断
    btype = str(row.get('building', '')).lower()
    type_height = {
        'apartments': 24, 'residential': 18, 'house': 6,
        'commercial':  20, 'office':      30, 'retail': 10,
        'industrial':  10, 'warehouse':   8,  'school': 12,
        'hospital':    20, 'hotel':       40, 'church': 15,
        'public':      15,
    }
    return type_height.get(btype, 12.0)   # 默认12m（约4层）

# 安全提取字段
for col in ['name', 'building', 'building:levels', 'height']:
    if col not in buildings.columns:
        buildings[col] = np.nan

buildings['height_m'] = buildings.apply(estimate_height, axis=1)
buildings['height_m'] = buildings['height_m'].clip(3, 200)
buildings['levels']   = (buildings['height_m'] / 3.0).round().astype(int)

# 计算建筑底面积
buildings_proj = buildings.to_crs(epsg=32651)   # UTM 51N
buildings['area_m2'] = buildings_proj.geometry.area
buildings = buildings[buildings['area_m2'] > 30]   # 剔除<30m²的噪声多边形
buildings = buildings.reset_index(drop=True)
print(f"  After cleaning:  {len(buildings):,} buildings")
print(f"  Mean height:     {buildings['height_m'].mean():.1f} m")
print(f"  Mean footprint:  {buildings['area_m2'].mean():.1f} m²")

# ══════════════════════════════════════════════
# 3. 为每栋建筑赋温度（模拟示例数据）
# ══════════════════════════════════════════════
# 在实际项目中：用 Physics-XGB 预测的 LST 栅格通过 rasterstats.zonal_stats
# 计算每栋建筑覆盖范围的温度均值。这里为方便演示，先用基于地理位置和
# 建筑密度的合成温度。后续接入真实LST栅格替换此段即可。
# ══════════════════════════════════════════════
print("\n[3] Mapping LST to each building (synthetic demo data)...")

# 城市中心位置（近人民广场）
center_lon, center_lat = 121.48, 31.235

# 每栋建筑的中心点
buildings['centroid_lon'] = buildings.geometry.centroid.x
buildings['centroid_lat'] = buildings.geometry.centroid.y

# 到城市中心的距离（km）
buildings['dist_km'] = np.sqrt(
    ((buildings['centroid_lon'] - center_lon) * 95)**2 +
    ((buildings['centroid_lat'] - center_lat) * 111)**2
)

# ── 合成LST：距中心越近、建筑越密、楼越高 → 温度越高 ──
# 基础温度40°C，随距离衰减
np.random.seed(42)
base_temp = 41.0 - buildings['dist_km'] * 1.3
height_bonus = (buildings['height_m'] - 12) * 0.06
noise = np.random.normal(0, 0.8, len(buildings))
buildings['LST_C'] = base_temp + height_bonus + noise
buildings['LST_C'] = buildings['LST_C'].clip(28, 45)

print(f"  LST range: {buildings['LST_C'].min():.1f}°C ~ "
      f"{buildings['LST_C'].max():.1f}°C")
print(f"  LST mean:  {buildings['LST_C'].mean():.1f}°C")

# ── 真实数据集成示例（注释，按需启用）─────────────
# from rasterstats import zonal_stats
# LST_RASTER_PATH = "data/physics_xgb_lst_2024summer.tif"
# stats = zonal_stats(buildings.geometry, LST_RASTER_PATH,
#                     stats=['mean'], nodata=-9999)
# buildings['LST_C'] = [s['mean'] if s['mean'] else np.nan for s in stats]

# ══════════════════════════════════════════════
# 4. 保存为GeoJSON（CesiumJS可直接读取）
# ══════════════════════════════════════════════
print("\n[4] Saving GeoJSON output...")

# 只保留CesiumJS需要的字段，减小文件体积
out_cols = ['height_m', 'levels', 'area_m2', 'LST_C',
            'centroid_lon', 'centroid_lat', 'building', 'geometry']
buildings_out = buildings[out_cols].copy()

# CesiumJS建议把字段名简化为英文小写
buildings_out = buildings_out.rename(columns={
    'height_m':    'height',
    'levels':      'levels',
    'area_m2':     'area',
    'LST_C':       'lst',
    'centroid_lon':'lon',
    'centroid_lat':'lat',
    'building':    'type',
})

# 处理任何剩余的非标量字段
for col in buildings_out.columns:
    if col == 'geometry': continue
    buildings_out[col] = buildings_out[col].apply(
        lambda x: str(x) if isinstance(x, (list, dict)) else x)

geojson_path = OUTPUT_DIR / "shanghai_buildings_with_lst.geojson"
buildings_out.to_file(geojson_path, driver='GeoJSON')
print(f"  Saved: {geojson_path}")
print(f"  File size: {geojson_path.stat().st_size / 1024:.1f} KB")

# CSV副本（便于Excel查看）
csv_path = OUTPUT_DIR / "shanghai_buildings_lst.csv"
buildings_out.drop(columns='geometry').to_csv(csv_path, index=False)
print(f"  CSV  : {csv_path}")

# ══════════════════════════════════════════════
# 5. 生成2D俯视预览图
# ══════════════════════════════════════════════
print("\n[5] Plotting 2D preview...")

# 红蓝色阶（蓝=冷，红=热）
heat_cmap = LinearSegmentedColormap.from_list(
    'urban_heat',
    ['#1565C0', '#42A5F5', '#FFC107', '#FF7043', '#D32F2F', '#B71C1C'],
    N=256)

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# ── (a) 按温度着色 ──
ax = axes[0]
buildings_out.plot(column='lst', cmap=heat_cmap, ax=ax,
                   edgecolor='white', linewidth=0.05,
                   legend=True,
                   legend_kwds={'label': 'Building LST (°C)',
                                'shrink': 0.7})
ax.set_xlim(BBOX_WEST, BBOX_EAST)
ax.set_ylim(BBOX_SOUTH, BBOX_NORTH)
ax.set_aspect('equal')
ax.set_xlabel('Longitude (°E)', fontsize=11)
ax.set_ylabel('Latitude (°N)',  fontsize=11)
ax.set_title(f'(a)  Buildings Colored by LST\n'
             f'n = {len(buildings_out):,}, '
             f'LST range: {buildings_out["lst"].min():.1f}°C – '
             f'{buildings_out["lst"].max():.1f}°C',
             fontsize=12, fontweight='bold')
ax.grid(alpha=0.25, linestyle='--')

# ── (b) 按建筑高度着色 ──
ax = axes[1]
buildings_out.plot(column='height', cmap='viridis', ax=ax,
                   edgecolor='white', linewidth=0.05,
                   legend=True,
                   legend_kwds={'label': 'Building Height (m)',
                                'shrink': 0.7})
ax.set_xlim(BBOX_WEST, BBOX_EAST)
ax.set_ylim(BBOX_SOUTH, BBOX_NORTH)
ax.set_aspect('equal')
ax.set_xlabel('Longitude (°E)', fontsize=11)
ax.set_ylabel('Latitude (°N)',  fontsize=11)
ax.set_title(f'(b)  Buildings Colored by Height\n'
             f'Mean: {buildings_out["height"].mean():.1f} m, '
             f'Max: {buildings_out["height"].max():.1f} m',
             fontsize=12, fontweight='bold')
ax.grid(alpha=0.25, linestyle='--')

fig.suptitle('Shanghai Central District — Building Footprints with Thermal Attributes',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_building_preview.png',
            dpi=200, bbox_inches='tight')
plt.close()
print(f"  Preview saved: fig_building_preview.png")

# ══════════════════════════════════════════════
# 6. 汇总
# ══════════════════════════════════════════════
print("\n" + "="*65)
print("Step 1 Complete — Output Summary")
print("="*65)
print(f"  Buildings   : {len(buildings_out):,}")
print(f"  Bounding box: ~{(BBOX_EAST-BBOX_WEST)*(BBOX_NORTH-BBOX_SOUTH)*111*97:.0f} km²")
print(f"  Output dir  : {OUTPUT_DIR.resolve()}")
print()
print("  Generated files:")
print("    shanghai_buildings_with_lst.geojson  ← input for Step 2")
print("    shanghai_buildings_lst.csv           ← Excel-readable")
print("    fig_building_preview.png             ← 2D top-down view")
print()
print("Next: Run Step 2 to generate CesiumJS / Deck.gl 3D viewer.")