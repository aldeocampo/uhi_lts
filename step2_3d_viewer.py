"""
Step 2: 生成 Deck.gl 三维热力图 (HTML)
=========================================
功能：
  读取Step 1生成的 GeoJSON 建筑数据，生成一个独立的HTML文件，
  可直接用浏览器打开，无需后端服务器。

特点：
  - 基于 Deck.gl（Uber 开源，纯前端，免账号）
  - 建筑按LST着色，高度按真实楼高拉伸
  - 鼠标悬停显示建筑信息
  - 支持鼠标拖拽旋转/缩放
  - 可截图作为论文插图

依赖：
  无需Python额外库（pure file I/O）
  浏览器自动加载 deck.gl + maplibre 的 CDN
"""

import json
from pathlib import Path

# ══════════════════════════════════════════════
INPUT_GEOJSON = Path("outputs_3d/shanghai_buildings_with_lst.geojson")
OUTPUT_DIR    = Path("outputs_3d")
OUTPUT_HTML   = OUTPUT_DIR / "shanghai_3d_thermal.html"

# ══════════════════════════════════════════════
print("="*65)
print("Step 2: Building 3D thermal visualization (HTML)")
print("="*65)

# 读取数据并提取统计范围
with open(INPUT_GEOJSON, 'r', encoding='utf-8') as f:
    geojson_data = json.load(f)

features    = geojson_data['features']
lst_vals    = [f['properties']['lst']    for f in features
               if f['properties'].get('lst') is not None]
heights     = [f['properties']['height'] for f in features
               if f['properties'].get('height') is not None]

lst_min, lst_max = min(lst_vals), max(lst_vals)
lst_mean = sum(lst_vals) / len(lst_vals)

# 视角中心点
all_lons = [f['properties']['lon'] for f in features]
all_lats = [f['properties']['lat'] for f in features]
center_lon = sum(all_lons) / len(all_lons)
center_lat = sum(all_lats) / len(all_lats)

print(f"  Buildings:    {len(features):,}")
print(f"  LST range:    {lst_min:.1f}°C ~ {lst_max:.1f}°C")
print(f"  Mean height:  {sum(heights)/len(heights):.1f} m")
print(f"  Map center:   ({center_lon:.4f}, {center_lat:.4f})")

# 内嵌GeoJSON到HTML（小数据集这样最方便）
geojson_str = json.dumps(geojson_data, ensure_ascii=False)

# ══════════════════════════════════════════════
# HTML模板
# ══════════════════════════════════════════════
html = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Shanghai 3D Urban Heat — Physics-XGB LST</title>
<script src="https://unpkg.com/deck.gl@^8.9.0/dist.min.js"></script>
<script src="https://unpkg.com/maplibre-gl@^3.0.0/dist/maplibre-gl.js"></script>
<link  href="https://unpkg.com/maplibre-gl@^3.0.0/dist/maplibre-gl.css" rel="stylesheet">
<style>
  body { margin: 0; padding: 0; font-family: Arial, sans-serif;
         overflow: hidden; background: #1a1a1a; color: #fff; }
  #map { position: absolute; top: 0; bottom: 0; left: 0; right: 0; }
  #title {
    position: absolute; top: 20px; left: 20px; z-index: 10;
    background: rgba(0,0,0,0.78); padding: 14px 20px;
    border-radius: 8px; max-width: 360px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
  }
  #title h2 { margin: 0 0 6px 0; font-size: 17px; color: #FF7043; }
  #title p  { margin: 4px 0; font-size: 13px; color: #ddd; }

  #legend {
    position: absolute; top: 20px; right: 20px; z-index: 10;
    background: rgba(0,0,0,0.78); padding: 14px 18px;
    border-radius: 8px; min-width: 200px;
  }
  #legend h3 { margin: 0 0 10px 0; font-size: 14px; color: #FF7043; }
  .colorbar {
    width: 200px; height: 22px;
    background: linear-gradient(to right,
      #1565C0 0%, #42A5F5 22%, #FFC107 50%, #FF7043 75%, #D32F2F 100%);
    border-radius: 3px; margin: 6px 0;
  }
  .scale-labels { display: flex; justify-content: space-between;
                  font-size: 11px; color: #ccc; }

  #info {
    position: absolute; bottom: 20px; left: 20px; z-index: 10;
    background: rgba(0,0,0,0.78); padding: 12px 16px;
    border-radius: 8px; min-width: 230px; min-height: 100px;
    font-size: 12px;
  }
  #info h4 { margin: 0 0 8px 0; color: #FF7043; font-size: 13px; }
  #info p  { margin: 3px 0; color: #ddd; }
  .placeholder { color: #777; font-style: italic; }

  #controls {
    position: absolute; bottom: 20px; right: 20px; z-index: 10;
    background: rgba(0,0,0,0.78); padding: 10px 14px;
    border-radius: 8px; font-size: 11px; color: #aaa;
  }
  #controls p { margin: 2px 0; }
</style>
</head>
<body>

<div id="map"></div>

<div id="title">
  <h2>Shanghai 3D Urban Heat Island</h2>
  <p>Geo-Semantic-Thermal Data Cube — Building-Level LST Visualization</p>
  <p>Buildings: <b>__N_BUILD__</b> &nbsp;|&nbsp; LST: <b>__LST_RANGE__</b></p>
  <p style="font-size:11px; color:#999;">Data: OpenStreetMap + Physics-XGB LST</p>
</div>

<div id="legend">
  <h3>Land Surface Temperature</h3>
  <div class="colorbar"></div>
  <div class="scale-labels">
    <span>__LST_MIN__°C</span>
    <span>__LST_MID__°C</span>
    <span>__LST_MAX__°C</span>
  </div>
  <p style="font-size:11px; margin-top:8px; color:#bbb;">
    Higher temperature = warmer color
  </p>
</div>

<div id="info">
  <h4>Building Info</h4>
  <p class="placeholder">Hover a building...</p>
</div>

<div id="controls">
  <p><b>Mouse controls</b></p>
  <p>Left-drag: pan</p>
  <p>Right-drag: rotate</p>
  <p>Wheel: zoom</p>
</div>

<script>
const GEOJSON_DATA = __GEOJSON_DATA__;
const LST_MIN = __LST_MIN_VAL__;
const LST_MAX = __LST_MAX_VAL__;

// ── LST → RGB 色阶（蓝→红）─────────────────
function lstToColor(lst) {
  if (lst === null || lst === undefined) return [128, 128, 128, 200];
  const t = Math.max(0, Math.min(1, (lst - LST_MIN) / (LST_MAX - LST_MIN)));
  // 5个锚点
  const stops = [
    [0.00, [21, 101, 192]],
    [0.22, [66, 165, 245]],
    [0.50, [255, 193,   7]],
    [0.75, [255, 112,  67]],
    [1.00, [211,  47,  47]],
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const [t1, c1] = stops[i];
    const [t2, c2] = stops[i + 1];
    if (t >= t1 && t <= t2) {
      const f = (t - t1) / (t2 - t1);
      return [
        Math.round(c1[0] + f * (c2[0] - c1[0])),
        Math.round(c1[1] + f * (c2[1] - c1[1])),
        Math.round(c1[2] + f * (c2[2] - c1[2])),
        230
      ];
    }
  }
  return stops[stops.length - 1][1].concat([230]);
}

// ── 初始视角 ─────────────────────────────
const INITIAL_VIEW_STATE = {
  longitude: __CENTER_LON__,
  latitude:  __CENTER_LAT__,
  zoom:      13.5,
  pitch:     55,
  bearing:   30
};

// ── Deck.gl ─────────────────────────────
const buildingLayer = new deck.GeoJsonLayer({
  id: 'buildings',
  data: GEOJSON_DATA,
  extruded: true,
  wireframe: false,
  filled: true,
  getElevation: f => f.properties.height || 12,
  getFillColor: f => lstToColor(f.properties.lst),
  getLineColor: [255, 255, 255, 30],
  lineWidthMinPixels: 0.3,
  material: {
    ambient: 0.65, diffuse: 0.7,
    shininess: 36, specularColor: [60, 60, 60]
  },
  pickable: true,
  autoHighlight: true,
  highlightColor: [255, 255, 255, 120],
  onHover: info => {
    const el = document.getElementById('info');
    if (info.object) {
      const p = info.object.properties;
      el.innerHTML = `
        <h4>Building Info</h4>
        <p><b>LST:</b> ${(p.lst || 0).toFixed(2)} °C</p>
        <p><b>Height:</b> ${(p.height || 0).toFixed(1)} m  (~${p.levels || '?'} floors)</p>
        <p><b>Footprint:</b> ${(p.area || 0).toFixed(0)} m²</p>
        <p><b>Type:</b> ${p.type || 'unknown'}</p>
        <p><b>Location:</b> ${(p.lat || 0).toFixed(4)}°N, ${(p.lon || 0).toFixed(4)}°E</p>
      `;
    } else {
      el.innerHTML = '<h4>Building Info</h4><p class="placeholder">Hover a building...</p>';
    }
  }
});

const MAP_STYLE = {
  version: 8,
  sources: {
    'carto-dark': {
      type: 'raster',
      tiles: ['https://a.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}.png'],
      tileSize: 256, attribution: '© CARTO © OpenStreetMap'
    }
  },
  layers: [{ id: 'carto-dark-layer', type: 'raster',
             source: 'carto-dark', minzoom: 0, maxzoom: 22 }]
};

new deck.DeckGL({
  container: 'map',
  mapStyle: MAP_STYLE,
  initialViewState: INITIAL_VIEW_STATE,
  controller: true,
  layers: [buildingLayer]
});
</script>
</body>
</html>'''

# 文本替换
html = html.replace('__GEOJSON_DATA__', geojson_str)
html = html.replace('__LST_MIN_VAL__',  f'{lst_min:.2f}')
html = html.replace('__LST_MAX_VAL__',  f'{lst_max:.2f}')
html = html.replace('__LST_MIN__',      f'{lst_min:.0f}')
html = html.replace('__LST_MID__',      f'{(lst_min+lst_max)/2:.0f}')
html = html.replace('__LST_MAX__',      f'{lst_max:.0f}')
html = html.replace('__LST_RANGE__',
                    f'{lst_min:.1f}°C ~ {lst_max:.1f}°C')
html = html.replace('__N_BUILD__',      f'{len(features):,}')
html = html.replace('__CENTER_LON__',   f'{center_lon:.4f}')
html = html.replace('__CENTER_LAT__',   f'{center_lat:.4f}')

with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n  HTML saved: {OUTPUT_HTML}")
print(f"  File size : {OUTPUT_HTML.stat().st_size / 1024:.1f} KB")

# ══════════════════════════════════════════════
print("\n" + "="*65)
print("DONE — How to view:")
print("="*65)
print(f"  Open in browser: {OUTPUT_HTML.resolve()}")
print(f"  (Just double-click the HTML file — no server needed)")
print()
print("Browser compatibility:")
print("  ✓ Chrome / Edge / Firefox  (recommended)")
print("  ✓ Safari")
print()
print("Tips for screenshots (for paper figure):")
print("  - Hold right mouse button + drag to rotate to perspective view")
print("  - Find an angle that shows building height contrast")
print("  - Use F11 fullscreen, then Win+Shift+S to capture")