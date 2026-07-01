"""
Figure A: Dual-view comparison
=================================
左右双地图同步显示：
  左：建筑按 LST 温度着色（红蓝色阶）
  右：建筑按建筑高度着色（viridis色阶）
两个视角完全同步（旋转、平移、缩放联动），便于对比
"建筑物理高度" 与 "热环境强度" 在空间上的差异。

核心论点：高建筑 ≠ 高温度，城市热岛与下垫面属性的关系
比与建筑高度的关系更密切。
"""

import json
from pathlib import Path

INPUT_GEOJSON = Path("outputs_3d/shanghai_buildings_with_lst.geojson")
OUTPUT_HTML   = Path("outputs_3d/fig_A_dual_view.html")

# 读取数据
with open(INPUT_GEOJSON, 'r', encoding='utf-8') as f:
    geojson_data = json.load(f)

features  = geojson_data['features']
lst_vals  = [f['properties']['lst']    for f in features if f['properties'].get('lst') is not None]
heights   = [f['properties']['height'] for f in features if f['properties'].get('height') is not None]
all_lons  = [f['properties']['lon']    for f in features]
all_lats  = [f['properties']['lat']    for f in features]

lst_min,  lst_max  = min(lst_vals),  max(lst_vals)
h_min,    h_max    = min(heights),   max(heights)
center_lon = sum(all_lons) / len(all_lons)
center_lat = sum(all_lats) / len(all_lats)

print(f"Buildings: {len(features):,}")
print(f"LST range: {lst_min:.1f}°C ~ {lst_max:.1f}°C")
print(f"Height   : {h_min:.1f} ~ {h_max:.1f} m")

# 注意：把GeoJSON内嵌到HTML中，避免双地图重复加载
geojson_str = json.dumps(geojson_data, ensure_ascii=False)

html = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Figure A — LST vs Building Height</title>
<script src="https://unpkg.com/deck.gl@^8.9.0/dist.min.js"></script>
<script src="https://unpkg.com/maplibre-gl@^3.0.0/dist/maplibre-gl.js"></script>
<link  href="https://unpkg.com/maplibre-gl@^3.0.0/dist/maplibre-gl.css" rel="stylesheet">
<style>
  body { margin:0; padding:0; font-family:Arial,sans-serif;
         overflow:hidden; background:#0a0a0a; color:#fff; }
  #container { display:flex; width:100vw; height:100vh; }
  .pane { flex:1; position:relative; border-left:1px solid #333; }
  .pane:first-child { border-left:none; }

  .pane-title {
    position:absolute; top:18px; left:18px; z-index:10;
    background:rgba(0,0,0,0.8); padding:10px 16px;
    border-radius:6px; min-width:280px;
  }
  .pane-title h2 { margin:0 0 4px 0; font-size:15px; }
  .pane-title p  { margin:2px 0; font-size:11px; color:#aaa; }

  .legend {
    position:absolute; bottom:20px; left:18px; z-index:10;
    background:rgba(0,0,0,0.8); padding:10px 14px;
    border-radius:6px;
  }
  .legend p { margin:2px 0; font-size:10px; color:#bbb; }
  .colorbar { width:220px; height:18px; border-radius:3px; margin:4px 0; }
  .cb-lst    { background:linear-gradient(to right,
               #1565C0 0%, #42A5F5 22%, #FFC107 50%, #FF7043 75%, #D32F2F 100%); }
  .cb-height { background:linear-gradient(to right,
               #440154 0%, #3B528B 25%, #21908C 50%, #5DC863 75%, #FDE725 100%); }
  .scale { display:flex; justify-content:space-between;
           font-size:10px; color:#ccc; }

  #title {
    position:absolute; top:18px; left:50%; transform:translateX(-50%);
    z-index:11; background:rgba(0,0,0,0.85);
    padding:8px 22px; border-radius:6px;
    border:1px solid #FF7043;
  }
  #title h1 { margin:0; font-size:15px; color:#FF7043; }
  #title p  { margin:3px 0 0 0; font-size:11px; color:#bbb; text-align:center; }
</style>
</head>
<body>

<div id="title">
  <h1>Figure A &nbsp;|&nbsp; Building LST  ↔  Building Height</h1>
  <p>Synchronized dual-view comparison &middot; Shanghai central district</p>
</div>

<div id="container">

  <div class="pane" id="pane-left">
    <div id="map-left"></div>
    <div class="pane-title">
      <h2 style="color:#FF7043;">(a)  Colored by LST</h2>
      <p>Buildings extruded to real height, color = land surface temperature</p>
    </div>
    <div class="legend">
      <p style="color:#FF7043; font-weight:bold;">Land Surface Temperature</p>
      <div class="colorbar cb-lst"></div>
      <div class="scale">
        <span>__LST_MIN__°C</span>
        <span>__LST_MID__°C</span>
        <span>__LST_MAX__°C</span>
      </div>
    </div>
  </div>

  <div class="pane" id="pane-right">
    <div id="map-right"></div>
    <div class="pane-title">
      <h2 style="color:#21908C;">(b)  Colored by Height</h2>
      <p>Buildings extruded to real height, color = physical height</p>
    </div>
    <div class="legend">
      <p style="color:#21908C; font-weight:bold;">Building Height</p>
      <div class="colorbar cb-height"></div>
      <div class="scale">
        <span>__H_MIN__ m</span>
        <span>__H_MID__ m</span>
        <span>__H_MAX__ m</span>
      </div>
    </div>
  </div>

</div>

<script>
const GEOJSON_DATA = __GEOJSON_DATA__;
const LST_MIN = __LST_MIN_VAL__, LST_MAX = __LST_MAX_VAL__;
const H_MIN   = __H_MIN_VAL__,   H_MAX   = __H_MAX_VAL__;

function lstToColor(lst) {
  if (lst == null) return [128,128,128,200];
  const t = Math.max(0, Math.min(1, (lst - LST_MIN) / (LST_MAX - LST_MIN)));
  const stops = [
    [0.00, [21,101,192]],  [0.22, [66,165,245]],
    [0.50, [255,193,7]],   [0.75, [255,112,67]],
    [1.00, [211,47,47]]
  ];
  for (let i=0;i<stops.length-1;i++){
    if (t>=stops[i][0] && t<=stops[i+1][0]){
      const f = (t-stops[i][0])/(stops[i+1][0]-stops[i][0]);
      return [Math.round(stops[i][1][0]+f*(stops[i+1][1][0]-stops[i][1][0])),
              Math.round(stops[i][1][1]+f*(stops[i+1][1][1]-stops[i][1][1])),
              Math.round(stops[i][1][2]+f*(stops[i+1][1][2]-stops[i][1][2])), 230];
    }
  }
  return stops[stops.length-1][1].concat([230]);
}

function heightToColor(h) {
  if (h == null) return [128,128,128,200];
  const t = Math.max(0, Math.min(1, (h - H_MIN) / (H_MAX - H_MIN)));
  // viridis色阶
  const stops = [
    [0.00, [68,1,84]],     [0.25, [59,82,139]],
    [0.50, [33,144,140]],  [0.75, [93,200,99]],
    [1.00, [253,231,37]]
  ];
  for (let i=0;i<stops.length-1;i++){
    if (t>=stops[i][0] && t<=stops[i+1][0]){
      const f = (t-stops[i][0])/(stops[i+1][0]-stops[i][0]);
      return [Math.round(stops[i][1][0]+f*(stops[i+1][1][0]-stops[i][1][0])),
              Math.round(stops[i][1][1]+f*(stops[i+1][1][1]-stops[i][1][1])),
              Math.round(stops[i][1][2]+f*(stops[i+1][1][2]-stops[i][1][2])), 230];
    }
  }
  return stops[stops.length-1][1].concat([230]);
}

const INITIAL_VIEW = {
  longitude: __CENTER_LON__,
  latitude:  __CENTER_LAT__,
  zoom: 13.0, pitch: 55, bearing: 30
};

const MAP_STYLE = {
  version: 8,
  sources: { 'carto-dark': {
    type: 'raster',
    tiles: ['https://a.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}.png'],
    tileSize: 256, attribution: '© CARTO © OpenStreetMap'
  }},
  layers: [{ id: 'carto-dark-layer', type: 'raster', source: 'carto-dark' }]
};

// 左：温度着色
const deckLeft = new deck.DeckGL({
  container: 'pane-left',
  mapStyle: MAP_STYLE,
  initialViewState: INITIAL_VIEW,
  controller: true,
  layers: [ new deck.GeoJsonLayer({
    id: 'b-lst', data: GEOJSON_DATA,
    extruded: true, filled: true, wireframe: false,
    getElevation: f => f.properties.height || 12,
    getFillColor: f => lstToColor(f.properties.lst),
    getLineColor: [255,255,255,30],
    material: { ambient: 0.65, diffuse: 0.7, shininess: 36 },
    pickable: false
  })],
  onViewStateChange: ({ viewState }) => {
    deckLeft.setProps({ viewState });
    deckRight.setProps({ viewState });
  }
});

// 右：高度着色
const deckRight = new deck.DeckGL({
  container: 'pane-right',
  mapStyle: MAP_STYLE,
  initialViewState: INITIAL_VIEW,
  controller: true,
  layers: [ new deck.GeoJsonLayer({
    id: 'b-h', data: GEOJSON_DATA,
    extruded: true, filled: true, wireframe: false,
    getElevation: f => f.properties.height || 12,
    getFillColor: f => heightToColor(f.properties.height),
    getLineColor: [255,255,255,30],
    material: { ambient: 0.65, diffuse: 0.7, shininess: 36 },
    pickable: false
  })],
  onViewStateChange: ({ viewState }) => {
    deckLeft.setProps({ viewState });
    deckRight.setProps({ viewState });
  }
});
</script>
</body>
</html>'''

# 替换占位符
html = (html
    .replace('__GEOJSON_DATA__', geojson_str)
    .replace('__LST_MIN_VAL__', f'{lst_min:.2f}')
    .replace('__LST_MAX_VAL__', f'{lst_max:.2f}')
    .replace('__LST_MIN__',     f'{lst_min:.0f}')
    .replace('__LST_MID__',     f'{(lst_min+lst_max)/2:.0f}')
    .replace('__LST_MAX__',     f'{lst_max:.0f}')
    .replace('__H_MIN_VAL__',   f'{h_min:.2f}')
    .replace('__H_MAX_VAL__',   f'{h_max:.2f}')
    .replace('__H_MIN__',       f'{h_min:.0f}')
    .replace('__H_MID__',       f'{(h_min+h_max)/2:.0f}')
    .replace('__H_MAX__',       f'{h_max:.0f}')
    .replace('__CENTER_LON__',  f'{center_lon:.4f}')
    .replace('__CENTER_LAT__',  f'{center_lat:.4f}'))

with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\nFigure A saved: {OUTPUT_HTML.resolve()}")
print("Open it in browser, find the best angle, then screenshot (Win+Shift+S).")
print()
print("Tips for paper screenshot:")
print("  - Use Chrome F11 fullscreen mode")
print("  - Find an angle showing CBD (Lujiazui) clearly")
print("  - Both views rotate synchronously")