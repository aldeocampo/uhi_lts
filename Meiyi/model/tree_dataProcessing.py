# ===================== 适配论文：Stacking集成融合模型 数据集拆分代码 =====================
# 完全匹配论文：7:1.5:1.5分层抽样规范 | 7项核心特征体系 | 无冗余LSTM逻辑
# 与你最终的Stacking/XGBoost模型训练代码100%兼容
# ======================================================================================
import os
import geopandas as gpd
import rasterio
from rasterio.warp import reproject
from rasterio.mask import mask
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import Point, mapping
import joblib
import warnings
warnings.filterwarnings('ignore')

# -------------------------- 1. 论文核心参数配置 --------------------------
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
# 热岛强度分层阈值，适配分层抽样，保证训练/验证/测试集分布一致
UHI_BINS = [-1, 1.5, 3.5, 5]
UHI_LABELS = ["Low", "Medium", "High"]

# -------------------------- 2. 路径配置（与原代码完全一致，无需修改） --------------------------
BASE_PATH = r"D:\paper\A1096\code\data"
RESULT_PATH = r"D:\paper\A1096\code\model\result"
os.makedirs(RESULT_PATH, exist_ok=True)

PATHS = {
    "building": os.path.join(BASE_PATH, "building", "xiamen.shp"),
    "mod11a1_dir": os.path.join(BASE_PATH, "MOD11A1"),
    "ndvi_dir": os.path.join(BASE_PATH, "NDVI"),
    "mcd12q1_dir": os.path.join(BASE_PATH, "MCD12Q1"),
    "xiamen_boundary": os.path.join(BASE_PATH, "vector", "350200.shp"),
    "save_path": RESULT_PATH
}
TARGET_YEARS = [2020, 2021, 2022, 2023, 2024]

# -------------------------- 3. 论文核心工具函数（无修改，完全保留） --------------------------
def load_and_align_raster(raster_path, boundary_gdf, target_size, target_transform, target_crs):
    """栅格加载、重投影对齐、边界裁剪、无效值处理"""
    if not os.path.exists(raster_path):
        print(f"⚠️ 文件不存在: {os.path.basename(raster_path)}")
        return np.full(target_size, np.nan)
        
    try:
        with rasterio.open(raster_path) as src:
            out_raster = np.full(target_size, np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=out_raster,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=rasterio.enums.Resampling.bilinear
            )
        
        # 无效值替换
        out_raster[out_raster == -9999] = np.nan
        
        # 研究区边界裁剪
        with rasterio.MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff", width=target_size[1], height=target_size[0],
                count=1, dtype=np.float32, transform=target_transform, crs=target_crs
            ) as dataset:
                dataset.write(out_raster, 1)
                clipped, _ = mask(dataset, [mapping(boundary_gdf.geometry.iloc[0])], crop=False, filled=False)
        clipped = clipped[0].filled(np.nan) if np.ma.is_masked(clipped[0]) else clipped[0]
        clipped[clipped == -9999] = np.nan
        
        return clipped
    except Exception as e:
        print(f"⚠️ 栅格加载失败: {os.path.basename(raster_path)}, 错误: {str(e)[:80]}")
        return np.full(target_size, np.nan)

def calculate_uhi_intensity(lst_data, landuse_data):
    """论文标准热岛强度计算：对照区差值法，完全符合学术规范"""
    # 郊区对照区：林地、农田等自然植被区域
    suburban_mask = np.isin(landuse_data, [1, 10, 12, 16, 17])
    suburban_lst = lst_data[suburban_mask & ~np.isnan(lst_data)]
    # 对照区样本不足时用全域均值兜底
    if len(suburban_lst) < 50:
        suburban_mean = np.nanmean(lst_data)
    else:
        suburban_mean = np.nanmean(suburban_lst)
    uhi_data = lst_data - suburban_mean
    return uhi_data

def extract_paper_features(ndvi_data, landuse_data, lst_data, uhi_data, building_data, transform, year, sample_step=10):
    """提取论文核心7项特征体系，与模型输入100%匹配"""
    features = []
    height, width = ndvi_data.shape
    
    for i in range(0, height, sample_step):
        for j in range(0, width, sample_step):
            # 跳过无效像元
            if np.isnan(ndvi_data[i,j]) or np.isnan(lst_data[i,j]) or np.isnan(landuse_data[i,j]):
                continue
            # 计算像元坐标，构建300m缓冲区统计建筑密度
            x, y = rasterio.transform.xy(transform, i, j)
            point = Point(x, y).buffer(300)
            
            # 核心特征提取
            ndvi = max(-1.0, min(1.0, ndvi_data[i,j]))
            landuse = int(landuse_data[i,j])
            building_intersect = building_data[building_data.geometry.intersects(point)]
            building_density = building_intersect.area.sum() / (np.pi * 300**2) if len(building_intersect) > 0 else 0
            raw_lst = lst_data[i,j]
            uhi = uhi_data[i,j]
            
            # 二值标识特征，与论文完全一致
            impervious_flag = 1 if landuse == 13 else 0
            water_flag = 1 if landuse == 17 else 0
            vegetation_flag = 1 if landuse in [1,2,3,4,5] else 0
            
            features.append([
                year, ndvi, landuse, building_density, raw_lst, impervious_flag,
                water_flag, vegetation_flag, uhi, x, y
            ])
    
    # 构建DataFrame，剔除异常值
    df = pd.DataFrame(features, columns=[
        "Year", "NDVI", "Landuse_IGBP", "Building_Density", "Raw_LST",
        "Impervious_Flag", "Water_Flag", "Vegetation_Flag", "UHI_Intensity", "Lon", "Lat"
    ])
    df = df[(df["UHI_Intensity"] >= -1) & (df["UHI_Intensity"] <= 5)]
    return df.dropna()

# -------------------------- 4. 核心主流程（精简优化，适配最终模型） --------------------------
if __name__ == "__main__":
    print("="*70)
    print("📊 适配论文：厦门热岛模型数据集构建与7:1.5:1.5拆分")
    print("="*70)

    # 1. 加载矢量基础数据
    xiamen_boundary = gpd.read_file(PATHS["xiamen_boundary"]).to_crs(epsg=4326)
    building_data = gpd.read_file(PATHS["building"]).to_crs(epsg=4326)
    print(f"✅ 厦门边界加载完成，建筑矢量共{len(building_data)}个要素")

    # 2. 加载基准栅格，确定统一空间参数
    sample_ndvi = os.path.join(PATHS["ndvi_dir"], "Xiamen_Sentinel2_MeanNDVI_30m_2020.tif")
    with rasterio.open(sample_ndvi) as src:
        target_size = (src.height, src.width)
        target_transform = src.transform
        target_crs = src.crs
    print(f"✅ 基准空间参数：尺寸{target_size}，坐标系WGS84")

    # 3. 逐年处理数据，构建全量数据集
    all_year_data = []
    for year in TARGET_YEARS:
        print(f"\n📅 处理{year}年数据...")
        ndvi_path = os.path.join(PATHS["ndvi_dir"], f"Xiamen_Sentinel2_MeanNDVI_30m_{year}.tif")
        lst_path = os.path.join(PATHS["mod11a1_dir"], f"BoxXiaMen_MOD11A1_Day_MeanLST_{year}-1.tif")
        landuse_path = os.path.join(PATHS["mcd12q1_dir"], f"China_MCD12Q1_IGBP_500m_{year}.tif")
        
        # 校验文件完整性
        if not all([os.path.exists(p) for p in [ndvi_path, lst_path, landuse_path]]):
            print(f"⚠️ {year}年数据缺失，跳过")
            continue
        
        # 栅格加载与空间对齐
        ndvi_data = load_and_align_raster(ndvi_path, xiamen_boundary, target_size, target_transform, target_crs)
        lst_data = load_and_align_raster(lst_path, xiamen_boundary, target_size, target_transform, target_crs)
        landuse_data = load_and_align_raster(landuse_path, xiamen_boundary, target_size, target_transform, target_crs)
        
        # 校验数据有效性
        if np.all(np.isnan(ndvi_data)) or np.all(np.isnan(lst_data)):
            print(f"⚠️ {year}年数据无效，跳过")
            continue
        
        # 热岛强度计算与特征提取
        uhi_data = calculate_uhi_intensity(lst_data, landuse_data)
        year_df = extract_paper_features(ndvi_data, landuse_data, lst_data, uhi_data, building_data, target_transform, year)
        
        if len(year_df) > 0:
            all_year_data.append(year_df)
            print(f"✅ {year}年处理完成，有效样本{len(year_df)}个")

    # 4. 全量数据集合并与分层标签构建
    full_dataset = pd.concat(all_year_data, ignore_index=True)
    print(f"\n📦 全量数据集构建完成，总样本量：{len(full_dataset)}个，特征维度：7维")

    # 构建分层抽样标签，保证训练/验证/测试集的年份、热岛强度分布一致
    full_dataset["UHI_Class"] = pd.cut(full_dataset["UHI_Intensity"], bins=UHI_BINS, labels=UHI_LABELS, include_lowest=True)
    full_dataset["Stratify_Label"] = full_dataset["Year"].astype(str) + "_" + full_dataset["UHI_Class"].astype(str)
    
    # 处理稀有标签，保证分层抽样有效性
    full_dataset = full_dataset.dropna(subset=["Stratify_Label"])
    stratify_counts = full_dataset["Stratify_Label"].value_counts()
    rare_labels = stratify_counts[stratify_counts < 2].index
    full_dataset.loc[full_dataset["Stratify_Label"].isin(rare_labels), "Stratify_Label"] = "Other"
    print(f"✅ 分层标签构建完成，有效分层数：{full_dataset['Stratify_Label'].nunique()}")

    # 5. 论文标准7:1.5:1.5分层抽样
    feature_cols = ["NDVI", "Landuse_IGBP", "Building_Density", "Impervious_Flag", "Water_Flag", "Vegetation_Flag", "Raw_LST"]
    X = full_dataset[feature_cols]
    y = full_dataset["UHI_Intensity"]
    stratify_y = full_dataset["Stratify_Label"]

    # 第一步：拆分训练集 + 临时集（验证+测试）
    sss_train = StratifiedShuffleSplit(n_splits=1, test_size=1-TRAIN_RATIO, random_state=RANDOM_SEED)
    train_idx, temp_idx = next(sss_train.split(X, stratify_y))
    X_train, X_temp = X.iloc[train_idx], X.iloc[temp_idx]
    y_train, y_temp = y.iloc[train_idx], y.iloc[temp_idx]
    full_temp = full_dataset.iloc[temp_idx].reset_index(drop=True)

    # 第二步：拆分验证集 + 测试集（各15%）
    sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_SEED)
    val_idx, test_idx = next(sss_val_test.split(X_temp, full_temp["Stratify_Label"]))
    X_val, X_test = X_temp.iloc[val_idx], X_temp.iloc[test_idx]
    y_val, y_test = y_temp.iloc[val_idx], y_temp.iloc[test_idx]

    # 6. 特征标准化，与模型训练代码完全匹配
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 构建最终数据集，适配Stacking/XGBoost模型
    train_set = pd.DataFrame(X_train_scaled, columns=feature_cols)
    train_set["UHI_Intensity"] = y_train.values
    val_set = pd.DataFrame(X_val_scaled, columns=feature_cols)
    val_set["UHI_Intensity"] = y_val.values
    test_set = pd.DataFrame(X_test_scaled, columns=feature_cols)
    test_set["UHI_Intensity"] = y_test.values

    # 7. 保存数据集与标准化器
    print("\n💾 保存数据集与标准化器...")
    train_set.to_csv(os.path.join(PATHS["save_path"], "train_set_70%.csv"), index=False)
    val_set.to_csv(os.path.join(PATHS["save_path"], "val_set_15%.csv"), index=False)
    test_set.to_csv(os.path.join(PATHS["save_path"], "test_set_15%.csv"), index=False)
    joblib.dump(scaler, os.path.join(PATHS["save_path"], "scaler_feature.pkl"))

    # 8. 论文级输出统计
    print("\n" + "="*70)
    print("✅ 数据集拆分完成！严格遵循7:1.5:1.5论文规范")
    print("="*70)
    print(f"训练集：{len(train_set)}条 | 占比：{len(train_set)/len(full_dataset):.1%} | 特征维度：{X_train.shape[1]}")
    print(f"验证集：{len(val_set)}条 | 占比：{len(val_set)/len(full_dataset):.1%}")
    print(f"测试集：{len(test_set)}条 | 占比：{len(test_set)/len(full_dataset):.1%}")
    print(f"核心特征：{feature_cols}")
    print(f"文件保存路径：{PATHS['save_path']}")
    print("="*70)