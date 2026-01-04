
from osgeo import gdal
import os

def geotiff_to_kmz(tif_path: str, kmz_path: str, out_size: int = 2048) -> None:
    """
    将 GeoTIFF 转为 KMZ（Google Earth 影像叠加层）。
    - 自动重投影到 EPSG:4326（KML 要求经纬度）
    - 自动生成一个 PNG 覆盖图并打包成 KMZ
    - out_size 控制输出影像最大边长，避免文件太大
    """
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"输入文件不存在: {tif_path}")

    src = gdal.Open(tif_path, gdal.GA_ReadOnly)
    if src is None:
        raise RuntimeError(f"无法打开 tif: {tif_path}")

    # gdal.Translate 的 KMLSUPEROVERLAY 输出实际上会生成一个目录，
    # 里面有 doc.kml + 多级瓦片；如果你给 .kmz，会自动打包成 KMZ
    options = gdal.TranslateOptions(
        format="KMLSUPEROVERLAY",
        outputSRS="EPSG:4326",
        width=out_size,   # 控制输出大小（按宽度缩放，保持比例）
    )

    out_ds = gdal.Translate(kmz_path, src, options=options)
    if out_ds is None:
        raise RuntimeError("生成 KMZ 失败")

    out_ds = None
    src = None

    print(f"✅ 已生成 KMZ: {kmz_path}")
    print("在 Google Earth 中直接打开即可。")


if __name__ == "__main__":
    geotiff_to_kmz(r"E:\forest\Project\DEM\找dem\slope_use1.tif", r"E:\forest\Project\DEM\找dem\output.kmz", out_size=2048)