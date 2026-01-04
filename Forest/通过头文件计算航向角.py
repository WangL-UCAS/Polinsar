import os
import glob
import math
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Tuple, Optional


# ============================================================
# 1) 你只需要改这里：master 路径
# ============================================================
master = r"E:\ShanXi_data_saocom\陕西林业\原始数据\厚畛子镇\区域二\S1B_OPER_SAR_EOSSP__CORE_L1A_OLF_20240917T181846"


# ============================================================
# 2) 是否递归搜索 xemt（如果 xemt 不在 master 目录下）
# ============================================================
RECURSIVE_SEARCH = True


def parse_iso_utc(s: str) -> datetime:
    s = s.strip().replace("Z", "")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def find_text_by_suffix(root: ET.Element, suffix: str) -> Optional[str]:
    for el in root.iter():
        if el.tag.endswith(suffix) and el.text:
            return el.text.strip()
    return None


def find_vals_under(root: ET.Element, parent_suffix: str) -> List[float]:
    parent = None
    for el in root.iter():
        if el.tag.endswith(parent_suffix):
            parent = el
            break
    if parent is None:
        return []

    vals = []
    for el in parent.iter():
        if el.tag.endswith("val") and el.text:
            vals.append(float(el.text.strip()))
    return vals


def group_xyz(vals: List[float]) -> List[Tuple[float, float, float]]:
    if len(vals) % 3 != 0:
        raise ValueError(f"Length not multiple of 3: {len(vals)}")
    return [(vals[i], vals[i+1], vals[i+2]) for i in range(0, len(vals), 3)]


def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def mul(s, a):
    return (s*a[0], s*a[1], s*a[2])


def ecef_to_lon_lat_spherical(r: Tuple[float, float, float]) -> Tuple[float, float]:
    x, y, z = r
    lon = math.atan2(y, x)
    hyp = math.sqrt(x*x + y*y)
    lat = math.atan2(z, hyp)
    return lon, lat


def enu_basis(lon: float, lat: float):
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)

    e = (-sin_lon, cos_lon, 0.0)
    n = (-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat)
    u = (cos_lat*cos_lon, cos_lat*sin_lon, sin_lat)
    return e, n, u


def interp_state(t_ref: datetime, dt: float,
                 r_list: List[Tuple[float, float, float]],
                 v_list: List[Tuple[float, float, float]],
                 t: datetime) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    ds = (t - t_ref).total_seconds()
    if ds <= 0:
        return r_list[0], v_list[0]

    idxf = ds / dt
    i0 = int(math.floor(idxf))
    i1 = i0 + 1

    if i0 >= len(r_list) - 1:
        return r_list[-1], v_list[-1]

    w = idxf - i0
    r0, r1 = r_list[i0], r_list[i1]
    v0, v1 = v_list[i0], v_list[i1]

    r = (r0[0] + w*(r1[0]-r0[0]),
         r0[1] + w*(r1[1]-r0[1]),
         r0[2] + w*(r1[2]-r0[2]))

    v = (v0[0] + w*(v1[0]-v0[0]),
         v0[1] + w*(v1[1]-v0[1]),
         v0[2] + w*(v1[2]-v0[2]))

    return r, v


def compute_heading(r: Tuple[float, float, float],
                    v: Tuple[float, float, float]) -> float:
    lon, lat = ecef_to_lon_lat_spherical(r)
    e_hat, n_hat, u_hat = enu_basis(lon, lat)

    v_up = dot(v, u_hat)
    v_h = sub(v, mul(v_up, u_hat))

    v_e = dot(v_h, e_hat)
    v_n = dot(v_h, n_hat)

    alpha = math.degrees(math.atan2(v_e, v_n))
    if alpha < 0:
        alpha += 360.0
    return alpha


def find_xemt_file(master_dir: str) -> str:
    if not os.path.isdir(master_dir):
        raise FileNotFoundError(f"master 不是一个目录: {master_dir}")

    pattern = os.path.join(master_dir, "*.xemt")
    files = glob.glob(pattern)

    if not files and RECURSIVE_SEARCH:
        pattern = os.path.join(master_dir, "**", "*.xemt")
        files = glob.glob(pattern, recursive=True)

    if not files:
        raise FileNotFoundError(f"在 master 路径下没有找到 .xemt 文件: {master_dir}")

    # 如果找到多个，优先选择带 “_AN_” 的 annotated 版本，其次选择第一个
    files_sorted = sorted(files)
    for f in files_sorted:
        if "_AN_" in os.path.basename(f):
            return f

    return files_sorted[0]


def main():
    xemt_path = find_xemt_file(master)
    print(f"找到 xemt: {xemt_path}")

    root = ET.parse(xemt_path).getroot()

    side = find_text_by_suffix(root, "sideLooking")
    if not side:
        raise ValueError("找不到 <sideLooking>")
    side = side.strip().lower()

    taz0 = find_text_by_suffix(root, "taz0_Utc")
    if not taz0:
        raise ValueError("找不到 <taz0_Utc>")
    t_target = parse_iso_utc(taz0)

    t_ref_s = find_text_by_suffix(root, "t_ref_Utc")
    dt_sv_s = find_text_by_suffix(root, "dtSV_s")
    if not t_ref_s or not dt_sv_s:
        raise ValueError("找不到 <t_ref_Utc> 或 <dtSV_s>")
    t_ref = parse_iso_utc(t_ref_s)
    dt_sv = float(dt_sv_s)

    p_vals = find_vals_under(root, "pSV_m")
    v_vals = find_vals_under(root, "vSV_mOs")
    if not p_vals or not v_vals:
        raise ValueError("找不到 <pSV_m> 或 <vSV_mOs>")

    r_list = group_xyz(p_vals)
    v_list = group_xyz(v_vals)

    r_t, v_t = interp_state(t_ref, dt_sv, r_list, v_list, t_target)
    alpha_a = compute_heading(r_t, v_t)

    if side.startswith("r"):
        alpha_r = (alpha_a + 90.0) % 360.0
        lr = "Right"
    elif side.startswith("l"):
        alpha_r = (alpha_a - 90.0) % 360.0
        lr = "Left"
    else:
        alpha_r = (alpha_a + 90.0) % 360.0
        lr = side

    print("\n=== 结果 ===")
    print(f"使用 xemt: {os.path.basename(xemt_path)}")
    print(f"使用时刻 taz0_Utc = {taz0}")
    print(f"sideLooking = {lr}")
    print(f"方位向方向角 / 航向角 alpha_a (deg, 从北顺时针) = {alpha_a:.6f}")
    print(f"距离向方向角 alpha_r (deg) = {alpha_r:.6f}")


if __name__ == "__main__":
    main()
