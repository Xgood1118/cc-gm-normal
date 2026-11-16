# -*- coding: utf-8 -*-
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from normalmap.core import (  # noqa: E402
    Gray,
    height_to_normal,
    load_gray_png,
    load_rgb_png,
    save_png,
    _srgb_to_linear,
)


def test_gray_default():
    g = Gray(4, 4)
    assert g.at(1, 1) == 0
    assert len(g.data) == 16


def test_normal_flat():
    g = Gray(4, 4, [128] * 16)
    nm = height_to_normal(g)
    r, gg, b = nm.at(1, 1)
    assert abs(b - 255) <= 1
    # 平面区域浮点法线严格指向正上方 (0,0,1)
    nx, ny, nz = nm.normals[1 * 4 + 1]
    assert (nx, ny) == (0.0, 0.0)
    assert nz == 1.0


def test_normal_shape():
    g = Gray(8, 8, [i % 255 for i in range(64)])
    nm = height_to_normal(g)
    assert nm.width == 8 and nm.height == 8
    assert len(nm.data) == 64


# ---------------------------------------------------------------------------
# 方向：凸起与凹陷的颜色方向相反（OpenGL 惯例，Y 向上）
# ---------------------------------------------------------------------------
def _linear_to_srgb_code(v):
    """sRGB 传输函数反函数：线性光值 -> 8 位码值。"""
    if v <= 0.0031308:
        c = 12.92 * v
    else:
        c = 1.055 * v ** (1.0 / 2.4) - 0.055
    return int(round(c * 255))


def _bump_height(w=7, h=7):
    """中心一个尖峰（码值 255=线性 1），四周为 0（线性 0）。

    只用 0/255 两个码值，凹陷图取反后线性高度恰好为 1-h，
    梯度严格反号。
    """
    data = [0] * (w * h)
    c = w // 2
    data[c * w + c] = 255
    return Gray(w, h, data)


def test_bump_and_dent_opposite():
    bump = _bump_height()
    nm_bump = height_to_normal(bump)
    # 凹陷 = 高度取反（255 - h），梯度整体反号
    dent = Gray(bump.width, bump.height, [255 - v for v in bump.data])
    nm_dent = height_to_normal(dent)

    w = bump.width
    # 尖峰右侧像素：高度向左（朝峰顶）上升，法线水平分量与梯度反向，
    # 故法线 X 分量为正（朝右、下坡侧）；凹陷整体反号。
    c = w // 2
    bx, by, _ = nm_bump.normals[c * w + (c + 1)]
    dx, dy, _ = nm_dent.normals[c * w + (c + 1)]
    assert bx > 0 and dx < 0
    assert math.copysign(1.0, bx) != math.copysign(1.0, dx)
    # 尖峰下方像素：高度向屏幕上方（朝峰顶）上升，ny=+dy_screen 应为负；
    # 凹陷反号为正。
    bx2, by2, _ = nm_bump.normals[(c + 1) * w + c]
    dx2, dy2, _ = nm_dent.normals[(c + 1) * w + c]
    assert by2 < 0 and dy2 > 0
    # 除正上方外，凹凸对应像素的 X/Y 分量全部反号
    for i in range(w * w):
        nx1, ny1, _ = nm_bump.normals[i]
        nx2, ny2, _ = nm_dent.normals[i]
        assert abs(nx1 + nx2) < 1e-12
        assert abs(ny1 + ny2) < 1e-12


def test_bump_encodes_upward_y():
    """OpenGL Y 向上惯例：尖峰上方的坡面法线 ny > 0，G 通道 > 128；
    下方坡面 ny < 0，G 通道 < 128。"""
    bump = _bump_height()
    nm = height_to_normal(bump)
    c = bump.width // 2
    _, ny_up, _ = nm.normals[(c - 1) * bump.width + c]
    _, ny_down, _ = nm.normals[(c + 1) * bump.width + c]
    assert ny_up > 0 and ny_down < 0
    assert nm.at(c, c - 1)[1] > 128
    assert nm.at(c, c + 1)[1] < 128


# ---------------------------------------------------------------------------
# 长度：所有法线为单位向量，误差 <= 1e-3；strength 只改倾斜角不改长度
# ---------------------------------------------------------------------------
def test_unit_length():
    g = Gray(8, 8, [(x * 17 + y * 31) % 256 for y in range(8) for x in range(8)])
    nm = height_to_normal(g, strength=2.5)
    for nx, ny, nz in nm.normals:
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        assert abs(length - 1.0) <= 1e-3


def test_strength_changes_tilt_not_length():
    g = Gray(6, 6, [(x * 40) % 256 for y in range(6) for x in range(6)])
    n1 = height_to_normal(g, strength=0.5).normals
    n2 = height_to_normal(g, strength=3.0).normals
    i = 3 * 6 + 2  # 内部像素
    t1 = math.acos(max(-1.0, min(1.0, n1[i][2])))
    t2 = math.acos(max(-1.0, min(1.0, n2[i][2])))
    assert t2 > t1  # strength 越大倾斜角越大
    for n in (n1, n2):
        for nx, ny, nz in n:
            assert abs(math.sqrt(nx * nx + ny * ny + nz * nz) - 1.0) <= 1e-3


# ---------------------------------------------------------------------------
# 边界：单侧差分，斜坡贴边时边缘像素仍倾斜，不成片平坦
# ---------------------------------------------------------------------------
def test_boundary_no_flat_band():
    # 每行从左到右线性升高的斜坡（线性空间等间隔，贴到左右边界）
    w, h = 6, 5
    g = Gray(w, h, [_linear_to_srgb_code(x / (w - 1))
                    for y in range(h) for x in range(w)])
    nm = height_to_normal(g)
    for y in range(h):
        for x in (0, w - 1):
            nx, _, nz = nm.normals[y * w + x]
            assert nx != 0.0           # 边缘不是平坦法线
            assert nz < 1.0
            r = nm.at(x, y)[0]
            assert r != 128            # 编码后也不是中性色平坦带
    # 线性等间隔斜坡上，单侧差分（边缘）与中心差分（内部）梯度一致
    # （容差来自反 sRGB 码值的 8 位量化）
    nx_in = nm.normals[2 * w + 3][0]
    nx_edge = nm.normals[2 * w + 0][0]
    assert abs(nx_in - nx_edge) < 5e-3


def test_flat_image_stays_flat_at_boundary():
    g = Gray(5, 5, [100] * 25)
    nm = height_to_normal(g)
    for i, (nx, ny, nz) in enumerate(nm.normals):
        assert (nx, ny, nz) == (0.0, 0.0, 1.0)
        assert nm.data[i] == (128, 128, 255)


# ---------------------------------------------------------------------------
# 突变边缘：归一化保证分量不超界，无颜色溢出
# ---------------------------------------------------------------------------
def test_no_overflow_at_discontinuity():
    w = h = 5
    data = [0] * (w * h)
    for y in range(h):
        for x in range(w):
            data[y * w + x] = 255 if x >= 2 else 0  # 黑白硬边
    nm = height_to_normal(Gray(w, h, data), strength=4.0)
    for r, g, b in nm.data:
        assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255
    for nx, ny, nz in nm.normals:
        assert -1.0 <= nx <= 1.0 and -1.0 <= ny <= 1.0 and -1.0 <= nz <= 1.0


# ---------------------------------------------------------------------------
# 色彩：sRGB 先转线性；输出线性 PNG，读回数值不被二次转换
# ---------------------------------------------------------------------------
def test_srgb_decode():
    assert _srgb_to_linear(0) == 0.0
    assert abs(_srgb_to_linear(255) - 1.0) < 1e-15
    mid = _srgb_to_linear(128)
    assert abs(mid - ((128 / 255.0 + 0.055) / 1.055) ** 2.4) < 1e-15
    assert mid < 128 / 255.0  # sRGB 中段解码后低于原值


def test_gradient_uses_linear_values():
    # 两个不同 sRGB 码值，线性化后梯度必须基于线性值而非码值
    g = Gray(3, 1, [0, 128, 255])
    nm = height_to_normal(g)
    lin_128 = _srgb_to_linear(128)
    # 左边缘单侧差分，梯度基于线性高度
    expect_gx = lin_128 - 0.0
    nx = nm.normals[0][0]
    expect_nx = -expect_gx / math.sqrt(expect_gx ** 2 + 1.0)
    assert abs(nx - expect_nx) < 1e-12
    # 若直接按 sRGB 码值差分，会得到明显不同的梯度，证明走了线性化
    naive_gx = 128 / 255.0
    assert abs(expect_gx - naive_gx) > 0.1


def test_png_roundtrip_exact(tmp_path):
    g = Gray(4, 3, [(0, 0, 0), (255, 128, 1), (17, 200, 99), (255, 255, 255),
                    (1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12),
                    (250, 128, 6), (0, 0, 255), (128, 128, 128), (33, 66, 99)])
    p = str(tmp_path / "n.png")
    save_png(g, p)
    back = load_rgb_png(p)
    assert back.width == g.width and back.height == g.height
    assert back.data == g.data  # 字节级一致，无二次 gamma


def test_output_png_declares_linear(tmp_path):
    g = Gray(2, 2, [(128, 128, 255)] * 4)
    p = str(tmp_path / "n.png")
    save_png(g, p)
    raw = open(p, "rb").read()
    assert b"gAMA" in raw and b"cHRM" in raw
    # gAMA 块数据为 100000（gamma 1.0，线性）
    idx = raw.index(b"gAMA")
    import struct
    assert struct.unpack(">I", raw[idx + 4:idx + 8])[0] == 100000


def test_end_to_end_on_synthetic(tmp_path):
    # 合成图走完整读写链路，量化后长度误差在 8 位量化容差内
    g = Gray(10, 10, [(x * 23 + y * 47) % 256 for y in range(10) for x in range(10)])
    nm = height_to_normal(g, strength=1.5)
    p = str(tmp_path / "normal.png")
    save_png(nm, p)
    back = load_rgb_png(p)
    for r, gg, b in back.data:
        nx, ny, nz = (r - 128) / 127.5, (gg - 128) / 127.5, (b - 128) / 127.5
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        assert abs(length - 1.0) < 0.02  # 8 位量化容差
