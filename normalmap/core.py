# -*- coding: utf-8 -*-
"""高度图转法线贴图（仅使用 Python 标准库）。

约定与实现要点（详见 README）：

- 切线空间采用 OpenGL 惯例：X 向右、Y 向上、Z 指向观察者，法线 (0,0,1)
  编码为 (128,128,255)。
- 高度图按 sRGB 解码为线性光值后再求中心差分梯度；``strength`` 只缩放
  线性梯度，不影响法线长度。
- 每条法线都归一化为单位向量，因此分量恒在 [-1,1] 内，突变边缘不会
  出现颜色溢出。
- 图像边界改用单侧差分（边缘像素用前向/后向差分），不钳制邻点，因此
  四周不会出现成片的错误平坦区。
- 输出 PNG 带 gAMA=1.0 与 cHRM（sRGB 原色）块，声明为线性数据，读回
  时像素值不做二次 gamma 转换。
"""

import math
import struct
import zlib


def _srgb_to_linear(c):
    """sRGB 传输函数（IEC 61966-2-1）：8 位码值 -> 线性光值 0..1。"""
    c = c / 255.0
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


class Gray(object):
    def __init__(self, width, height, data=None):
        self.width = int(width)
        self.height = int(height)
        self.data = data if data is not None else [0] * (self.width * self.height)

    def at(self, x, y):
        return self.data[y * self.width + x]


# ---------------------------------------------------------------------------
# PNG 读取（8 位；灰度 0 / 灰度+alpha 4 / RGB 2 / RGBA 6；支持全部 5 种行过滤）
# ---------------------------------------------------------------------------
def _load_png(path):
    with open(path, "rb") as f:
        raw_file = f.read()
    if raw_file[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("不是 PNG 文件")
    pos = 8
    idat = b""
    width = height = bit = ctype = None
    while pos < len(raw_file):
        length = struct.unpack(">I", raw_file[pos:pos + 4])[0]
        tag = raw_file[pos + 4:pos + 8]
        body = raw_file[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, bit, ctype = struct.unpack(">IIBB", body[:10])
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
    if bit != 8:
        raise ValueError("只支持 8 位")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(ctype)
    if channels is None:
        raise ValueError("不支持的颜色类型 %d" % ctype)
    raw = zlib.decompress(idat)
    stride = width * channels
    rows = []
    prev = bytearray(stride)
    for y in range(height):
        ft = raw[y * (stride + 1)]
        line = bytearray(raw[y * (stride + 1) + 1:
                             y * (stride + 1) + 1 + stride])
        if ft == 1:          # Sub
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif ft == 2:        # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ft == 3:        # Average
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif ft == 4:        # Paeth
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                if pa <= pb and pa <= pc:
                    pr = a
                elif pb <= pc:
                    pr = b
                else:
                    pr = c
                line[i] = (line[i] + pr) & 0xFF
        elif ft != 0:
            raise ValueError("未知 PNG 行过滤类型 %d" % ft)
        rows.append(bytes(line))
        prev = line
    return width, height, channels, rows


def load_gray_png(path):
    """读取灰度高度图，返回 Gray（取每个像素第一个通道的原始 8 位值）。

    不做色彩空间转换，sRGB 解码在 :func:`height_to_normal` 内完成。
    """
    width, height, channels, rows = _load_png(path)
    data = []
    for line in rows:
        for x in range(width):
            data.append(line[x * channels])
    return Gray(width, height, data)


def load_rgb_png(path):
    """读取 8 位 RGB/RGBA/灰度 PNG，返回 Gray，每像素存 (R, G, B) 三元组。

    不做任何 gamma/色彩空间转换：文件里存什么字节就返回什么字节，
    用于验证法线贴图读回后数值不被二次转换。
    """
    width, height, channels, rows = _load_png(path)
    data = []
    for line in rows:
        for x in range(width):
            i = x * channels
            if channels in (1, 2):
                v = line[i]
                data.append((v, v, v))
            else:
                data.append((line[i], line[i + 1], line[i + 2]))
    return Gray(width, height, data)


# ---------------------------------------------------------------------------
# PNG 写入（8 位；channels=3 RGB 或 0 灰度；无过滤）
# gAMA=1.0 + cHRM(sRGB/BT.709 原色) 声明输出为线性数据
# ---------------------------------------------------------------------------
def _chunk(tag, body):
    return (struct.pack(">I", len(body)) + tag + body +
            struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF))


# sRGB 原色色度（ITU-R BT.709），配合 gAMA=1.0 使用
_C_R = (0.640, 0.330)
_C_G = (0.300, 0.600)
_C_B = (0.150, 0.060)
_C_W = (0.3127, 0.3290)


def save_png(img, path, channels=3):
    """保存 Gray。channels=3 时每像素为 (R,G,B)，否则按单通道灰度保存。

    输出为线性数据：写入 gAMA(1.0) 与 cHRM 块声明色彩空间，任何合规
    读取方都不会再对像素做 gamma 转换。
    """
    raw = bytearray()
    for y in range(img.height):
        raw.append(0)  # filter type 0: None
        for x in range(img.width):
            if channels == 3:
                raw.extend(img.at(x, y))
            else:
                raw.append(img.at(x, y))
    ihdr = struct.pack(">IIBBBBB", img.width, img.height, 8,
                       2 if channels == 3 else 0, 0, 0, 0)
    gama = struct.pack(">I", 100000)  # gamma 1.0 -> 1/1.0 * 1e5
    chrm = struct.pack(">IIIIIIII",
                       int(_C_R[0] * 1e5), int(_C_R[1] * 1e5),
                       int(_C_G[0] * 1e5), int(_C_G[1] * 1e5),
                       int(_C_B[0] * 1e5), int(_C_B[1] * 1e5),
                       int(_C_W[0] * 1e5), int(_C_W[1] * 1e5))
    data = (b"\x89PNG\r\n\x1a\n" +
            _chunk(b"IHDR", ihdr) +
            _chunk(b"gAMA", gama) +
            _chunk(b"cHRM", chrm) +
            _chunk(b"IDAT", zlib.compress(bytes(raw), 9)) +
            _chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(data)


# ---------------------------------------------------------------------------
# 高度图 -> 切线空间法线贴图
# ---------------------------------------------------------------------------
def height_to_normal(height, strength=1.0):
    """由灰度高度图生成单位长度切线空间法线贴图。

    步骤：
      1. 8 位高度值按 sRGB 传输函数解码为线性高度（0..1）；
      2. 中心差分求梯度，边界像素改用单侧（前向/后向）差分；
      3. 法线取 (-gx, +gy_screen, 1)（OpenGL 惯例，Y 向上）并归一化；
      4. 按 n*0.5+0.5 量化为 8 位 RGB。

    几何说明：表面 z=h(x,y) 的法线水平分量与高度梯度反向。屏幕坐标
    y 向下、切线空间 Y 向上，故 nx=-dh/dx、ny=+dh/dy_screen。

    返回 Gray（RGB）。归一化后的浮点法线同时挂在 ``out.normals`` 上，
    元素为 (nx, ny, nz) 单位向量，便于无量化误差地校验长度与方向。
    """
    w, h = height.width, height.height
    lin = [_srgb_to_linear(v) for v in height.data]

    def lin_at(x, y):
        return lin[y * w + x]

    out = Gray(w, h, [(0, 0, 0)] * (w * h))
    normals = [(0.0, 0.0, 1.0)] * (w * h)

    for y in range(h):
        for x in range(w):
            # X 方向梯度：内部中心差分，边界单侧差分
            if w == 1:
                dx = 0.0
            elif x == 0:
                dx = lin_at(1, y) - lin_at(0, y)
            elif x == w - 1:
                dx = lin_at(w - 1, y) - lin_at(w - 2, y)
            else:
                dx = (lin_at(x + 1, y) - lin_at(x - 1, y)) * 0.5
            # Y 方向梯度（屏幕坐标，行号 y 向下增大）：同样内部中心、边界单侧
            if h == 1:
                dy_screen = 0.0
            elif y == 0:
                dy_screen = lin_at(x, 1) - lin_at(x, 0)
            elif y == h - 1:
                dy_screen = lin_at(x, h - 1) - lin_at(x, h - 2)
            else:
                dy_screen = (lin_at(x, y + 1) - lin_at(x, y - 1)) * 0.5

            gx = dx * strength
            gy = dy_screen * strength
            # OpenGL 切线空间：X 右、Y 上、Z 朝外。
            # 几何法线（切向量叉乘）水平分量与高度梯度反向；
            # 屏幕 y 向下而切线空间 Y 向上，故 nx=-gx、ny=+gy_screen。
            # 峰顶上方坡面 ny>0（G>128），下方坡面 ny<0（G<128）。
            nx, ny, nz = -gx, gy, 1.0
            inv_len = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
            nx *= inv_len
            ny *= inv_len
            nz *= inv_len
            normals[y * w + x] = (nx, ny, nz)

            def encode(v):
                return max(0, min(255, int(round(v * 127.5 + 127.5))))

            out.data[y * w + x] = (encode(nx), encode(ny), encode(nz))

    out.normals = normals
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="normalmap")
    ap.add_argument("height")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--out", default="normal.png")
    args = ap.parse_args(argv)
    height = load_gray_png(args.height)
    nm = height_to_normal(height, args.strength)
    save_png(nm, args.out)
    print("法线贴图 %dx%d -> %s" % (height.width, height.height, args.out))
    return 0
