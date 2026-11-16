# -*- coding: utf-8 -*-
"""高度图转法线贴图骨架。

梯度用中心差分但 Y 轴方向按屏幕坐标算，边界不补边，法线不做归一。
"""

import struct
import zlib


class Gray(object):
    def __init__(self, width, height, data=None):
        self.width = int(width)
        self.height = int(height)
        self.data = data or [0] * (self.width * self.height)

    def at(self, x, y):
        return self.data[y * self.width + x]


def load_gray_png(path):
    with open(path, "rb") as f:
        raw_file = f.read()
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
    raw = zlib.decompress(idat)
    ch = 4 if ctype == 6 else (3 if ctype == 2 else 1)
    stride = width * ch
    data = []
    for y in range(height):
        ft = raw[y * (stride + 1)]
        line = raw[y * (stride + 1) + 1: y * (stride + 1) + 1 + stride]
        if ft != 0:
            raise ValueError("只支持无过滤")
        for x in range(width):
            data.append(line[x * ch])
    return Gray(width, height, data)


def save_png(img, path, channels=3):
    raw = bytearray()
    for y in range(img.height):
        raw.append(0)
        for x in range(img.width):
            if channels == 3:
                raw.extend(img.at(x, y))
            else:
                raw.append(img.at(x, y))
    def chunk(tag, body):
        return (struct.pack(">I", len(body)) + tag + body +
                struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", img.width, img.height, 8, 2 if channels == 3 else 0, 0, 0, 0)
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) +
            chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(data)


def height_to_normal(height, strength=1.0):
    """中心差分求梯度，转成法线。当前 Y 方向反了，且不归一。"""
    w, h = height.width, height.height
    out = Gray(w, h, [(0, 0, 0)] * (w * h))
    for y in range(h):
        for x in range(w):
            xm = height.at(max(0, x - 1), y)
            xp = height.at(min(w - 1, x + 1), y)
            ym = height.at(x, max(0, y - 1))
            yp = height.at(x, min(h - 1, y + 1))
            dx = (xp - xm) / 255.0 * strength
            dy = (ym - yp) / 255.0 * strength
            n = (-dx, dy, 1.0)
            out.data[y * w + x] = (int((n[0] * 0.5 + 0.5) * 255),
                                   int((n[1] * 0.5 + 0.5) * 255),
                                   int((n[2] * 0.5 + 0.5) * 255))
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
