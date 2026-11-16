# -*- coding: utf-8 -*-
"""生成高度图：一个圆台加一条斜脊，边缘有突变。"""
import math
import os
import struct
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 160, 160


def main():
    raw = bytearray()
    for y in range(H):
        raw.append(0)
        for x in range(W):
            d = math.hypot(x - 70, y - 70)
            v = 0
            if d < 40:
                v = int(255 * (1 - d / 40.0))
            ridge = abs((x - 120) * 0.6 - (y - 20))
            if ridge < 12:
                v = max(v, int(255 * (1 - ridge / 12.0)))
            raw.extend((v, v, v))
    def chunk(tag, body):
        return (struct.pack(">I", len(body)) + tag + body +
                struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) +
            chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    path = os.path.join(HERE, "data", "height.png")
    with open(path, "wb") as f:
        f.write(data)
    print("已生成", path, W, "x", H)


if __name__ == "__main__":
    main()
