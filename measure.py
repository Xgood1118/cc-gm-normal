# -*- coding: utf-8 -*-
"""测量法线贴图的长度范围与圆台、斜脊处的倾斜角。

用法：
    python measure.py data/height.png normal.png [--strength 1.0]

输出：
- 浮点法线（量化前）全图长度最小/最大值；
- 从 PNG 读回的 8 位法线长度最小/最大值（验证无二次转换）；
- 圆台坡面与斜脊坡面的最大倾斜角及采样点倾斜角。
倾斜角定义为法线与 +Z 轴的夹角：acos(nz)，单位度。
"""
import argparse
import math

from normalmap.core import height_to_normal, load_gray_png, load_rgb_png


def tilt_deg(n):
    nx, ny, nz = n
    return math.degrees(math.acos(max(-1.0, min(1.0, nz))))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("height")
    ap.add_argument("normal")
    ap.add_argument("--strength", type=float, default=1.0)
    args = ap.parse_args(argv)

    h = load_gray_png(args.height)
    nm = height_to_normal(h, args.strength)
    w, hgt = h.width, h.height

    lengths = [math.sqrt(nx * nx + ny * ny + nz * nz) for nx, ny, nz in nm.normals]
    print("== 量化前浮点法线 ==")
    print("全图长度 min=%.6f max=%.6f" % (min(lengths), max(lengths)))

    back = load_rgb_png(args.normal)
    dec = [((r - 128) / 127.5, (g - 128) / 127.5, (b - 128) / 127.5)
           for r, g, b in back.data]
    dlen = [math.sqrt(x * x + y * y + z * z) for x, y, z in dec]
    print("== PNG 读回的 8 位法线 ==")
    print("全图长度 min=%.6f max=%.6f" % (min(dlen), max(dlen)))

    # 区域划分与 gen_samples.py 保持一致：
    # 圆台：到 (70,70) 距离 < 40；斜脊：|(x-120)*0.6-(y-20)| < 12
    cone, ridge = [], []
    for y in range(hgt):
        for x in range(w):
            d = math.hypot(x - 70, y - 70)
            r = abs((x - 120) * 0.6 - (y - 20))
            i = y * w + x
            if d < 40:
                cone.append((x, y, nm.normals[i]))
            if r < 12:
                ridge.append((x, y, nm.normals[i]))

    def report(name, pts):
        tilted = [(x, y, tilt_deg(n)) for x, y, n in pts]
        tilted.sort(key=lambda t: t[2], reverse=True)
        xm, ym, amax = tilted[0]
        print("== %s ==" % name)
        print("最大倾斜角 %.3f 度 @ (%d,%d)" % (amax, xm, ym))
        # 坡面中段代表点（倾斜角居中的点）
        mid = tilted[len(tilted) // 2]
        print("坡面中段代表点 (%d,%d) 倾斜角 %.3f 度" % mid)
        # 圆台外缘附近（半径 30~38 环带）采样
        ring = [t for t in tilted if 30 <= math.hypot(t[0] - 70, t[1] - 70) <= 38]
        if ring:
            ring.sort(key=lambda t: t[2], reverse=True)
            xm2, ym2, a2 = ring[0]
            print("圆台外缘环带最大倾斜角 %.3f 度 @ (%d,%d)" % (a2, xm2, ym2))

    report("圆台（中心 70,70 半径 40）", cone)
    report("斜脊", ridge)


if __name__ == "__main__":
    main()
