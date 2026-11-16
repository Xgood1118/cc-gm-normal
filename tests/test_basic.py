# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from normalmap.core import Gray, height_to_normal  # noqa: E402


def test_gray_default():
    g = Gray(4, 4)
    assert g.at(1, 1) == 0
    assert len(g.data) == 16


def test_normal_flat():
    g = Gray(4, 4, [128] * 16)
    nm = height_to_normal(g)
    r, gg, b = nm.at(1, 1)
    assert abs(b - 255) <= 1


def test_normal_shape():
    g = Gray(8, 8, [i % 255 for i in range(64)])
    nm = height_to_normal(g)
    assert nm.width == 8 and nm.height == 8
    assert len(nm.data) == 64
