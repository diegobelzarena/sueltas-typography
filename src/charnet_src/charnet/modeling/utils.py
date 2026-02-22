# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import math
import numpy as np


def rotate_rect(x1, y1, x2, y2, degree, center_x, center_y):
    points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    new_points = list()
    for point in points:
        dx = point[0] - center_x
        dy = point[1] - center_y
        new_x = center_x + dx * math.cos(degree) - dy * math.sin(degree)
        new_y = center_y + dx * math.sin(degree) + dy * math.cos(degree)
        new_points.append([(new_x), (new_y)])
    return new_points


def rotate_rect_vectorized(x1, y1, x2, y2, orient, cx, cy):
    """Vectorised version of :func:`rotate_rect`.

    All arguments are 1-D numpy arrays of length N (one entry per
    detection).  Returns an (N, 8) array where each row is
    [p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y].
    """
    cos_o = np.cos(orient)
    sin_o = np.sin(orient)

    # Four corners of the axis-aligned rectangle
    # (x1,y1)  (x2,y1)  (x2,y2)  (x1,y2)
    dxs = np.stack([x1 - cx, x2 - cx, x2 - cx, x1 - cx], axis=1)  # (N, 4)
    dys = np.stack([y1 - cy, y1 - cy, y2 - cy, y2 - cy], axis=1)  # (N, 4)

    cos_o = cos_o[:, None]  # (N, 1)
    sin_o = sin_o[:, None]

    new_x = cx[:, None] + dxs * cos_o - dys * sin_o  # (N, 4)
    new_y = cy[:, None] + dxs * sin_o + dys * cos_o  # (N, 4)

    # Interleave x and y: [x0, y0, x1, y1, x2, y2, x3, y3]
    result = np.empty((x1.shape[0], 8), dtype=np.float32)
    result[:, 0::2] = new_x
    result[:, 1::2] = new_y
    return result
