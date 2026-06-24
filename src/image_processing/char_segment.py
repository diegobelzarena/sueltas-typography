import numpy as np
import skimage as ski
from skimage.graph import route_through_array
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage import measure



# ---------------------------------------------------------------------------
# box_init segmentation (original CharNet path)
# ---------------------------------------------------------------------------

def find_hor_paths(img: np.ndarray,
                   tblrs: np.ndarray,
                   pen: float = 0.1
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Find paths delineating top and bottom edges of a text line.

    A list of character boxes in the same line is provided. Computed paths are
    incentivized to stay close to the boxes' top (resp. bottom) edges and to
    wrap around the text.

    Args:
        img: Grayscale text image, shape (h, w) and values in [0, 1].
        tblrs: Array of top, bottom, left, right edges of each box, shape (n, 4)
            and integer dtype.
        pen: Penalty parameter for deviating from box edges.

    Returns:
        Top and bottom paths, as arrays of (y, x) points of shape (L, 2) (where
        L >= w is the length of each path).
    """
    h, w = img.shape

    ## Extrapolate top and bottom edges of boxes into image-wide plots
    y_mins, y_maxs = np.zeros(w+2, dtype=int)+h-1, np.zeros(w+2, dtype=int) # horizontal 1-padding
    box_coverage = np.zeros(w+2, dtype=int) # int is important
    for t, b, l, r in tblrs:
        y_mins[l+1:r+2] = np.minimum(y_mins[l+1:r+2], t)
        y_maxs[l+1:r+2] = np.maximum(y_maxs[l+1:r+2], b)
        box_coverage[l+1:r+2] = 1
    # Ins (resp. outs): first position after getting in (resp. out) of box coverage
    # Padding ensures that it starts with an in and ends with an out
    io = box_coverage[1:] - box_coverage[:-1]
    ins, outs = np.nonzero(io==1)[0]+1, np.nonzero(io==-1)[0]+1

    # Handle edge case: no boxes or no valid transitions
    if len(ins) == 0 or len(outs) == 0:
        # No valid box coverage - use middle of image as fallback
        mid_y = h // 2
        y_mins[:] = mid_y - 10
        y_maxs[:] = mid_y + 10
    else:
        # Fill in the gaps between boxes
        for o, i in zip(outs[:-1], ins[1:]):
            y_mins[o:i] = (y_mins[o-1]+y_mins[i])/2
            y_maxs[o:i] = (y_maxs[o-1]+y_maxs[i])/2
        # Fill the left-most and right-most void
        y_mins[:ins[0]] = y_mins[ins[0]]
        y_maxs[:ins[0]] = y_maxs[ins[0]]
        y_mins[outs[-1]:] = y_mins[outs[-1]-1]
        y_maxs[outs[-1]:] = y_maxs[outs[-1]-1]
    # Remove padding
    y_mins, y_maxs = y_mins[1:-1], y_maxs[1:-1]

    # Build a mid-line barrier that shall not be crossed
    barrier = []
    for x, y in enumerate( ((y_mins+y_maxs)/2).astype(int) ):
        # Close vertical jumps
        if x > 0 and (y0 := barrier[-1][0]) != y:
            step = 1 if y0 < y else -1
            barrier.extend((y1, x) for y1 in range(y0, y, step))
        barrier.append([y, x])
    barrier = np.array(barrier) # shape (N, 2) with N >= w

    # Cost map for top and bottom paths
    yy, _ = np.indices((h, w))
    costs = 1-img

    # Avoid divide by zero: ensure denominators are at least 1
    denom_bottom = np.maximum(h - y_maxs, 1)
    denom_top = np.maximum(y_mins + 1, 1)
    denom_mid = np.maximum(y_maxs - y_mins + 1, 1)

    costs += pen * np.maximum((yy-y_maxs)/denom_bottom,
                              np.maximum((y_mins-yy)/denom_top, 0))
    costs += 3*pen * (np.maximum(0, np.minimum(yy-y_mins, y_maxs-yy))
                      / denom_mid * 2 )
    costs[*(barrier.T)] += 2*w

    # Clip start/end points to valid range [0, h-1]
    y_min_start = int(np.clip(y_mins[0], 0, h-1))
    y_min_end = int(np.clip(y_mins[-1], 0, h-1))
    y_max_start = int(np.clip(y_maxs[0], 0, h-1))
    y_max_end = int(np.clip(y_maxs[-1], 0, h-1))

    # Draw top and bottom paths
    tpath, _ = route_through_array(costs, [y_min_start, 0], [y_min_end, w-1],
                                   fully_connected=False, geometric=True)
    bpath, _ = route_through_array(costs, [y_max_start, 0], [y_max_end, w-1],
                                   fully_connected=False, geometric=True)

    return np.array(tpath), np.array(bpath)

def find_vert_paths(img: np.ndarray,
                    lrs: np.ndarray,
                    tpath: np.ndarray | None = None,
                    bpath: np.ndarray | None = None,
                    bgcost_init : float = 0.05,
                    bgcost_step : float = 0.15,
                    bgcost_max : float = 1.2
                    ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Find paths delineating left and right edges of characters in a line.

    Minimal cost paths are found iteratively by increasing the background cost
    (i.e. forcing more rigidity), until they are "nice" enough.

    Args:
        img: Grayscale text image, shape (h, w) and values in [0, 1].
        lrs: Array of left, right edges of each character box, shape (n, 2) and
            integer dtype.
        tpath: Array of (y, x) points on the line's top edge, shape (Lt, 2) and
            integer dtype.
        bpath: Array of (y, x) points on the line's top edge, shape (Lb, 2) and
            integer dtype.
        bgcost_init: Initial background cost.
        bgcost_step: Background cost increase at each iteration.
        bgcost_max: Maximum background cost.

    Returns:
        List of left and right paths, as an arrays of (y, x) points of shape
        (L, 2) (where L is the length of each path), for each character.
    """
    h, w = img.shape

    # Make paths into functions x -> y, by keeping extremal values
    if tpath is None:
        tops = np.zeros(w, dtype=int)
    else:
        fwd, bck = np.zeros(w, dtype=int), np.zeros(w, dtype=int)
        fwd[tpath[::-1, 1]] = tpath[::-1, 0]
        bck[tpath[:, 1]] = tpath[:, 0]
        tops = np.maximum(fwd, bck)
    if bpath is None:
        bots = np.zeros(w, dtype=int) + h-1
    else:
        fwd, bck = np.zeros(w, dtype=int)+h-1, np.zeros(w, dtype=int)+h-1
        fwd[bpath[::-1, 1]] = bpath[::-1, 0]
        bck[bpath[:, 1]] = bpath[:, 0]
        bots = np.minimum(fwd, bck)

    # Each left (resp. right) edge will be moved towards, in spirit, the right
    # (resp. left) edge of the previous (rep. next) box. Here we find these
    # target edges.
    r_targets = np.zeros(len(lrs), dtype=float)
    rpos, rxlocs = zip(*sorted(enumerate(lrs), key=lambda x: x[1][1]))
    rxlocs = np.array(rxlocs+([w-1, w-1],))
    for idx, (pos, (_, r_curr)) in enumerate(zip(rpos, rxlocs[:-1])):
        argmin = np.argmin(np.abs(rxlocs[idx+1:, :]-r_curr))
        r_targets[pos] = ( rxlocs[idx+1:, :].flatten()[argmin]
                         + r_curr )/2
    l_targets = np.zeros(len(lrs), dtype=float)
    lpos, lxlocs = zip(*sorted(enumerate(lrs), key=lambda x: x[1][0],
                               reverse=True))
    lxlocs = np.array(lxlocs+([0, 0],))
    for idx, (pos, (l_curr, _)) in enumerate(zip(lpos, lxlocs[:-1])):
        argmin = np.argmin(np.abs(lxlocs[idx+1:, :]-l_curr))
        l_targets[pos] = ( lxlocs[idx+1:, :].flatten()[argmin]
                         + l_curr )/2

    # Find left and right paths for each box
    paths_list = []
    gamma = 0.3
    for (l_init, r_init), l_target, r_target in zip(lrs, l_targets, r_targets):

        # Initialize while stop conditions variables
        cond1, cond2, cond3, cond4 = False, False, False, False
        # Initialize background cost
        bgcost = bgcost_init
        # Initialize starting points
        l, r = l_init, r_init

        while not((cond1 and cond2 and cond3) or cond4):

            # Find left and right minimum cost path
            lpath, _ = route_through_array((img**gamma)+bgcost,
                                           [tops[l], l], [bots[l], l],
                                           fully_connected=True,
                                           geometric=True)
            lpath = np.array(lpath)
            rpath, _ = route_through_array((img**gamma)+bgcost,
                                           [tops[r], r], [bots[r], r],
                                           fully_connected=True,
                                           geometric=True)
            rpath = np.array(rpath)

            # Compute the mean width of the paths
            aux_l = -1 + np.zeros(h, dtype=int)
            aux_r = -1 + np.zeros(h, dtype=int)
            aux_l[lpath[:,0]] = lpath[:,1]
            aux_r[rpath[:,0]] = rpath[:,1]
            lr_dif = (aux_r - aux_l)[(aux_l>=0)&(aux_r>=0)]+1
            # Compute the width of the starting points
            width =  r_init-l_init+1

            if len(lr_dif) == 0:
                cond1, cond2, cond3 = False, False, False
            else:
                # Condition 1: The maximum horizontal distance between the left and right paths is less than 4/3 of the width of the starting points
                cond1 = max(lr_dif) <= (4/3)*(width)
                # Condition 2: The minimum horizontal distance between the left and right paths is greater than 2/3 of the width of the starting points
                cond2 = min(lr_dif) >= (2/3)*(width)
                # Condition 3: The mean width of the paths is within 10% of the width of the starting points
                cond3 = abs((np.mean(lr_dif) - width)/width) <= 0.1
            # Condition 4: The vertical cost exceeds 1.2
            cond4 = bgcost > bgcost_max

            # If the conditions are not met, increase the background cost
            bgcost += bgcost_step
            # If the conditions are not met, move the starting points closer to those of the contiguous characters
            l += 2*int(np.sign(np.round((l_target - l)/2)))
            r += 2*int(np.sign(np.round((r_target - r)/2)))

            gamma = min(0.9, gamma + 0.05)

        paths_list.append([lpath, rpath])

    return paths_list

def char_segment(img_c, tblrs, box_clu, refwidth):

    # Global padding
    img_c = np.pad(img_c, pad_width=1, mode='constant', constant_values=1)
    tblrs += 1

    h, w = img_c.shape
    # Produce character segmentation, for each line element
    char_data, idxs_input = [], []
    for i in range(1, box_clu.max()+1):
        boxes = tblrs[box_clu==i].copy()
        idxs, = np.nonzero(box_clu==i)
        if len(boxes) == 0:
            continue
        # Crop the line element
        l0 = max(0, boxes[boxes[:, 2].argmin(), 2] - int(refwidth))
        r0 = min(w, boxes[boxes[:, 3].argmax(), 3] + int(refwidth) + 1)
        crop = img_c[:, l0:r0]
        boxes[:, 2:] -= l0
        # Find top and bottom edges of the line element, draw them on canvas
        tpath, bpath = find_hor_paths(crop, boxes, pen=0.1)
        canv = np.zeros(crop.shape, dtype=bool)
        canv[*(tpath.T)] = True
        canv[*(bpath.T)] = True
        # Find left and right edges of each character
        loop = zip(idxs, find_vert_paths((1-crop), boxes[:, 2:], tpath, bpath))
        for idx, (lpath, rpath) in loop:
            # Draw a character canvas, with 1-padding for easy CC finding
            canv_char = canv.copy()
            canv_char[*(lpath.T)] = True
            canv_char[*(rpath.T)] = True
            l, r = lpath[:, 1].min(), rpath[:, 1].max()
            canv_char = np.pad(canv_char[:, l:r+1], pad_width=1)
            # Get mask as the complementary of the edge connected component
            char_comps = ski.measure.label(~canv_char, connectivity=1)
            premask = (char_comps != char_comps[0, 0])[1:-1, 1:-1] # Padding removed
            # Remove white pixels from mask
            premask &= (crop[:, l:r+1] != 1)
            # If mask becomes empty, discard character
            if not premask.any():
                continue
            # Update character bounding box
            hprojs, = np.nonzero(np.any(premask, axis=1))
            vprojs, = np.nonzero(np.any(premask, axis=0))
            t, b = np.min(hprojs), np.max(hprojs)+1
            pre_l, pre_r = np.min(vprojs), np.max(vprojs)+1
            r = l0 + l + pre_r # Cropping undone
            l += l0 + pre_l # Cropping undone
            mask = premask[t:b, pre_l:pre_r]
            # Remove global padding
            dt, dl = max(0, t-1)-t, max(0, l-1)-l # values 0 (if 0) or -1 (else)
            db, dr = min(h-2, b-1)-b, min(w-2, r-1)-r # values -2 (if h/w) or -1 (else)
            mask = mask[dt+1:b-t+db+1, dl+1:r-l+dr+1]
            t, b, l, r = t+dt, b+db, l+dl, r+dr
            # Store
            char_data.append(((t, b, l, r), mask.astype(bool)))
            idxs_input.append(idx.item())
    return char_data, idxs_input


# ---------------------------------------------------------------------------
# logit_init segmentation (DocTR path — uses CRNN-logit-derived boxes)
# ---------------------------------------------------------------------------

def find_hor_paths_logit_init(
    img: np.ndarray,
    tblrs: np.ndarray,
    top_pad: float = 1,
    bottom_pad: float = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Find top/bottom text-line edges using blur-based midline + multi-path routing.

    Unlike :func:`find_hor_paths` (which extrapolates box edges into a barrier),
    this variant estimates a midline from a blurred intensity profile and
    selects the cheapest non-crossing paths above and below it.

    Args:
        img: Grayscale text image, shape (h, w) and values in [0, 1].
        tblrs: Character boxes, shape (n, 4), integer.
        top_pad: Vertical padding above text (pixels).
        bottom_pad: Vertical padding below text (pixels).

    Returns:
        Top and bottom paths as arrays of (y, x) points.
    """
    h, w = img.shape

    len_c = int(w / len(tblrs))
    len_c += len_c % 2 + 1

    # Approximate midline from blurred intensity
    blur = cv2.GaussianBlur(
        img, (len_c, 17), sigmaX=len_c / 2, sigmaY=9,
        borderType=cv2.BORDER_REFLECT,
    )
    top_pad_int = int(top_pad)
    bottom_pad_int = int(bottom_pad)
    weights = 1 - blur[top_pad_int:-(bottom_pad_int), :]
    y = np.arange(weights.shape[0])[:, None]
    denom = weights.sum(axis=0) + 1e-12
    yc = np.int8((weights * y).sum(axis=0) / denom) + top_pad_int

    midline = np.clip(
        np.stack([yc + i for i in np.arange(-h // 5, h // 5 + 1)]),
        0, h - 1,
    )
    # Cost image: blend of blur and raw, with midline barrier
    cost_img = 1 - ((3 * blur + img) / 4)
    cost_img[midline, np.arange(w)] = 1000

    costs = []
    paths = []
    for y_start in range(top_pad_int // 2, h - (bottom_pad_int // 2)):
        path, cost = route_through_array(
            cost_img, [y_start, 0], [y_start, w - 1],
            fully_connected=True, geometric=True,
        )
        path = np.array(path)
        paths.append(path)
        costs.append(cost * (1 - cost_img[path[:, 0], path[:, 1]]).min())

    # Build path-density image
    path_img = np.zeros_like(img, dtype=int)
    for path in paths:
        path_img[*(path.T)] += 1

    # Remove paths crossing midline
    remove_idxs = []
    for i, path in enumerate(paths):
        for py, px in path:
            if abs(py - yc[px]) < h // 5:
                remove_idxs.append(i)
                break
    for idx in remove_idxs[::-1]:
        paths.pop(idx)
        costs.pop(idx)

    # Rebuild density after filtering
    path_img = np.zeros_like(img, dtype=int)
    for path in paths:
        path_img[*(path.T)] += 1

    if len(paths) == 0:
        default_top = np.column_stack([np.full(w, top_pad_int), np.arange(w)])
        default_bot = np.column_stack([np.full(w, h - bottom_pad_int - 1), np.arange(w)])
        return default_top, default_bot

    # Normalize costs by mean path density
    cost_agg = np.array(costs)
    for i, path in enumerate(paths):
        cost_agg[i] /= path_img[path[:, 0], path[:, 1]].mean()

    clas = np.array([path[0, 0] > yc[0] for path in paths])

    has_top = np.any(clas == 0)
    has_bottom = np.any(clas == 1)
    if not has_top or not has_bottom:
        default_top = np.column_stack([np.full(w, top_pad_int), np.arange(w)])
        default_bot = np.column_stack([np.full(w, h - bottom_pad_int - 1), np.arange(w)])
        return default_top, default_bot

    c1_s = np.argmin(clas == 0)
    p1 = np.argmin(cost_agg[clas == 0])
    p2 = np.argmin(cost_agg[clas == 1]) + c1_s

    return np.array(paths[p1]), np.array(paths[p2])


def find_vert_paths_logit_init(
    img: np.ndarray,
    lrs: np.ndarray,
    tpath: np.ndarray | None = None,
    bpath: np.ndarray | None = None,
    bgcost_init: float = 0.3,
    bgcost_step: float = 0.3,
    bgcost_max: float = 1.75,
    widths: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    """Find left/right character edges with italic-aware path routing.

    Like :func:`find_vert_paths` but adds italic-slanted start-point
    offsets and returns a third *proximity path* per character that can
    optionally replace the right boundary.

    Returns:
        List of (lpath, rpath, prox_path) tuples for each character.
        *prox_path* is the next character's left path for boundary
        negotiation, or ``None`` for the last character.
    """
    h, w = img.shape
    tan_theta = 0.2679491924311227  # tan(pi/12)
    cos_theta = 0.9659258262890683  # cos(pi/12)

    # Make paths into functions x -> y
    if tpath is None:
        tops = np.zeros(w, dtype=int)
    else:
        fwd, bck = np.zeros(w, dtype=int), np.zeros(w, dtype=int)
        fwd[tpath[::-1, 1]] = tpath[::-1, 0]
        bck[tpath[:, 1]] = tpath[:, 0]
        tops = np.maximum(fwd, bck)
    if bpath is None:
        bots = np.zeros(w, dtype=int) + h - 1
    else:
        fwd, bck = np.zeros(w, dtype=int) + h - 1, np.zeros(w, dtype=int) + h - 1
        fwd[bpath[::-1, 1]] = bpath[::-1, 0]
        bck[bpath[:, 1]] = bpath[:, 0]
        bots = np.minimum(fwd, bck)

    # Target edges for iterative adjustment
    r_targets = np.zeros(len(lrs), dtype=float)
    rpos, rxlocs = zip(*sorted(enumerate(lrs), key=lambda x: x[1][1]))
    rxlocs = np.array(rxlocs + ([w - 1, w - 1],))
    for idx, (pos, (_, r_curr)) in enumerate(zip(rpos, rxlocs[:-1])):
        argmin = np.argmin(np.abs(rxlocs[idx + 1:, :] - r_curr))
        r_targets[pos] = (rxlocs[idx + 1:, :].flatten()[argmin] + r_curr) / 2
    l_targets = np.zeros(len(lrs), dtype=float)
    lpos, lxlocs = zip(*sorted(enumerate(lrs), key=lambda x: x[1][0], reverse=True))
    lxlocs = np.array(lxlocs + ([0, 0],))
    for idx, (pos, (l_curr, _)) in enumerate(zip(lpos, lxlocs[:-1])):
        argmin = np.argmin(np.abs(lxlocs[idx + 1:, :] - l_curr))
        l_targets[pos] = (lxlocs[idx + 1:, :].flatten()[argmin] + l_curr) / 2

    paths_list = []
    for (l_init, r_init), l_target, r_target in zip(lrs, l_targets, r_targets):
        cond1, cond2, cond3, cond4 = False, False, False, False
        bgcost = bgcost_init
        l, r = l_init, r_init
        gamma = 5
        try_italic = False
        h_ = min(bots[l], bots[r]) - max(tops[l], tops[r])
        got_right = False
        while (not ((cond1 and cond2 and cond3) or cond4) or try_italic) and not (not try_italic and got_right):
            l_top = min(l + int(1.5 * tan_theta * h_ / 2) * try_italic, w - 1)
            l_bot = max(l - int(0.5 * tan_theta * h_ / 2) * try_italic, 0)
            r_top = min(r + int(1.5 * tan_theta * h_ / 2) * try_italic, w - 1)
            r_bot = max(r - int(0.5 * tan_theta * h_ / 2) * try_italic, 0)

            cost_img = (img[:, :] + (1 - bgcost)) ** gamma
            cost_img += bgcost
            lpath, cost_l = route_through_array(
                cost_img, [tops[l_top], l_top], [bots[l_bot], l_bot],
                fully_connected=True, geometric=True,
            )
            lpath = np.array(lpath)
            rpath, cost_r = route_through_array(
                cost_img, [tops[r_top], r_top], [bots[r_bot], r_bot],
                fully_connected=True, geometric=True,
            )
            rpath = np.array(rpath)

            aux_l = -1 + np.zeros(h, dtype=int)
            aux_r = -1 + np.zeros(h, dtype=int)
            aux_l[lpath[:, 0]] = lpath[:, 1]
            aux_r[rpath[:, 0]] = rpath[:, 1]
            lr_dif = (aux_r - aux_l)[(aux_l >= 0) & (aux_r >= 0)] + 1
            width = max(r_bot - l_bot, r_top - l_top) + 1

            if len(lr_dif) == 0:
                cond1, cond2, cond3 = False, False, False
            else:
                cond1 = (max(lr_dif) <= (3 / 2) * width) and (lpath[:, 1].max() < (r_top + r_bot) / 2) and (lpath[:, 1].min() > ((l_top + l_bot) / 2) - 3 * width / 4)
                cond2 = (min(lr_dif) >= (1 / 2) * width) and (rpath[:, 1].min() > (l_top + l_bot) / 2) and (rpath[:, 1].max() < ((r_top + r_bot) / 2) + 3 * width / 4)
                cond3 = abs((np.mean(lr_dif) - width) / width) <= 0.33
            cond4 = bgcost > bgcost_max

            bgcost += bgcost_step * try_italic
            gamma = int(max(1, gamma - 1 * try_italic))

            if cond1 and cond2 and cond3:
                if not try_italic:
                    got_right = True
                    right_cost = cost_l * img[lpath[:, 0], lpath[:, 1]].max() + cost_r * img[rpath[:, 0], rpath[:, 1]].max()
                    right_l = lpath
                    right_r = rpath
                elif got_right:
                    if (cost_l * img[lpath[:, 0], lpath[:, 1]].max() + cost_r * img[rpath[:, 0], rpath[:, 1]].max()) * cos_theta > right_cost:
                        lpath = right_l
                        rpath = right_r
                else:
                    got_right = True
            elif got_right:
                lpath = right_l
                rpath = right_r
            try_italic = not try_italic

            l = int(l_init)
            r = int(r_init)

        if len(paths_list) > 0:
            paths_list[-1].extend([lpath])
        paths_list.append([lpath, rpath])
    paths_list[-1].extend([None])

    return paths_list


def char_segment_logit_init(img_c, tblrs, box_clu, refwidth, top_pad, bottom_pad, widths,
                            img_flat=None):
    """Character segmentation using logit-initialised boxes (DocTR path).

    Same contract as :func:`char_segment` but uses
    :func:`find_hor_paths_logit_init` / :func:`find_vert_paths_logit_init`
    and includes proximity-path boundary negotiation.

    Args:
        img_c: Grayscale image, shape (h, w), values in [0, 1].
            Used for path finding (horizontal + vertical cost paths).
        tblrs: Character boxes, shape (n, 4).
        box_clu: Line cluster assignments, shape (n,).
        refwidth: Mean character width (float).
        top_pad: Top padding size (pixels).
        bottom_pad: Bottom padding size (pixels).
        widths: Per-character widths array.
        img_flat: Optional bg-flattened image, shape (h, w), values in
            [0, 1].  When provided, mask filtering (white-pixel removal)
            uses this image instead of *img_c*.  This lets path finding
            exploit full grayscale contrast while masks are cleaned
            against the flattened background.

    Returns:
        char_data: list of ((t, b, l, r), mask) tuples.
    """
    if img_flat is None:
        img_flat = img_c
    h, w = img_c.shape
    char_data = []
    for i in range(1, box_clu.max() + 1):
        boxes = tblrs[box_clu == i].copy()
        idxs, = np.nonzero(box_clu == i)
        if len(boxes) == 0:
            continue
        l0 = max(0, boxes[boxes[:, 2].argmin(), 2] - int(refwidth))
        r0 = min(w, boxes[boxes[:, 3].argmax(), 3] + int(refwidth) + 1)
        crop = img_c[:, l0:r0]
        crop_flat = img_flat[:, l0:r0]
        boxes[:, 2:] -= l0
        tpath, bpath = find_hor_paths_logit_init(crop, boxes, top_pad, bottom_pad)
        canv = np.zeros(crop.shape, dtype=bool)
        canv[*(tpath.T)] = True
        canv[*(bpath.T)] = True
        loop = zip(idxs, find_vert_paths_logit_init(1 - crop, boxes[:, 2:], tpath, bpath, widths=widths))
        last_pth = None
        for idx, (lpath, rpath, prox_path) in loop:
            # Boundary negotiation: prefer proximity path if it has lower cost
            if prox_path is not None:
                if 1 - crop[prox_path[:, 0], prox_path[:, 1]].mean() < 1 - crop[rpath[:, 0], rpath[:, 1]].mean():
                    aux_l = -1 + np.zeros(h, dtype=int)
                    aux_r = -1 + np.zeros(h, dtype=int)
                    aux_l[lpath[:, 0]] = lpath[:, 1]
                    aux_r[prox_path[:, 0]] = prox_path[:, 1]
                    lr_dif = (aux_r - aux_l)[(aux_l >= 0) & (aux_r >= 0)] + 1
                    lr_dif = np.clip(lr_dif, a_min=1, a_max=None)
                    box_w = boxes[idxs == idx, 3] - boxes[idxs == idx, 2] + 1
                    if max(lr_dif) <= 2 * box_w and min(lr_dif) >= (1 / 3) * box_w:
                        rpath = prox_path
            if last_pth is not None:
                if 1 - crop[last_pth[:, 0], last_pth[:, 1]].mean() < 1 - crop[lpath[:, 0], lpath[:, 1]].mean():
                    aux_l = -1 + np.zeros(h, dtype=int)
                    aux_r = -1 + np.zeros(h, dtype=int)
                    aux_l[last_pth[:, 0]] = last_pth[:, 1]
                    aux_r[rpath[:, 0]] = rpath[:, 1]
                    lr_dif = (aux_r - aux_l)[(aux_l >= 0) & (aux_r >= 0)] + 1
                    lr_dif = np.clip(lr_dif, a_min=1, a_max=None)
                    box_w = boxes[idxs == idx, 3] - boxes[idxs == idx, 2] + 1
                    if max(lr_dif) <= 2 * box_w and min(lr_dif) >= (1 / 3) * box_w:
                        lpath = last_pth
            last_pth = rpath
            canv_char = canv.copy()
            canv_char[*(lpath.T)] = True
            canv_char[*(rpath.T)] = True
            l, r = lpath[:, 1].min(), rpath[:, 1].max()
            canv_char = np.pad(canv_char[:, l:r + 1], pad_width=1)
            char_comps = ski.measure.label(~canv_char, connectivity=1)
            premask = (char_comps != char_comps[0, 0])[1:-1, 1:-1]
            premask &= (crop_flat[:, l:r + 1] != 1)
            if not premask.any():
                continue
            hprojs, = np.nonzero(np.any(premask, axis=1))
            vprojs, = np.nonzero(np.any(premask, axis=0))
            t, b = np.min(hprojs), np.max(hprojs) + 1
            pre_l, pre_r = np.min(vprojs), np.max(vprojs) + 1
            r = l0 + l + pre_r
            l += l0 + pre_l
            mask = premask[t:b, pre_l:pre_r]
            char_data.append(((t, b, l, r), mask.astype(bool)))
    return char_data



# ---------------------------------------------------------------------------
# kraken_init segmentation (Kraken path — uses Kraken-logit-derived boxes)
# ---------------------------------------------------------------------------

def extend_baseline(baseline: np.ndarray, new_x1: int, new_x2: int) -> np.ndarray:
    """
    Extend a 2-point baseline to new X coordinates.
    
    Args:
        baseline: Array of shape (2, 2) representing [[x1, y1], [x2, y2]]
        new_x1: Target left X coordinate
        new_x2: Target right X coordinate
        
    Returns:
        Array of shape (2, 2) with the extended baseline points
    """
    xs = baseline[:, 0]
    ys = baseline[:, 1]
    
    # np.interp handles extrapolation automatically when given values
    # outside the original xs range
    new_ys = np.interp([new_x1, new_x2], xs, ys).astype(int)
    
    return np.array([[new_x1, new_ys[0]], 
                     [new_x2, new_ys[1]]])


def find_hor_paths_kraken_init(
    img: np.ndarray,
    tblrs: np.ndarray,
    boundary: np.ndarray,
    baseline: np.ndarray,
    top_pad: float = 1,
    bottom_pad: float = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Find top/bottom text-line edges using kraken baseline + multi-path routing.

    Args:
        img: Grayscale text image, shape (h, w) and values in [0, 1].
        tblrs: Character boxes, shape (n, 4), integer.
        boundary: Kraken boundary, shape (L, 2), integer.
        baseline: Kraken baseline, shape (2, 2), integer.
        top_pad: Vertical padding above text (pixels).
        bottom_pad: Vertical padding below text (pixels).

    Returns:
        Top and bottom paths as arrays of (y, x) points.
    """
    h, w = img.shape

    len_c = int(w / len(tblrs))
    len_c += len_c % 2 + 1

    # extend bounndary to image width
    bx_min, bx_max = boundary[:, 0].min(), boundary[:, 0].max()
    ex_boundary = boundary.copy()
    ex_boundary[:, 0] = (ex_boundary[:, 0] - bx_min) * (w - 1) / (bx_max - bx_min)

    # extend baseline to image width
    ex_baseline = extend_baseline(baseline, 0, w)

    # blur original boundary
    blur = np.zeros(img.shape, dtype=np.float32)
    blur = cv2.polylines(blur, [ex_boundary], isClosed=True, color=1, thickness=1)
    blur = cv2.GaussianBlur(
        blur, (len_c*2 + 1, len_c*2 + 1), 0,
        borderType=cv2.BORDER_REFLECT,
    )
    # invert so minimum cost is given to original boundary
    blur = blur.max() - blur
    # add img so that the cost is not only based on original boundary
    blur+= cv2.GaussianBlur(1 - img, (len_c, 3), 0, borderType=cv2.BORDER_REPLICATE)

    # normalize 
    cost_img = (blur.copy() - blur.min()) / (blur.max() - blur.min() + 1e-12)

    # add basline as no-crossing barrier
    baseline = cv2.line(np.zeros_like(cost_img), tuple(ex_baseline[0]), tuple(ex_baseline[1]), color=1e3, thickness=2)
    # Convert to a boolean mask (True where the line is drawn)
    baseline_mask = baseline > 0 

    # add baseline barrier
    cost_img[baseline_mask] = np.inf
    
    costs = []
    paths = []

    for y_start in range(top_pad // 2, h - (bottom_pad // 2)):
        # skip paths that start or end in baselie
        if baseline_mask[y_start, 0] or baseline_mask[y_start, w - 1]:
            continue
        try:
            path, cost = route_through_array(
                cost_img, [y_start, 0], [y_start, w - 1],
                fully_connected=True, geometric=False,
            )
            path = np.array(path)
            # If ANY point in the path lands on a True pixel in the mask, skip to the next y_start
            if baseline_mask[path[:, 0], path[:, 1]].any():
                continue
            paths.append(path)
            costs.append(cost * (1 - cost_img[path[:, 0], path[:, 1]]).min())
        except ValueError:
            continue



    # Build density after filtering
    path_img = np.zeros_like(img, dtype=int)
    for path in paths:
        path_img[*(path.T)] += 1


    if len(paths) == 0:
        default_top = np.column_stack([np.full(w, top_pad), np.arange(w)])
        default_bot = np.column_stack([np.full(w, h - bottom_pad - 1), np.arange(w)])
        return default_top, default_bot

    # Normalize costs by mean path density
    cost_agg = np.array(costs)
    for i, path in enumerate(paths):
        cost_agg[i] /= path_img[path[:, 0], path[:, 1]].mean()

    clas = np.array([path[0, 0] > ex_baseline[0][1] for path in paths])

    has_top = np.any(clas == 0)
    has_bottom = np.any(clas == 1)
    if not has_top or not has_bottom:
        default_top = np.column_stack([np.full(w, top_pad), np.arange(w)])
        default_bot = np.column_stack([np.full(w, h - bottom_pad - 1), np.arange(w)])
        return default_top, default_bot

    c1_s = np.argmin(clas == 0)
    p1 = np.argmin(cost_agg[clas == 0])
    p2 = np.argmin(cost_agg[clas == 1]) + c1_s

    return np.array(paths[p1]), np.array(paths[p2])

def get_connected_components(crop_gray_norm, tb_mask):
    """Computes the connected components. This only needs to be run ONCE."""
    thld, thldd = cv2.threshold((crop_gray_norm*255).astype(np.uint8), 0, 255, 
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    thldd = crop_gray_norm < ((thld/255.) ** (3/4))
    
    label_img = measure.label(thldd * tb_mask, connectivity=1)
    
    cc_areas = np.bincount(label_img.ravel())
    
    return label_img, thld/255., cc_areas

def assign_ccs_to_tblrs(label_img, tblrs, chars_labels, threshold=0.75):
    """Assigns CCs to the provided bounding boxes based on area overlap."""
    h, w = label_img.shape
    num_tblrs = len(tblrs)
    
    # Create the map of current TBLRs
    tblr_map = np.zeros((h, w), dtype=np.uint8)
    for i, tblr in enumerate(tblrs):
        if chars_labels[i] == ' ':
            continue
        t, b, l, r = tblr
        cv2.rectangle(tblr_map, (int(l), int(t)), (int(r), int(b)), color=i+1, thickness=-1)
        
    max_cc_label = label_img.max()
    if max_cc_label == 0:
        return np.zeros_like(label_img, dtype=int), tblr_map
        
    # Compute intersections using bincount
    combined_ids = tblr_map.astype(np.int32) * (max_cc_label + 1) + label_img.astype(np.int32)
    counts = np.bincount(combined_ids.ravel(), minlength=(num_tblrs + 1) * (max_cc_label + 1))
    intersection_matrix = counts.reshape(num_tblrs + 1, max_cc_label + 1)
    
    cc_total_areas = np.bincount((label_img * (tblr_map!=0)).ravel(), minlength=max_cc_label + 1)
    ratio_matrix = intersection_matrix / (cc_total_areas + 1e-6)
    
    # Assign CCs
    aux_labels = np.zeros_like(label_img, dtype=int)
    for t in range(1, num_tblrs + 1):
        valid_ccs = np.where(ratio_matrix[t, 1:] > threshold)[0] + 1 
        for cc_id in valid_ccs:
            aux_labels[label_img == cc_id] = t
            
    return aux_labels, tblr_map

def adjust_tblrs(tblrs, aux_labels):
    """Adjusts the left/right borders of the TBLRs based on assigned CCs and neighbors."""
    num_tblrs = len(tblrs)
    adjusted_tblrs = tblrs.copy()
    
    ys, xs = np.where(aux_labels > 0)
    labels = aux_labels[ys, xs]
    
    min_x = np.full(num_tblrs + 1, np.inf)
    max_x = np.full(num_tblrs + 1, -np.inf)
    
    # Safety check: only calculate extremes if there are actually assigned pixels
    if len(labels) > 0:
        np.minimum.at(min_x, labels, xs)
        np.maximum.at(max_x, labels, xs)
    
    for t in range(1, num_tblrs + 1):
        if min_x[t] != np.inf:
            orig_l = tblrs[t-1, 2]
            orig_r = tblrs[t-1, 3]
            
            cc_left = min_x[t]
            cc_right = max_x[t]

            # Calculate new Left border
            if (t-1 >= 0) and max_x[t-1] != -np.inf:
                # find mean of current left and previous right
                new_l = (max_x[t-1] + min_x[t]) // 2
                diff_l = (new_l - orig_l) // 3
            else:
                # find mean of current left and assigned CC left
                new_l = (orig_l + int(cc_left)) // 2
                diff_l = (new_l - orig_l) // 2
                
            # Calculate new Right border
            if (t+1 < num_tblrs + 1) and min_x[t+1] != np.inf:
                new_r = (min_x[t+1] + max_x[t]) // 2
                diff_r = (new_r - orig_r) // 3
            else:
                new_r = (orig_r + int(cc_right)) // 2
                diff_r = (new_r - orig_r) // 2
            
            # Apply to current box
            adjusted_tblrs[t-1, 2] = orig_l + diff_l
            adjusted_tblrs[t-1, 3] = orig_r + diff_r
            
            # Apply to neighbors to ensure contiguous borders
            if t-2 >= 0:
                adjusted_tblrs[t-2, 3] = tblrs[t-2, 3] + diff_l
            if t < num_tblrs:
                adjusted_tblrs[t, 2] = tblrs[t, 2] + diff_r
                
    return adjusted_tblrs

def visualize_results(tblrs, aux_labels, chars_labels, shape, title="Assigned CCs"):
    """Helper function to visualize the current state."""
    h, w = shape
    col_labels = np.zeros((h, w, 3), dtype=np.uint8)
    num_tblrs = len(tblrs)
    
    for t in range(1, num_tblrs + 1):
        t_coord, b_coord, l_coord, r_coord = tblrs[t-1]
        col_labels = cv2.rectangle(col_labels, (int(l_coord), int(t_coord)), (int(r_coord), int(b_coord)), color=(255, 255, 255), thickness=1)
        
        if chars_labels and t-1 < len(chars_labels):
            col_labels = cv2.putText(col_labels, str(chars_labels[t-1]), (int(l_coord), int(t_coord)-2), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        mask = aux_labels == t
        color = np.random.randint(0, 255, size=3).tolist()
        col_labels[mask] = color

    plt.figure(figsize=[20,6])
    plt.imshow(col_labels)
    plt.title(title)
    plt.show()


def resolve_divided_ccs(orig_img, label_img, aux_labels, tblrs, chars_labels, thld_iters=1, thld=0.5):
    """
    For CCs that span multiple TBLRs (divided), cut higher level-sets to find seeds,
    use watershed to properly split them, and assign pieces to TBLRs.
    Only processes the ROI of each divided CC for efficiency.
    
    Returns updated (label_img, aux_labels) so split pieces become 
    independent CCs for future iterations.
    """
    h, w = label_img.shape
    new_aux_labels = aux_labels.copy()
    new_label_img = label_img.copy()
    num_tblrs = len(tblrs)
    
    # --- Precompute a full-image TBLR map for fast lookups ---
    tblr_map = np.zeros((h, w), dtype=np.uint16)
    for t in range(num_tblrs):
        if chars_labels[t] == ' ':
            continue
        t_coord, b_coord, l_coord, r_coord = tblrs[t]
        cv2.rectangle(tblr_map, (int(l_coord), int(t_coord)), 
                      (int(r_coord), int(b_coord)), color=t+1, thickness=-1)
    
    # --- Find mean size and std dev of assigned CCs ---
    assigned_mask = (label_img > 0) & (aux_labels > 0)
    assigned_tblrs = aux_labels[assigned_mask]
    tblr_pixel_counts = np.bincount(assigned_tblrs, minlength=num_tblrs + 1)

    assigned_cc_sizes = []
    for t in range(num_tblrs):
        if chars_labels[t] != ' ' and t + 1 < len(tblr_pixel_counts):
            size = tblr_pixel_counts[t + 1]
            if size > 0:
                assigned_cc_sizes.append(size)
    
    if assigned_cc_sizes:
        mean_size = np.mean(assigned_cc_sizes)
        std_dev = np.std(assigned_cc_sizes)
    else:
        mean_size = 0
        std_dev = 0

    # --- Find unassigned CCs (exist in label_img but not in aux_labels) ---
    unassigned_mask = (label_img > 0) & (aux_labels == 0)
    if not unassigned_mask.any():
        return new_label_img, new_aux_labels
    
    unassigned_cc_ids = np.unique(label_img[unassigned_mask])
    next_label = new_label_img.max() + 1
    
    for cc_id in unassigned_cc_ids:
        cc_mask = (new_label_img == cc_id)
        ys, xs = np.where(cc_mask)
        
        if len(ys) == 0:
            continue
        
        # --- QUICK FILTER: Is this CC actually divided across TBLRs? ---
        cc_tblrs = tblr_map[ys, xs]
        unique_tblrs = np.unique(cc_tblrs)
        unique_tblrs = unique_tblrs[unique_tblrs > 0]  # Remove background (0)
        
        if len(unique_tblrs) < 2:
            # Not divided. If it touches one TBLR, assign it there.
            if len(unique_tblrs) == 1:
                new_aux_labels[ys, xs] = unique_tblrs[0]
                new_label_img[ys, xs] = next_label
                next_label += 1
            continue
        
        # ============================================================
        # THIS CC IS DIVIDED — run level_cut + watershed on its ROI
        # ============================================================
        pad = 3
        min_y = max(0, ys.min() - pad)
        max_y = min(h, ys.max() + pad + 1)
        min_x = max(0, xs.min() - pad)
        max_x = min(w, xs.max() + pad + 1)
        
        roi = orig_img[min_y:max_y, min_x:max_x]
        roi_mask = cc_mask[min_y:max_y, min_x:max_x].astype(np.uint8)
        roi_tblr_map = tblr_map[min_y:max_y, min_x:max_x]
        
        # 1. Threshold at higher level to separate touching pieces → these become seeds
        thlded = (roi < (thld**(thld_iters)) ) * roi_mask 
        seeds = measure.label(thlded, connectivity=1)
        num_seeds = seeds.max()
        seed_areas = np.bincount(seeds.ravel())
        
        # --- FILTER SEEDS BASED ON SIZE ---
        valid_seed_ids = []
        invalid_seed_ids = []
        for seed_id in range(1, num_seeds + 1):
            area = seed_areas[seed_id]
            if mean_size * 0.1 <= area <= mean_size * 1.75:
                valid_seed_ids.append(seed_id)
            else:
                invalid_seed_ids.append(seed_id)

        
        # If erosion produced seeds but NONE passed the size filter, fallback
        if (len(valid_seed_ids) < 1) or (num_seeds < 2):
            continue
            
        # Put all non valid seeds to single value (so watershed grows invalid as a single piece)
        filtered_seeds =np.zeros_like(seeds)
        for seed_id in range(1, num_seeds + 1):
            if seed_id in valid_seed_ids:
                filtered_seeds[seeds == seed_id] = seed_id
            else:
                filtered_seeds[seeds == seed_id] = num_seeds + 2
        

        # 2. Watershed on the ROI only (super fast!)
        # Because we passed filtered_seeds, watershed will ONLY grow from valid seeds
        distance = orig_img[min_y:max_y, min_x:max_x].astype(np.float32)
        ws_labels = watershed(distance, markers=filtered_seeds, mask=roi_mask)
        

        # filter only good sized watershed pieces
        ws_areas = np.bincount(ws_labels.ravel())
        valid_ws_ids = []
        for ws_id in range(1, ws_labels.max() + 1):
            if ws_id == num_seeds + 2:
                continue
            area = ws_areas[ws_id]
            if mean_size * 0.33 <= area <= mean_size * 1.75:
                valid_ws_ids.append(ws_id)
            
        
        # 3. Assign each watershed piece to the TBLR with most overlap
        # We only iterate over the valid IDs
        for piece_id in valid_ws_ids:
            piece_mask = (ws_labels == piece_id)
            if not piece_mask.any():
                continue
            

            piece_tblrs = roi_tblr_map[piece_mask]
            if len(piece_tblrs) == 0:
                continue
            
            counts = np.bincount(piece_tblrs)
            if len(counts) > 1:
                counts[0] = 0  # ignore background
            best_tblr = np.argmax(counts)
            
            piece_ys, piece_xs = np.where(piece_mask)
            full_ys = piece_ys + min_y
            full_xs = piece_xs + min_x
            
            if best_tblr > 0:
                # new_aux_labels[full_ys, full_xs] = best_tblr
                new_label_img[full_ys, full_xs] = next_label  # New unique CC ID!
                next_label += 1
            else:
                # Piece is outside all TBLRs → remove from label_img 
                # so it doesn't get reprocessed in future iterations
                new_label_img[full_ys, full_xs] = 0
    
    return new_label_img, aux_labels

def char_segment_kraken_init(img_c, tblrs, chars_labels, len_no_sp, top_pad, bottom_pad,
                            img_flat=None, boundary=None, baseline=None):
    """Character segmentation using logit-initialised boxes (Kraken path).

    Same contract as :func:`char_segment` but uses
    :func:`find_hor_paths_kraken_init` / :func:`find_vert_paths_kraken_init`
    and includes proximity-path boundary negotiation.

    Args:
        img_c: Grayscale image, shape (h, w), values in [0, 1].
            Used for path finding (horizontal + vertical cost paths).
        tblrs: Character boxes, shape (n, 4).
        chars_labels: List of character labels, shape (n,).
        len_no_sp: Number of non-space characters.
        top_pad: Top padding size (pixels).
        bottom_pad: Bottom padding size (pixels).
        widths: Per-character widths array.
        img_flat: Optional bg-flattened image, shape (h, w), values in
            [0, 1].  When provided, mask filtering (white-pixel removal)
            uses this image instead of *img_c*.  This lets path finding
            exploit full grayscale contrast while masks are cleaned
            against the flattened background.
        boundary: Kraken boundary, shape (L, 2), integer.
        baseline: Kraken baseline, shape (2, 2), integer.

    Returns:
        char_data: list of ((t, b, l, r), mask) tuples.
    """
    if img_flat is None:
        img_flat = img_c
    tblrs = np.array(tblrs)
    char_data = []
    tpath, bpath = find_hor_paths_kraken_init(img_c, tblrs, boundary, baseline, top_pad, bottom_pad)
    tb_mask = cv2.fillPoly(np.zeros_like(img_c),[np.concatenate((tpath, bpath[::-1]))[:, ::-1]], 1)

    # 1. Precompute Connected Components (Run this ONLY ONCE)
    label_img_orig, thld, cc_areas = get_connected_components(img_c, tb_mask)
    cc_ids = np.argsort(cc_areas)[::-1]
    label_img = np.isin(label_img_orig, cc_ids[:len_no_sp//2]).astype(np.uint8) * label_img_orig

    # 2. Initialize with your original bounding boxes
    current_tblrs = tblrs.copy()

    # 3. Iterate!
    num_iterations = 15  # Change this to however many passes you want

    delay = 0
    for i in range(num_iterations):        
        # Step A: Assign CCs based on the CURRENT bounding boxes
        aux_labels, tblr_map = assign_ccs_to_tblrs(label_img, current_tblrs, chars_labels, threshold=0.8 - (i * 0.05))
                
        # Step B: Adjust the bounding boxes based on the assignments
        current_tblrs = adjust_tblrs(current_tblrs, aux_labels)
        
        # Step D: Try to separate CCs by eroding the mask
        # label_img = cv2.erode((label_img>0).astype(np.uint8), np.ones((2, 2), np.uint8), iterations=1) * label_img
        label_old = label_img.copy()
        label_img, aux_labels = resolve_divided_ccs(
            img_c, label_img, aux_labels, current_tblrs, chars_labels, thld_iters=(i+4)/ 4, thld=thld
        )

        if (label_old == label_img).all():
            if delay == 2:
                break
            else: 
                delay+= 1
        else:
            delay = 0
        # add some of the components back to the label_img that were removed in cc_ids
        new_adds = (np.isin(label_img_orig, cc_ids[(len_no_sp//2)+((len_no_sp//3)*i): (len_no_sp//2)+((len_no_sp//3)*(i+1))]).astype(np.uint8) * label_img_orig)
        new_uq = np.unique(new_adds[new_adds!=0])
        for k, uq in enumerate(new_uq):
            label_img[new_adds == uq] = label_img.max() + k + 1
        
    
    ws = watershed(img_flat, aux_labels, mask=(img_flat !=1)*tb_mask)
    for t in range(len(tblrs)):
        mask = ws == (t + 1)
        if not mask.any():
            # TO DO: Do something with this, we should signal which bboxes are unusual and which neighbour took the cc.
            char_data.append(None)
            continue
        hprojs, = np.nonzero(np.any(mask, axis=1))
        vprojs, = np.nonzero(np.any(mask, axis=0))
        t_coord, b_coord = np.min(hprojs), np.max(hprojs) + 1
        l_coord, r_coord = np.min(vprojs), np.max(vprojs) + 1
        mask = mask[t_coord:b_coord, l_coord:r_coord]
        char_data.append(((t_coord, b_coord, l_coord, r_coord), mask.astype(bool)))

    return char_data, tpath, bpath



# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def segment_characters(img_c, tblrs, box_clu=None, refwidth=None, *,
                       mode="box_init",
                       top_pad=1, bottom_pad=1, widths=None,
                       img_flat=None, boundary=None, baseline=None,
                       chars_labels=None, len_no_sp=None
                       ):
    """Unified entry point for character segmentation.

    Args:
        img_c: Grayscale image, shape (h, w), values in [0, 1].
            Used for path finding in ``logit_init`` mode.
        tblrs: Character boxes, shape (n, 4).
        box_clu: Line cluster assignments, shape (n,).
        refwidth: Mean character width (float).
        mode: ``"box_init"`` (detector-box-based, original) or
              ``"logit_init"`` (CRNN-logit-based, DocTR variant) or
              ``"kraken_init"`` (Kraken-logit-based, Kraken variant).
        top_pad: Top padding (only used by ``logit_init``).
        bottom_pad: Bottom padding (only used by ``logit_init``).
        widths: Per-character widths (only used by ``logit_init``).
        img_flat: Optional bg-flattened image for mask filtering
            (only used by ``logit_init``).
        boundary: Optional boundary for mask filtering
            (only used by ``kraken_init``).
        baseline: Optional baseline for mask filtering
            (only used by ``kraken_init``).

    Returns:
        For ``box_init``: (char_data, idxs_input) — list of ((t,b,l,r), mask)
            tuples and corresponding input indices.
        For ``logit_init``: char_data — list of ((t,b,l,r), mask) tuples
            (no idxs_input).
    """
    if mode == "box_init":
        return char_segment(img_c, tblrs, box_clu, refwidth)
    elif mode == "logit_init":
        return char_segment_logit_init(
            img_c, tblrs, box_clu, refwidth, top_pad, bottom_pad, widths,
            img_flat=img_flat,
        )
    elif mode == "kraken_init":
        return char_segment_kraken_init(
            img_c, tblrs, chars_labels, len_no_sp, top_pad, bottom_pad,
            img_flat=img_flat,
            boundary=boundary,
            baseline=baseline
        )
    else:
        raise ValueError(f"Unknown segmentation mode: {mode!r}. "
                         f"Expected 'box_init' or 'logit_init' or 'kraken_init'.")
