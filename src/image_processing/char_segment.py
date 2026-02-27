import numpy as np
import skimage as ski
from skimage.graph import route_through_array
import cv2

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
