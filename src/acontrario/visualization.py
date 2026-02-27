"""Visualization helpers for a contrario analysis results.

- ``plot_matrix``: discrete heatmap of *n̂₁* with printer / remark overlays.
- ``plot_graph``: UMAP‑laid‑out graph with edge weights from *n̂₁ / n*.
"""

from __future__ import annotations

from itertools import product

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from colorspacious import cspace_converter
from matplotlib.colors import ListedColormap


# ===================================================================
# Internal helpers
# ===================================================================

def _validate_style_kwargs(default_kw: dict, user_kw: dict) -> dict:
    """Merge Matplotlib style kwargs avoiding alias conflicts."""
    _alias = {
        "ls": "linestyle", "c": "color", "ec": "edgecolor",
        "fc": "facecolor", "lw": "linewidth", "mec": "markeredgecolor",
        "mfcalt": "markerfacecoloralt", "ms": "markersize",
        "mew": "markeredgewidth", "mfc": "markerfacecolor",
        "aa": "antialiased", "ds": "drawstyle", "font": "fontproperties",
        "family": "fontfamily", "name": "fontname", "size": "fontsize",
        "stretch": "fontstretch", "style": "fontstyle",
        "variant": "fontvariant", "weight": "fontweight",
        "ha": "horizontalalignment", "va": "verticalalignment",
        "ma": "multialignment",
    }
    for short, full in _alias.items():
        if short in user_kw and full in user_kw:
            raise TypeError(
                f"Got both {short} and {full}, which are aliases of one another"
            )
    merged = default_kw.copy()
    for key, val in user_kw.items():
        merged[_alias.get(key, key)] = val
    return merged


def _build_edge_cmap() -> ListedColormap:
    """BrBG diverging cmap with widened neutral range, luminance‑normalised."""
    x = np.linspace(-1, 1, 1000)
    x = 1 - 2 * np.arccos(x) / np.pi
    x = 1 - np.arccos(x) / np.pi
    rgb = mpl.colormaps["BrBG"](x)[:, :3]
    rgb /= cspace_converter("sRGB1", "CAM02-UCS")(rgb)[:, :1] / 25
    return ListedColormap(rgb)


# ===================================================================
# Matrix plot
# ===================================================================

def plot_matrix(
    cm: np.ndarray,
    *,
    printers: np.ndarray | None = None,
    remarks: np.ndarray | None = None,
    printer_to_color: dict | None = None,
    remark_to_shape: dict | None = None,
    display_labels=None,
    include_values: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
    xticks_rotation: str = "vertical",
    colorbar: bool = True,
    colorbar_step: int = 5,
    im_kw: dict | None = None,
    text_kw: dict | None = None,
    values_format: str | None = None,
    marker_edge_color: str = "white",
    marker_edge_width: float = 0.4,
    marker_sizes: dict | None = None,
    legend_marker_size: float = 250,
    legend_fontsize: str = "xx-large",
    legend_loc: str = "upper right",
) -> mpl.figure.Figure:
    """Discrete heatmap of a score matrix with optional printer overlays.

    Parameters
    ----------
    cm : np.ndarray, shape (n, n)
        Score matrix (typically *n̂₁*).
    printers, remarks : array‑like or None
        Per‑document printer names / remark strings for overlay markers.
    printer_to_color, remark_to_shape : dict or None
        Mappings from names to matplotlib colours / marker codes.
    """
    figsize = (8, 6) if figsize is None else figsize
    fig, ax = plt.subplots(figsize=figsize)

    n_classes = cm.shape[0]
    vmin, vmax = cm.min(), cm.max()
    n = int(vmax - vmin + 1)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = sm.to_rgba(np.linspace(vmin, vmax, n))
    cmap_obj = mpl.colors.ListedColormap(colors)

    default_im_kw = dict(interpolation="nearest", cmap=cmap_obj,
                         vmin=vmin - 0.5, vmax=vmax + 0.5)
    im_kw = _validate_style_kwargs(default_im_kw, im_kw or {})
    text_kw = text_kw or {}

    im_ = ax.imshow(cm, **im_kw)
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(1.0)

    # ---- Printer / remark overlay markers ----
    if printers is not None:
        if marker_sizes is None:
            marker_sizes = {}
        _default_marker_size = 100
        if printer_to_color is None:
            cmap_pr = plt.get_cmap("tab10")
            printer_to_color = {
                p: cmap_pr(i / len(set(printers)))
                for i, p in enumerate(sorted(set(printers)))
            }
        if remarks is None:
            remarks = np.array(["nan"] * len(printers))
        if remark_to_shape is None:
            remark_to_shape = {r: "o" for r in set(remarks)}

        for i, (printer, remark) in enumerate(zip(printers, remarks)):
            if printer == "unknown":
                continue
            remark = "known" if remark == "nan" else remark
            color = printer_to_color.get(printer, "gray")
            shape = remark_to_shape.get(remark, "o")
            s = marker_sizes.get(remark, _default_marker_size)
            ax.scatter(
                i, i, color=color, marker=shape, label=None,
                edgecolors=marker_edge_color,
                linewidths=marker_edge_width,
                s=s,
            )

    # ---- Cell values ----
    if include_values:
        thresh = (cm.max() + cm.min()) / 2.0
        for i, j in product(range(n_classes), range(n_classes)):
            color = cmap_max if cm[i, j] < thresh else cmap_min
            if values_format is None:
                text_cm = format(cm[i, j], ".2g")
                if cm.dtype.kind != "f":
                    text_d = format(cm[i, j], "d")
                    if len(text_d) < len(text_cm):
                        text_cm = text_d
            else:
                text_cm = format(cm[i, j], values_format)
            kw = _validate_style_kwargs(
                dict(ha="center", va="center", color=color), text_kw
            )
            ax.text(j, i, text_cm, **kw)

    # ---- Ticks ----
    if display_labels is None:
        ticks = np.arange(0, n_classes, 5)
        display_labels = np.arange(0, n_classes, 5)
    else:
        ticks = np.arange(n_classes)

    # ---- Colorbar ----
    if colorbar:
        tickvals = np.arange(vmin, vmax + 1, colorbar_step)
        cbar = fig.colorbar(im_, ax=ax, fraction=0.0456, pad=0.04,
                            ticks=tickvals)
        cbar.ax.set_yticklabels(tickvals)

    ax.set(xticks=ticks, yticks=ticks,
           xticklabels=display_labels, yticklabels=display_labels,)
        #    title=title)

    # ---- Legend ----
    if printers is not None:
        for printer in sorted(set(printers)):
            if printer == "unknown":
                continue
            ax.scatter([], [], color=printer_to_color.get(printer, "gray"),
                       marker="o", label=printer, edgecolors=marker_edge_color,
                       s=legend_marker_size, linewidths=marker_edge_width)
        # Only show remark legend entries when shapes actually differ
        unique_shapes = set(remark_to_shape.values())
        if len(unique_shapes) > 1:
            for remark in ["known", "new"]:
                shape = remark_to_shape.get(remark, "o")
                ax.scatter([], [], color="gray", marker=shape,
                           label=f"$\\it{{{remark}}}$",
                           edgecolors=marker_edge_color,
                           s=legend_marker_size * 0.6 if shape == "D" else legend_marker_size * 1.2)
        ax.legend(loc=legend_loc, fontsize=legend_fontsize)

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
    return fig


# ===================================================================
# Graph plot
# ===================================================================

_DFLT_GRAPH_KW = dict(
    n_neighbors=5,
    min_dist=0.1,
    n_components=2,
    random_state=0,
    scale_factor=1.0,
    shift_dir="se",
    shift_len=0.3,
    font_size=9,
    node_size=24,
    tf_geom=np.eye(2),
    cmap=None,
    figsize=(8, 8),
    edge_width=0.5,
    legend_marker_size=120,
    legend_fontsize="x-large",
    legend_loc="upper right",
)


def plot_graph(
    names: np.ndarray,
    dists: np.ndarray,
    weights: np.ndarray,
    *,
    printers: np.ndarray | None = None,
    remarks: np.ndarray | None = None,
    printer_to_color: dict | None = None,
    remark_to_shape: dict | None = None,
    leans: np.ndarray | None = None,
    idxs: np.ndarray | None = None,
    **kwargs,
) -> mpl.figure.Figure:
    """UMAP‑laid‑out weighted graph of document similarities.

    Parameters
    ----------
    names : np.ndarray of str
        Document display names.
    dists : np.ndarray, shape (n, n)
        Full distance / dissimilarity matrix.
    weights : np.ndarray, shape (n, n)
        Edge weights (e.g. *n̂₁ / n*).  Zero means no edge.
    printers, remarks : array‑like
        Per‑document metadata for colouring / shape.
    leans : np.ndarray or None
        Per‑edge colour value ∈ [0, 1] (e.g. italic fraction).
    idxs : np.ndarray or None
        Subset of document indices to show.
    **kwargs
        Override any entry in ``_DFLT_GRAPH_KW``.
    """
    kw = _DFLT_GRAPH_KW.copy()
    kw.update(kwargs)
    kw_umap = {k: kw[k] for k in
               ["n_neighbors", "min_dist", "n_components", "random_state"]}

    import umap as umap_lib  # lazy import – numba JIT is slow on first load

    edge_cmap = kw["cmap"] if kw["cmap"] is not None else _build_edge_cmap()

    # ---- Subset ----
    if idxs is None:
        idxs, = np.nonzero(np.sum(weights > 0, axis=0) > 1)
    if printers is None:
        printers = np.array(["unknown"] * len(names))
    if printer_to_color is None:
        cmap_pr = plt.get_cmap("tab10")
        printer_to_color = {
            p: cmap_pr(i / max(len(set(printers)), 1))
            for i, p in enumerate(sorted(set(printers)))
        }
        printer_to_color.setdefault("unknown", "black")
    if remarks is None:
        remarks = np.array(["nan"] * len(names))
    if remark_to_shape is None:
        remark_to_shape = {r: "s" for r in set(remarks)}

    mesh = np.meshgrid(idxs, idxs)
    names_sub = names[idxs]
    printers_sub = printers[idxs]
    remarks_sub = remarks[idxs]
    dists_sub = dists[mesh[0], mesh[1]]
    weights_sub = weights[mesh[0], mesh[1]]
    if leans is not None:
        leans_sub = leans[mesh[0], mesh[1]]
    else:
        leans_sub = 0.5 + np.zeros_like(weights_sub)

    # ---- UMAP layout ----
    red = umap_lib.UMAP(metric="precomputed", **kw_umap)
    pts = red.fit_transform(dists_sub)
    pts = pts @ kw["tf_geom"]

    # ---- Build graph ----
    G = nx.Graph()
    for i, label in enumerate(names_sub):
        G.add_node(i, label=label)
    for i in range(len(names_sub)):
        for j in range(i + 1, len(names_sub)):
            if weights_sub[i, j] > 0:
                G.add_edge(i, j, weight=weights_sub[i, j],
                           lean=leans_sub[i, j])

    sf = kw["scale_factor"]
    pts *= sf
    fig, ax = plt.subplots(figsize=kw["figsize"])

    # ---- Edges ----
    edges = G.edges(data=True)
    wts = [d["weight"] for (_, _, d) in edges]
    lns = [d["lean"] for (_, _, d) in edges]
    wt_max = max(wts) if wts else 1
    for (u, v, _), wt, ln in zip(G.edges(data=True), wts, lns):
        ec = edge_cmap(ln)[:3]
        nx.draw_networkx_edges(
            G, dict(enumerate(pts)), edgelist=[(u, v)],
            width=kw["edge_width"], alpha=wt / wt_max, edge_color=ec, ax=ax,
        )

    # ---- Vertex markers ----
    markers = np.array([remark_to_shape.get(r, "s") for r in remarks_sub])
    markersizes = np.full(len(markers), kw["node_size"] / 2)
    colors = np.array([printer_to_color.get(p, "black") for p in printers_sub])

    sd = kw["shift_dir"].lower()
    shift = np.array([sd.count("e") - sd.count("w"),
                      sd.count("n") - sd.count("s")])
    norm = np.linalg.norm(shift)
    shift = (shift / norm if norm > 0 else shift) * kw["shift_len"] * sf

    nx.draw_networkx_labels(
        G, dict(enumerate(pts + shift)),
        labels=dict(enumerate(names_sub)),
        font_color=dict(enumerate(colors)),
        font_size=kw["font_size"], ax=ax,
    )

    for marker in np.unique(markers):
        idx, = np.nonzero(markers == marker)
        ax.scatter(pts[idx, 0], pts[idx, 1],
                   s=markersizes[idx], marker=marker, color=colors[idx])

    # ---- Legend ----
    for printer in sorted(set(printers_sub)):
        if printer == "unknown":
            continue
        ax.scatter([], [], color=printer_to_color.get(printer, "gray"),
                   marker="o", label=printer, edgecolors="white",
                   s=kw["legend_marker_size"])
    # Only show remark legend entries when shapes actually differ
    unique_shapes = set(remark_to_shape.values())
    if len(unique_shapes) > 1:
        for remark in ["known", "new"]:
            shape = remark_to_shape.get(remark, "o")
            ax.scatter([], [], color="gray", marker=shape,
                       label=f"$\\it{{{remark}}}$", edgecolors="white",
                       s=kw["legend_marker_size"] * 0.6 if shape == "D" else kw["legend_marker_size"] * 1.25)
    ax.legend(loc=kw["legend_loc"], fontsize=kw["legend_fontsize"])
    ax.set_aspect("auto")
    ax.axis("off")
    return fig
