from itertools import product

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import umap
from colorspacious import cspace_converter
from matplotlib.colors import ListedColormap
from sklearn.utils._plotting import _validate_style_kwargs


### MATRIX ###


def plot_matrix(cm,
                display_labels=None,
                include_values=True,
                title=None,
                figsize=None,
                cmap="viridis",
                xticks_rotation="vertical",
                colorbar=True,
                im_kw=None,
                text_kw=None,
                values_format=None):
    """Adapted from sklearn.metrics.ConfusionMatrixDisplay.plot()."""
    figsize = (8, 6) if figsize is None else figsize
    fig, ax = plt.subplots(figsize=figsize)

    n_classes = cm.shape[0]

    vmin, vmax = cm.min(), cm.max()
    n = vmax-vmin+1
    # Discrete colormap
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = sm.to_rgba(np.linspace(vmin, vmax, n))
    cmap = mpl.colors.ListedColormap(colors)

    default_im_kw = dict(interpolation="nearest",
                         cmap=cmap, vmin=vmin-0.5, vmax=vmax+0.5)
    im_kw = im_kw or {}
    im_kw = _validate_style_kwargs(default_im_kw, im_kw)
    text_kw = text_kw or {}

    im_ = ax.imshow(cm, **im_kw)
    text_ = None
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(1.0)

    if include_values:
        text_ = np.empty_like(cm, dtype=object)

        # print text with appropriate color depending on background
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

            default_text_kwargs = dict(ha="center", va="center", color=color)
            text_kwargs = _validate_style_kwargs(default_text_kwargs, text_kw)

            text_[i, j] = ax.text(j, i, text_cm, **text_kwargs)

    if display_labels is None:
        ticks = np.arange(0, n_classes, 5)
        display_labels = np.arange(0, n_classes, 5)
    elif display_labels is 'start1':
        ticks = np.arange(4, n_classes, 5)
        display_labels = np.arange(5, n_classes+1, 5)
    else:
        ticks = np.arange(n_classes)
    if colorbar:
        tickvals = np.arange(vmin, vmax+1, 5)
        cbar = fig.colorbar(im_, ax=ax,
                            fraction=0.0456, pad=0.04, ticks=tickvals)
        cbar.ax.set_yticklabels(tickvals)
    ax.set(
        xticks=ticks,
        yticks=ticks,
        xticklabels=display_labels,
        yticklabels=display_labels,
        title=title,
    )

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)


### GRAPH ###


_DFLT_KWARGS = dict(
    # meta
    title=None,
    savepath=None,
    # umap
    n_neighbors=15,
    min_dist=0.5,
    n_components=2,
    random_state=0,
    # drawing
    scale_factor=1.0,
    shift_dir='se',
    shift_len=0.3,
    font_size=9,
    node_size=24,
    tf_geom=np.eye(2),
    cmap=None,
    figsize=(8, 8)
)


def plot_graph(names,
               dists,
               weights,
               leans: np.ndarray | None = None,
               idxs: np.ndarray | None = None,
               markers: list | None = None,
               markersizes: list | None = None,
               colors: list | None = None,
               **kwargs) -> None:
    kwargs, new_kwargs = _DFLT_KWARGS.copy(), kwargs.copy()
    kwargs.update(new_kwargs)
    kwargs_umap = {k: kwargs[k] for k in
                   ['n_neighbors', 'min_dist', 'n_components', 'random_state']}
    
    # Edge colormap
    if kwargs['cmap'] is None:
        # Rescale colormap to widen neutral range
        x = np.linspace(-1, 1, 1000)
        x = 1-2*np.arccos(x)/np.pi
        x = 1-np.arccos(x)/np.pi
        rgb = mpl.colormaps['BrBG'](x)[:, :3]
        # Luminance normalization
        rgb /= cspace_converter("sRGB1", "CAM02-UCS")(rgb)[:, :1]/50
        cmap = ListedColormap(rgb)
    else:
        cmap = kwargs['cmap']
    
    # Remove isolated vertices
    if idxs is None:
        idxs, = np.nonzero(np.sum(weights>0, axis=0) > 1) # not np.any() because of diagonal
    inds = np.meshgrid(idxs, idxs)
    names = names[idxs]
    dists = dists[inds]
    weights = weights[inds]
    leans = leans[idxs] if leans is not None else 0.5+np.zeros_like(weights)
    # if leans is None:
    #     leans = 0.5+np.zeros_like(weights)

    # Determine geometry
    red = umap.UMAP(metric='precomputed', **kwargs_umap)
    pts = red.fit_transform(dists)
    pts = pts @ kwargs['tf_geom']

    # Prepare graph
    G = nx.Graph()
    for i, label in enumerate(names):
        G.add_node(i, label=label)
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            if weights[i, j] > 0:
                G.add_edge(i, j, weight=weights[i, j], lean=leans[i, j])

    # Draw setup
    scale_factor = kwargs['scale_factor']  # increase spacing between vertices
    pts *= scale_factor
    plt.figure(figsize=kwargs['figsize'])

    # Draw edges
    edges = G.edges(data=True)
    wts = [d['weight'] for (_, _, d) in edges]
    lns = [d['lean'] for (_, _, d) in edges]
    wt_max = max(wts)
    for (u, v, _), wt, ln in zip(edges, wts, lns):
        color = cmap(ln)[:3]
        nx.draw_networkx_edges(G, dict(enumerate(pts)), edgelist=[(u, v)],
                               width=0.5, alpha=wt/wt_max, edge_color=color)

    markers = markers[idxs] if markers is not None else 's'
    markersizes = markersizes[idxs] if markersizes is not None else kwargs['node_size']
    colors = colors[idxs] if colors is not None else 'black'

    # Draw labels
    sd = kwargs['shift_dir'].lower()
    shift = np.array([sd.count('e')-sd.count('w'), sd.count('n')-sd.count('s')])
    shift = shift/np.linalg.norm(shift) if np.linalg.norm(shift) > 0 else shift
    shift *= kwargs['shift_len'] * scale_factor # x, y
    nx.draw_networkx_labels(G, dict(enumerate(pts+shift)),
                            labels=dict(enumerate(names)),
                            font_color=dict(enumerate(colors)),
                            font_size=kwargs['font_size'])

    # Draw vertices
    if isinstance(markers, np.ndarray):
        for marker in np.unique(markers):
            idxs, = np.nonzero(markers==marker)
            plt.scatter(pts[idxs, 0], pts[idxs, 1],
                        s=markersizes[idxs], marker=marker, color=colors[idxs])
    else:
        plt.scatter(pts[:, 0], pts[:, 1],
                    s=markersizes, marker=markers, color=colors)
        
    # Finishing touches
    plt.axis('equal')
    plt.axis('off')
    if kwargs['title'] is not None:
        plt.title(kwargs['title'])

    if kwargs['savepath'] is not None:
        plt.savefig(kwargs['savepath'], bbox_inches='tight')
    # plt.show()