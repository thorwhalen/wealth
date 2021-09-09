"""Getting animations of multi-dimensional timeseries data"""

from typing import Callable, Union, Iterable

import numpy as np
from scipy import interpolate
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from celluloid import Camera


def remove_ticks():
    """Remove x and y ticks and labels from current figure"""
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,  # labels along the bottom edge are off
        labelleft=False,
    )


def get_colors_for_xys(xys, n_colors=7, colors=None, method='first_slice_clusters'):
    if colors is None:
        if n_colors <= 10:
            colors = 'tab10'
        elif n_colors <= 20:
            colors = 'tab20'
    if isinstance(colors, str):
        try:
            colors = plt.cm.get_cmap(colors).colors
        except AttributeError:
            pass  # leave colors as is (it might be single character color spec)

    if method == 'first_slice_clusters':
        from sklearn.cluster import KMeans

        color_indices = KMeans(n_clusters=n_colors).fit_predict(xys[0])
        color_indices = [i % len(colors) for i in color_indices]
        return np.array(colors)[color_indices]
    else:
        raise ValueError(f'Unknown method: {method}')


def interpolated_xys(xys, n_frames):
    from scipy.interpolate import InterpolatedUnivariateSpline

    xys = np.array(xys)
    assert xys.ndim == 3, f'xys needs to have 3 dimensions. Had {xys.ndim}'

    n, n_pts, ax = xys.shape

    if n == n_frames:
        return xys

    x = np.linspace(0, n - 1, n_frames)

    def gen():
        for i in range(n_pts):

            def ggen():
                for j in range(ax):
                    arr = xys[:, i, j]
                    ius = InterpolatedUnivariateSpline(range(n), arr)
                    yield list(ius(x))

            yield list(ggen())

    return np.transpose(list(gen()), (2, 0, 1))


def xys_to_swarm_animation(
    xys,
    n_frames=None,
    figsize=(9, 9),
    marker='.',
    marker_size=None,
    color: Union[str, Callable, Iterable] = get_colors_for_xys,
):
    """Make a swarm animation from a sequence of xy matrices"""
    xys = np.array(xys)
    n_frames = n_frames or len(xys)

    if n_frames != len(xys):
        xys = interpolated_xys(xys, n_frames)

    n_pts = len(xys[0])
    assert all(len(x) == n_pts for x in xys), "Some xys don't have the same size!"

    if marker_size is None:
        # TODO: Better keeping fig_area / marker_area constant
        marker_size = max(1, int(5000 / n_pts))
    if isinstance(color, str):
        color = [color] * n_pts
    elif isinstance(color, Callable):
        color = color(xys)

    # get_color = lambda i: color[i % len(color)]

    fig = plt.figure(figsize=figsize)
    camera = Camera(fig)

    for i in range(n_frames):
        plt.scatter(xys[i][:, 0], xys[i][:, 1], marker=marker, s=marker_size, c=color)
        remove_ticks()
        camera.snap()

    animation = camera.animate(blit=False)
    return animation


# from IPython.display import HTML
# animation = embeddings_to_animation(embeddings)
# HTML(animation.to_html5_video())
