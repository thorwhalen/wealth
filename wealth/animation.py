"""Getting animations of multi-dimensional timeseries data"""

import matplotlib.pyplot as plt
from celluloid import Camera


def remove_ticks():
    """Remove x and y ticks and labels from current figure"""
    plt.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        left=False,
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,  # labels along the bottom edge are off
        labelleft=False,
    )


def embeddings_to_swarm_animation(
    embeddings,
    figsize=(9, 9),
    marker=".",
    s=2,
    c="black",
):
    """"""
    fig = plt.figure(figsize=figsize)
    camera = Camera(fig)
    for i in range(len(embeddings)):
        plt.scatter(embeddings[i][:, 0], embeddings[i][:, 1], marker=marker, s=s, c=c)
        remove_ticks()
        camera.snap()

    animation = camera.animate(blit=False)
    return animation


# from IPython.display import HTML
# animation = embeddings_to_animation(embeddings)
# HTML(animation.to_html5_video())
