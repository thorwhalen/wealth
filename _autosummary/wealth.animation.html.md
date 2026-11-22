# wealth.animation

Getting animations of multi-dimensional timeseries data

### Functions

| `get_colors_for_xys`(xys[, n_colors, colors, ...])                                            |                                                       |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------|
| `interpolated_xys`(xys, n_frames)                                                             |                                                       |
| [`remove_ticks`](#wealth.animation.remove_ticks)()                               | Remove x and y ticks and labels from current figure   |
| [`xys_to_swarm_animation`](#wealth.animation.xys_to_swarm_animation)(xys[, n_frames, ...]) | Make a swarm animation from a sequence of xy matrices |

### wealth.animation.remove_ticks()

Remove x and y ticks and labels from current figure

### wealth.animation.xys_to_swarm_animation(xys, n_frames=None, figsize=(9, 9), marker='.', marker_size=None, color=<function get_colors_for_xys>)

Make a swarm animation from a sequence of xy matrices
