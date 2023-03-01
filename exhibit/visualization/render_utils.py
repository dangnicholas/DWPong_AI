import numpy as np
import tkinter as tk

"""
This class is a relatively straightforward set of utilities used in the inference visualization proof-of-concept.
"""

TITLE_FONT = ("Helvetica", "32")
WEIGHT_COLOR = "#666666"
WEIGHT_COLOR_ACTIVE = "#BB6666"
NEURON_COLOR = "#000000"
NEURON_COLOR_ACTIVE = "#DD2222"
ACTIVE_WEIGHT_THRESHOLD = 1


def create_circle(x, y, r, ctx, color=NEURON_COLOR): #center coordinates, radius
    """
    Draw a circle on the canvas
    :param x: x position
    :param y: y position
    :param r: radius
    :param ctx: Tkinter canvas context
    :param color: RGB color
    :return:
    """
    x0 = x - r
    y0 = y - r
    x1 = x + r
    y1 = y + r
    return ctx.create_oval(x0, y0, x1, y1, fill=color)


def get_intensity(val):
    """
    Maps a float between 0 and 1 to a red-based color ranging from gray to bright red
    :param val: float between 0 and 1
    :return: Red color that is redder as val approaches 1
    """
    val = hex(max(int(val * 255), 0x22))
    color = f"#{val[2:]}2222"
    return color


def render_layer(canvas, neurons, top_y, bottom_y, x, neuron_size, activations=None, labels=None, activation_intensities=None):
    # Scale and normalize biases around 1 to represent useful node scale factors (ranging from ~0.3 - ~1.7)
    neurons *= 10
    neurons += 1
    coordinates = []
    padding = ((bottom_y - top_y) - (len(neurons) * neuron_size)) / (len(neurons) + 1)
    for i in range(len(neurons)):
        fill = NEURON_COLOR
        if activation_intensities is not None:
            fill = get_intensity(activation_intensities[i])
        if activations is not None:
            if activations[i]:
                fill = NEURON_COLOR_ACTIVE
        b = neurons[i]
        y = top_y + (i * neuron_size) + ((i+1) * padding) + (neuron_size / 2)
        create_circle(x, y, (neuron_size / 2) * b, canvas, color=fill)
        if labels:
            canvas.create_text(x+25, y, text=labels[i], font=TITLE_FONT, anchor=tk.W)
        coordinates.append((x, y))
    return coordinates


def render_rescale(data, magnitude=2):
    scale = max(np.max(data), -1 * np.min(data))
    return magnitude * data / scale


# Simple util for returning bool array of active neurons
def is_firing(values, threshold=0.2):
    return values >= threshold


# Given weights and preceding layer firing booleans, returns array of booleans indicating weight activity
def is_weight_active(w, l1):
    w_active = np.zeros_like(w)
    for l1_index in range(len(l1)):
        if l1[l1_index]: w_active[l1_index] = 1
    return w_active.astype(np.bool)


# Return bool array of "significant" weights given weight strength
def is_significant(w, threshold=0.2):
    # Find indices of top threshold% weight values
    render_cap = int(len(w) * threshold)

    # Unravels the weights, create a new array of the top n indices, and transforms the indices into 2d weight indices
    # Then stacks up the array, squeezes out the extra dim, and cuts off everything after the max render cap
    significant = np.dstack(np.unravel_index(np.argpartition(w.ravel(), -render_cap)[-render_cap:], w.shape)).squeeze()[:render_cap]

    significant_w = np.zeros_like(w)
    for l1, l2 in significant:
        significant_w[l1, l2] = 1
    return significant_w.astype(np.bool)


def render_weights(canvas, l1_positions, l2_positions, w, significant=None, needs_update=None, activations=None):
    # Set up filtering
    should_render = np.ones(w.shape, dtype=bool)
    if needs_update is not None: should_render = np.logical_and(should_render, needs_update)
    if significant is not None: should_render = np.logical_and(should_render, significant)
    if activations is not None: should_render = np.logical_and(should_render, activations)
    should_render = np.argwhere(should_render)

    for l1, l2 in should_render:
        l2_pos = l2_positions[l2]
        l1_pos = l1_positions[l1]
        weight = w[l1][l2]
        fill = WEIGHT_COLOR
        weight = abs(weight)
        np.set_printoptions(threshold=np.inf)
        # Render activation if l1 activation values are supplied
        if activations is not None:
            active = activations[l1, l2]
            if active: fill = WEIGHT_COLOR_ACTIVE
        canvas.create_line(*l1_pos, *l2_pos, width=weight, fill=fill)
