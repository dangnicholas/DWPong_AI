import tkinter as tk
from exhibit.visualization.render_utils import render_weights, render_layer, render_rescale, TITLE_FONT, is_significant, is_firing, is_weight_active
import numpy as np
import cv2
from PIL import Image, ImageTk
from keras.models import Model


class RealtimeVisualizer:
    """
    This class handles rendering the pong game and agent network state
    over the course of a game. It gets injected into the simulator, which will call it as appropriate.

    This proof of concept implementation uses Tkinter for rendering. I recommend that any actual implementation
    use the HTML5 canvas, which seems faster and much easier to work with.
    """
    # Rendering constants
    MIN_PADDING = 3
    HIDDEN_LAYER_X = 1000
    OUTPUT_LAYER_X = 1600
    CANVAS_SIZE_720P = (1280, 720)
    CANVAS_SIZE_1080P = (1920, 1080)
    IMAGE_X = 400
    IMAGE_UPSCALE = 5
    OUTPUT_LABELS = ["UP", "DOWN"]

    def __init__(self, model):
        # Create canvas setup
        self.master = tk.Tk()
        self.canvas_width, self.canvas_height = self.CANVAS_SIZE_1080P
        self.base_ctx = tk.Canvas(self.master,
                   width=self.canvas_width,
                   height=self.canvas_height)
        self.base_ctx.pack()
        self.model = model
        self.last_hw_activation = None
        self.last_ow_activation = None

    def base_render(self, frame):
        """
        Called after the first frame is rendered
        """
        base_ctx = self.base_ctx
        # Render game frame
        frame *= 256
        pixel_dims = frame.shape
        render_frame = cv2.resize(frame, (frame.shape[1] * self.IMAGE_UPSCALE, frame.shape[0] * self.IMAGE_UPSCALE))
        img = ImageTk.PhotoImage(image=Image.fromarray(render_frame))
        # Save rendering id of image to later inspect attributes
        self.img_id = self.base_ctx.create_image(self.IMAGE_X, self.canvas_height / 2, anchor="c", image=img)

        # Store image-related rendering measurements
        img_coords = base_ctx.coords(self.img_id)
        img_size = (img.width(), img.height())
        top_corner = (np.subtract(img_coords, np.divide(img_size, 2)))
        pixel_step = img_size[0] / pixel_dims[1]

        # Calculate positions of individual pixels, for use as a "layer" to connect edges
        self.pixel_pos = []
        for y in range(pixel_dims[0]):
            for x in range(pixel_dims[1]):
                x_pos = (x * pixel_step) + top_corner[0] + (pixel_step / 2)
                y_pos = (y * pixel_step) + top_corner[1] + (pixel_step / 2)
                self.pixel_pos.append((x_pos, y_pos))

        # Get neurons.
        # Structure: list of numpy arrays, [hidden weight, hidden bias, output weight, output bias]
        model = self.model
        weights = model.get_weights()
        self.hidden_biases = weights[1]
        self.output_biases = weights[3]

        # Normalize weights for use as line thickness
        self.output_weights = weights[2]
        self.hidden_weights = weights[0]

        self.significant_hw = is_significant(self.hidden_weights, 0.2)
        self.significant_ow = is_significant(self.output_weights, 1)

        # Determine appropriate base size for a neuron based on minimum allowable padding for a specific layer
        self.neuron_size = (self.canvas_height - (len(self.hidden_biases) * self.MIN_PADDING)) / len(self.hidden_biases)

        # Create model for collecting hidden layer activations
        self.hl_model = Model(inputs=self.model.inputs, outputs=self.model.layers[0].output)
        # Render neuron nodes, saving calculated positions for weight rendering
        self.hidden_pos = render_layer(self.base_ctx, render_rescale(self.hidden_biases, magnitude=1), 0,
                                       self.canvas_height, self.HIDDEN_LAYER_X, self.neuron_size)
        self.out_pos = render_layer(self.base_ctx, render_rescale(self.output_biases, magnitude=1), 0,
                               self.canvas_height, self.OUTPUT_LAYER_X, self.neuron_size, labels=self.OUTPUT_LABELS)


        # Create dynamic labels for inference confidence
        up_pos = self.out_pos[0]
        down_pos = self.out_pos[1]
        self.up_prob = tk.StringVar()
        self.down_prob = tk.StringVar()
        self.up_label = tk.Label(self.master, textvariable=self.up_prob, font=TITLE_FONT)
        self.down_label = tk.Label(self.master, textvariable=self.down_prob, font=TITLE_FONT)
        self.down_label.pack()
        self.up_label.pack()
        self.up_prob.set("0%")
        self.down_prob.set("0%")

        base_ctx.update()
        self.down_label.place(x=down_pos[0] + 25, y=down_pos[1] + 25)
        self.up_label.place(x=up_pos[0] + 25, y=up_pos[1] + 25)


    def render_frame(self, state_frame, render_frame, prob):
        render_frame = np.copy(render_frame)

        # Update rendered probabilities
        percent_prob = prob * 100
        self.up_prob.set(f'{"{0:.2f}".format(percent_prob[0])}%')
        self.down_prob.set(f'{"{0:.2f}".format(percent_prob[1])}%')

        # Render game frame
        render_frame *= 256
        render_frame = cv2.resize(render_frame, (render_frame.shape[1] * self.IMAGE_UPSCALE, render_frame.shape[0] * self.IMAGE_UPSCALE))
        img = ImageTk.PhotoImage(image=Image.fromarray(render_frame))
        #self.base_ctx.itemconfig(self.img_id, image=img)
        self.base_ctx.create_image(self.IMAGE_X, self.canvas_height / 2, anchor="c", image=img)
        X = state_frame.reshape([1, state_frame.shape[0] * state_frame.shape[1]])

        # Render image weights
        image_activations = is_firing(state_frame.ravel())
        hw_activations = is_weight_active(self.hidden_weights, image_activations)
        hw_needs_update = None
        if self.last_hw_activation is not None:
            hw_needs_update = hw_activations != self.last_hw_activation
        self.last_hw_activation = hw_activations
        render_weights(self.base_ctx, self.pixel_pos, self.hidden_pos, render_rescale(self.hidden_weights),
                       significant=self.significant_hw, activations=hw_activations)

        # Re-compute hidden activations for rendering
        hl_activations = is_firing(self.hl_model.predict(X, batch_size=1).squeeze())
        ow_activations = is_weight_active(self.output_weights, hl_activations)
        # Render hidden weights
        ow_needs_update = None
        if self.last_ow_activation is not None: ow_needs_update = ow_activations != self.last_ow_activation
        render_weights(self.base_ctx, self.hidden_pos, self.out_pos, render_rescale(self.output_weights),
                       significant=self.significant_ow, activations=ow_activations)

        render_layer(self.base_ctx, render_rescale(self.hidden_biases, magnitude=1), 0,
                                       self.canvas_height, self.HIDDEN_LAYER_X, self.neuron_size, activations=hl_activations)
        render_layer(self.base_ctx, render_rescale(self.output_biases, magnitude=1), 0,
                     self.canvas_height, self.OUTPUT_LAYER_X, self.neuron_size,
                     labels=self.OUTPUT_LABELS, activation_intensities=prob)
        # Open window
        self.base_ctx.update()
        self.base_ctx.delete(tk.ALL)
