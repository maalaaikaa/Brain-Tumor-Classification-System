from __future__ import annotations

import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import colormaps


def find_last_conv_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if isinstance(layer, tf.keras.Model):
            try:
                return find_last_conv_layer(layer)
            except ValueError:
                continue
    raise ValueError("No convolutional layer found for Grad-CAM.")


def make_gradcam_heatmap(
    image_batch: np.ndarray,
    model: tf.keras.Model,
    class_index: int | None = None,
    layer_name: str | None = None,
) -> np.ndarray:
    layer_name = layer_name or find_last_conv_layer(model)
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.42,
    colormap: str = "jet",
) -> np.ndarray:
    heatmap_image = Image.fromarray(np.uint8(255 * heatmap)).resize(
        (image_rgb.shape[1], image_rgb.shape[0]),
        Image.Resampling.BILINEAR,
    )
    normalized = np.asarray(heatmap_image).astype("float32") / 255.0
    try:
        color_map = np.uint8(255 * colormaps[colormap](normalized)[..., :3])
    except KeyError:
        color_map = np.uint8(255 * colormaps["jet"](normalized)[..., :3])
        
    overlay = (image_rgb.astype("float32") * (1 - alpha)) + (color_map.astype("float32") * alpha)
    return np.clip(overlay, 0, 255).astype("uint8")
