"""
gradcam.py

Grad-CAM (Gradient-weighted Class Activation Mapping) for the trained wound
classifier. Instead of guessing what the model "looks at," this computes it
directly: it traces how much each region of the final convolutional feature
map contributed to the predicted class, then overlays that as a heatmap on
the original image. Red/yellow = influenced the prediction most. Blue/dark
= had little influence.

Requires the model to already be trained (see train_model.py) - this file
only loads and inspects it, never retrains it.

Usage:
    python gradcam.py path/to/image.jpg
        -> saves a single heatmap overlay for that image

    python gradcam.py
        -> no image given: automatically grabs one sample image per class
           from data/test/ and saves a grid overview, one Grad-CAM panel
           per class

    python gradcam.py path/to/image.jpg --class abrasion
        -> forces the heatmap to explain a specific class instead of the
           model's own top prediction (useful for seeing what the model
           would have needed to see for a class it did NOT pick)

Requires: pip install tensorflow pillow matplotlib
"""

import os
import sys
import argparse

import numpy as np
import tensorflow as tf
import matplotlib
from PIL import Image

from model import load_trained_model, IMG_SIZE
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "gradcam_output")


def find_base_model(model):
    """Finds the nested MobileNetV2 submodel inside the outer classifier
    model (model.py builds the outer model with the pretrained backbone as
    a single nested layer, per build_model())."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError(
        "Could not find a nested backbone model inside the loaded model. "
        "Is this the model built by model.py's build_model()?"
    )


def find_last_conv_layer_name(base_model):
    """Picks the target layer for Grad-CAM: the last 4D (spatial feature
    map) output in the backbone. For MobileNetV2 this is normally
    'out_relu' - falling back to a manual search keeps this working even
    if the backbone architecture changes later."""
    preferred = "out_relu"
    layer_names = [l.name for l in base_model.layers]
    if preferred in layer_names:
        return preferred

    for layer in reversed(base_model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue

    raise ValueError("Could not find a convolutional layer to target for Grad-CAM.")


def build_grad_model(model):
    """
    Builds two smaller models instead of one combined graph:
      1. conv_model: base_model's input -> the target conv layer's output
      2. classifier_model: takes that conv output as its own Input, and
         replays the remaining layers (GlobalAveragePooling2D, Dropout,
         Dense) that come after the backbone in the outer model.

    This two-step split is necessary because the backbone is used as a
    single nested layer inside the outer model - the backbone's own layer
    tensors (e.g. base_model.get_layer('out_relu').output) belong to the
    backbone's own standalone graph, not the outer model's graph, so trying
    to reference them directly in a Model(inputs=model.inputs, ...) fails
    with "Output ... is not connected to inputs". Rebuilding the head as
    its own small model on a fresh Input sidesteps that entirely.
    """
    base_model = find_base_model(model)
    target_layer_name = find_last_conv_layer_name(base_model)
    target_layer = base_model.get_layer(target_layer_name)

    conv_model = tf.keras.Model(inputs=base_model.input, outputs=target_layer.output)

    # Everything in the outer model that comes after the backbone layer,
    # in order (GlobalAveragePooling2D -> Dropout -> Dense, per model.py).
    classifier_layers = []
    found_base = False
    for layer in model.layers:
        if layer is base_model:
            found_base = True
            continue
        if found_base:
            classifier_layers.append(layer)

    classifier_input = tf.keras.Input(shape=target_layer.output.shape[1:])
    x = classifier_input
    for layer in classifier_layers:
        x = layer(x, training=False)
    classifier_model = tf.keras.Model(classifier_input, x)

    return conv_model, classifier_model, target_layer_name


def compute_heatmap(conv_model, classifier_model, img_array, class_index=None):
    """
    Runs the image through conv_model, then classifier_model, and computes
    how much each spatial location in the target conv layer's feature map
    contributed to the chosen class's score.

    Returns (heatmap [0-1, HxW], class_index used, confidence for that class).
    """
    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_score = predictions[:, class_index]

    # How much would this class's score change if each pixel in the
    # feature map changed slightly? That's the signal Grad-CAM uses.
    grads = tape.gradient(class_score, conv_outputs)

    # Average the gradient over height/width -> one importance weight per
    # feature-map channel, representing how much that channel mattered
    # for this specific class.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Only positive influence is kept (negative values would mean "this
    # region argued against the class"), then normalized to [0, 1].
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    confidence = float(predictions[0][class_index])
    return heatmap.numpy(), class_index, confidence


def overlay_heatmap(original_img, heatmap, alpha=0.45):
    """Resizes the (small) heatmap up to the original image size, colors it
    with a red-yellow 'jet'-style colormap, and blends it over the image."""
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(
        original_img.size, resample=Image.BILINEAR
    )
    heatmap_arr = np.array(heatmap_img)

    colormap = matplotlib.colormaps["jet"]
    colored_heatmap = colormap(heatmap_arr / 255.0)[:, :, :3]  # drop alpha channel
    colored_heatmap = np.uint8(colored_heatmap * 255)
    colored_heatmap_img = Image.fromarray(colored_heatmap).convert("RGB")

    original_rgb = original_img.convert("RGB")
    blended = Image.blend(original_rgb, colored_heatmap_img, alpha=alpha)
    return blended


def run_single_image(image_path, class_name=None, output_path=None):
    model, class_names = load_trained_model()
    conv_model, classifier_model, target_layer_name = build_grad_model(model)

    class_index = None
    if class_name is not None:
        if class_name not in class_names:
            print(f"'{class_name}' is not a known class. Options: {class_names}")
            return
        class_index = class_names.index(class_name)

    original_img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(original_img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    heatmap, used_index, confidence = compute_heatmap(conv_model, classifier_model, img_array, class_index)
    overlay = overlay_heatmap(original_img, heatmap)

    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_gradcam_{class_names[used_index]}.png")

    overlay.save(output_path)

    print(f"Target layer: {target_layer_name}")
    print(f"Explaining class: {class_names[used_index]} (confidence {confidence:.1%})")
    print(f"Saved heatmap overlay to {output_path}")


def run_class_grid():
    """Grabs one sample image per class from data/test/ and builds a single
    grid image showing the Grad-CAM heatmap for each - a quick overview of
    what the model is keying on across all categories at once."""
    import matplotlib.pyplot as plt

    if not os.path.isdir(TEST_DIR):
        print(f"No test set found at {TEST_DIR}. Run train_model.py first, or pass a single image path.")
        return

    model, class_names = load_trained_model()
    conv_model, classifier_model, target_layer_name = build_grad_model(model)
    print(f"Target layer: {target_layer_name}\n")

    class_dirs = sorted(
        d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))
    )
    if not class_dirs:
        print(f"No class folders found under {TEST_DIR}.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cols = 4
    rows = (len(class_dirs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, class_dir in enumerate(class_dirs):
        class_path = os.path.join(TEST_DIR, class_dir)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        ax = axes[i]

        if not image_files:
            ax.set_title(f"{class_dir}\n(no test images)")
            ax.axis("off")
            continue

        image_path = os.path.join(class_path, image_files[0])
        original_img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(original_img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        heatmap, used_index, confidence = compute_heatmap(conv_model, classifier_model, img_array)
        overlay = overlay_heatmap(original_img, heatmap)

        predicted_name = class_names[used_index]
        correct = predicted_name == class_dir
        title_color = "green" if correct else "red"

        ax.imshow(overlay)
        ax.set_title(f"true: {class_dir}\npred: {predicted_name} ({confidence:.0%})", color=title_color, fontsize=10)
        ax.axis("off")

        print(f"{class_dir}: predicted '{predicted_name}' at {confidence:.1%} confidence "
              f"({'correct' if correct else 'MISCLASSIFIED'})")

    for j in range(len(class_dirs), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    grid_path = os.path.join(OUTPUT_DIR, "gradcam_grid.png")
    plt.savefig(grid_path, dpi=150)
    print(f"\nSaved class overview grid to {grid_path}")


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for the wound classifier")
    parser.add_argument("image", nargs="?", default=None, help="Path to a single image. Omit to run one-per-class grid.")
    parser.add_argument("--class", dest="class_name", default=None, help="Force explaining a specific class instead of the model's top prediction.")
    parser.add_argument("--output", dest="output_path", default=None, help="Where to save the single-image heatmap (default: gradcam_output/).")
    args = parser.parse_args()

    if args.image:
        run_single_image(args.image, class_name=args.class_name, output_path=args.output_path)
    else:
        run_class_grid()


if __name__ == "__main__":
    main()