from __future__ import annotations

import tensorflow as tf

from config import CLASS_NAMES, IMAGE_SIZE
from preprocessing.data_loader import build_augmentation


@tf.keras.utils.register_keras_serializable(package="BrainTumor")
class ApplicationPreprocess(tf.keras.layers.Layer):
    """Serializable preprocessing layer for transfer-learning backbones."""

    def __init__(self, backbone_name: str, **kwargs):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name.lower()

    def call(self, inputs):
        x = inputs * 255.0
        if self.backbone_name == "mobilenetv2":
            return tf.keras.applications.mobilenet_v2.preprocess_input(x)
        if self.backbone_name == "efficientnetb0":
            return tf.keras.applications.efficientnet.preprocess_input(x)
        if self.backbone_name == "resnet50":
            return tf.keras.applications.resnet50.preprocess_input(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"backbone_name": self.backbone_name})
        return config


def _compile(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_custom_cnn(dropout: float = 0.35, learning_rate: float = 1e-3) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = build_augmentation()(inputs)
    for filters in [32, 64, 128, 256]:
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return _compile(tf.keras.Model(inputs, outputs, name="custom_cnn"), learning_rate)


def build_transfer_model(
    backbone_name: str,
    dropout: float = 0.35,
    learning_rate: float = 1e-4,
    trainable: bool = False,
) -> tf.keras.Model:
    backbones = {
        "mobilenetv2": tf.keras.applications.MobileNetV2,
        "efficientnetb0": tf.keras.applications.EfficientNetB0,
        "resnet50": tf.keras.applications.ResNet50,
    }
    if backbone_name.lower() not in backbones:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    backbone_key = backbone_name.lower()
    backbone_cls = backbones[backbone_key]
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = ApplicationPreprocess(backbone_key, name=f"{backbone_key}_preprocess")(inputs)
    base = backbone_cls(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = trainable
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return _compile(tf.keras.Model(inputs, outputs, name=backbone_name.lower()), learning_rate)


def get_model(name: str, dropout: float = 0.35, learning_rate: float = 1e-4) -> tf.keras.Model:
    if name == "custom_cnn":
        return build_custom_cnn(dropout=dropout, learning_rate=learning_rate)
    return build_transfer_model(name, dropout=dropout, learning_rate=learning_rate)
