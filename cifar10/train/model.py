# -*- coding: utf-8 -*-
"""
Model Building Module:
Supports multiple architectures:
- Custom CNN (3 conv blocks, Conv->BN->ReLU)
- Residual CNN (custom 6-block ResNet for 32x32 input)
- ResNet50 (ImageNet-pretrained transfer learning)
- EfficientNetB0 (ImageNet-pretrained transfer learning)
"""

import tensorflow as tf
from tensorflow.keras import Input, applications, layers, models, regularizers

from config import Config


class CIFARModel:
    """Factory that builds the configured model architecture for CIFAR classification."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = cfg.logger

    def build(self) -> models.Model:
        """Build and return an (uncompiled) Keras model per cfg.model_type."""
        model_type = self.cfg.model_type.lower()
        self.logger.info(f"Building {model_type} model...")

        builders = {
            "cnn": self._build_cnn,
            "resnet": self._build_residual_cnn,
            "resnet50": self._build_resnet50,
            "efficientnetb0": self._build_efficientnet,
        }
        if model_type not in builders:
            raise ValueError(f"Unsupported model_type: {self.cfg.model_type}")
        return builders[model_type]()

    # ==========================
    # Building Blocks
    # ==========================
    def _conv_bn_relu(self, x, filters: int, kernel_size: int, name: str):
        """Standard Conv2D -> BatchNormalization -> ReLU block.

        BN sits between the linear transform and the nonlinearity, which is
        the ordering from the original ResNet paper (Conv(relu) -> BN would
        normalize post-activation values and loses part of BN's benefit).
        """
        x = layers.Conv2D(filters, kernel_size, padding="same",
                          name=f"{name}_conv")(x)
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
        return layers.Activation("relu", name=f"{name}_relu")(x)

    # ==========================
    # Standard CNN
    # ==========================
    def _build_cnn(self) -> models.Model:
        """3 conv blocks (2x Conv-BN-ReLU + pool + dropout each) + MLP head.

        32x32 input -> 3 pooling stages -> 4x4x128 -> dense classifier.
        """
        inputs = Input(shape=self.cfg.input_shape, name="input")
        x = inputs

        # (filters, kernel_size) per block
        conv_blocks = [(32, 3), (64, 3), (128, 3)]

        for i, (filters, kernel_size) in enumerate(conv_blocks, 1):
            x = self._conv_bn_relu(x, filters, kernel_size, f"block{i}_1")
            x = self._conv_bn_relu(x, filters, kernel_size, f"block{i}_2")
            x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
            x = layers.Dropout(0.25, name=f"dropout{i}")(x)

        # Classifier head
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(128, name="fc1")(x)
        x = layers.BatchNormalization(name="bn_fc1")(x)
        x = layers.Activation("relu", name="fc1_relu")(x)
        x = layers.Dropout(0.3, name="dropout_fc1")(x)

        outputs = layers.Dense(self.cfg.num_classes, activation="softmax", name="output")(x)
        return models.Model(inputs, outputs, name="CIFAR10_CNN")

    # ==========================
    # Residual CNN
    # ==========================
    def _residual_block(self, x, filters: int, block_name: str,
                        downsample: bool = False):
        """Post-activation residual block (ResNet v1) with identity/projection shortcut.

        Structure: [Conv3x3-BN-ReLU] -> [Conv3x3-BN] -> (+shortcut) -> ReLU.
        Dropout inside residual branches is intentionally omitted: it injects
        noise into the identity mapping and is not part of standard ResNet
        designs for CIFAR.
        """
        stride = 2 if downsample else 1
        shortcut = x

        # Projection shortcut when spatial/channel dims change
        if downsample or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding="same",
                                     kernel_regularizer=regularizers.l2(1e-4),
                                     name=f"{block_name}_shortcut_conv")(shortcut)
            shortcut = layers.BatchNormalization(name=f"{block_name}_shortcut_bn")(shortcut)

        # Main path
        x = layers.Conv2D(filters, (3, 3), strides=stride, padding="same",
                          kernel_regularizer=regularizers.l2(1e-4),
                          name=f"{block_name}_conv1")(x)
        x = layers.BatchNormalization(name=f"{block_name}_bn1")(x)
        x = layers.Activation("relu", name=f"{block_name}_relu1")(x)

        x = layers.Conv2D(filters, (3, 3), padding="same",
                          kernel_regularizer=regularizers.l2(1e-4),
                          name=f"{block_name}_conv2")(x)
        x = layers.BatchNormalization(name=f"{block_name}_bn2")(x)

        # Residual connection
        x = layers.Add(name=f"{block_name}_add")([x, shortcut])
        return layers.Activation("relu", name=f"{block_name}_output")(x)

    def _build_residual_cnn(self) -> models.Model:
        """Custom ResNet for 32x32 inputs: stem + 3 stages x 2 blocks + GAP head."""
        inputs = Input(shape=self.cfg.input_shape, name="input")

        # Stem
        x = layers.Conv2D(64, (3, 3), padding="same",
                          kernel_regularizer=regularizers.l2(1e-4),
                          name="initial_conv")(inputs)
        x = layers.BatchNormalization(name="initial_bn")(x)
        x = layers.Activation("relu", name="initial_relu")(x)

        # Stage 1: 32x32, 32 filters
        x = self._residual_block(x, 32, "stage1_block1")
        x = self._residual_block(x, 32, "stage1_block2")

        # Stage 2: 16x16, 64 filters
        x = self._residual_block(x, 64, "stage2_block1", downsample=True)
        x = self._residual_block(x, 64, "stage2_block2")

        # Stage 3: 8x8, 128 filters
        x = self._residual_block(x, 128, "stage3_block1", downsample=True)
        x = self._residual_block(x, 128, "stage3_block2")

        # Global pooling and classifier (GAP instead of Flatten: fewer
        # parameters and better generalization)
        x = layers.GlobalAveragePooling2D(name="global_pool")(x)
        x = layers.Dense(128, name="fc1")(x)
        x = layers.BatchNormalization(name="bn_fc1")(x)
        x = layers.Activation("relu", name="fc1_relu")(x)
        x = layers.Dropout(0.5, name="dropout_fc")(x)

        outputs = layers.Dense(self.cfg.num_classes, activation="softmax", name="output")(x)
        return models.Model(inputs, outputs, name="ResidualCNN_CIFAR")

    # ==========================
    # Transfer Learning Backbones
    # ==========================
    def _build_transfer_model(self, base_model: tf.keras.Model,
                              model_name: str) -> models.Model:
        """Wrap a frozen ImageNet backbone with a GAP + Dropout + Dense head.

        ``training=False`` on the backbone call keeps BatchNormalization in
        inference mode so the pretrained moving statistics are preserved
        while the head trains.
        """
        inputs = Input(shape=self.cfg.input_shape, name="input")
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D(name="global_pool")(x)
        x = layers.Dropout(0.4, name="dropout")(x)
        outputs = layers.Dense(self.cfg.num_classes, activation="softmax", name="output")(x)
        return models.Model(inputs, outputs, name=model_name)

    def _build_resnet50(self) -> models.Model:
        """ResNet50 with pretrained ImageNet weights, backbone initially frozen."""
        base_model = applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=self.cfg.input_shape,
        )
        base_model.trainable = False
        return self._build_transfer_model(base_model, "ResNet50_CIFAR10")

    def _build_efficientnet(self) -> models.Model:
        """EfficientNetB0 with pretrained ImageNet weights, backbone initially frozen."""
        base_model = applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=self.cfg.input_shape,
        )
        base_model.trainable = False
        return self._build_transfer_model(base_model, "EfficientNetB0_CIFAR10")


# ============================================================
# Test Section
# ============================================================
if __name__ == "__main__":
    cfg = Config()
    for model_type in ["cnn", "resnet", "resnet50", "efficientnetb0"]:
        cfg.model_type = model_type
        cfg.input_shape = None
        cfg.__post_init__()  # re-resolve input shape for the new model type
        print("=" * 60)
        print(f"Building model: {model_type}")
        print("=" * 60)
        model = CIFARModel(cfg).build()
        model.summary()
        print(f"Model {model_type} built successfully.\n")
