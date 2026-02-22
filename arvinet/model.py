import tensorflow as tf
import os


# ==========================================================
# Residual Block
# ==========================================================

def residual_block(inputs, filters, downsample=False):
    """
    Basic ResNet-style residual block.

    Args:
        inputs: input tensor
        filters: number of output filters
        downsample: whether to reduce spatial resolution

    Returns:
        Output tensor
    """

    stride = 2 if downsample else 1

    # ----- Main path -----
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        strides=1,  # IMPORTANT: always 1 here
        padding="same",
        use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # ----- Shortcut path -----
    shortcut = inputs

    if downsample or inputs.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    # ----- Merge -----
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)

    return x


# ==========================================================
# Backbone Architecture
# ==========================================================

def build_arvinet_backbone(input_shape=(32, 32, 3)):
    """
    Builds the ArviNet backbone.

    Returns:
        tf.keras.Model that outputs a 512-dimensional feature vector.
    """

    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # ----- Stem -----
    x = tf.keras.layers.Conv2D(
        64,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        name="stem_conv"
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)

    # ----- Stage 1 -----
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # ----- Stage 2 -----
    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 128)

    # ----- Stage 3 -----
    x = residual_block(x, 256, downsample=True)
    x = residual_block(x, 256)

    # ----- Stage 4 -----
    x = residual_block(x, 512, downsample=True)
    x = residual_block(x, 512)

    # ----- Global Pooling -----
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    model = tf.keras.Model(inputs, x, name="ArviNet_Backbone")

    return model


# ==========================================================
# Public API
# ==========================================================

def ArviNet(pretrained=False, input_shape=(32, 32, 3)):
    """
    Public function to load ArviNet backbone.

    Args:
        pretrained (bool): If True, loads pretrained weights.
        input_shape (tuple): Input image shape.

    Returns:
        tf.keras.Model
    """

    model = build_arvinet_backbone(input_shape=input_shape)

    if pretrained:
        weights_path = os.path.join(
            os.path.dirname(__file__),
            "weights",
            "arvinet_pretrained.weights.h5"
        )

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Pretrained weights not found at {weights_path}"
            )

        model.load_weights(weights_path)

    return model
