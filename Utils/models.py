from .blocks import decoder_block, conv_bn_block, transpose_bn_block, aspp_block
import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D


def Unet(
    input_shape,
    backbone="efficientnetb3",
    classes=1,
    decoder_activation="relu",
    final_activation="sigmoid",
    filters=[256, 128, 64, 32, 16],
):

    if backbone == "efficientnetb3":

        encoder = tf.keras.applications.efficientnet.EfficientNetB3(
            include_top=False, input_shape=input_shape
        )
        layers = [
            "block6a_expand_activation",
            "block4a_expand_activation",
            "block3a_expand_activation",
            "block2a_expand_activation",
        ]
        x = encoder.get_layer("top_activation").output

    elif backbone == "mobilenetv2":

        encoder = tf.keras.applications.mobilenet_v2.MobileNetV2(
            include_top=False, input_shape=input_shape
        )
        layers = [
            "block_13_expand_relu",
            "block_6_expand_relu",
            "block_3_expand_relu",
            "block_1_expand_relu",
        ]
        x = encoder.get_layer("out_relu").output

    elif backbone == "resnet50":

        encoder = tf.keras.applications.resnet50.ResNet50(
            include_top=False, input_shape=input_shape
        )
        layers = [
            "conv4_block6_out",
            "conv3_block4_out",
            "conv2_block3_out",
            "conv1_relu",
        ]
        x = encoder.get_layer("conv5_block3_out").output

    elif backbone == "efficientnetv2_b3":
        encoder = tf.keras.applications.efficientnet_v2.EfficientNetV2B3(
            include_top=False, input_shape=input_shape
        )
        layers = [
            "block6a_expand_activation",
            "block4a_expand_activation",
            "block2c_add",
            "block1b_add",
        ]
        x = encoder.get_layer("top_activation").output

    else:
        list_of_backbones = [
            "efficientnetb3",
            "mobilenetv2",
            "resnet50",
            "efficientnetv2_b3",
        ]
        raise ValueError(f"Valid backbones are: {list_of_backbones}")

    skip_connections = []
    for layer in layers:
        skip_connections.append(encoder.get_layer(layer).output)

    for i, skip in enumerate(skip_connections):
        if backbone == "efficientnetv2_b3" and i > 1:
            skip = BatchNormalization(axis=-1, name=f"BatchNorm_{i}")(skip)
            skip = Activation("swish", name=f"Activation_{i}")(skip)
        x = decoder_block(
            inputs=x,
            filters=filters[i],
            stage=i,
            skip=skip,
            activation=decoder_activation,
        )

    x = decoder_block(
        inputs=x, filters=filters[-1], stage=4, activation=decoder_activation
    )

    x = Conv2D(filters=classes, kernel_size=(3, 3), padding="same", name="final_conv")(
        x
    )
    x = Activation(final_activation, name=final_activation)(x)

    model = tf.keras.models.Model(encoder.input, x)

    return model
