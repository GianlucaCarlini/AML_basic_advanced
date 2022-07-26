from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import AveragePooling2D


def transpose_bn_block(inputs, filters, stage, activation="relu"):

    transpose_name = f"decoder_stage_{stage}a_transpose"
    bn_name = f"decoder_stage_{stage}a_bn"
    act_name = f"decoder_stage_{stage}a_{activation}"

    x = inputs
    x = Conv2DTranspose(
        filters,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding="same",
        name=transpose_name,
        use_bias=False,
    )(inputs)

    x = BatchNormalization(axis=-1, name=bn_name)(x)

    x = Activation(activation, name=act_name)(x)

    return x


def conv_bn_block(inputs, filters, stage, activation="relu"):

    conv_block_name = f"decoder_stage_{stage}b"

    x = inputs
    x = Conv2D(
        filters,
        kernel_size=(3, 3),
        kernel_initializer="he_uniform",
        padding="same",
        name=f"{conv_block_name}_conv",
        use_bias=False,
    )(x)

    x = BatchNormalization(axis=-1, name=f"{conv_block_name}_bn")(x)

    x = Activation(activation, name=f"{conv_block_name}_{activation}")(x)

    return x


def decoder_block(inputs, filters, stage, skip=None, activation="relu"):

    x = inputs
    x = transpose_bn_block(x, filters, stage, activation)

    if skip is not None:
        x = Concatenate(axis=-1, name=f"decoder_stage_{stage}concat")([x, skip])

    x = conv_bn_block(x, filters, stage, activation)

    return x


def aspp_block(input, filters):

    y1 = AveragePooling2D(pool_size=(input.shape[1], input.shape[2]))(input)
    y1 = Conv2D(filters, 1, padding="same", name="ASPP_conv1")(y1)
    y1 = BatchNormalization(name="ASPP_bn1")(y1)
    y1 = Activation("relu", name="ASPP_act1")(y1)
    y1 = Conv2DTranspose(
        filters,
        (8, 8),
        activation="relu",
        strides=(input.shape[1], input.shape[2]),
        name="ASPP_transp1",
    )(y1)

    y2 = Conv2D(
        filters, 1, dilation_rate=1, padding="same", use_bias=False, name="ASPP_conv2"
    )(input)
    y2 = BatchNormalization(name="ASPP_bn2")(y2)
    y2 = Activation("relu", name="ASPP_act2")(y2)

    y3 = Conv2D(
        filters, 3, dilation_rate=6, padding="same", use_bias=False, name="ASPP_conv3"
    )(input)
    y3 = BatchNormalization(name="ASPP_bn3")(y3)
    y3 = Activation("relu", name="ASPP_act3")(y3)

    y4 = Conv2D(
        filters, 3, dilation_rate=12, padding="same", use_bias=False, name="ASPP_conv4"
    )(input)
    y4 = BatchNormalization(name="ASPP_bn4")(y4)
    y4 = Activation("relu", name="ASPP_act4")(y4)

    y5 = Conv2D(
        filters, 3, dilation_rate=18, padding="same", use_bias=False, name="ASPP_conv5"
    )(input)
    y5 = BatchNormalization(name="ASPP_bn5")(y5)
    y5 = Activation("relu", name="ASPP_act5")(y5)

    y = Concatenate(name="ASPP_concat")([y1, y2, y3, y4, y5])

    y = Conv2D(
        filters,
        1,
        dilation_rate=1,
        padding="same",
        use_bias=False,
        name="ASPP_conv_final",
    )(y)
    y = BatchNormalization(name="ASPP_bn_final")(y)
    y = Activation("relu", name="ASPP_act_final")(y)

    return y
