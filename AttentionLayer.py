import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.channel_attn = Conv2D((1,1), kernel_size=1)
        self.activation = Activation('sigmoid')

    def call(self, x):
        # Pass the input tensor through the channel attention module
        x_attn = self.channel_attn(x)
        x_attn = self.activation(x_attn)

        # Multiply the input tensor by the attention weights
        x = x * x_attn

        return x