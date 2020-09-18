from kerastuner import HyperModel
from config import network_param_dict
from kerastuner import HyperParameters
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization

from network.custom import InputSanitizerLayer, OneHotLayer

class HyperNetwork(HyperModel):
    def __init__(self, n_regular_inputs, min_values, max_values):
        super().__init__()
        self.n_regular_inputs = n_regular_inputs
        self.min_values = min_values
        self.max_values = max_values

    def build(self, hp):
        activation = hp.Choice('activation', ['relu', 'swish'])
        layers = hp.Int('layers', 3, 10)
        use_one_hot = hp.Boolean("one_hot")

        _kernel_regularization = network_param_dict["kernel_regularization"]
        _kernel_initializer = network_param_dict["kernel_initializer"]

        reg_inp = Input(self.n_regular_inputs, name="regular_input_layer")
        cat_inp = Input(1, name="categorical_input_layer")

        sanitized_inp = InputSanitizerLayer(self.min_values, self.max_values)(reg_inp)
        if use_one_hot:
            one_hot_inp = OneHotLayer()(cat_inp)
        else:
            one_hot_inp = cat_inp
        x = Concatenate()([sanitized_inp, one_hot_inp])

        for i in range(layers):
            x = Dense(units=hp.Choice(f'dense_{i+1}_units',
                           [256, 128, 64, 32]),
                      activation=activation,
                      kernel_regularizer=_kernel_regularization,
                      kernel_initializer=_kernel_initializer)(x)
            x = BatchNormalization()(x)

        # Bias initializer is the "good initializer" giving the expectation value for two
        # equally balanced classes. See "init well" from A.Karpathy https://karpathy.github.io/2019/04/25/recipe/
        out = Dense(units=1,
                    activation='sigmoid',
                    kernel_initializer=_kernel_initializer,
                    kernel_regularizer=_kernel_regularization,
                    bias_initializer=tf.keras.initializers.Constant(0.5))(x)

        model = tf.keras.Model([reg_inp, cat_inp], out)

        model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                      loss="binary_crossentropy",
                      metrics=[tf.keras.metrics.AUC()])

        return model
