import tensorflow as tf

class InputSanitizerLayer(tf.keras.layers.Layer):
    '''
    Layer that performes preprocessing of all regular (non-categorical) inputs.

    This includes clipping values to the range contained in the training samples, taking absolute values of some
    variables where the sign shouldn't matter, log-scaling inputs that are spread out to a huge range of values (like pT)
    and finally squeezing everything to a range between [0, 1] for faster and more stable training.
    Incorporating preprocessing to the network like this makes deployment easier when one does not have to do
    preprocessing on the inputs separately.

    NOTE: There is a shameful amount of hard-coding in the call function based on the order of the inputs. So if messing
    with the order of config.columns_to_use_in_training, make sure the ranges in the call function are adjusted if necessary

    :param
    min_values : minimum values for each input variable seen in the training set in the order config.columns_to_use_in_training
    max_values : maximum values for each input variable seen in the training set in the order config.columns_to_use_in_training
    '''

    def __init__(self, preprocessed_min_values, preprocessed_max_values, **kwargs):
        super(InputSanitizerLayer, self).__init__(**kwargs)
        self._min_values = preprocessed_min_values
        self._max_values = preprocessed_max_values
        self._min_values_tensor = tf.cast(tf.convert_to_tensor(tf.reshape(self._min_values, (1, self._min_values.shape[-1]))), dtype=tf.keras.backend.floatx())
        self._max_values_tensor = tf.cast(tf.convert_to_tensor(tf.reshape(self._max_values, (1, self._max_values.shape[-1]))), dtype=tf.keras.backend.floatx())

    def build(self, input_shape):
        super(InputSanitizerLayer, self).build(input_shape)

    def call(self, input):
        #absolute values
        _abs = tf.math.abs(input[:, :19])
        _rest = input[:, 19:]
        input = tf.concat([_abs, _rest], axis=-1)

        #log values (f: x -> log(x+1) )
        _log = tf.math.log1p(input[:, :17])
        _rest = input[:, 17:]
        input = tf.concat([_log, _rest], axis=-1)

        #clipping
        clipped = tf.math.maximum(tf.math.minimum(input, self._max_values_tensor), self._min_values_tensor)

        #scaling
        sanitized_inputs = 2.0 * tf.math.divide(clipped - self._min_values_tensor, self._max_values_tensor - self._min_values_tensor) - 1.0

        return sanitized_inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "preprocessed_min_values": self._min_values,
            "preprocessed_max_values": self._max_values
         })
        return config