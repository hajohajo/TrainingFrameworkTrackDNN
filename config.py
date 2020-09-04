import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Add, Concatenate
from network.custom import InputSanitizerLayer, OneHotLayer

'''
Collects parameters and variables one might often change while testing different training setups into one place
'''

'''
Commonly tuned fit parameters when training a network. NOTE: Batch size is multiplied with the number of logical gpus,
because each minibatch is split evenly for each gpu when running distributed workloads with many GPUs available
'''
n_gpus = len(tf.config.list_logical_devices('GPU'))
minibatch_multiplier = n_gpus if n_gpus > 0 else 1

network_callbacks =[]
network_fit_param = {
    "batch_size":       2048*minibatch_multiplier,
    "epochs":           15,
    "callbacks":        network_callbacks,
}

'''
Commonly tuned parameters when training network
'''
network_param_dict = {
    "lr":                       3e-4,
    "loss":                     tf.keras.losses.binary_crossentropy,
    "kernel_regularization":    tf.keras.regularizers.l2(1e-4),
    "kernel_initializer":       tf.keras.initializers.lecun_normal()
}

'''
Model architecture
'''
def make_model(n_regular_inputs, n_categories, min_values, max_values):
    _kernel_regularization = network_param_dict["kernel_regularization"]
    _kernel_initializer = network_param_dict["kernel_initializer"]

    reg_inp = Input(n_regular_inputs, name="regular_input_layer")
    cat_inp = Input(1, name="categorical_input_layer")
    # x = Concatenate()([reg_inp, cat_inp])

    # min_values = tf.cast(tf.convert_to_tensor(tf.reshape(min_values, (1, min_values.shape[-1]))),
    #         dtype=tf.keras.backend.floatx())
    # max_values = tf.cast(tf.convert_to_tensor(tf.reshape(max_values, (1, max_values.shape[-1]))),
    #         dtype=tf.keras.backend.floatx())
    sanitized_inp = InputSanitizerLayer(min_values, max_values)(reg_inp)
    # sanitized_inp = InputSanitizerLayer()(reg_inp)
    one_hot_inp = OneHotLayer()(cat_inp)
    x = Concatenate()([sanitized_inp, one_hot_inp])
    # x = Concatenate()([reg_inp, cat_inp])

    x = Dense(units=512,
              activation='relu',
              kernel_regularizer=_kernel_regularization,
              kernel_initializer=_kernel_initializer)(x)
    x = BatchNormalization()(x)
    x = Dense(units=256,
              activation='relu',
              kernel_regularizer=_kernel_regularization,
              kernel_initializer=_kernel_initializer)(x)
    x = BatchNormalization()(x)
    x = Dense(units=128,
              activation='relu',
              kernel_regularizer=_kernel_regularization,
              kernel_initializer=_kernel_initializer)(x)
    x = BatchNormalization()(x)
    x = Dense(units=64,
              activation='relu',
              kernel_regularizer=_kernel_regularization,
              kernel_initializer=_kernel_initializer)(x)
    x = BatchNormalization()(x)
    x = Dense(units=32,
              activation='relu',
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
    return model


'''
Variables to load from the *.root files into the pandas DataFrames
'''
columns_to_load = [
    "trk_pt", "trk_eta", "trk_phi", "trk_ptErr", "trk_etaErr", "trk_phiErr", "trk_dxyClosestPV","trk_dzClosestPV",
    "trk_nChi2", "trk_nPixel", "trk_nStrip", "trk_ndof", "trk_dxy", "trk_dz", "trk_dxyErr", "trk_dzErr", "trk_nLostLay",
    "trk_nCluster", "trk_originalAlgo", "trk_mva", "trk_isTrue", "trk_nInnerInactive", "trk_nOuterInactive",
    "trk_nInnerLost", "trk_nOuterLost", "trk_nChi2_1Dmod","trk_px", "trk_py", "trk_pz", "trk_inner_px", "trk_inner_py",
    "trk_inner_pz", "trk_inner_pt", "trk_outer_px", "trk_outer_py", "trk_outer_pz", "trk_outer_pt"
]

'''
These variables need to be in same order when evaluating the trained network in deployment environment (i.e. CMSSW)
'''
columns_of_regular_inputs = [
    "trk_pt", "trk_inner_px", "trk_inner_py", "trk_inner_pz", "trk_inner_pt",
    "trk_outer_px", "trk_outer_py", "trk_outer_pz", "trk_outer_pt", "trk_ptErr",
    "trk_dxyClosestPV", "trk_dzClosestPV", "trk_dxy", "trk_dz", "trk_dxyErr", "trk_dzErr",
    "trk_nChi2", "trk_eta", "trk_phi", "trk_etaErr", "trk_phiErr", "trk_nPixel", "trk_nStrip", "trk_ndof",
    "trk_nInnerLost", "trk_nOuterLost", "trk_nInnerInactive", "trk_nOuterInactive", "trk_nLostLay"
]

phase1_iterations = [
    "InitialStep",
    "LowPtQuadStep",
    "HighPtTripletStep",
    "LowPtTripletStep",
    "DetachedQuadStep",
    "DetachedTripletStep",
    "PixelPairStep",
    "MixedTripletStep",
    "PixelLessStep",
    "TobTecStep",
    "JetCoreRegionalStep",
]

test_set_size = 500000
training_set_size = int(6e7)