import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # #Silences unnecessary spam from TensorFlow libraries. Set to 0 for full output
import gpusetter

from helper_functions import balance_true_and_fakes, balance_iterations
from timer import Timer
from dataset_loader import DatasetLoader
from network import ModelManager
import config as cfg
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import tensorflow as tf

def main():

    #Load and prepare datasets used in the training
    loader = DatasetLoader(max_workers=12)
    loader.columns_to_load = cfg.columns_to_load
    with Timer("Load data to memory"):
        qcd_dataframe = loader.load_into_dataframe("/work/data/tracking/QCD_Flat_15to3000/results/*.root")
        test_qcd_dataframe = qcd_dataframe[-cfg.test_set_size:]
        qcd_dataframe = qcd_dataframe[:-cfg.test_set_size]
        qcd_high_pt = loader.load_into_dataframe("/work/data/tracking/QCD_1800to2400/results/*.root")
        displaced_dataframe = loader.load_into_dataframe("/work/data/tracking/Displaced_SUSY/results/*.root")
        ZEE_dataframe = loader.load_into_dataframe("/work/data/tracking/ZEE/results/*.root")
        test_ZEE_dataframe = ZEE_dataframe[-cfg.test_set_size:]
        ZEE_dataframe = ZEE_dataframe[:-cfg.test_set_size]
        test_displaced_dataframe = displaced_dataframe[-cfg.test_set_size:]
        displaced_dataframe = displaced_dataframe[:-cfg.test_set_size]
        training_frame = pd.concat([qcd_dataframe,
                                    qcd_high_pt.sample(n=int(1e7), replace=True),
                                    displaced_dataframe.sample(n=int(1e7), replace=True),
                                    ZEE_dataframe.sample(n=int(1e7), replace=True)])

    training_frame = balance_iterations(training_frame)
    validation_frame = training_frame[-int(training_frame.shape[0]*0.1):]
    training_frame = training_frame[:-int(training_frame.shape[0]*0.1)]

    with Timer("Calculate min, max values"):
        min_abs_log, max_abs_log = training_frame.loc[:, cfg.columns_of_regular_inputs[:19]].abs().agg([min, max]).values
        min_, max_ = training_frame.loc[:, cfg.columns_of_regular_inputs[19:]].agg([min, max]).values
        min_abs_log[:17] = np.log1p(min_abs_log[:17])
        max_abs_log[:17] = np.log1p(max_abs_log[:17])
        min_abs_log = np.concatenate((min_abs_log, min_))
        max_abs_log = np.concatenate((max_abs_log, max_))
    sample_weights = np.log1p(np.clip(training_frame.loc[:, "trk_pt"], 0.0, 50.0))
    sample_weights = tf.convert_to_tensor(sample_weights, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices(({"regular_input_layer":training_frame.loc[:, cfg.columns_of_regular_inputs],
                                                   "categorical_input_layer":training_frame.loc[:, "trk_originalAlgo"]},
                                                  training_frame.loc[:, "trk_isTrue"], sample_weights))
    val_dataset = tf.data.Dataset.from_tensor_slices(({"regular_input_layer":validation_frame.loc[:, cfg.columns_of_regular_inputs],
                                                   "categorical_input_layer":validation_frame.loc[:, "trk_originalAlgo"]},
                                                  validation_frame.loc[:, "trk_isTrue"]))
    dataset = dataset.batch(cfg.network_fit_param['batch_size'], drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(cfg.network_fit_param['batch_size'])


    model_manager = ModelManager(n_regular_inputs=len(cfg.columns_of_regular_inputs),
                                 min_values=min_abs_log,
                                 max_values=max_abs_log)

    model_manager.initalize_model(reinitialize_model=True)

    model = model_manager.get_model()

    with Timer("Training"):
        model.fit(dataset,
                  validation_data=val_dataset,
                  **cfg.network_fit_param
        )

    with Timer("Save"):
        model_manager.save_model(model)

    with Timer("Predictions"):
        test_displaced_dataframe.loc[:, "prediction"] = model.predict([test_displaced_dataframe.loc[:, cfg.columns_of_regular_inputs],
                                                                  test_displaced_dataframe.loc[:, "trk_originalAlgo"]], batch_size=cfg.network_fit_param['batch_size'],
                                                                  use_multiprocessing=True, workers=12)
        test_qcd_dataframe.loc[:, "prediction"] = model.predict([test_qcd_dataframe.loc[:, cfg.columns_of_regular_inputs],
                                                                  test_qcd_dataframe.loc[:, "trk_originalAlgo"]], batch_size=cfg.network_fit_param['batch_size'],
                                                                  use_multiprocessing=True, workers=12)
        test_ZEE_dataframe.loc[:, "prediction"] = model.predict([test_ZEE_dataframe.loc[:, cfg.columns_of_regular_inputs],
                                                                  test_ZEE_dataframe.loc[:, "trk_originalAlgo"]], batch_size=cfg.network_fit_param['batch_size'],
                                                                  use_multiprocessing=True, workers=12)

    with Timer("Plotting"):
        model_manager.make_plots([test_displaced_dataframe, test_qcd_dataframe, test_ZEE_dataframe],
                                 ["Displaced_SUSY", "QCD", "ZEE"])

if __name__ == "__main__":
    main()
