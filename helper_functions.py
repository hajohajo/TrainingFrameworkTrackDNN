import numpy as np

from argparse import ArgumentParser
from datetime import datetime
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow as tf

def balance_true_and_fakes(dataframe, target):
    '''
    Resamples the dataframe so that the number of examples of both classes are equal by upsampling the smaller class.
    Assumes binary targets with 1 or 0 describing the class.
    :param
        dataframe: Pandas dataframe
        target: Name of the column containing the class label e.g. "trk_isTrue"

    :return
        a pandas dataframe
    '''


    seed = 7 #lock a random seed to get same sampling each time
    np.random.seed(seed)

    true_indices = dataframe.loc[(dataframe.loc[:, target].values == 1), :].index.values
    fake_indices = dataframe.loc[(dataframe.loc[:, target].values == 0), :].index.values
    n_true_tracks = true_indices.size
    n_fake_tracks = fake_indices.size

    if n_true_tracks > n_fake_tracks:
        balanced_indices = np.concatenate([true_indices, np.random.choice(fake_indices, size=n_true_tracks, replace=True)])
    else:
        balanced_indices = np.concatenate([true_indices, np.random.choice(true_indices, size=n_fake_tracks, replace=True)])
    np.random.shuffle(balanced_indices)
    dataframe = dataframe.iloc[balanced_indices, :]
    dataframe.reset_index(inplace=True, drop=True)
    return dataframe

def get_timestamp():
    date_time = datetime.now()
    stamp = "{year:02}{month:02}{day:02}_{hour:02}{minute:02}{second:02}".format(year=date_time.year,
                                                               month=date_time.month,
                                                               day=date_time.day,
                                                               hour=date_time.hour,
                                                               minute=date_time.minute,
                                                               second=date_time.second)

    return stamp