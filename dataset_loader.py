import pandas as pd
from glob import glob
from uproot import open
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

class DatasetLoader():
    '''
    Class for loading datasets stored as *.root files into pandas DataFrames using uproot library.

    :param:
    max_workers: Allows parallelizing data load operators into multiple processes, taking advantage of multicore CPUs to load data faster to memory
    columns_to_load: A list of TTree branch names to load into the dataframe.

    '''

    def __init__(self, max_workers=1):
        self.max_workers = max_workers
        self.columns_to_load = None

    def read_to_dataframe(self, path, columns):
        tree = open(path)["trackingNtuple/tree"].arrays(columns, namedecode='utf-8', flatten=True)
        dataframe = pd.DataFrame(tree, columns=columns, dtype='float32') #Explicitly using float32 to save memory during training
        dataframe = self.preprocess_dataframe(dataframe)
        return dataframe

    def preprocess_dataframe(self, dataframe):
        dataframe.loc[:, "trk_mva"] = (dataframe.loc[:, "trk_mva"] + 1.0) / 2.0
        return dataframe

    def load_into_dataframe(self, path_to_directory):
        '''

        :param path_to_directory:
        :return:
        '''
        if self.columns_to_load is None:
            raise TypeError("Set columns_to_load to contain a list of TTree branch names to be loaded into the dataframe as columns")
        file_paths = glob(path_to_directory)
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self.read_to_dataframe, file_paths, repeat(self.columns_to_load)))
        if len(results) == 1:
            dataframe = results[0]
        else:
            dataframe = results[0].append(results[1:]) #Joins the individual dataframes loaded by separate workers together
        return dataframe
