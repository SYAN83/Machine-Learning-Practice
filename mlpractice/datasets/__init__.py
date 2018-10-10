

import pandas
import os


path_to_data = './mlpractice/datasets/'


def load_iris():
    return pandas.read_csv(filepath_or_buffer=os.path.join(path_to_data, 'iris.csv'))