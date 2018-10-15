
import pandas
import os


PATH_TO_DATASETS = './mlpractice/datasets/'


class DataSet(object):

    def __init__(self, dir_name, extensions=['.csv'], path_to_datasets=PATH_TO_DATASETS):
        data_dir = os.path.join(path_to_datasets, dir_name)
        for file_name in os.listdir(data_dir):
            name, ext = os.path.splitext(file_name)
            if ext in extensions:
                data = pandas.read_csv(filepath_or_buffer=os.path.join(data_dir, file_name))
                setattr(self, name, data)


def load_iris():
    return DataSet(dir_name='iris/')


def load_movieLens():
    return DataSet(dir_name='ml-latest-small/')