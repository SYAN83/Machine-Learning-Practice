
from abc import ABCMeta, abstractmethod
import numpy
import pandas
import copy


class Params(object):

    params = set()

    def __init__(self, **kwargs):
        self.set_params(**kwargs)

    def get_param(self, param):
        return getattr(self, param)

    def get_params(self):
        return {param: getattr(self, param) for param in self.params}

    def set_params(self, **kwargs):
        self.params.update(kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def freeze(self):
        return FrozenParams(**copy.deepcopy(self.get_params()))


class FrozenParams(object):

    params = set()

    def __init__(self, **kwargs):
        self._set_params(**kwargs)

    def get_params(self):
        return {param: getattr(self, param) for param in self.params}

    def _set_params(self, **kwargs):
        self.params.update(kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)


class Algorithm(object):
    """
    Abstract class for Machine Learning algorithm.
    """

    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.params = Params(**kwargs)

    @abstractmethod
    def _fit(self, X: numpy.ndarray, y: numpy.ndarray=None):
        """
        Fits providing an ML algorithm (that is, the learning algorithm) with training data to learn from
        :param dataset: input dataset, which is an instance of numpy.ndarray
        :returns: fitted model
        """
        raise NotImplementedError()

    def fit(self, dataset: pandas.DataFrame, **kwargs):
        """
        Fits a model to the input dataset with optional parameters.
        :param dataset: input dataset, which is an instance of pandas.DataFrame
        :param params: an optional param map that overrides default params.
        :returns: fitted model(s)
        """
        self.params.set_params(**kwargs)
        if self.params.get_param('label'):
            labelCol = self.params.label
            y = dataset.loc[:, labelCol].values
        else:
            labelCol = None
            y = None
        if self.params.get_param('features'):
            featuresCol = [col for col in dataset.columns if col in self.params.features]
        else:
            featuresCol = [col for col in dataset.columns if col != labelCol]
        X = dataset.loc[:, featuresCol].values
        self.params.set_params(featuresCol=featuresCol, labelCol=labelCol)
        return self._fit(X, y)


class Model(object):
    """
    Abstract class for Model that make prediction on dataset.
    """

    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def _predict(self, X):
        """
        Transforms the input dataset.
        :param dataset: input dataset, which is an instance of :py:class:`pyspark.sql.DataFrame`
        :returns: transformed dataset
        """
        raise NotImplementedError()

    def predict(self, dataset):
        """
        Transforms the input dataset with optional parameters.
        :param dataset: input dataset, which is an instance of :py:class:`pyspark.sql.DataFrame`
        :param params: an optional param map that overrides embedded params.
        :returns: transformed dataset
        """
        X = dataset.loc[:, self.params.featuresCol].values
        dataset[self.params.prediction] = self._predict(X)
        return dataset
