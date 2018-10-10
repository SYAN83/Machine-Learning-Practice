import numpy
from .base import Algorithm, Model
from collections import OrderedDict


class LinearRegression(Algorithm):

    def __init__(self, features=[], label='label', prediction='prediction', fit_intercept=True):
        super().__init__(features=features, label=label, prediction=prediction, fit_intercept=fit_intercept)

    def _fit(self, X, y):
        if self.params.fit_intercept:
            X = numpy.insert(X, 0, 1, axis=1)
        X_t = numpy.transpose(X)
        XX_inv = numpy.linalg.inv(numpy.dot(X_t, X))
        _coef = numpy.dot(numpy.dot(XX_inv, X_t), y)
        if self.params.fit_intercept:
            coef = OrderedDict(zip(['intercept'] + self.params.featuresCol, _coef))
        else:
            coef = OrderedDict(zip(self.params.featuresCol, _coef))
        model = LinearRegressionModel(coef=coef, _coef=_coef, params=self.params.freeze())
        return model


class LinearRegressionModel(Model):

    def _predict(self, X):
        if self.params.fit_intercept:
            X = numpy.insert(X, 0, 1, axis=1)
        return numpy.dot(X, self._coef)