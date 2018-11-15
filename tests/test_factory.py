import unittest

import numpy as np

from mario.factory.transformer_factory import make_transformer


def identity(x):
    """Identity function"""
    return x


def scale(x, factor=1.):
    """Scaler function"""
    return x * factor


def shift_scale(x, offset=0., factor=1.):
    """Apply offset and scale"""
    return offset + x * factor


class TestTransformerFactory(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [1, 2],
            [3, 4],
        ], dtype=float)

    def test_transformers(self):
        """transformers created with default kwargs"""
        functions = [identity, scale, shift_scale]
        for func in functions:
            X_answer = func(self.X)
            transformer = make_transformer(func)
            X_test = transformer.fit_transform(self.X)
            np.testing.assert_array_equal(X_test, X_answer)

    def test_transformers_kwargs_at_init(self):
        """kwargs passed to init"""
        functions = [identity, scale, shift_scale]
        kwargss = [{}, {'factor': 2}, {'offset': 1}]
        for func, kwargs in zip(functions, kwargss):
            X_answer = func(self.X, **kwargs)
            transformer = make_transformer(func, **kwargs)
            X_test = transformer.fit_transform(self.X)
            np.testing.assert_array_equal(X_test, X_answer)

    def test_transformers_set_params(self):
        """kwargs passed to set_params"""
        functions = [identity, scale, shift_scale]
        kwargss = [{}, {'factor': 2}, {'offset': 1}]
        for func, kwargs in zip(functions, kwargss):
            X_answer = func(self.X, **kwargs)
            transformer = make_transformer(func)
            transformer.set_params(**kwargs)
            X_test = transformer.fit_transform(self.X)
            np.testing.assert_array_equal(X_test, X_answer)


if __name__ == '__main__':
    unittest.main()
