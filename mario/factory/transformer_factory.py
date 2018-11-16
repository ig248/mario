import inspect

from decorator import FunctionMaker
from sklearn.base import BaseEstimator, TransformerMixin


def identity(x):
    return x


def make_transformer_class(func=identity):
    """Make an improved FunctionTransformer class from a function."""
    signature = inspect.signature(func)
    # args = [
    #     name for name, param in signature.parameters.items()
    #     if param.default is inspect._empty
    # ]
    kwargs_defaults = [
        (name, param.default)
        for name, param in signature.parameters.items()
        if param.default is not inspect._empty
    ]
    if kwargs_defaults:
        kwargs, defaults = zip(*kwargs_defaults)
    else:
        kwargs, defaults = [], []
    # all_args = list(args) + list(kwargs)

    init_signature = '__init__(self, {args})'.format(args=', '.join(kwargs))
    init_kwarg_string = '\n'.join(
        ['self.{kwarg}={kwarg}'.format(kwarg=kwarg) for kwarg in kwargs]
    )
    init_body = 'self.func = func\n{init_kwarg_string}'.format(
        init_kwarg_string=init_kwarg_string
    )

    proto__init = FunctionMaker.create(
        init_signature, init_body, {'func': func}, defaults=defaults
    )
    proto_fit = FunctionMaker.create('fit(self, x)', 'return self', {})

    kwarg_string = ', '.join(
        ['{kwarg}=self.{kwarg}'.format(kwarg=kwarg) for kwarg in kwargs]
    )
    transform_body = 'return self.func(x, {kwarg_string})'.format(
        kwarg_string=kwarg_string
    )
    proto_transform = FunctionMaker.create(
        'transform(self, x)', transform_body, {}
    )

    proto_dict = {
        '__init__': proto__init,
        '__doc__': func.__doc__,
        'fit': proto_fit,
        'transform': proto_transform
    }

    new_class = type(
        'FunctionTransformer_' + func.__name__,
        (BaseEstimator, TransformerMixin), proto_dict
    )

    return new_class


def make_transformer(func=identity, **kwargs):
    """Make an improved FunctionTransformer from a function."""
    transformer_class = make_transformer_class(func=func)
    return transformer_class(**kwargs)
