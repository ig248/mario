# FunctionTransformer factory
Create the equivalent of `sklearn.preprocessing.FunctionTransformer` that
inherits keyword arguments from the underlying function.

No more parameter grids over lists of `kw_args` dictionaries!

## Usage
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn_transformer_factory import make_transformer

def scale(x, factor=1.):
    """Scale array x by given factor"""
    return x * factor

pipeline = Pipeline([
    ('scaler0', FunctionTransformer(scale, kw_args={'factor': 2})),
    ('scaler1', make_transformer(scale, factor=2))
])
pipeline.set_params(scaler0__kw_args={'factor': 2})
pipeline.set_params(scaler1__factor=2)

X = np.array([
    [1, 2],
    [3, 4],
], dtype=float)

X_new = pipeline.fit_transform(X)
```

## Advantages
Apart from the more compact notation, the key advantage is when using `GridSearchCV` with transformers built from functions with multiple parameters.

Compare:

```python
grid = {
    'transformer__kw_args': [dict(arg1=1, arg2=1),
                         dict(arg1=1, arg2=2),
                         dict(arg1=2, arg2=1),
                         dict(arg1=2, arg2=2)]
}
```

with

```
grid = {
    'transformer__arg1': [1, 2],
    'transformer__arg2': [1, 2],
}
```

## Implementation
The implementation relies on `inspect.signature` to parse the signature of the underlying function, `type()` built-in for dynamically creating a class, and `decorator.FunctionMaker` to create class methods dynamically.

At the end of the day, the latter relies on constructing code strings and using `eval()`. This feels wrong, but seems to be the only way in the current Python universe.
