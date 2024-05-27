# Spherinator for Developers

## Poetry

Poetry installs the package and all dependencies in a virtual environment.

```bash
poetry install
```

The project is installed in an editable mode. This means that changes to the source code are immediately available to the installed package.


## Publish to TestPyPi

```bash
poerty build
poetry publish -r testpypi -u __token__ -p <TOKEN>
```

## Install from TestPyPi

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple spherinator
```
