# Spherinator for Developers

## Publish to TestPyPi

```bash
poerty build
poetry publish -r testpypi -u __token__ -p <TOKEN>
```



## Install from TestPyPi

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple spherinator
```
