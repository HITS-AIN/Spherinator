# Contributing to Spherinator & HiPSter

We welcome contributions to the Spherinator & HiPSter project. This document outlines the guidelines for contributing to the project. Don't worry about the length of this document. There are various ways to contribute to the project at different levels. Just get started, and we will help you through the process. Finally, all the guidelines and checks are in place to ensure the quality of the project.

## Code of Conduct

The project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). Please read the code of conduct before contributing to the project.

## Use best practices for a clean codebase

The Python code must adhere to the [PEP8 Python coding style guide](https://peps.python.org/pep-0008/) for formatting and syntax. The [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) provides additional guidelines for writing clean and readable code. The Spherinator is using the design principles of the [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) library.

## Python dependencies

The project uses [Poetry](https://python-poetry.org/) for dependency management. The dependencies are listed in the `pyproject.toml` file. To add new dependencies, use Poetry's `poetry add` command. It is recommended to keep the number of dependencies to a minimum.

## Documentation

Write docstrings for all public modules, functions, classes, and methods. Follow docstring conventions as outlined by [PEP 257](https://peps.python.org/pep-0257/).
The user documentation is located in the `docs` directory and will be generated with [Sphinx](https://www.sphinx-doc.org/en/master/index.html) and hosted on [Read The Docs](https://spherinator.readthedocs.io/en/latest/index.html).

## Testing

The code is tested using [pytest](https://docs.pytest.org). All tests are located in the `tests` directory. Please write tests for all new code and ensure all tests pass before creating a pull request. The CI workflow will run the tests and fail if any test fails.

## Continuous Integration

GitHub Actions is used for continuous integration. The CI workflow is defined in [.github/workflows/python-package.yml](.github/workflows/python-package.yml). The CI workflow runs the tests and checks the code style and the code formatting on every push to the repository.

## Code Formatting

The Python code must be formatted with [black](https://black.readthedocs.io/en/stable/).
This can be done manually by running `black .` in the project directory or by installing the vscode extension `ms-python.black-formatter`. The format on save option is enabled for `vscode` in the `.vscode/settings.json` file.
The CI workflow will check the code formatting and fail if the code is not formatted correctly.

## Static code analysis

The code will be analyzed using [flake8](https://flake8.pycqa.org/en/latest/).
The linting rules are defined in the `.flake8` file. The linter can be run manually by running `flake8` in the project directory.
The CI workflow will check the code for any issues reported by flake8 and fail if any issues are found.

## Pull requests

The project uses the [GitHub flow](https://guides.github.com/introduction/flow/) to manage pull requests.

1. Create a fork of the repository. Although you can create a branch in the main repository, creating a fork for better isolation is recommended.
2. Create a branch in your fork. Let the branch name describe the changes you are making.
3. Make the changes in the branch.
4. Check if all the checks are passing. If not, fix the issues and push the changes to the branch.
5. Create a pull request from your branch to the `main` repository and describe your changes.
6. The pull request will be reviewed by the maintainers. The review process may involve multiple iterations of changes and reviews.
7. Please keep the branch up to date with the `main` repository by rebasing it on the main branch if necessary.
8. Once the pull request is approved, it will be merged into the `main` repository, and the feature branch will be deleted.

## Issues

If you find a bug or have a feature request, please create an issue in the [GitHub issue tracker](https://github.com/HITS-AIN/Spherinator/issues).

Thank you for contributing to the Spherinator & HiPSter project!
