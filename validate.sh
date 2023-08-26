set -ex
black src
flake8 src
mypy src
