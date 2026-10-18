# Contributing

Issues and pull requests are welcome.

## Setup

Python 3.10 or newer.

```bash
pip install -e ".[dev,paper]"
```

## Checks

```bash
make check
```

That runs the tests and rebuilds the manuscript tables and figures. A change that moves those numbers should update `paper/causalgeo.tex` in the same pull request.

## Conduct

This project follows the [code of conduct](CODE_OF_CONDUCT.md). Report a problem to puttaparthy.aditya@gmail.com.
