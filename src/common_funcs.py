import sys


def xprint(*args, **kwargs) -> None:
    if "flush" not in kwargs:
        kwargs["flush"] = True

    if "file" not in kwargs:
        kwargs["file"] = sys.stderr

    print(*args, **kwargs)
