import codecs
import json
import os
import pathlib
import sys
from typing import Any, Optional, Union


def dump_json(
        path_to_direc: Union[str, pathlib.Path],
        fn: str,
        variable: Any,
        *,
        indent: Optional[int] = None,
) -> None:
    """Dump given variable to given JSON filename on given path.
    :param path_to_direc: path to the directory in which the JSON should be dumped (non-existing directories included in
    the path are automatically created).
    :param fn: filename the JSON should receive.
    :param variable: variable to be dumped in the JSON.
    :param indent: indentation to be applied in the JSON.
    :return: `None`
    """

    if not os.path.isdir(path_to_direc):
        os.makedirs(path_to_direc)

    with codecs.open(os.path.join(path_to_direc, fn), "w", "utf-8") as f:
        json.dump(variable, f, indent=indent) if indent is not None else json.dump(variable, f)
    f.close()


def load_json(
        path: Union[str, pathlib.Path],
        *,
        encoding: str = "utf-8"
) -> Any:
    """Load JSON file from given path.
    :param path: path at which the JSON is located.
    :param encoding: encoding of the JSON file.
    :return: the variable which was dumped in the JSON file.
    """

    with codecs.open(path, "r", encoding) as f:
        f_loaded = json.load(f)
    f.close()

    return f_loaded
