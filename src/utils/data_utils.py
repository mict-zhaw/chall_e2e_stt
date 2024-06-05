import glob
import os
from typing import Any, List, TypeVar, Callable, Type, cast, Union

T = TypeVar("T")


def from_float(x: Any) -> float:
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_bool(x: Any) -> bool:
    assert isinstance(x, bool)
    return x


def from_list2(x: Any) -> list:
    assert isinstance(x, list)
    return x


def from_dict(x: Any) -> dict:
    assert isinstance(x, dict)
    return x


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def from_class(f: Callable[[Any], T], x: Any) -> T:
    return f(x)


def from_class_or_none(f: Callable[[Any], T], x: Any) -> T:
    if x is None:
        return None
    return f(x)


def to_class(c: Type[T], x: Any, *args, **kwargs) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


def to_float(x: Any) -> float:
    assert isinstance(x, float)
    return x


def from_float_or_none(x: Any) -> Union[float, None]:
    if x is None:
        return None
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)


def from_str_or_none(x: Any) -> Union[str, None]:
    if x is None:
        return None
    assert isinstance(x, str)
    return x


def from_int_or_none(x: Any) -> Union[int, None]:
    if x is None:
        return None
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_list_or_none(x: Any) -> Union[list, None]:
    if x is None:
        return None
    assert isinstance(x, list)
    return x


def to_class_or_none(c: Type[T], x: Any) -> Union[T, None]:
    if x is None:
        return None
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


def load_files(input_dir: str, input_formats: list = None) -> list[str]:
    """
    Load files with specified formats from the given input directory
    :param input_dir:  The directory path to scan for files
    :param input_formats:  (Optional) A list of file formats to consider. Defaults to ["txt"] if None
    :return:  A list of file paths matching the specified formats
    """

    files: list = []

    if input_formats is None:
        input_formats = ["txt"]

    for input_format in input_formats:
        for input_file in glob.glob(f'{input_dir}/**/*.{input_format}', recursive=True):
            files.append(input_file)

    return files


def get_filename(input_file: str, with_extension: bool = False):
    """
    Extracts the base filename from a given file path
    :param input_file:  The input file path
    :param with_extension:  If True, includes the file extension in the returned filename. Defaults to False
    :return:  The base filename without the path and optionally without the extension
    """
    if with_extension:
        return os.path.basename(input_file)
    else:
        return os.path.splitext(os.path.basename(input_file))[0]
