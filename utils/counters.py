import sys


def print_multiple_of(counter: int, multiple_of: int) -> None:
    """Counter which prints every multiple of a given number.
    :param counter: current value of the counter.
    :param multiple_of: number whose multiples should be printed.
    :return: `None`
    """
    if (counter / multiple_of).is_integer():
        print(counter)
