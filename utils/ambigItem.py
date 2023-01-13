import sys
from typing import Tuple


def split_ambig_item_code(ambig_item_code: str) -> Tuple[str, str, str]:
    """Split the ambiguous item code into its meaningful units.
    :param ambig_item_code: ambiguous item code.
    :return: the meaningful units, i.e. the ambiguous item, the part-of-speech tag and the gender.
    """
    ambig_item_code_split = ambig_item_code.split("|")
    assert len(ambig_item_code_split) == 3, f"Invalid ambiguous item code: {ambig_item_code}."
    ambig_item = ambig_item_code_split[0]
    pos = ambig_item_code_split[1]
    gender = ambig_item_code_split[2]

    return ambig_item, pos, gender
