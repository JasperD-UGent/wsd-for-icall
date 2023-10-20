from .process_JSONs import dump_json, load_json
import copy
import h5py
import numpy as np
import os
import pathlib
import sys
from typing import Dict, Tuple, Union


def define_type_name_query_support_data_and_update_meta(
        direc_outp: str,
        direc_meta: str,
        direc_item_dep_info: str,
        ambig_item_code_fns: str,
        fn_meta: str,
        d_criteria: Dict
) -> str:
    """Define the type name of the support data criteria defined in the query and update the meta dictionary.
    :param direc_outp: name of the directory in which all output of the method is saved.
    :param direc_meta: name of the directory in which meta information is saved.
    :param direc_item_dep_info: name of the directory in which information dependent on the ambiguous item is saved.
    :param ambig_item_code_fns: the ambiguous item code used in filenames.
    :param fn_meta: filename of the file in which the meta information is saved.
    :param d_criteria: dictionary containing the support data criteria defined in this query.
    :return: the type name.
    """
    path_direc_ambig_item = os.path.join(direc_outp, direc_meta, direc_item_dep_info, ambig_item_code_fns)

    if not os.path.exists(os.path.join(path_direc_ambig_item, fn_meta)):
        dump_json(path_direc_ambig_item, fn_meta, {}, indent=2)

    d_meta = load_json(os.path.join(path_direc_ambig_item, fn_meta))
    d_meta_copy = copy.deepcopy(d_meta)

    if d_criteria in d_meta_copy.values():

        for type_name in d_meta_copy:

            if d_meta_copy[type_name] == d_criteria:
                type_name_query = type_name

    else:
        l_type_ids = []

        for type_name in d_meta_copy:
            l_type_ids.append(int(type_name.replace("type", "")))

        if l_type_ids:
            type_id = str((max(l_type_ids) + 1))
            type_name_query = "type" + type_id
        else:
            type_name_query = "type1"

        d_meta_copy[type_name_query] = d_criteria
        dump_json(path_direc_ambig_item, fn_meta, d_meta_copy, indent=2)

    return type_name_query


def load_support_data(
        path_direc_support_data_dataset_type: Union[str, pathlib.Path],
        fn_support_data_l_sims: str,
        fn_support_data_d_lab_data: str,
        fn_support_data_d_mapping_new_orig: str,
        fn_support_data_d_mapping_orig_new: str
) -> Tuple[np.ndarray, Dict, Dict, Dict]:
    """Load the support data to apply the WSD method.
    :param path_direc_support_data_dataset_type: path at which the directory containing the support data for the given
        query is located.
    :param fn_support_data_l_sims: filename of the file containing the list of cosine similarity values between all
        sentences of the dataset.
    :param fn_support_data_d_lab_data: filename of the file in which the processed version of the original labelled data
        dictionary is saved.
    :param fn_support_data_d_mapping_new_orig: filename of the file containing new IDs of the sentences linked to their
        original data.
    :param fn_support_data_d_mapping_orig_new: filename of the file containing the original data of the sentences linked
        to their new IDs.
    :return: the list of cosine similarities (in a Numpy array), the labelled data dictionary and the mapping
        dictionaries.
    """
    hf = h5py.File(os.path.join(path_direc_support_data_dataset_type, fn_support_data_l_sims), "r")
    l_sims = np.array(hf.get("x"))
    hf.close()

    d_lab_data = load_json(os.path.join(path_direc_support_data_dataset_type, fn_support_data_d_lab_data))
    d_mapping_new_orig = load_json(
        os.path.join(path_direc_support_data_dataset_type, fn_support_data_d_mapping_new_orig)
    )
    d_mapping_orig_new = load_json(
        os.path.join(path_direc_support_data_dataset_type, fn_support_data_d_mapping_orig_new)
    )

    return l_sims, d_lab_data, d_mapping_new_orig, d_mapping_orig_new


def save_support_data(
        path_direc_support_data_dataset_type: Union[str, pathlib.Path],
        fn_support_data_l_sims: str,
        fn_support_data_d_lab_data: str,
        fn_support_data_d_mapping_new_orig: str,
        fn_support_data_d_mapping_orig_new: str,
        l_sims: np.ndarray,
        d_lab_data: Dict,
        d_mapping_new_orig: Dict,
        d_mapping_orig_new: Dict
) -> None:
    """Save the support data to apply the WSD method.
    :param path_direc_support_data_dataset_type: path to the directory in which the support data for the given query
        should be saved.
    :param fn_support_data_l_sims: filename of the file in which the list of cosine similarity values between all
        sentences of the dataset should be saved.
    :param fn_support_data_d_lab_data: filename of the file in which the processed version of the original labelled data
        dictionary should be saved.
    :param fn_support_data_d_mapping_new_orig: filename of the file in which the new IDs of the sentences should be
        linked to their original data.
    :param fn_support_data_d_mapping_orig_new: filename of the file in which the original data of the sentences should
        be linked to their new IDs.
    :param l_sims: list of cosine similarity values.
    :param d_lab_data: processed version of the original labelled data dictionary.
    :param d_mapping_new_orig: mapping dictionary in which the new sentence IDs are linked to their original data.
    :param d_mapping_orig_new: mapping dictionary in which the original sentence data are linked to their new IDs.
    :return: `None`
    """
    hf = h5py.File(os.path.join(path_direc_support_data_dataset_type, fn_support_data_l_sims), "w")
    hf.create_dataset("x", data=l_sims, compression="gzip")
    hf.close()

    dump_json(path_direc_support_data_dataset_type, fn_support_data_d_lab_data, d_lab_data)
    dump_json(path_direc_support_data_dataset_type, fn_support_data_d_mapping_new_orig, d_mapping_new_orig)
    dump_json(path_direc_support_data_dataset_type, fn_support_data_d_mapping_orig_new, d_mapping_orig_new)


def support_data_check_lab_data(
        direc_temp: str,
        direc_support_data: str,
        ambig_item_code: str,
        direc_support_data_dataset: str,
        dataset_name: str,
        type_support_data: str,
        d_lab_data: Dict
) -> bool:
    """Check if labelled data of the saved support data corresponds to the data included in the original labelled data
    dictionary.
    :param direc_temp: name of the directory in which all temp data generated by the method are saved.
    :param direc_support_data: name of the directory in which all support data (i.e. the processed input files) are
        saved.
    :param ambig_item_code: ambiguous item code.
    :param direc_support_data_dataset: name of the directory in which the support data for the target and rest sets are
        saved.
    :param dataset_name: name of the dataset.
    :param type_support_data: the type name of the support data criteria defined in the query.
    :param d_lab_data: original labelled data dictionary.
    :return: `True` if correspondence, `False` if no correspondence.
    """
    ambig_item_code_fns = ambig_item_code.replace("|", "_")
    path_direc_type_support_data = os.path.join(
        direc_temp, direc_support_data, ambig_item_code_fns, direc_support_data_dataset, dataset_name, type_support_data
    )

    if not os.path.isdir(path_direc_type_support_data):
        os.makedirs(path_direc_type_support_data)

    direc_support_data_check_lab_data = direc_support_data_dataset + "_check_lab_data"
    path_to_direc_support_data_check_lab_data = os.path.join(
        direc_temp, direc_support_data, ambig_item_code_fns, direc_support_data_check_lab_data
    )
    path_to_direc_support_data_check_lab_data_dataset = os.path.join(
        path_to_direc_support_data_check_lab_data, dataset_name
    )
    path_to_direc_support_data_check_lab_data_dataset_type = os.path.join(
        path_to_direc_support_data_check_lab_data_dataset, type_support_data
    )

    d_example_sents = {}
    fn_d_example_sents = "d_example_sents.json"
    d_lab_data_ambig_item = d_lab_data[ambig_item_code]

    for sense in d_lab_data_ambig_item:
        d_example_sents[sense] = {}
        d_example_sents[sense]["orig"] = d_lab_data_ambig_item[sense]["l_example_sents"]
        d_example_sents[sense]["autom"] = d_lab_data_ambig_item[sense]["l_example_sents_autom_added"] \
            if "l_example_sents_autom_added" in d_lab_data_ambig_item[sense] else []

    if not os.path.exists(os.path.join(path_to_direc_support_data_check_lab_data_dataset_type, fn_d_example_sents)):
        lab_data_correspondence = False
        dump_json(path_to_direc_support_data_check_lab_data_dataset_type, fn_d_example_sents, d_example_sents)
    else:
        d_example_sents_json = load_json(
            os.path.join(path_to_direc_support_data_check_lab_data_dataset_type, fn_d_example_sents)
        )

        if d_example_sents == d_example_sents_json:
            lab_data_correspondence = True
        else:
            lab_data_correspondence = False
            dump_json(path_to_direc_support_data_check_lab_data_dataset_type, fn_d_example_sents, d_example_sents)

    return lab_data_correspondence
