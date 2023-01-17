from .process_JSONs import dump_json, load_json
import codecs
import copy
import os
import sys
from typing import Dict


def define_type_name_query_preds_and_update_meta(
        enrich_type: str,
        direc_outp: str,
        direc_meta: str,
        direc_type_names_preds: str,
        fn_meta: str,
        d_criteria: Dict
) -> str:
    """Define the prediction type name of the criteria used in the query and update the meta dictionary.
    :param enrich_type: way in which the automatic addition of extra training data should be performed.
    :param direc_outp: name of the directory in which all output of the method is saved.
    :param direc_meta: name of the directory in which meta information is saved.
    :param direc_type_names_preds: name of the directory in which information on the prediction type names is saved.
    :param fn_meta: filename of the file in which the meta information is saved.
    :param d_criteria: dictionary containing the criteria used in the query.
    :return: the type name.
    """
    path_direc_type_names_preds = os.path.join(direc_outp, direc_meta, direc_type_names_preds)

    if not os.path.exists(os.path.join(path_direc_type_names_preds, fn_meta)):
        dump_json(path_direc_type_names_preds, fn_meta, {}, indent=2)

    d_meta = load_json(os.path.join(path_direc_type_names_preds, fn_meta))
    d_meta_copy = copy.deepcopy(d_meta)

    if enrich_type not in d_meta_copy:
        d_meta_copy[enrich_type] = {}

    if d_criteria in d_meta_copy[enrich_type].values():

        for type_name in d_meta_copy[enrich_type]:

            if d_meta_copy[enrich_type][type_name] == d_criteria:
                type_name_query = type_name

    else:
        l_type_ids = []

        for type_name in d_meta_copy[enrich_type]:
            l_type_ids.append(int(type_name.replace("type", "")))

        if l_type_ids:
            type_id = str((max(l_type_ids) + 1))
            type_name_query = f"type{type_id}"
        else:
            type_name_query = "type1"

        d_meta_copy[enrich_type][type_name_query] = d_criteria
        dump_json(path_direc_type_names_preds, fn_meta, d_meta_copy, indent=2)

    return type_name_query


def write_to_txt(
        proj: str,
        d_lab_data: Dict,
        ambig_item_code: str,
        d_target: Dict,
        direc_outp: str,
        direc_preds: str,
        fn: str,
        d_outp: Dict
) -> None:
    """Write predictions to TXT file.
    :param proj: name of the project (with `enrich_type` extension).
    :param d_lab_data: original labelled data dictionary.
    :param ambig_item_code: ambiguous item code.
    :param d_target: dictionary containing target set.
    :param direc_outp: name of the directory in which all output of the method is saved.
    :param direc_preds: name of the directory in which the predictions should be saved.
    :param fn: filename of the file in which the predictions (in the form of output triplets) for the target set are
    saved.
    :param d_outp: dictionary containing the output to be written to the TXT file.
    :return: `None`
    """
    path_direc_preds = os.path.join(direc_outp, direc_preds, proj)

    if not os.path.isdir(path_direc_preds):
        os.makedirs(path_direc_preds)

    with codecs.open(os.path.join(path_direc_preds, fn), "w", "utf-8") as f:
        f.write(
            "sentence_ID\tpredicted_sense (ID | description)\thighest_maintained_value\t"
            "difference_with_second_highest_maintained_value\tsentence_text\n"
        )

        for sent in d_outp:
            sense = d_outp[sent]["output_triplet"][0]
            prediction = f"{sense} | {d_lab_data[ambig_item_code][sense]['description_ES']}"
            sim = str(round(d_outp[sent]['output_triplet'][1], 4))
            diff = str(round(d_outp[sent]["output_triplet"][2], 4))
            sent_text = d_target[sent]["text"] if "text" in d_target[sent] else " ".join(d_target[sent]["toks"])
            f.write("\t".join([sent, prediction, sim, diff, sent_text]) + "\n")

    f.close()
