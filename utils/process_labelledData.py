from .process_JSONs import load_json
import copy
import os
import sys
from typing import Dict


def define_d_lab_data_to_be_upd(
        proj: str,
        ambig_item_code: str,
        d_lab_data: Dict,
        direc_temp: str,
        direc_enriched_lab_data: str,
        meth: str,
        fn: str
) -> Dict:
    """Define labelled data dictionary dictionary to be updated.
    :param proj: name of the project (with `enrich_type` extension).
    :param ambig_item_code: ambiguous item code.
    :param d_lab_data: labelled data dictionary from the previous iteration (used if the dictionary to be updated does
        not exist yet).
    :param direc_temp: name of the directory in which all temp data generated by the method are saved.
    :param direc_enriched_lab_data: name of the directory in which the enriched labelled data dictionaries are saved.
    :param meth: way in which the cosine similarity should be calculated.
    :param fn: filename of the labelled data dictionary to be updated.
    :return: the labelled data dictionary dictionary in question.
    """

    if os.path.exists(os.path.join(direc_temp, direc_enriched_lab_data, proj, meth, fn)):
        d_lab_data_to_be_upd = load_json(os.path.join(direc_temp, direc_enriched_lab_data, proj, meth, fn))
    else:
        d_lab_data_to_be_upd = {ambig_item_code: d_lab_data[ambig_item_code]}

    return d_lab_data_to_be_upd


def enrich_d_lab_data(
        ambig_item_code: str,
        enrich_type: str,
        d_rest: Dict,
        d_sims: Dict,
        d_lab_data_inp: Dict,
        d_lab_data_to_be_upd: Dict,
        sim_thresh_aat: float,
        diff_thresh_aat: float,
        n_sents_added_top_n: int,
        d_mapping_orig_new: Dict
) -> Dict:
    """Enrich labelled data dictionary.
    :param ambig_item_code: ambiguous item code.
    :param enrich_type: way in which the automatic addition of extra training data should be performed.
    :param d_rest: dictionary containing the rest set.
    :param d_sims: dictionary containing all similarity data (cosine similarity values, rankings, etc.) for each rest
        set sentence.
    :param d_lab_data_inp: labelled data dictionary used as input.
    :param d_lab_data_to_be_upd: labelled data dictionary to be updated.
    :param sim_thresh_aat: minimal cosine similarity value a rest set sentence should have before it can be added as
        additional training data.
    :param diff_thresh_aat: minimal difference between the top two maintained cosine similarity values a rest set
        sentence should have before it can be added as additional training data for the top sense.
    :param n_sents_added_top_n: number of sentences added for each sense in the top-N setup.
    :param d_mapping_orig_new: mapping dictionary in which the original sentence data are linked to their new IDs.
    :return: the enriched dictionary.
    """
    n_senses = len(d_lab_data_inp[ambig_item_code])
    d_lab_data_inp_copy = copy.deepcopy(d_lab_data_inp)
    d_lab_data_enriched = d_lab_data_inp_copy[ambig_item_code]

    for loop in range(n_senses):
        id_sense = str(loop + 1)
        example_sent_found = False
        l_sel_example_sents = []

        if enrich_type == "AAT":

            for sent in d_sims:

                if d_sims[sent]["max"][0] == id_sense \
                        and d_sims[sent]["max"][1] >= sim_thresh_aat \
                        and d_sims[sent]["max"][2] >= diff_thresh_aat:
                    example_sent_found = True
                    l_sel_example_sents.append(sent)

        if enrich_type == "top-N":

            for sent in d_sims:

                if d_sims[sent]["ranking_sense"]["rankings_sum"][0] == id_sense \
                        and d_sims[sent]["ranking_sense"]["rankings_sum"][1] <= n_sents_added_top_n:
                    example_sent_found = True
                    l_sel_example_sents.append(sent)

        if example_sent_found:
            print(f"{ambig_item_code} - Sense {id_sense} - Number example sentences added: {len(l_sel_example_sents)}")
            l_sel_example_sents_sorted = sorted(l_sel_example_sents, key=lambda x: x[1])

            for sel_example_sent in l_sel_example_sents_sorted:
                source = d_mapping_orig_new[sel_example_sent]["source"]
                l_toks_sel_example_sent = d_rest[sel_example_sent]["toks"]
                """text_sel_example_sent = " ".join(l_toks_sel_example_sent)"""
                idx_ambig_item = d_rest[sel_example_sent]["idx_ambig_item"]
                sent_id_new = d_mapping_orig_new[sel_example_sent]["id_new"]
                d_sel_example_sent = {
                    "toks": l_toks_sel_example_sent,
                    "idx_ambig_item": idx_ambig_item,
                    "source": source,
                    "sent_ID": sent_id_new
                }

                if "l_example_sents_autom_added" not in d_lab_data_enriched[id_sense]:
                    d_lab_data_enriched[id_sense]["l_example_sents_autom_added"] = [d_sel_example_sent]
                else:
                    d_lab_data_enriched[id_sense]["l_example_sents_autom_added"].append(d_sel_example_sent)

        else:
            print(f"{ambig_item_code} - Sense {id_sense} - No new example sentences found")

    d_lab_data_to_be_upd_copy = copy.deepcopy(d_lab_data_to_be_upd)
    d_lab_data_to_be_upd_copy[ambig_item_code] = {} if ambig_item_code not in d_lab_data_to_be_upd_copy \
        else d_lab_data_to_be_upd_copy[ambig_item_code]
    d_lab_data_to_be_upd_copy[ambig_item_code] = d_lab_data_enriched

    return d_lab_data_to_be_upd_copy
