from .process_JSONs import dump_json, load_json
from .process_labelledData import define_d_lab_data_to_be_upd, enrich_d_lab_data
import numpy as np
import operator
import os
import statistics
import sys
from typing import Dict, List, Tuple


def link_sims_to_rest_iteration(
        ambig_item_code: str,
        d_lab_data: Dict,
        d_rest: Dict,
        meth: str,
        l_cosine_sims: np.ndarray,
        d_mapping_new_orig: Dict,
        d_mapping_orig_new: Dict,
        l_d_entries: List
) -> Tuple[Dict, List]:
    """Link precalculated similarities to rest set sentences for an individual iteration.
    :param ambig_item_code: ambiguous item code.
    :param d_lab_data: (enriched) labelled data dictionary from the previous iteration.
    :param d_rest: dictionary containing the rest set.
    :param meth: way in which the cosine similarity should be calculated.
    :param l_cosine_sims: list of cosine similarities.
    :param d_mapping_new_orig: mapping dictionary in which the new sentence IDs are linked to their original data.
    :param d_mapping_orig_new: mapping dictionary in which the original sentence data are linked to their new IDs.
    :param l_d_entries: list of entries which should be consulted to know which sentences are included as labelled data
    for each sense. The original, manually labelled sentences are stored under the 'l_example_sents' entry, and the
    automatically added sentences are stored under the 'l_example_sents_autom_added' entry.
    :return: a dictionary containing all similarity data (cosine similarity values, rankings, etc.) for each sentence in
    the set and a list containing all data on the difference between the top two maintained values for each sentence in
    the set.
    """
    n_senses = len(d_lab_data[ambig_item_code])

    # loop over labelled data dictionary to extract all necessary information
    l_ids_example_sents = []
    d_ids_example_sents_per_sense = {}

    for sense in d_lab_data[ambig_item_code]:
        d_ids_example_sents_per_sense[sense] = []

        for entry in l_d_entries:

            if entry in d_lab_data[ambig_item_code][sense]:

                for d_sent in d_lab_data[ambig_item_code][sense][entry]:
                    sent_id = d_sent["sent_ID"]
                    l_ids_example_sents.append(sent_id)
                    d_ids_example_sents_per_sense[sense].append(sent_id)

    # extract similarities from 'l_cosine_sims'
    d_sims_linked = {}

    for target_sent_orig in d_rest:
        target_sent_new = d_mapping_orig_new[target_sent_orig]["id_new"]

        if target_sent_new not in l_ids_example_sents:
            d_sims_linked[target_sent_orig] = {"sim_scores": {}, "ranking_sense": {}}

    for target_sent_orig in d_rest:
        target_sent_new = d_mapping_orig_new[target_sent_orig]["id_new"]
        idx_target_sent_new = d_mapping_orig_new[target_sent_orig]["idx_l_sims"]

        if target_sent_new not in l_ids_example_sents:

            for sense in d_lab_data[ambig_item_code]:
                l_sims = []

                for example_sent_new in d_ids_example_sents_per_sense[sense]:
                    idx_example_sent_new = d_mapping_new_orig[example_sent_new]["idx_l_sims"]
                    partial_sim = l_cosine_sims[idx_target_sent_new][idx_example_sent_new]
                    l_sims.append(partial_sim)

                sim_max = max(l_sims)
                sim_avg_score = statistics.mean(l_sims)

                if meth == "cs_max":
                    d_sims_linked[target_sent_orig]["sim_scores"][sense] = sim_max

                if meth == "cs_avg":
                    d_sims_linked[target_sent_orig]["sim_scores"][sense] = sim_avg_score

                if meth == "cs_max_plus_avg":
                    d_sims_linked[target_sent_orig]["sim_scores"][sense] = (sim_max + sim_avg_score) / 2

    # determine for each sentence the corresponding output triplet information ('max_sense', 'max_sim' and 'diff')
    l_max = []
    l_diff = []

    for sent in d_sims_linked:
        dic_scores = d_sims_linked[sent]["sim_scores"].items()
        max_sense = max(dic_scores, key=operator.itemgetter(1))[0]
        max_sim = max(dic_scores, key=operator.itemgetter(1))[1]
        second_max_sense = sorted(dic_scores, key=operator.itemgetter(1), reverse=True)[1][0]
        second_max_sim = sorted(dic_scores, key=operator.itemgetter(1), reverse=True)[1][1]
        diff = max_sim - second_max_sim
        d_sims_linked[sent]["max"] = (max_sense, max_sim, diff)
        l_max.append((sent, max_sense, max_sim, diff))
        l_diff.append((sent, max_sense, max_sim, second_max_sense, second_max_sim, diff))

    # determine rankings (one according to the cosine similarity values and one according to the differences between the
    # two top maintained values)
    d_ranking = {}

    for loop in range(n_senses):
        id_sense = str(loop + 1)
        d_ranking[id_sense] = {}
        l_max_sense_score = []
        l_max_sense_diff = []

        for tup in l_max:
            sent_id = tup[0]
            max_sense = tup[1]
            max_sim = tup[2]
            diff = tup[3]

            if max_sense == id_sense:
                l_max_sense_score.append([sent_id, max_sense, max_sim])
                l_max_sense_diff.append([sent_id, max_sense, diff])

        l_max_sense_score_sorted = sorted(l_max_sense_score, key=lambda x: x[2], reverse=True)
        l_max_sense_diff_sorted = sorted(l_max_sense_diff, key=lambda x: x[2], reverse=True)

        d_score_ranked = {}

        for idx, l_sent in enumerate(l_max_sense_score_sorted):
            ranking = idx + 1
            l_sent.append(ranking)

        for l_sent in l_max_sense_score_sorted:
            d_score_ranked[l_sent[0]] = (l_sent[1], l_sent[2], l_sent[3])

        d_diff_ranked = {}

        for idx, l_sent in enumerate(l_max_sense_diff_sorted):
            ranking = idx + 1
            l_sent.append(ranking)

        for l_sent in l_max_sense_diff_sorted:
            d_diff_ranked[l_sent[0]] = (l_sent[1], l_sent[2], l_sent[3])

        d_ranking[id_sense]["score"] = d_score_ranked
        d_ranking[id_sense]["diff"] = d_diff_ranked

    # combine the two rankings into one single ranking by taking the average of the two ranking positions
    d_sum_ranking_prov = {}

    for sent in d_sims_linked:
        n_occurrences = 0
        sum_rankings_sent = 0
        l_sent_senses = []

        for sense in d_ranking:

            if sense not in d_sum_ranking_prov:
                d_sum_ranking_prov[sense] = []

            if sent in d_ranking[sense]["score"]:
                d_sims_linked[sent]["ranking_sense"]["score"] = (sense, d_ranking[sense]["score"][sent][2])
                n_occurrences += 1
                sum_rankings_sent += d_ranking[sense]["score"][sent][2]
                l_sent_senses.append(sense)

            if sent in d_ranking[sense]["diff"]:
                d_sims_linked[sent]["ranking_sense"]["diff"] = (sense, d_ranking[sense]["diff"][sent][2])
                n_occurrences += 1
                sum_rankings_sent += d_ranking[sense]["diff"][sent][2]
                l_sent_senses.append(sense)

        assert n_occurrences == 2, \
            f"Invalid number of sentence occurrences ({n_occurrences}) in d_ranking for sentence {sent}."
        assert len(set(l_sent_senses)) == 1
        sense = l_sent_senses[0]
        d_sum_ranking_prov[sense].append([sent, sense, sum_rankings_sent])

    d_sum_ranking = {}

    for sense in d_sum_ranking_prov:
        d_sum_ranking[sense] = {}
        l_sorted = sorted(d_sum_ranking_prov[sense], key=lambda x: x[2])

        for idx, l_sent in enumerate(l_sorted):
            ranking = idx + 1
            l_sent.append(ranking)

        for l_sent in l_sorted:
            assert l_sent[0] not in d_sum_ranking[sense], f"Sentence cannot be in dictionary yet: {l_sent}."
            d_sum_ranking[sense][l_sent[0]] = (l_sent[1], l_sent[2], l_sent[3])

    # assert that all sentences occur with only one sense and store final ranking as separate entry
    for sent in d_sims_linked:
        n_occurrences = 0

        for sense in d_sum_ranking:

            if sent in d_sum_ranking[sense]:
                d_sims_linked[sent]["ranking_sense"]["rankings_sum"] = (sense, d_sum_ranking[sense][sent][2])
                n_occurrences += 1

        assert n_occurrences == 1, \
            f"Invalid number of sentence occurrences ({n_occurrences}) in d_ranking for sentence {sent}."

    return d_sims_linked, l_diff


def link_sims_to_rest(
        root_proj: str,
        ambig_item_code: str,
        d_rest: Dict,
        enrich_type: str,
        sim_calc_meth: str,
        sim_thresh_aat: float,
        diff_thresh_aat: float,
        n_iters: int,
        n_sents_added_per_iter_top_n: int,
        direc_temp: str,
        direc_enriched_lab_data: str,
        direc_iter: str,
        direc_sims: str,
        f_enriched_lab_data: str,
        fn_sims_rest: str,
        l_cosine_sims: np.ndarray,
        d_lab_data: Dict,
        d_mapping_new_orig: Dict,
        d_mapping_orig_new: Dict
) -> Dict:
    """Link precalculated similarities to rest set sentences and enrich labelled data dictionary with sentences which
    were selected as additional training data by the method.
    :param root_proj: name of the project.
    :param ambig_item_code: ambiguous item code.
    :param d_rest: dictionary containing the rest set.
    :param enrich_type: way in which the automatic addition of extra training data should be performed.
    :param sim_calc_meth: way in which the cosine similarity should be calculated.
    :param sim_thresh_aat: minimal cosine similarity value a rest set sentence should have before it can be added as
    additional training data.
    :param diff_thresh_aat: minimal difference between the top two maintained cosine similarity values a rest set
    sentence should have before it can be added as additional training data for the top sense.
    :param n_iters: number of iterations which will be performed.
    :param n_sents_added_per_iter_top_n: number of sentences added for each sense in the top-N setup.
    :param direc_temp: name of the directory in which all temp data generated by the method are saved.
    :param direc_enriched_lab_data: name of the directory in which the enriched labelled data dictionaries are saved.
    :param direc_iter: fixed part of directory name of the directories in which the results per iteration are saved.
    :param direc_sims: name of the directory in which the linked similarities are saved.
    :param f_enriched_lab_data: fixed part of the filenames of the files in which the enriched labelled data
    dictionaries per iteration are saved.
    :param fn_sims_rest: filename of the file in which the similarity calculations for the rest set are saved.
    :param l_cosine_sims: list of cosine similarities.
    :param d_lab_data: processed version of the original labelled data dictionary.
    :param d_mapping_new_orig: mapping dictionary in which the new sentence IDs are linked to their original data.
    :param d_mapping_orig_new: mapping dictionary in which the original sentence data are linked to their new IDs.
    :return: the last enriched labelled data dictionary.
    """
    proj = "_".join([root_proj, enrich_type])
    ambig_item_code_fns = ambig_item_code.replace("|", "_")
    l_d_entries_example_sents = ["l_example_sents", "l_example_sents_autom_added"]

    for idx_iter in range(n_iters):
        iter_number = idx_iter + 1
        print(f"\n-----Running iteration {iter_number} for {proj}.-----\n")

        # define labelled data dictionary which serves as input for the similarity calculations
        if idx_iter == 0:
            d_lab_data_inp = d_lab_data
        else:
            fn_d_lab_data_inp = f"{f_enriched_lab_data}{str(iter_number - 1)}.json"
            d_lab_data_inp = load_json(
                os.path.join(direc_temp, direc_enriched_lab_data, proj, sim_calc_meth, fn_d_lab_data_inp)
            )

        # define labelled data dictionary which needs to be updated after the similarity calculations
        fn_d_lab_data_to_be_upd = f"{f_enriched_lab_data}{str(iter_number)}.json"
        d_lab_data_to_be_upd = define_d_lab_data_to_be_upd(
            proj, ambig_item_code, d_lab_data, direc_temp, direc_enriched_lab_data, sim_calc_meth,
            fn_d_lab_data_to_be_upd
        )

        # link similarities, enrich labelled data dictionary and dump resulting variables to JSON files
        d_sims_rest, l_diff = link_sims_to_rest_iteration(
            ambig_item_code, d_lab_data_inp, d_rest, sim_calc_meth, l_cosine_sims, d_mapping_new_orig,
            d_mapping_orig_new, l_d_entries_example_sents
        )
        dump_json(os.path.join(
            direc_temp, proj, sim_calc_meth, f"{direc_iter}{str(iter_number)}", direc_sims, ambig_item_code_fns
        ), fn_sims_rest, d_sims_rest)

        d_lab_data_outp = enrich_d_lab_data(
            ambig_item_code, enrich_type, d_rest, d_sims_rest, d_lab_data_inp, d_lab_data_to_be_upd, sim_thresh_aat,
            diff_thresh_aat, n_sents_added_per_iter_top_n, d_mapping_orig_new
        )
        dump_json(
            os.path.join(direc_temp, direc_enriched_lab_data, proj, sim_calc_meth), fn_d_lab_data_to_be_upd,
            d_lab_data_outp
        )

    return d_lab_data_outp if n_iters > 0 else d_lab_data


def link_sims_to_target(
        ambig_item_code: str,
        d_lab_data: Dict,
        d_target: Dict,
        meth: str,
        l_cosine_sims: np.ndarray,
        d_mapping_new_orig: Dict,
        d_mapping_orig_new: Dict
) -> Dict:
    """Link precalculated similarities to target set sentences in order to generate the predictions.
    :param ambig_item_code: ambiguous item code.
    :param d_lab_data: the enriched labelled data dictionary.
    :param d_target: dictionary containing the target set.
    :param meth: way in which the cosine similarity should be calculated.
    :param l_cosine_sims: list of cosine similarities.
    :param d_mapping_new_orig: mapping dictionary in which the new sentence IDs are linked to their original data.
    :param d_mapping_orig_new: mapping dictionary in which the original sentence data are linked to their new IDs.
    :return: dictionary containing cosine similarity values and output triplets for each sentence in the set.
    """
    l_d_entries_example_sents = ["l_example_sents", "l_example_sents_autom_added"]

    # variables to be defined while looping over the labelled data dictionary
    l_ids_example_sents = []
    d_ids_example_sents_per_sense = {}

    # loop over labelled data dictionary to extract all necessary information
    for sense in d_lab_data[ambig_item_code]:
        d_ids_example_sents_per_sense[sense] = []

        for entry in l_d_entries_example_sents:

            if entry in d_lab_data[ambig_item_code][sense]:

                for d_sent in d_lab_data[ambig_item_code][sense][entry]:
                    sent_id = d_sent["sent_ID"]
                    l_ids_example_sents.append(sent_id)
                    d_ids_example_sents_per_sense[sense].append(sent_id)

    # extract similarities from l_cosine_sims
    l_target_sents = [target_sent_orig for target_sent_orig in d_target]
    d_sims_linked = {
        target_sent_orig: {"sim_scores": {}, "output_triplet": Tuple[str, str, str]}
        for target_sent_orig in l_target_sents
    }

    for target_sent_orig in l_target_sents:
        idx_target_sent_new = d_mapping_orig_new[target_sent_orig]["idx_l_sims"]

        for sense in d_lab_data[ambig_item_code]:
            l_sims = []

            for example_sent_new in d_ids_example_sents_per_sense[sense]:
                idx_example_sent_new = d_mapping_new_orig[example_sent_new]["idx_l_sims"]
                partial_sim = l_cosine_sims[idx_target_sent_new][idx_example_sent_new]
                l_sims.append(partial_sim)

            sim_max = max(l_sims)
            sim_avg_score = statistics.mean(l_sims)

            if meth == "cs_max":
                d_sims_linked[target_sent_orig]["sim_scores"][sense] = sim_max

            if meth == "cs_avg":
                d_sims_linked[target_sent_orig]["sim_scores"][sense] = sim_avg_score

            if meth == "cs_max_plus_avg":
                d_sims_linked[target_sent_orig]["sim_scores"][sense] = (sim_max + sim_avg_score) / 2

    # determine for each sentence the output triplet (predicted sense, highest maintained cosine similarity value
    # corresponding to the predicted sense, difference with the second highest maintained value)
    for sent in d_sims_linked:
        max_sense = max(d_sims_linked[sent]["sim_scores"].items(), key=operator.itemgetter(1))[0]
        max_sim = max(d_sims_linked[sent]["sim_scores"].items(), key=operator.itemgetter(1))[1]
        second_max_sim = sorted(d_sims_linked[sent]["sim_scores"].values(), reverse=True)[1]
        diff = max_sim - second_max_sim
        d_sims_linked[sent]["output_triplet"] = (max_sense, max_sim, diff)

    return d_sims_linked
