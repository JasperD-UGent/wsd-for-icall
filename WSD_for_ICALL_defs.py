from utils.ambigItem import split_ambig_item_code
from utils.calculate_similarity import link_sims_to_rest, link_sims_to_target
from utils.process_datasets import define_dataset_name_and_update_meta, extract_sents_from_ud_treebank, \
    process_custom_sents_plain_text, process_custom_sents_preprocsd
from utils.process_JSONs import dump_json, load_json
from utils.process_output import define_type_name_query_preds_and_update_meta, write_to_txt
from utils.process_supportData import define_type_name_query_support_data_and_update_meta, load_support_data, \
    save_support_data, support_data_check_lab_data
from utils.wordEmbeddings import calculate_cosine_sims
import os
import pathlib
import pyconll
import shutil
import spacy
import sys
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertConfig, RobertaConfig, XLMRobertaConfig
from typing import Dict, Optional, Tuple, Union


def process_dataset(
        d_lab_data: Dict,
        source: str,
        *,
        custom_dataset_name: Optional[str] = None,
        ud_version: Optional[str] = None,
        ud_treebank: Optional[str] = None
) -> Tuple[Union[str, pathlib.Path], Union[str, pathlib.Path]]:
    """Process the raw dataset and generate the target and rest sets to which the WSD method should be applied.
    :param d_lab_data: dictionary containing the manually labelled example sentences which represent the senses included
    in the sense inventory.
    :param source: source of the dataset. Choose between: 'custom_plain_text', 'custom_preprocessed', and 'UD'.
    :param custom_dataset_name: name of the custom dataset. Only needs to be defined if the source is
    'custom_plain_text' or 'custom_preprocessed'.
    :param ud_version: UD version. Only needs to be defined if the source is 'UD'.
    :param ud_treebank: name of the UD treebank. Only needs to be defined if the source is 'UD'.
    :return: the path to the raw dataset files and the path to the processed dataset files (so that they do not need to
    be defined again in the `apply_wsd_method` function).
    """
    print("Running Step_2 ...")

    # assertions
    assert source in ["custom_plain_text", "custom_preprocessed", "UD"], f"Unsupported source: {source}."

    if source in ["custom_plain_text", "custom_preprocessed"]:
        assert custom_dataset_name is not None, \
            "Please provide the name of the custom dataset through the 'custom_dataset_name' argument."

    if source == "UD":
        assert ud_version is not None and ud_treebank is not None, \
            "Please provide the UD version and the name of the UD treebank through the 'ud_version' and " \
            "'ud_treebank' arguments."

    # parameter-independent directory names
    direc_inp = "input"
    direc_datasets_procsd = "datasets_procsd"
    direc_datasets_raw = "datasets_raw"

    direc_outp = "output"
    direc_meta = "meta"
    direc_dataset_names = "datasetNames"

    # parameter-independent filenames
    fn_dataset_target = "target.json"
    fn_dataset_rest = "rest.json"
    fn_meta_dataset_names = "meta_datasetNames.json"

    # process dataset
    if source == "UD":
        path_dataset_procsd = os.path.join(direc_inp, direc_datasets_procsd, source, ud_version, ud_treebank)
        path_dataset_raw = os.path.join(direc_inp, direc_datasets_raw, source, ud_version, ud_treebank)

        assert len([doc for doc in os.listdir(path_dataset_raw) if doc.endswith("test.conllu")]) == 1, \
            f"{path_dataset_raw} contains no or more than one test file (to be used for target set)."
        fn_target = [doc for doc in os.listdir(path_dataset_raw) if doc.endswith("test.conllu")][0]
        treebank_target = pyconll.load_from_file(os.path.join(path_dataset_raw, fn_target))

        assert len([doc for doc in os.listdir(path_dataset_raw) if doc.endswith("train.conllu")]) == 1, \
            f"{path_dataset_raw} contains no or more than one train file (to be used for rest set)."
        fn_rest = [doc for doc in os.listdir(path_dataset_raw) if doc.endswith("train.conllu")][0]
        treebank_rest = pyconll.load_from_file(os.path.join(path_dataset_raw, fn_rest))

        # loop over ambiguous items in labelled data dictionary
        for ambig_item_code in d_lab_data:
            print(f"\nRunning script for {ambig_item_code}.")
            ambig_item, pos, gender = split_ambig_item_code(ambig_item_code)
            ambig_item_code_fns = ambig_item_code.replace("|", "_")

            # check if processed sets already exist and extract sets if not
            if not os.path.exists(os.path.join(path_dataset_procsd, ambig_item_code_fns, fn_dataset_target)):
                print(f"Extracting target set from {os.path.join(path_dataset_raw, fn_target)}.")
                d_target_item = extract_sents_from_ud_treebank(treebank_target, ambig_item, pos, gender)
                dump_json(os.path.join(path_dataset_procsd, ambig_item_code_fns), fn_dataset_target, d_target_item)
            else:
                print(f"Target set already exists at "
                      f"{os.path.join(path_dataset_procsd, ambig_item_code_fns, fn_dataset_target)}.")

            if not os.path.exists(os.path.join(path_dataset_procsd, ambig_item_code_fns, fn_dataset_rest)):
                print(f"Extracting rest set from {os.path.join(path_dataset_raw, fn_rest)}.")
                d_rest_item = extract_sents_from_ud_treebank(treebank_rest, ambig_item, pos, gender)
                dump_json(os.path.join(path_dataset_procsd, ambig_item_code_fns), fn_dataset_rest, d_rest_item)
            else:
                print(f"Rest set already exists at "
                      f"{os.path.join(path_dataset_procsd, ambig_item_code_fns, fn_dataset_rest)}.")

    if source == "custom_plain_text":
        path_dataset_procsd = os.path.join(direc_inp, direc_datasets_procsd, source, custom_dataset_name)
        path_dataset_raw = os.path.join(direc_inp, direc_datasets_raw, source, custom_dataset_name)

        # preparatory steps

        #   - load NLP tools (spaCy pipeline)
        nlp_spacy = spacy.load("es_core_news_md")

        # loop over ambiguous items in labelled data dictionary
        for ambig_item_code in d_lab_data:
            print(f"\nRunning script for {ambig_item_code}.")
            ambig_item, pos, gender = split_ambig_item_code(ambig_item_code)
            ambig_item_code_fns = ambig_item_code.replace("|", "_")

            # parameter-dependent filenames (with extension) which depend on ambiguous item
            fn_target = ambig_item_code_fns + "_target.txt"
            fn_rest = ambig_item_code_fns + "_rest.txt"

            # check if processed sets already exist and make sets if not
            assert os.path.exists(os.path.join(path_dataset_raw, fn_target)), \
                f"For {ambig_item_code} no target data is provided in the {path_dataset_raw} directory."

            if not os.path.exists(os.path.join(path_dataset_procsd, ambig_item_code_fns, fn_dataset_target)):
                print(f"Processing {os.path.join(path_dataset_raw, fn_target)}.")
                d_target_item = process_custom_sents_plain_text(
                    os.path.join(path_dataset_raw, fn_target), ambig_item, pos, gender, nlp_spacy
                )
                dump_json(os.path.join(path_dataset_procsd, ambig_item_code_fns), fn_dataset_target, d_target_item)
            else:
                print(f"Target set already exists at "
                      f"{os.path.join(path_dataset_procsd, ambig_item_code_fns, fn_dataset_target)}.")

            assert os.path.exists(os.path.join(path_dataset_raw, fn_rest)), \
                f"For {ambig_item_code} no rest data is provided in the {path_dataset_raw} directory."

            if not os.path.exists(os.path.join(path_dataset_procsd, ambig_item_code_fns, fn_dataset_rest)):
                print(f"Processing {os.path.join(path_dataset_raw, fn_rest)}.")
                d_rest_item = process_custom_sents_plain_text(
                    os.path.join(path_dataset_raw, fn_rest), ambig_item, pos, gender, nlp_spacy
                )
                dump_json(os.path.join(path_dataset_procsd, ambig_item_code_fns), fn_dataset_rest, d_rest_item)
            else:
                print(f"Rest set already exists at "
                      f"{os.path.join(path_dataset_procsd, ambig_item_code_fns, fn_dataset_rest)}.")

    if source == "custom_preprocessed":
        path_dataset_procsd = os.path.join(direc_inp, direc_datasets_procsd, source, custom_dataset_name)
        path_dataset_raw = os.path.join(direc_inp, direc_datasets_raw, source, custom_dataset_name)

        # loop over ambiguous items in labelled data dictionary
        for ambig_item_code in d_lab_data:
            print(f"\nRunning script for {ambig_item_code}.")
            ambig_item_code_fns = ambig_item_code.replace("|", "_")

            # parameter-dependent filenames (with extension) which depend on ambiguous item
            fn_target = ambig_item_code_fns + "_target.txt"
            fn_rest = ambig_item_code_fns + "_rest.txt"

            # check if processed sets already exist and make sets if not
            assert os.path.exists(os.path.join(path_dataset_raw, fn_target)), \
                f"For {ambig_item_code} no target data is provided in the {path_dataset_raw} directory."

            if not os.path.exists(os.path.join(path_dataset_procsd, ambig_item_code_fns, fn_dataset_target)):
                print(f"Processing {os.path.join(path_dataset_raw, fn_target)}.")
                d_target_item = process_custom_sents_preprocsd(os.path.join(path_dataset_raw, fn_target))
                dump_json(os.path.join(path_dataset_procsd, ambig_item_code_fns), fn_dataset_target, d_target_item)
            else:
                print(f"Target set already exists at "
                      f"{os.path.join(path_dataset_procsd, ambig_item_code_fns, fn_dataset_target)}.")

            assert os.path.exists(os.path.join(path_dataset_raw, fn_rest)), \
                f"For {ambig_item_code} no rest data is provided in the {path_dataset_raw} directory."

            if not os.path.exists(os.path.join(path_dataset_procsd, ambig_item_code_fns, fn_dataset_rest)):
                print(f"Processing {os.path.join(path_dataset_raw, fn_rest)}.")
                d_rest_item = process_custom_sents_preprocsd(os.path.join(path_dataset_raw, fn_rest))
                dump_json(os.path.join(path_dataset_procsd, ambig_item_code_fns), fn_dataset_rest, d_rest_item)
            else:
                print(f"Rest set already exists at "
                      f"{os.path.join(path_dataset_procsd, ambig_item_code_fns, fn_dataset_rest)}.")

    # assign name to dataset and save it to meta file
    define_dataset_name_and_update_meta(
        path_dataset_raw, direc_outp, direc_meta, direc_dataset_names, fn_meta_dataset_names
    )

    print("\nFinished running Step_2.\n\n-----------\n")

    return path_dataset_raw, path_dataset_procsd


def apply_wsd_method(
        proj: str,
        d_lab_data_orig: Dict,
        dataset_source: str,
        path_dataset_raw: Union[str, pathlib.Path],
        path_dataset_procsd: Union[str, pathlib.Path],
        enrich_type: str,
        *,
        sim_calc_meth: str = "cs_max",
        sim_thresh_aat: float = 0.0,
        diff_thresh_aat: float = 0.1,
        n_iters_top_n: int = 5,
        n_sents_added_per_iter_top_n: int = 5,
        transformer_model_name: str = "PlanTL-GOB-ES/roberta-base-bne",
        save_temp: bool = True
) -> None:
    """Apply the WSD method.
    :param proj: name of the project.
    :param d_lab_data_orig: dictionary containing the manually labelled example sentences which represent the senses
    included in the sense inventory.
    :param dataset_source: source of the dataset. Choose between: 'custom_plain_text', 'custom_preprocessed', and 'UD'.
    :param path_dataset_raw: path to the directory in which the raw dataset files are located.
    :param path_dataset_procsd: path to the directory in which the processed dataset files are located.
    :param enrich_type: way in which the automatic addition of extra training data should be performed. Choose between:
    'AAT' (all-above-threshold; for each sense, all rest set sentences for which both the cosine similarity value and
    the difference with the second highest maintained value exceed the predefined thresholds in `sim_thresh_aat` and
    `diff_thresh_aat` will be added), and 'top-N' (for each sense, the N rest set sentences with the highest cosine
    similarity value and difference with the second highest maintained value will be added, with N being defined in
    `n_sents_added_per_iter_top_n`).
    :param sim_calc_meth: way in which the cosine similarity should be calculated. Choose between: 'cs_max'
    (maintained value = highest value amongst all individual values), 'cs_avg' (maintained value = average of all
    individual values), and 'cs_max_plus_avg' (maintained value = average of `cs_max` and `cs_avg`).
    :param sim_thresh_aat: minimal cosine similarity value a rest set sentence should have before it can be added as
    additional training data.
    :param diff_thresh_aat: minimal difference between the top two maintained cosine similarity values a rest set
    sentence should have before it can be added as additional training data for the top sense.
    :param n_iters_top_n: number of times the top-N procedure should be repeated. Each iteration takes into account the
    training data automatically added during the previous iteration(s).
    :param n_sents_added_per_iter_top_n: number of sentences added for each sense in the top-N setup.
    :param transformer_model_name: name of the Transformer model to be used for calculating the contextualised word
    embeddings. Available models can be consulted at: https://huggingface.co/models.
    :param save_temp: indicates whether or not the temp files should be saved for later reuse.
    :return: `None`
    """
    print("Running Step_3 ...")

    # assertions
    assert sim_calc_meth in ["cs_max", "cs_avg", "cs_max_plus_avg"]

    # parameter-dependent variables

    #   - number of iterations
    if enrich_type == "AAT":
        n_iters = 1

    if enrich_type == "top-N":
        n_iters = n_iters_top_n

    #   - criteria used in the query
    d_criteria_query = {
        "similarity_calculation_method": sim_calc_meth, "n_iterations": n_iters,
        "Transformers_model": transformer_model_name
    }

    if enrich_type == "AAT":
        d_criteria_query["similarity_threshold"] = sim_thresh_aat
        d_criteria_query["difference_threshold"] = diff_thresh_aat

    if enrich_type == "top-N":
        d_criteria_query["n_sents_added_per_iteration"] = n_sents_added_per_iter_top_n

    #   - type of Transformer model
    if type(AutoConfig.from_pretrained(transformer_model_name)) == type(BertConfig()):
        transformers_type = "BERT"
    elif type(AutoConfig.from_pretrained(transformer_model_name)) == type(RobertaConfig()):
        transformers_type = "RoBERTa"
    elif type(AutoConfig.from_pretrained(transformer_model_name)) == type(XLMRobertaConfig()):
        transformers_type = "XLM-RoBERTa"
    else:
        raise Exception("Transformer model should be a BERT, RoBERTa or XLM-RoBERTa model.")

    # parameter-independent directory names
    direc_inp = "input"
    direc_sis = "senseInventories"

    direc_outp = "output"
    direc_meta = "meta"
    direc_dataset_names = "datasetNames"
    direc_item_dep_info = "item-dependentInformation"
    direc_type_names_preds = "typeNames_predictions"
    direc_preds = "predictions"

    direc_temp = "temp"
    direc_enriched_lab_data = "enrichedLabelledData"
    direc_iter = "iter"
    direc_sims = "sims"
    direc_support_data = "supportData"
    direc_support_data_target_rest = "target_rest_supportData"

    # parameter-independent filenames
    fn_dataset_target = "target.json"
    fn_dataset_rest = "rest.json"
    fn_meta_dataset_names = "meta_datasetNames.json"
    fn_meta_type_names_preds = "meta_typeNames_predictions.json"
    fn_meta_support_data = "meta_target_rest_supportData.json"
    fn_support_data_l_cosine_sims = "l_cosine_sims.h5"
    fn_support_data_d_lab_data = "d_lab_data.json"
    fn_support_data_d_mapping_new_orig = "d_mapping_new_orig.json"
    fn_support_data_d_mapping_orig_new = "d_mapping_orig_new.json"

    # preparatory steps

    #   - parameter-dependent variables
    d_meta_dataset_names = load_json(os.path.join(direc_outp, direc_meta, direc_dataset_names, fn_meta_dataset_names))
    dataset_name = d_meta_dataset_names[path_dataset_raw]
    f_enriched_lab_data = f"{dataset_name}_enrichedLabData_"

    #   - load NLP tools (Transformer model)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if transformers_type in ["BERT", "XLM-RoBERTa"]:
        tokeniser_pretokenised = AutoTokenizer.from_pretrained(transformer_model_name)

    if transformers_type == "RoBERTa":
        tokeniser_pretokenised = AutoTokenizer.from_pretrained(transformer_model_name, add_prefix_space=True)

    model = AutoModel.from_pretrained(transformer_model_name, output_hidden_states=True).to(device)
    configuration = model.config
    tup_nlp_transformers = (tokeniser_pretokenised, model, configuration, device)

    # loop over ambiguous items in labelled data dictionary
    for ambig_item_code in d_lab_data_orig:
        print(f"\nRunning script for {ambig_item_code}.")
        ambig_item_code_fns = ambig_item_code.replace("|", "_")

        # parameter-dependent filenames (with extension) which depend on ambiguous item
        f_preds_target = "_".join([dataset_name, ambig_item_code_fns, "predictions"])
        fn_sims_rest = f"{'_'.join([dataset_name, ambig_item_code_fns, 'sims_rest'])}.json"

        # preparatory steps

        #   - load target and rest sets
        d_target = load_json(os.path.join(path_dataset_procsd, ambig_item_code_fns, fn_dataset_target))
        d_rest = load_json(os.path.join(path_dataset_procsd, ambig_item_code_fns, fn_dataset_rest))

        # generate (+ save if 'save_temp' is True) or load the support data for the similarity calculations
        path_to_si_file = os.path.join(pathlib.Path(os.path.join(direc_inp, direc_sis, proj + ".json")).absolute())
        d_support_data_query = {"si_file": path_to_si_file, "Transformer_model": transformer_model_name}
        type_name_query_support_data = define_type_name_query_support_data_and_update_meta(
            direc_outp, direc_meta, direc_item_dep_info, ambig_item_code_fns, fn_meta_support_data, d_support_data_query
        )
        lab_data_correspondence = support_data_check_lab_data(
            direc_temp, direc_support_data, ambig_item_code, direc_support_data_target_rest, dataset_name,
            type_name_query_support_data, d_lab_data_orig
        )
        l_fns_direc_support_data = [
            fn_support_data_l_cosine_sims, fn_support_data_d_lab_data, fn_support_data_d_mapping_new_orig,
            fn_support_data_d_mapping_orig_new
        ]
        path_direc_support_data_dataset_type = os.path.join(
            direc_temp, direc_support_data, ambig_item_code_fns, direc_support_data_target_rest, dataset_name,
            type_name_query_support_data
        )
        l_docs_support_data = os.listdir(path_direc_support_data_dataset_type)

        if sorted(l_docs_support_data) == sorted(l_fns_direc_support_data) and lab_data_correspondence:
            print(
                f"Load from files (support data target and rest)?\t{True} "
                f"(loading from {path_direc_support_data_dataset_type})"
            )

            # load list of similarities, labelled data dictionary and mapping dictionaries
            l_cosine_sims, d_lab_data, d_mapping_new_orig, d_mapping_orig_new = load_support_data(
                path_direc_support_data_dataset_type, fn_support_data_l_cosine_sims, fn_support_data_d_lab_data,
                fn_support_data_d_mapping_new_orig, fn_support_data_d_mapping_orig_new
            )
        else:
            print(f"Load from files (support data test and rest)?\t{False}")
            l_cosine_sims, d_lab_data, d_mapping_new_orig, d_mapping_orig_new = calculate_cosine_sims(
                ambig_item_code, d_lab_data_orig, dataset_source, tup_nlp_transformers, d_target=d_target, d_rest=d_rest
            )

            if save_temp:

                # save list of similarities, labelled data dictionary and mapping dictionaries
                save_support_data(
                    path_direc_support_data_dataset_type, fn_support_data_l_cosine_sims, fn_support_data_d_lab_data,
                    fn_support_data_d_mapping_new_orig, fn_support_data_d_mapping_orig_new, l_cosine_sims, d_lab_data,
                    d_mapping_new_orig, d_mapping_orig_new
                )

        # apply WSD method
        d_lab_data_enriched = link_sims_to_rest(
            proj, ambig_item_code, d_rest, enrich_type,
            sim_calc_meth, sim_thresh_aat, diff_thresh_aat, n_iters, n_sents_added_per_iter_top_n,
            direc_temp, direc_enriched_lab_data, direc_iter, direc_sims,
            f_enriched_lab_data, fn_sims_rest,
            l_cosine_sims, d_lab_data, d_mapping_new_orig, d_mapping_orig_new
        )
        d_output_triplets = link_sims_to_target(
            ambig_item_code, d_lab_data_enriched, d_target, sim_calc_meth, l_cosine_sims, d_mapping_new_orig,
            d_mapping_orig_new
        )

        # write output to TXT
        type_name_query_preds = define_type_name_query_preds_and_update_meta(
            enrich_type, direc_outp, direc_meta, direc_type_names_preds, fn_meta_type_names_preds, d_criteria_query
        )
        write_to_txt(
            "_".join([proj, enrich_type]), d_lab_data_orig, ambig_item_code, d_target, direc_outp, direc_preds,
            f"{f_preds_target}_{type_name_query_preds}.txt", d_output_triplets
        )

    print("\nFinished running Step_3.\n\n-----------\n")


def remove_temp(save_temp: bool) -> None:
    """Remove temp files if `save_temp` is set to `False`.
    :param save_temp: indicates whether or not the temp files should be saved for later reuse.
    :return: `None`
    """
    print("Running Step_4 ...")

    if not save_temp:
        shutil.rmtree(os.path.join("temp"))

    print("\nFinished running Step_4.\n\n-----------\n")
