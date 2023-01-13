from .process_JSONs import dump_json, load_json
import codecs
import copy
import os
import pathlib
import pyconll
import spacy
import sys
from typing import Dict, Union


def define_dataset_name_and_update_meta(
        path_dataset: Union[str, pathlib.Path],
        direc_outp: str,
        direc_meta: str,
        direc_dataset_names: str,
        fn_meta: str
) -> None:
    """Define the dataset name of the query and update the meta dictionary.
    :param path_dataset: path to the dataset.
    :param direc_outp: name of the directory in which all output of the method is saved.
    :param direc_meta: name of the directory in which meta information is saved.
    :param direc_dataset_names: name of the directory in which information on the dataset names is saved.
    :param fn_meta: filename of the file in which the meta information is saved.
    :return: `None`
    """
    path_direc_dataset_names = os.path.join(direc_outp, direc_meta, direc_dataset_names)

    if not os.path.exists(os.path.join(path_direc_dataset_names, fn_meta)):
        dump_json(path_direc_dataset_names, fn_meta, {}, indent=2)

    d_meta = load_json(os.path.join(path_direc_dataset_names, fn_meta))
    d_meta_copy = copy.deepcopy(d_meta)

    if path_dataset not in d_meta_copy:
        l_dataset_ids = []

        for dataset_name in list(d_meta_copy.values()):
            l_dataset_ids.append(int(dataset_name.replace("dataset", "")))

        if l_dataset_ids:
            dataset_id = str((max(l_dataset_ids) + 1))
            dataset_name_query = f"dataset{dataset_id}"
        else:
            dataset_name_query = "dataset1"

        d_meta_copy[path_dataset] = dataset_name_query
        dump_json(path_direc_dataset_names, fn_meta, d_meta_copy, indent=2)


def extract_sents_from_ud_treebank(
        treebank: pyconll.unit.conll, ambig_item: str, ambig_item_pos: str, ambig_item_gender: str
) -> Dict:
    """Extract all sentences which contain given ambiguous item from UD treebank.
    :param treebank: object containing the loaded treebank.
    :param ambig_item: lemma of the ambiguous item.
    :param ambig_item_pos: part-of-speech tag of the ambiguous item.
    :param ambig_item_gender: gender of the ambiguous item.
    :return: dictionary containing the processed target sentences.
    """
    ambig_item_gender = {"Masc"} if ambig_item_gender == "m" else {"Fem"}
    d_dataset = {}

    for sent in treebank:
        d_sent_prov = {
            "sent_ID_treebank": sent.id,
            "text": sent.text,
            "toks": [],
        }
        l_lems = []
        l_pos = []
        l_feats = []

        for tok in sent:

            if not tok.is_multiword() and "." not in str(tok.id):
                d_sent_prov["toks"].append(tok.form)
                l_lems.append(tok.lemma)
                l_pos.append(tok.upos)
                l_feats.append(tok.feats)

        for idx, (lem, pos, feats) in enumerate(zip(l_lems, l_pos, l_feats)):

            if lem == ambig_item and pos == ambig_item_pos:

                if "Gender" in feats and feats["Gender"] == ambig_item_gender:
                    d_sent = copy.deepcopy(d_sent_prov)
                    sent_id_dataset = "_".join([sent.id, str(idx)])
                    d_sent["idx_ambig_item"] = idx
                    d_dataset[sent_id_dataset] = d_sent

    return d_dataset


def process_custom_sents_plain_text(
        path_input_file: Union[str, pathlib.Path],
        ambig_item: str,
        ambig_item_pos: str,
        ambig_item_gender: str,
        nlp_spacy: spacy.Language
) -> Dict:
    """Extract sentences from custom input file (plain text).
    :param path_input_file:
    :param ambig_item: lemma of the ambiguous item.
    :param ambig_item_pos: part-of-speech tag of the ambiguous item.
    :param ambig_item_gender: gender of the ambiguous item.
    :param nlp_spacy: initialised spaCy `Language` object.
    :return: dictionary containing the processed sentences.
    """
    ambig_item_gender = {"Masc"} if ambig_item_gender == "m" else {"Fem"}

    with codecs.open(path_input_file, "r", "utf-8") as f:
        f_rl = f.readlines()
    f.close()

    d_dataset = {}

    for line in f_rl:
        procsd_line = line.strip().split("\t")

        if len(procsd_line) == 2:
            sent_id_orig = procsd_line[0]
            text = procsd_line[1]
            l_toks = []
            l_idxs_ambig_item = []

            for tok in nlp_spacy(text):
                idx = tok.i
                tok_text = tok.text
                l_toks.append(tok_text)
                pos = tok.pos_
                lem = tok.lemma_

                if lem == ambig_item and pos == ambig_item_pos:
                    feats_spacy_prov = str(tok.morph)
                    feats_spacy = {}

                    if feats_spacy_prov != "":

                        for entry in feats_spacy_prov.split("|"):
                            key = entry.split("=")[0]
                            value = set([item for item in entry.split("=")[1].split(",")])
                            feats_spacy[key] = value

                    if "Gender" in feats_spacy and feats_spacy["Gender"] == ambig_item_gender:
                        l_idxs_ambig_item.append(idx)

            for idx in l_idxs_ambig_item:
                sent_id_dataset = "_".join([sent_id_orig, str(idx)])
                d_dataset[sent_id_dataset] = {
                    "sent_ID_custom_dataset": sent_id_orig,
                    "toks": l_toks,
                    "idx_ambig_item": idx,
                    "text": text
                }

            if not l_idxs_ambig_item:
                print(f"For {sent_id_orig} no instance of {ambig_item} could be identified. - Sentence text: {text}")

    return d_dataset


def process_custom_sents_preprocsd(path_input_file: Union[str, pathlib.Path]) -> Dict:
    """Extract sentences from custom input file (preprocessed).
    :param path_input_file: path to the custom input file.
    :return: dictionary containing the processed sentences.
    """
    with codecs.open(path_input_file, "r", "utf-8") as f:
        f_rl = f.readlines()
    f.close()

    d_dataset = {}

    for line in f_rl:
        procsd_line = line.strip().split("\t")

        if len(procsd_line) == 4:
            sent_id_orig = procsd_line[0]
            l_toks = procsd_line[1].split()
            idx_ambig_item = int(procsd_line[2])
            text = procsd_line[3] if procsd_line[3] != "NA" else None
            sent_id_dataset = "_".join([sent_id_orig, str(idx_ambig_item)])
            d_dataset[sent_id_dataset] = {
                "sent_ID_custom_dataset": sent_id_orig,
                "toks": l_toks,
                "idx_ambig_item": idx_ambig_item
            }

            if text is not None:
                d_dataset[sent_id_dataset]["text"] = text

    return d_dataset
