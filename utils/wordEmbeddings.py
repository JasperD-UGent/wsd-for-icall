from .counters import print_multiple_of
import copy
import numpy as np
import sys
import torch
from typing import Dict, List, Optional, Tuple


def get_hidden_states(inputs, token_ids_word, model, layers):
    """Push input IDs through model. Stack and sum `layers`. Select only those subword token outputs that belong to our
    word of interest and average them."""

    with torch.no_grad():
        outputs = model(**inputs)

    states = outputs.hidden_states  # get all hidden states
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()  # stack and sum all requested layers
    word_tokens_output = output[token_ids_word]  # only select the tokens that constitute the requested word

    return word_tokens_output.mean(dim=0).to("cpu")


def get_word_vector(l_toks, idx, tokeniser, model, layers, device):
    """Get a word vector by first tokenising the input sentence, getting all token indices that make up the word of
    interest, and then `get_hidden_states`."""
    inputs = tokeniser(l_toks, is_split_into_words=True, return_tensors="pt").to(device)
    token_ids_word = np.where(np.array(inputs.word_ids()) == idx)

    return get_hidden_states(inputs, token_ids_word, model, layers)


def get_l_toks_cut(l_toks_orig: List, idx_ambig_item: int) -> Tuple[List, int]:
    """Cut sentence which exceeds maximum length of tokeniser.
    :param l_toks_orig: list of original tokens.
    :param idx_ambig_item: index of the ambiguous item.
    :return: list of tokens of cut sentence and new index of the ambiguous item.
    """
    idx_start = (idx_ambig_item - 100) if (idx_ambig_item - 100) >= 0 else 0
    idx_end = idx_ambig_item + 101
    l_toks_cut = l_toks_orig[idx_start:idx_end]

    return l_toks_cut, (idx_ambig_item - idx_start)


def update_d_nlp(
        source: str,
        d_dataset: Dict,
        data_type: str,
        tup_nlp_transformers: Tuple,
        d_nlp: Dict,
        d_mapping_new_orig: Dict,
        d_mapping_orig_new: Dict,
        idx_l_sims: int
) -> int:
    """Update the dictionary containing the word embeddings and update the mapping dictionaries.
    :param source: source of the dataset.
    :param d_dataset: dictionary containing the sentences of the target set or rest set.
    :param data_type: data type. Choose between: 'si' (for manually labelled sentences included in sense inventory),
    'target' (for target set sentences), and 'rest' (for rest set sentences)
    :param tup_nlp_transformers: tuple containing the Transformer resources (tokeniser at position 0, model at position
    1, configuration at position 2 and device at position 3).
    :param d_nlp: dictionary containing the word embeddings.
    :param d_mapping_new_orig: mapping dictionary in which the new sentence IDs are linked to their original data.
    :param d_mapping_orig_new: mapping dictionary in which the original sentence data are linked to their new IDs.
    :param idx_l_sims: current index in the list of cosine similarities
    :return: updated current index in the list of cosine similarities.
    """
    print(f"Calculating word embeddings for {data_type} ...")

    tokeniser = tup_nlp_transformers[0]
    model = tup_nlp_transformers[1]
    configuration = tup_nlp_transformers[2]
    device = tup_nlp_transformers[3]
    layers = [-4, -3, -2, -1]  # the hidden layers which have to be considered by the Transformer model
    data_id = 0
    counter = 0

    for sent in d_dataset:
        sent_id = f"{data_type}_{data_id}"
        data_id += 1
        d_nlp[sent_id] = {}
        l_toks_prov = copy.deepcopy(d_dataset[sent]["toks"])
        idx_ambig_item_prov = d_dataset[sent]["idx_ambig_item"]

        if len(tokeniser.tokenize(l_toks_prov, is_split_into_words=True)) < configuration.max_position_embeddings:
            l_toks = l_toks_prov
            idx_ambig_item = idx_ambig_item_prov
        else:
            print(f"Example sentence cut because it exceeds 'max_position_embeddings': {sent}.")
            l_toks, idx_ambig_item = get_l_toks_cut(l_toks_prov, idx_ambig_item_prov)

        # get contextualised word embedding with Transformer
        vec_ambig_item = get_word_vector(l_toks, idx_ambig_item, tokeniser, model, layers, device)
        d_nlp[sent_id]["vec_ambig_item"] = vec_ambig_item

        # mapping
        d_mapping_new_orig[sent_id] = {"source": source, "id_orig": sent, "idx_l_sims": idx_l_sims}

        if sent not in d_mapping_orig_new:
            d_mapping_orig_new[sent] = {"source": source, "id_new": sent_id, "idx_l_sims": idx_l_sims}
        else:
            print(f"{sent} already in d_mapping_orig_new as {d_mapping_orig_new[sent]}.")

        idx_l_sims += 1
        counter += 1
        print_multiple_of(counter, 250)

    print(f"Finished calculating word embeddings for {data_type}.")

    return idx_l_sims


def calculate_cosine_sims(
        ambig_item_code: str,
        d_lab_data_orig: Dict,
        dataset_source: str,
        tup_nlp_transformers: Tuple,
        *,
        d_target: Optional[Dict] = None,
        d_rest: Optional[Dict] = None,

) -> Tuple[np.ndarray, Dict, Dict, Dict]:
    """Calculate the cosine similarities.
    :param ambig_item_code: ambiguous item code.
    :param d_lab_data_orig: original labelled data dictionary.
    :param dataset_source: source of the dataset.
    :param tup_nlp_transformers: tuple containing the Transformer resources (tokeniser at position 0, model at position
    1, configuration at position 2 and device at position 3).
    :param d_target: dictionary containing the target set.
    :param d_rest: dictionary containing the rest set.
    :return: the list of cosine similarities (in a Numpy array), the labelled data dictionary and the mapping
    dictionaries.
    """
    tokeniser = tup_nlp_transformers[0]
    model = tup_nlp_transformers[1]
    configuration = tup_nlp_transformers[2]
    device = tup_nlp_transformers[3]
    layers = [-4, -3, -2, -1]  # the hidden layers which have to be considered by the Transformer model

    d_nlp_sents = {}
    d_mapping_sent_id_new_to_sent_id_orig = {}
    d_mapping_sent_id_orig_to_sent_id_new = {}
    si_id = 0
    d_lab_data_orig_copy = copy.deepcopy(d_lab_data_orig)
    l_d_entries_example_sents = ["l_example_sents"]
    idx_l_sims = 0
    counter = 0
    print("Calculating word embeddings for labelled data included in sense inventory ...")

    for sense in d_lab_data_orig_copy[ambig_item_code]:
        counter_n_sents_sense = 1

        for entry in l_d_entries_example_sents:

            if entry in d_lab_data_orig_copy[ambig_item_code][sense]:

                for sent in d_lab_data_orig_copy[ambig_item_code][sense][entry]:
                    l_toks_prov = sent["toks"]
                    idx_ambig_item_prov = sent["idx_ambig_item"]
                    source = sent["source"]
                    sent_id_orig = "_".join([sense, str(counter_n_sents_sense), str(idx_ambig_item_prov)])
                    counter_n_sents_sense += 1

                    sent_id = f"si_{si_id}"
                    sent["sent_ID"] = sent_id
                    d_nlp_sents[sent_id] = {}
                    si_id += 1

                    if len(tokeniser.tokenize(l_toks_prov, is_split_into_words=True)) \
                            < configuration.max_position_embeddings:
                        l_toks = l_toks_prov
                        idx_ambig_item = idx_ambig_item_prov
                    else:
                        print(f"Original example sentence cut because it exceeds 'max_position_embeddings': {sent}.")
                        l_toks, idx_ambig_item = get_l_toks_cut(l_toks_prov, idx_ambig_item_prov)

                    # get contextualised word embedding with Transformer
                    vec_ambig_item = get_word_vector(l_toks, idx_ambig_item, tokeniser, model, layers, device)
                    d_nlp_sents[sent_id]["vec_ambig_item"] = vec_ambig_item

                    # mapping
                    d_mapping_sent_id_new_to_sent_id_orig[sent_id] = {
                        "source": source, "id_orig": sent_id_orig, "idx_l_sims": idx_l_sims
                    }

                    if sent_id_orig not in d_mapping_sent_id_orig_to_sent_id_new:
                        d_mapping_sent_id_orig_to_sent_id_new[sent_id_orig] = {
                            "source": source, "id_new": sent_id, "idx_l_sims": idx_l_sims
                        }
                    else:
                        raise Exception(f"Duplicate sentence in sense inventory: {sent_id_orig}.")

                    idx_l_sims += 1
                    counter += 1
                    print_multiple_of(counter, 250)

    print("Finished calculating word embeddings for labelled data included in sense inventory.")

    if d_target is not None:
        idx_l_sims = update_d_nlp(
            dataset_source, d_target, "target", tup_nlp_transformers, d_nlp_sents,
            d_mapping_sent_id_new_to_sent_id_orig, d_mapping_sent_id_orig_to_sent_id_new, idx_l_sims
        )

    if d_rest is not None:
        _ = update_d_nlp(
            dataset_source, d_rest, "rest", tup_nlp_transformers, d_nlp_sents, d_mapping_sent_id_new_to_sent_id_orig,
            d_mapping_sent_id_orig_to_sent_id_new, idx_l_sims
        )

    if device.split(":")[0] == "cuda":
        import numba
        from numba import jit

        @jit(nopython=True)
        def vecs_to_sims_gpu(l_vecs) -> np.ndarray:
            """Calculate cosine similarities between all vectors (on GPU).
            :param l_vecs: list of vectors.
            :return: list of cosine similarities (in a Numpy array).
            """
            print("Calculating cosine similarities between all vectors ...")
            l_sims_gpu = np.zeros((len(l_vecs), len(l_vecs)))

            for idx_1, vec1 in enumerate(l_vecs):

                for idx_2, vec2 in enumerate(l_vecs):
                    l_sims_gpu[idx_1][idx_2] = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

            print("Finished calculating cosine similarities between all vectors.")

            return l_sims_gpu

        l_vecs_orig = numba.typed.List()

        for sent in d_nlp_sents:
            l_vecs_orig.append(d_nlp_sents[sent]["vec_ambig_item"].numpy())

        l_sims = vecs_to_sims_gpu(l_vecs_orig)

    if device == "cpu":

        def vecs_to_sims_cpu(l_vecs: List) -> np.ndarray:
            """Calculate cosine similarities between all vectors (on CPU).
            :param l_vecs: list of vectors.
            :return: list of cosine similarities (in a Numpy array).
            """
            print("Calculating cosine similarities between all vectors ...")
            l_sims_cpu = np.zeros((len(l_vecs), len(l_vecs)))

            for idx_1, vec1 in enumerate(l_vecs):

                for idx_2, vec2 in enumerate(l_vecs):
                    l_sims_cpu[idx_1][idx_2] = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

            print("Finished calculating cosine similarities between all vectors.")

            return l_sims_cpu

        l_vecs_orig = []

        for sent in d_nlp_sents:
            l_vecs_orig.append(d_nlp_sents[sent]["vec_ambig_item"].numpy())

        l_sims = vecs_to_sims_cpu(l_vecs_orig)

    return l_sims, d_lab_data_orig_copy, d_mapping_sent_id_new_to_sent_id_orig, d_mapping_sent_id_orig_to_sent_id_new
