# wsd-for-icall
This module allows you to perform word sense disambiguation (WSD) for Intelligent Computer-Assisted Language Learning (ICALL) purposes. As input, the method takes an [ICALL-tailored sense inventory](#step_1) in which the senses are represented by prototypical example sentences. These sentences are the only manually labelled data required to initialise the method. The method targets Spanish as a foreign language and can be applied to [Universal Dependencies (UD) treebank data](#ud-treebank-data), [custom preprocessed data](#custom-preprocessed-data) or [custom plain text data](#custom-plain-text-data) as data sources. This repository includes demo data for all three possible data sources, based on the [UD Spanish GSD treebank](https://universaldependencies.org/treebanks/es_gsd/index.html). Below you can find more information on the main steps performed by the method, as well as on the data sources and where to put the dataset files. 

**NOTE**: the WSD method is initialised through the ``WSD_for_ICALL.py`` script. Normally, this script should be the only one you need to modify in order to be able to apply the WSD method to your own data and to set the parameters to the values of your choice.

## Required Python modules
See ``requirements.txt``.

## Method
### Step_1
The goal of a WSD method is to disambiguate ambiguous words, so first we need to determine which ambiguous items the method should be applied to. Additionally, the senses to be distinguished for each ambiguous item also have to be defined, and for each sense a prototypical example sentence needs to be provided. As mentioned in the introduction, these sentences constitute the core of the WSD method. All this information should be gathered in a sense inventory file, which is loaded in the first step of the ``WSD_for_ICALL.py`` script.

The demo makes use of the sense inventory elaborated in the [NLP4CALL2022 paper](https://ecp.ep.liu.se/index.php/sltc/article/view/577) by Degraeuwe and Goethals (2022). However, as the demo only focuses on the ambiguous item _acción_, a new sense inventory file ``demoProject.json`` was created with ``"acción|NOUN|f"`` as the sole entry. To consult the full inventory and to find more details on its structure, you can visit the corresponding [GitHub repository](https://github.com/JasperD-UGent/sense-inventory-economics-50). For more details on its elaboration, please refer to the NLP4CALL2022 paper.

**NOTE**: when providing your own sense inventory, make sure the underlying dictionary follows the exact same structure as the demo file. For more information, you can again consult the [GitHub repository](https://github.com/JasperD-UGent/sense-inventory-economics-50) of the original inventory.

### Step_2
Secondly, the dataset (see below for more details on the data sources) is processed to arrive at a target set (containing the sentences for which the method should make predictions) and a rest set (containing the sentences which can be used by the method to automatically add more training data) in the form of Python dictionaries. To learn more about the dataset processing function and its parameters, have a look at the [source code](https://github.com/JasperD-UGent/wsd-for-icall/blob/2332822a8d0dcac470e11d513509ff0b2328ac47/WSD_for_ICALL_defs.py#L21).

### Step_3
Next, the WSD method is applied to the dataset, with the final predictions being saved in the ``output/predictions`` directory. The two principal building blocks of the method are the use of contextualised word embeddings (for which the ``transformers`` library from [Hugging Face](https://huggingface.co/) is used) and the calculation of cosine similarity values. For more details on these aspects, please refer to Section 3.3 of the [NLP4CALL2022 paper](https://ecp.ep.liu.se/index.php/sltc/article/view/577).

In brief, the goal of the method is to predict, for each sentence in the target set, the correct sense of the ambiguous instance in that sentence (choosing from the senses included in the sense inventory for the ambiguous item in question). However, as only a very limited amount of labelled training data is provided as input (i.e. the prototypical example sentences representing the senses in the sense inventory), the method will try to expand this training set by automatically adding sentences from the rest set. This addition can be done in two different setups: "all-above-threshold" (for each sense, all rest set sentences for which both the cosine similarity value and the difference with the second highest maintained value exceed predefined thresholds will be added) or "top-N" (for each sense, the N rest set sentences with the highest cosine similarity value and difference with the second highest maintained value will be added). For each of the setups, a set of parameters can be tweaked to adapt the method to your own needs. To learn more about the ``apply_wsd_method`` function and its parameters, have a look at the [source code](https://github.com/JasperD-UGent/wsd-for-icall/blob/2332822a8d0dcac470e11d513509ff0b2328ac47/WSD_for_ICALL_defs.py#L198).

### Step_4
Finally, the method offers the possibility to automatically delete all automatically generated temp files, such as the cosine similarity values between all sentences of the dataset and the enriched labelled data dictionaries (i.e. the files in which you can keep track of which rest set sentences are automatically added as extra training data). To learn more about the underlying function and its parameter, have a look at the [source code](https://github.com/JasperD-UGent/wsd-for-icall/blob/2332822a8d0dcac470e11d513509ff0b2328ac47/WSD_for_ICALL_defs.py#L414).

## Data sources
### UD treebank data
The first data source is UD treebank data in CoNNL-U format. Readily available treebanks can be downloaded from the [UD site](https://universaldependencies.org/#download). The treebank data need to be saved into the ``input/datasets_raw/UD`` directory. In this directory, first-level subdirectories should indicate the version of the UD data (e.g., "v2_11") and second-level subdirectories should indicate the name of the treebank (e.g., "UD_Spanish-AnCora"). The demo data are located in the ``input/dataset_raw/UD/demo/UD_Spanish-GSD`` directory.

**NOTE**: for the script to work, the treebank data need to include both a test set file ending on "test.conllu" (used for the target set) and a training set file ending on "train.conllu" (used for the rest set).

### Custom preprocessed data
The second data source is custom data which have already been preprocessed (i.e. tokenised and with the index of the ambiguous instance being known). The data need to be stored as TXT files (for each ambiguous item separately) in the ``input/dataset_raw/custom_preprocessed`` directory. In this directory, first-level subdirectories should indicate the name of the custom dataset. For each ambiguous item, two TXT files are required: one containing the target set and one containing the rest set. The names of the files should be the ambiguous item code followed by an underscore and "target.txt" or "rest.txt", with the ambiguous item code corresponding to the entry of the ambiguous item in the sense inventory (with underscores instead of pipes). Contentwise, the TXTs should adhere to the following format:
- One sentence per line
- Four items on one line (each of them separated by a tab)
  1. Sentence ID 
  2. Sentence tokens as a space-separated string 
  3. Index of the ambiguous item in the list of tokens 
  4. Sentence text in one string ("NA" if not available)

The demo data (for the ambiguous feminine noun _acción_, taken from the UD Spanish GSD treebank) are located in the ``input/dataset_raw/custom_preprocessed/demo`` directory.

### Custom plain text data
The third and final data source is custom data which have not been preprocessed yet. To tokenise and tag the sentences, [spaCy](https://spacy.io/) is used. As was the case for the preprocessed data, the plain text data also need to be stored in separate TXT files for each ambiguous item (in the ``input/dataset_raw/custom_plain_text`` directory). First-level subdirectories should again indicate the name of the custom dataset. For each ambiguous item, two TXT files are required: one containing the target set (see above for how to name the file) and one containing the rest set (see above for how to name the file). Contentwise, the TXTs should adhere to the following format:
- One sentence per line
- Two items on one line (each of them separated by a tab)
  1. Sentence ID
  2. Sentence text in one string

The demo data (for the ambiguous feminine noun _acción_, taken from the UD Spanish GSD treebank) are located in the ``input/dataset_raw/custom_plain_text/demo`` directory.

## References
- Degraeuwe, J., & Goethals, P. (2022). Interactive word sense disambiguation in foreign language learning. In D. Alfter, E. Volodina, T. François, P. Desmet, F. Cornillie, A. Jönsson, & E. Rennes (Eds.), _Proceedings of the 11th Workshop on Natural Language Processing for Computer-Assisted Language Learning (NLP4CALL 2022)_ (Vol. 190, pp. 46–54). https://doi.org/10.3384/ecp190005
