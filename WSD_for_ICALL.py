from WSD_for_ICALL_defs import process_dataset, apply_wsd_method, remove_temp
from utils.process_JSONs import load_json
import os
import sys


def main():
    # Step_1: load sense inventory (containing example sentences as labelled data)

    #   - parameters
    proj = "demoProject"  # name of the project; sense inventory file should be located at input/senseInventories/[name_of_the_project].json

    #   - call function
    d_lab_data = load_json(os.path.join("input", "senseInventories", f"{proj}.json"))

    # Step_2: process dataset

    #   - parameters
    dataset_source = "UD"
    custom_dataset_name = "demo"
    ud_version = "demo"
    ud_treebank = "UD_Spanish-GSD"

    #   - call function
    path_dataset_raw, path_dataset_procsd = process_dataset(
        d_lab_data, dataset_source, custom_dataset_name=custom_dataset_name, ud_version=ud_version,
        ud_treebank=ud_treebank
    )

    # Step_3: apply WSD method

    #   - parameters
    enrich_type = "AAT"
    sim_calc_meth = "cs_max"
    sim_thresh_aat = 0.0
    diff_thresh_aat = 0.1
    n_iters_top_n = 2
    n_sents_added_per_iter_top_n = 5
    save_temp = True

    #   - call function
    apply_wsd_method(
        proj,
        d_lab_data,
        dataset_source,
        path_dataset_raw,
        path_dataset_procsd,
        enrich_type,
        sim_calc_meth=sim_calc_meth,
        sim_thresh_aat=sim_thresh_aat,
        diff_thresh_aat=diff_thresh_aat,
        n_iters_top_n=n_iters_top_n,
        n_sents_added_per_iter_top_n=n_sents_added_per_iter_top_n,
        save_temp=save_temp
    )

    # Step_4: remove temporary files if parameter 'save_temp' is True

    #   - call function
    remove_temp(save_temp)


if __name__ == "__main__":
    main()
