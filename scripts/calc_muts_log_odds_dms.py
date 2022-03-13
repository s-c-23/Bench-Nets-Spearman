from cmath import exp
from pathlib import Path

import pandas as pd 
import numpy as np
from scipy.stats import spearmanr,pearsonr
from pd_utils import *


# exp_data = Path("/stor/home/dd32387/codes/ml/benchmarking/data/exp_data/dms_data/CALM1_HUMAN_Roth2017_Mutation_Data.csv ")

# nn_pred_dir = Path(
#     "/stor/home/dd32387/codes/ml/benchmarking/data/nn_pred/fireprotdb/negatron_nets"
# )

# df = calc_nn_log_odd_ratio(exp_data, nn_pred_dir, mut_log_odds)

# df.to_csv("/stor/home/dd32387/codes/ml/benchmarking/data/exp_data/fireprotdb/fireprotdb_esm1v-5m_nn-pred.csv", index=False)

AA_COLUMNS = [
       'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 
       'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
       'LEU', 'LYS', 'MET', 'PHE', 'PRO', 
       'SER', 'THR', 'TRP', 'TYR', 'VAL'
    ]

def clean_dms_csv(csv: Path) -> pd.DataFrame: 

    required_cols = [
        'pdb', 'chain_id', 'position', 'wtAA',
       'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 
       'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
       'LEU', 'LYS', 'MET', 'PHE', 'PRO', 
       'SER', 'THR', 'TRP', 'TYR', 'VAL'
    ]

    exp_df = pd.read_csv(csv)

    exp_df = exp_df[required_cols]

    exp_df.to_csv(csv, index=False)



    # print(exp_row)

    # exp_row[AA_COLUMNS].apply(lambda x: print(x))

    # exit(0) 

    # wild_type = gemmi.expand_protein_one_letter(exp_row.wtAA)
    # mutation = gemmi.expand_protein_one_letter(exp_row.mutation)
    # mut = Mutation(exp_row.chain, int(exp_row.position), wild_type, mutation)

    # if fp_pdb_map.get(mut.seq_id, None) is None:
    #     print(f"Unable to map {mut} to nn-pred seq-id. Skipping...")
    #     return exp_row

    # corr_mut = Mutation(mut.chain_id, fp_pdb_map[mut.seq_id], mut.comp_id, mut.mut_id)

    # log_odds = calculate_log_odds(nn_df, corr_mut, log_odd_func)

    # exp_row.update(log_odds)

    # return exp_row

    
def calc_nn_log_odd_mutation(exp_aa_data, wt_aa) -> float:

    print()


def calc_nn_log_odd_rows(
    exp_row, nn_df, seq_pdb_map, log_odd_func: Callable
) -> List:

    nn_log_odd_row = exp_row.copy()
    wildtype = exp_row.wtAA.upper()

    for aa in AA_COLUMNS:
        mutation = aa.upper()

        mut = Mutation(
            exp_row.chain_id, int(exp_row.position), wildtype, mutation
        )

        if seq_pdb_map.get(mut.seq_id, None) is None:
                print(f"Unable to map {mut} to nn-pred seq-id. Skipping...")
                return exp_row
                continue

        corr_mut = Mutation(mut.chain_id, fp_pdb_map[mut.seq_id], mut.comp_id, mut.mut_id)
        # print(corr_mut)
        
        row = find_mutant_rows(nn_df, corr_mut)

        # this returns either a series or a df. But we need a float. 
        try:
            log_odd = log_odd_func(row, wt=mut.comp_id, mut=mut.mut_id)
        except:
            print('Could not process data. Skipping...')
            log_odd = None
        # print(log_odd)

        nn_log_odd_row[aa] = log_odd


    return nn_log_odd_row

if __name__ == "__main__":

    uniprot_id = "P0DP23"

    exp_data = Path("data/exp_data/dms_data/CALM1_HUMAN_Roth2017_Mutation_Data.csv")

    nn_pred = Path(
        'data/ecnet/2jzi_EnsResNet.csv'
    )

    nn_df = pd.read_csv(nn_pred)

    exp_df = extract_relavant_cols(exp_data, pdb_col="pdb")

    corr_df = exp_df.copy()

    corr_df[AA_COLUMNS] = np.nan

    model_name = collect_model_names(nn_df, "model")[0]

    corr_df.assign(model=model_name)

    pdb, chain = exp_df["pdb"].tolist()[0], exp_df["chain_id"].tolist()[0]

    nn_rows = nn_df

    pdb_exp = exp_df

    # try:
    exp_seq = fetch_seq_from_uniprot(uniprot_id)
    fp_pdb_map = map_fp_seqnums_to_pdb_seqnums(exp_seq, nn_rows, chain)


    nn_log_odd_df = pdb_exp.apply(
        lambda row: calc_nn_log_odd_rows(
            row, nn_rows, fp_pdb_map, mut_log_odds
        ),
        axis=1,
    )


    corr= spearmanr(exp_df[AA_COLUMNS].squeeze(), nn_log_odd_df[AA_COLUMNS].squeeze())
    corr_np_array = np.array(corr[0])
    corr_df = pd.DataFrame(corr_np_array)
    corr_two = corr_df.squeeze()
    print(corr_two)


    # corr_df = exp_df.corrwith(nn_log_odd_df, axis = 1, method="spearman")


    # print(nn_log_odd_df.head(10))

    # print(nn_log_odd_df.shape)
    # print(pdb_exp.shape)

    # print(pdb_exp.head())
    # print(nn_log_odd_df.head())

    # except Exception as e:
    #     print(
    #         f"Failed to extract structure net predictions for {pdb}_{chain}\nError: {e}"
    #     )

    # else:
    #     pass
    #     # exp_nn_pred.append(pdb_exp_nn_pred)

    # finally:
    #     print(f"Finished processing {pdb}_{chain}")

    # df = calc_nn_log_odd_ratio(exp_data, nn_pred_dir, mut_log_odds)

    # # df.to_csv("/stor/home/dd32387/codes/ml/benchmarking/data/exp_data/fireprotdb/fireprotdb_esm1v-5m_nn-pred.csv", index=False)
