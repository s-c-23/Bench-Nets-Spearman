from pathlib import Path
from typing import List, Iterable, Tuple, Dict, Callable
# from curses.ascii import isalpha
from math import isnan
from dataclasses import dataclass
from io import StringIO

from Bio import SeqIO
import numpy as np
import pandas as pd
import requests as r
import gemmi


pd.set_option("display.width", None)
pd.set_option("display.max_rows", None)


EXP_COLUMNS = [
    "protein_name",
    "uniprot_id",
    "pdb_id",
    "chain",
    "position",
    "wild_type",
    "mutation",
    "tm",
    "dTm",
    "ddG",
    "asa",
    "is_in_catalytic_pocket",
    "sequence",
]

ESM_1V_COLUMNS = [
    "esm1v_t33_650M_UR90S_1",
    "esm1v_t33_650M_UR90S_2",
    "esm1v_t33_650M_UR90S_3",
    "esm1v_t33_650M_UR90S_4",
    "esm1v_t33_650M_UR90S_5",
]


@dataclass(frozen=True)
class Mutation:
    chain_id: str
    seq_id: int
    comp_id: str
    mut_id: str


def extract_relavant_cols(
    data: Path, columns: List[str] = None, pdb_col: str = "pdb_id"
) -> pd.DataFrame:
    def format_pdb_id(pdb_row):
        if isinstance(pdb_row, str):
            return pdb_row.lower()
        return pdb_row

    df = pd.read_csv(data, index_col=None, usecols=columns)

    if pdb_col in df.columns.to_list():
        df[pdb_col] = df[pdb_col].apply(lambda x: format_pdb_id(x))

    return df


# Deprecated - Tianlong got me new data with these columns already present.
def prepare_mutation_cols(df: pd.DataFrame, mutant_col: str = "mutant") -> pd.DataFrame:
    "Expecting mutant_col values to follow A167S mutation notation."

    column_order = ["position", "wtAA", "mutAA"] + df.columns.to_list()

    df["position"] = df[mutant_col].apply(lambda x: int(x[1:-1]))
    df["wtAA"] = df[mutant_col].apply(lambda x: gemmi.expand_protein_one_letter(x[0]))
    df["mutAA"] = df[mutant_col].apply(lambda x: gemmi.expand_protein_one_letter(x[-1]))

    df = df[column_order]

    print(df.head())


def collect_csv(dir: Path, filter: str = "*.csv") -> Iterable[Path]:

    assert isinstance(dir, Path)
    assert dir.is_dir()

    return [file for file in dir.rglob(filter)]


def generate_csv_df_pairs(CSVs: Iterable[Path]) -> Iterable[Tuple[Path, pd.DataFrame]]:

    return [(csv, pd.read_csv(csv)) for csv in CSVs]


def generate_multimodel_df(
    CSVs: Iterable[Path], remove_pdb_prefix: bool = True
) -> Iterable[pd.DataFrame]:

    DFs = [pd.read_csv(csv) for csv in CSVs]

    if remove_pdb_prefix:
        return [df.assign(model=csv.stem[5:]) for csv, df in zip(CSVs, DFs)]

    return [df.assign(model=csv.stem) for csv, df in zip(CSVs, DFs)]


def concat_df(DFs: Iterable[pd.DataFrame]) -> pd.DataFrame:

    return pd.concat(DFs, axis=0, ignore_index=True)


def concat_exp_data(
    csv_dir: Path, csv_glob: str = "*.csv", exp_columns=EXP_COLUMNS
) -> pd.DataFrame:

    csvs = collect_csv(csv_dir, csv_glob)

    dfs = [extract_relavant_cols(csv, exp_columns) for csv in csvs]

    return pd.concat(dfs, axis=0, ignore_index=True)


def sort_nn_preds(
    df: pd.DataFrame,
    sort_cols: List[str] = ["pos", "avg_log_ratio"],
    sort_orders: List[bool] = [True, False],
) -> pd.DataFrame:

    return df.sort_values(sort_cols, ascending=sort_orders)


def count_models(df: pd.DataFrame, model_col: str) -> int:

    return len(df[model_col].unique())


def collect_model_names(df: pd.DataFrame, model_col: str) -> int:

    return df[model_col].unique()


def write_excel_sheet(
    xc_writer: pd.ExcelWriter, name: str, df: pd.DataFrame, **kwargs
) -> None:

    df.to_excel(xc_writer, sheet_name=name, **kwargs)


def find_mutant_rows(nn_pred: pd.DataFrame, mut: Mutation) -> pd.DataFrame:

    assert isinstance(nn_pred, pd.DataFrame)
    assert isinstance(mut, Mutation)

    return nn_pred[
        (nn_pred["chain_id"] == mut.chain_id)
        & (nn_pred["pos"] == mut.seq_id)
        & (nn_pred["wtAA"] == mut.comp_id)
    ]


def mut_log_odds(row, wt: str, mut: str) -> float:
    assert len(wt) == 3
    assert len(mut) == 3

    numerator = row[f"pr{mut.upper()}"] + 1e-9
    denominator = row[f"pr{wt.upper()}"] + 1e-9

    # print(numerator)
    # print(denominator)

    odds = numerator / denominator
    # if odds.empty:
    #    return 0
    return float(np.log10(odds))


def wt_log_odds(row, wt: str, mut: str = None) -> float:
    assert len(wt) == 3

    if mut is not None:
        assert len(mut) == 3

    numerator = 1 - row[f"pr{wt.upper()}"] + 1e-9
    denominator = row[f"pr{wt.upper()}"] + 1e-9

    return np.log10(numerator / denominator)


def calculate_log_odds(
    nn_pred: pd.DataFrame, mut: Mutation, log_odds_func: Callable
) -> pd.Series:

    rows = find_mutant_rows(nn_pred, mut)

    rows["log_ratios"] = rows.apply(
        lambda x: log_odds_func(x, wt=mut.comp_id, mut=mut.mut_id), axis=1
    )

    log_odds_df = rows[["model", "log_ratios"]].set_index("model")

    return log_odds_df.T.squeeze()


def fetch_seq_from_uniprot(uniprot_id: str) -> str:

    url = f"http://www.uniprot.org/uniprot/{uniprot_id}.fasta"

    response = r.get(url)

    if response.ok:
        Seq = StringIO("".join(response.text))

        return str(SeqIO.read(Seq, "fasta").seq)

    return ""


def fetch_seq_from_nn_pred(
    nn_pred: pd.DataFrame, chain_id: str
) -> Tuple[List[int], str]:

    chain_rows = (
        nn_pred.sort_values(by="pos", ascending=True).drop_duplicates(["pos", "wtAA"])
    )

    # print(chain_rows)

    seq_nums, seq = list(zip(*chain_rows[["pos", "wtAA"]].values))

    return seq_nums, gemmi.one_letter_code(seq)

def fetch_seq_from_exp_df(exp_df: pd.DataFrame, seq_col: str = "sequence") -> str:
    """Returns sequence from the first row"""
    assert not exp_df.empty

    return exp_df[seq_col].tolist()[0]


def fetch_chain_id_from_exp_df(exp_df: pd.DataFrame, chain_col: str = "chain") -> str:
    """Returns the chain id of the first row"""
    return exp_df[chain_col].tolist()[0]


def map_fp_seqnums_to_pdb_seqnums(
    fp_seq: str, nn_pred: pd.DataFrame, chain_id: str
) -> Dict[int, int]:
    """
    Returns a dict where:
        the key is the residue position from the fireprot sequence
        the value is the residue position from the nn prediction csv
    """

    pdb_seq_ids, pdb_seq = fetch_seq_from_nn_pred(nn_pred, chain_id)
    
    alignment = gemmi.align_string_sequences(list(fp_seq), list(pdb_seq), [])

    fp_gaps = alignment.add_gaps(fp_seq, 1)
    pdb_gaps = alignment.add_gaps(pdb_seq, 2)

    pdb_seq_ids_gen = (num for num in pdb_seq_ids)
    fp_pdb_mapping = dict()
    fp_idx = 0
    for idx in range(len(fp_gaps)):
        fp_char = fp_gaps[idx]
        pdb_char = pdb_gaps[idx]

        pdb_seq_id = None
        if pdb_char.isalpha():
            pdb_seq_id = next(pdb_seq_ids_gen)

        if fp_char.isalpha():
            fp_idx += 1

        if (
            pdb_char.isalpha()
            and fp_char.isalpha()
            and pdb_char.upper() == fp_char.upper()
            and not isnan(pdb_seq_id)
        ):
            fp_pdb_mapping[fp_idx] = int(pdb_seq_id)

    return fp_pdb_mapping


def update_nn_log_odd_cols(
    exp_row, nn_df, fp_pdb_map, log_odd_func: Callable
) -> pd.DataFrame:

    wild_type = gemmi.expand_protein_one_letter(exp_row.wild_type)
    mutation = gemmi.expand_protein_one_letter(exp_row.mutation)
    mut = Mutation(exp_row.chain, int(exp_row.position), wild_type, mutation)

    if fp_pdb_map.get(mut.seq_id, None) is None:
        print(f"Unable to map {mut} to nn-pred seq-id. Skipping...")
        return exp_row

    corr_mut = Mutation(mut.chain_id, fp_pdb_map[mut.seq_id], mut.comp_id, mut.mut_id)

    log_odds = calculate_log_odds(nn_df, corr_mut, log_odd_func)

    exp_row.update(log_odds)

    return exp_row


def calc_nn_log_odd_ratio(
    exp_data: Path, nn_pred_dir: Path, log_odd_func: Callable
) -> pd.DataFrame:

    exp_df = extract_relavant_cols(exp_data)

    csvs = collect_csv(nn_pred_dir, f"*.csv")

    nn_dfs = generate_multimodel_df(csvs)

    nn_df = concat_df(nn_dfs)

    model_names = collect_model_names(nn_df, "model")

    for model in model_names:
        exp_df[model] = np.nan

    proteins = exp_df[["pdb_id", "chain"]].dropna().drop_duplicates().values

    exp_nn_pred = []
    for pdb, chain in proteins:

        nn_rows = nn_df[nn_df.pdb_id.str.lower().str.startswith(pdb.lower())]
        csvs = collect_csv(nn_pred_dir, f"{pdb.lower()}_*.csv")

        if len(csvs) == 0:
            print(f"No structure net csv present for {pdb.lower()}. Skipping...")
            continue

        pdb_exp = exp_df[(exp_df.pdb_id == pdb) & (exp_df.chain == chain)]

        if pdb_exp.empty:
            print(f"No experimental data present for {pdb.lower()}. Skipping...")
            continue

        try:
            exp_seq = fetch_seq_from_exp_df(pdb_exp)

            fp_pdb_map = map_fp_seqnums_to_pdb_seqnums(exp_seq, nn_rows, chain)

            pdb_exp_nn_pred = pdb_exp.apply(
                lambda row: update_nn_log_odd_cols(
                    row, nn_rows, fp_pdb_map, log_odd_func
                ),
                axis=1,
            )
        except Exception as e:
            print(
                f"Failed to extract structure net predictions for {pdb}_{chain}\nError: {e}"
            )

        else:
            exp_nn_pred.append(pdb_exp_nn_pred)

        finally:
            print(f"Finished processing {pdb}_{chain}")

    return concat_df(exp_nn_pred)
