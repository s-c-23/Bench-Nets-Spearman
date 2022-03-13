from pathlib import Path

import pandas as pd

from pd_utils import EXP_COLUMNS, ESM_1V_COLUMNS, concat_exp_data



if __name__ == "__main__":

    pd.set_option("display.width", None)

    exp_dir = Path("/stor/home/dd32387/codes/ml/benchmarking/data/exp_data/fireprotdb")

    csv_glob = "*_labeled.csv"

    out_file = exp_dir / "fireprotdb_esm1v-5m.csv"

    columns = [*EXP_COLUMNS, *ESM_1V_COLUMNS]

    df = concat_exp_data(exp_dir, csv_glob, columns)

    if "pdb_id" in df.columns.to_list():
        df.sort_values(["pdb_id"], ascending=[True], inplace=True)

    df.to_csv(out_file, index=False)

    print(f"Concatenated csv files: {out_file.resolve()}")
