from pathlib import Path
from typing import Dict, Tuple
from pprint import pprint
import re

import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)



def model_stability_correlation(
    data: Path, model_re: re.Pattern = re.compile("^esm|resnet*"), flags=re.IGNORECASE
) -> pd.DataFrame:

    assert isinstance(data, Path)
    assert data.is_file() and data.suffix == ".csv"

    df = pd.read_csv(data)

    model_cols = [col for col in df.columns if model_re.search(col)]

    model_corr = dict()
    for model in model_cols:
        model_df = df[~df[model].isnull()]

        dTm_count, ddG_count = model_df[["dTm", "ddG"]].count().values

        pearson_corr = model_df[["dTm", "ddG", model]].corr(method="pearson")
        spearman_corr = model_df[["dTm", "ddG", model]].corr(method="spearman")

        pearson_dTm, pearson_ddG = pearson_corr.iloc[:2, 2].values.round(5)
        spearman_dTm, spearman_ddG = spearman_corr.iloc[:2, 2].values.round(5)

        model_corr[model] = {
            "pearson_dTm": pearson_dTm,
            "pearson_ddG": pearson_ddG,
            "spearman_dTm": spearman_dTm,
            "spearman_ddG": spearman_ddG,
            "dTm_count": dTm_count,
            "ddG_count": ddG_count
        }

    return (
        pd.DataFrame.from_dict(model_corr, orient="index")
        .rename_axis("model")
        .reset_index()
    )


if __name__ == "__main__":

    data = Path(
        "/stor/home/dd32387/codes/ml/benchmarking/data/exp_data/fireprotdb/fireprotdb_esm1v-5m_nn-pred.csv"
    )

    out_dir = Path("/stor/home/dd32387/codes/ml/benchmarking/data/analysis")

    model_re = re.compile("^(esm|resnet|nohup)")

    corr_df = model_stability_correlation(data, model_re)

    corr_df.to_csv(out_dir / f"{data.stem}_corr.csv", index=False)

    print(corr_df.sort_values("spearman_ddG"))
