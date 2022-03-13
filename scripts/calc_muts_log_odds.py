from pathlib import Path
from pd_utils import calc_nn_log_odd_ratio, mut_log_odds



if __name__ == "__main__":

    exp_data = Path("/stor/home/dd32387/codes/ml/benchmarking/data/exp_data/fireprotdb/fireprotdb_esm1v-5m.csv")

    nn_pred_dir = Path(
        "/stor/home/dd32387/codes/ml/benchmarking/data/nn_pred/fireprotdb/negatron_nets"
    )

    df = calc_nn_log_odd_ratio(exp_data, nn_pred_dir, mut_log_odds)

    df.to_csv("/stor/home/dd32387/codes/ml/benchmarking/data/exp_data/fireprotdb/fireprotdb_esm1v-5m_nn-pred.csv", index=False)
