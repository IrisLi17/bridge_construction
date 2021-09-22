import pandas
import numpy as np
import matplotlib.pyplot as plt
import sys, os


def smooth(arr, window_size):
    arr = np.asarray(arr)
    assert len(arr) > window_size
    out = np.zeros(len(arr) - window_size + 1)
    for i in range(len(out)):
        out[i] = np.mean(arr[i: i + window_size])
    return out


if __name__ == "__main__":
    option = sys.argv[1]
    dirs = sys.argv[2:]
    window_size = 5
    label = dict(reward="ep_reward_mean",
                 entropy="entropy",
                 value_loss="value_loss",
                 success_rate="success_rate",
                 hard_ratio="hard_ratio",)
    assert option in label.keys()
    timestep_label = "total_timesteps"
    p = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for i, log_dir in enumerate(dirs):
        dataframe = pandas.read_csv(os.path.join(log_dir, "progress.csv"))
        timesteps = dataframe[timestep_label].values
        ys = dataframe[label[option]].values
        plt.plot(smooth(timesteps, window_size), smooth(ys, window_size), c=p[i % len(p)], label=log_dir)
        if option == "success_rate":
            hard_ratio = dataframe["hard_ratio"].values
            plt.plot(timesteps, hard_ratio, c=p[i % len(p)], linestyle="--")

    plt.legend(loc="upper right", bbox_to_anchor=(1.0, 0.1))
    plt.xlabel("timesteps")
    plt.ylabel(option)
    plt.grid()
    plt.show()
