import seaborn as sns
import pandas
import numpy as np
import sys, os
from scipy import interpolate
import matplotlib.pyplot as plt


def get_item(log_file, label):
    data = pandas.read_csv(log_file, index_col=None, comment='#', error_bad_lines=True)
    return data[label].values


def smooth(array, window):
    out = np.zeros(array.shape[0] - window)
    for i in range(out.shape[0]):
        out[i] = np.mean(array[i:i + window])
    return out


if __name__ == '__main__':
    folder_name = sys.argv[1]
    max_timesteps = 2e7
    df_timesteps, df_3obj_sr, df_5obj_sr, df_7obj_sr, df_legend = [], [], [], [], []
    df2_timesteps, df2_sr, df2_legend = [], [], []
    subfolders = ['ours', 'no_restart', 'no_aux', 'no_restart_no_aux']
    subfolders2 = ['ours', 'priority_auxerror']

    for folder_idx, subfolder in enumerate(subfolders):
        last_sr = []
        for i in range(3):
            if not os.path.exists(os.path.join(folder_name, subfolder, str(i), 'progress.csv')):
                continue
            progress_file = os.path.join(folder_name, subfolder, str(i), 'progress.csv')
            raw_total_timesteps = get_item(progress_file, 'total_timesteps')
            sr_3obj = get_item(progress_file, 'eval_3_sr')
            sr_5obj = get_item(progress_file, 'eval_5_sr')
            sr_7obj = get_item(progress_file, 'eval_7_sr')
            eval_timesteps = smooth(raw_total_timesteps, 5)
            eval_3obj_sr = smooth(sr_3obj, 5)
            eval_5obj_sr = smooth(sr_5obj, 5)
            eval_7obj_sr = smooth(sr_7obj, 5)
            df_timesteps.append(eval_timesteps)
            df_3obj_sr.append(eval_3obj_sr)
            df_5obj_sr.append(eval_5obj_sr)
            df_7obj_sr.append(eval_7obj_sr)
            df_legend.append(np.array([subfolder.upper()] * len(eval_timesteps)))
    for folder_idx, subfolder in enumerate(subfolders2):
        for i in range(3):
            if not os.path.exists(os.path.join(folder_name, subfolder, str(i), 'progress.csv')):
                continue
            progress_file = os.path.join(folder_name, subfolder, str(i), 'progress.csv')
            raw_total_timesteps = get_item(progress_file, 'total_timesteps')
            sr_7obj = get_item(progress_file, 'eval_7_sr')
            eval_timesteps = smooth(raw_total_timesteps, 5)
            eval_7obj_sr = smooth(sr_7obj, 5)
            df2_timesteps.append(eval_timesteps)
            df2_sr.append(eval_7obj_sr)
            df2_legend.append(np.array([subfolder.upper()] * len(eval_timesteps)))
    df_timesteps = np.concatenate(df_timesteps, axis=0).tolist()
    df_3obj_sr = np.concatenate(df_3obj_sr, axis=0).tolist()
    df_5obj_sr = np.concatenate(df_5obj_sr, axis=0).tolist()
    df_7obj_sr = np.concatenate(df_7obj_sr, axis=0).tolist()
    df_legend = np.concatenate(df_legend, axis=0).tolist()
    df2_timesteps = np.concatenate(df2_timesteps, axis=0).tolist()
    df2_sr = np.concatenate(df2_sr, axis=0).tolist()
    df2_legend = np.concatenate(df2_legend, axis=0).tolist()

    data = {'samples': df_timesteps, 'success_rate': df_7obj_sr, 'algo': df_legend}
    eval_sr_timesteps = pandas.DataFrame(data)

    data = {'samples': df2_timesteps, 'success_rate': df2_sr, 'algo': df2_legend}
    metric_sr = pandas.DataFrame(data)

    wspace = .3
    bottom = .3
    margin = .1
    # left = .08
    left = .12
    width = 2.15 / ((1. - left) / (2 + wspace + margin / 2))
    height = 1.5 / ((1. - bottom) / (1 + margin / 2))

    plt.style.use("ggplot")
    print(plt.rcParams['font.family'])
    # plt.rcParams.update({'legend.fontsize': 14})
    p = sns.color_palette()
    sns.set_palette([p[i] for i in range(len(subfolders))])
    f, axes = plt.subplots(1, 2, figsize=(width, height))
    # sns.lineplot(x='samples', y='success_rate', hue='algo', ax=axes[0], data=sr_timesteps)
    # axes[0].set_xlabel('samples')
    # axes[0].set_ylabel('train succ. rate')
    # axes[0].get_legend().remove()
    # sns.lineplot(x='samples', y='hard_ratio', hue='algo', ax=axes[1], data=hr_timesteps)
    sns.lineplot(x='samples', y='success_rate', hue='algo', ax=axes[0], data=eval_sr_timesteps)
    axes[0].set_xlabel('samples')
    axes[0].set_ylabel('succ. rate')
    axes[0].get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    f.legend(handles[:], ['ppg+reset+aux.', 'ppg+aux.', 'ppg+reset', 'ppg'], loc="upper left", ncol=1, bbox_to_anchor=(0.12, 0.97), title='')
    sns.lineplot(x='samples', y='success_rate', hue='algo', ax=axes[1], data=metric_sr)
    axes[1].set_xlabel('samples')
    axes[1].set_ylabel('succ. rate')
    axes[1].get_legend().remove()
    handles, labels = axes[1].get_legend_handles_labels()
    f.legend(handles[:], ['TD error', 'Aux. error'], loc="upper left", ncol=1, bbox_to_anchor=(0.6, 0.97), title='')
    f.subplots_adjust(top=1. - margin / height, bottom=0.21, wspace=wspace, left=left, right=1. - margin / width)
    plt.savefig(os.path.join(folder_name, 'compare_ablation.pdf'))
    plt.show()
