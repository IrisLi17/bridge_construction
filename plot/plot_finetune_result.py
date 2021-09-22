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
    max_timesteps = 1e7
    df_timesteps, df_3obj_sr, df_5obj_sr, df_7obj_sr, df_sr, df_legend = [], [], [], [], [], []
    subfolders = ['fine_tune']
    n_objs = [3, 5, 7]

    for i in range(3):
        if not os.path.exists(os.path.join(folder_name, "ours", str(i), "progress.csv")):
            continue
        if not os.path.exists(os.path.join(folder_name, "fine_tune", str(i), "progress.csv")):
            continue
        train_progress = os.path.join(folder_name, "ours", str(i), "progress.csv")
        raw_total_timesteps = get_item(train_progress, 'total_timesteps')
        success_rate = get_item(train_progress, 'eval_7_sr')
        finetune_progress = os.path.join(folder_name, "fine_tune", str(i), "progress.csv")
        finetune_timesteps = get_item(finetune_progress, 'total_timesteps') + 2e7
        finetune_success_rate = get_item(finetune_progress, "eval_7_sr")
        raw_total_timesteps = np.concatenate([raw_total_timesteps, finetune_timesteps])
        success_rate = np.concatenate([success_rate, finetune_success_rate])
        df_timesteps.append(smooth(raw_total_timesteps, 5))
        df_sr.append(smooth(success_rate, 5))
        df_legend.append(np.array(["fine_tune"] * len(df_timesteps[-1])))

        if not os.path.exists(os.path.join(folder_name, "adaptive_primitive", str(i), "progress.csv")):
            continue
        progress_file = os.path.join(folder_name, "adaptive_primitive", str(i), "progress.csv")
        raw_total_timesteps = get_item(progress_file, "total_timesteps")
        success_rate = get_item(progress_file, "eval_7_sr")
        df_timesteps.append(smooth(raw_total_timesteps, 5))
        df_sr.append(smooth(success_rate, 5))
        df_legend.append(np.array(["mixed"] * len(df_timesteps[-1])))
    '''
    for n_obj in n_objs:
        last_sr = []
        for i in range(3):
            if not os.path.exists(os.path.join(folder_name, str(i), 'progress.csv')):
                continue
            progress_file = os.path.join(folder_name, str(i), 'progress.csv')
            # raw_success_rate = get_item(progress_file, 'success_rate')
            raw_total_timesteps = get_item(progress_file, 'total_timesteps')
            # sr_3obj = get_item(progress_file, 'eval_3_sr')
            # sr_5obj = get_item(progress_file, 'eval_5_sr')
            # sr_7obj = get_item(progress_file, 'eval_7_sr')
            success_rate = get_item(progress_file, 'eval_' + str(n_obj) + '_sr')
            # raw_hard_ratio = get_item(progress_file, 'hard_ratio')
            # sr_f = interpolate.interp1d(raw_total_timesteps, raw_success_rate, fill_value="extrapolate")
            # hard_f = interpolate.interp1d(raw_total_timesteps, raw_hard_ratio, fill_value="extrapolate")
            # timesteps = np.arange(0, max_timesteps[env_name], max_timesteps[env_name] // 200)
            # print(timesteps[0], timesteps[-1], raw_total_timesteps[0], raw_total_timesteps[-1])
            # success_rate = sr_f(timesteps)
            # hard_ratio = hard_f(timesteps)
            # timesteps = smooth(timesteps, 5)
            # success_rate = smooth(success_rate, 5)
            # hard_ratio = smooth(hard_ratio, 5)
            eval_timesteps = smooth(raw_total_timesteps, 5)
            # eval_3obj_sr = smooth(sr_3obj, 5)
            # eval_5obj_sr = smooth(sr_5obj, 5)
            # eval_7obj_sr = smooth(sr_7obj, 5)
            eval_success_rate = smooth(success_rate, 5)
            df_timesteps.append(eval_timesteps)
            # df_3obj_sr.append(eval_3obj_sr)
            # df_5obj_sr.append(eval_5obj_sr)
            # df_7obj_sr.append(eval_7obj_sr)
            df_sr.append(eval_success_rate)
            df_legend.append(np.array([str(n_obj)] * len(eval_timesteps)))
    '''
    df_timesteps = np.concatenate(df_timesteps, axis=0).tolist()
    # df_3obj_sr = np.concatenate(df_3obj_sr, axis=0).tolist()
    # df_5obj_sr = np.concatenate(df_5obj_sr, axis=0).tolist()
    # df_7obj_sr = np.concatenate(df_7obj_sr, axis=0).tolist()
    df_sr = np.concatenate(df_sr, axis=0).tolist()
    df_legend = np.concatenate(df_legend, axis=0).tolist()

    # data = {'samples': df_timesteps, 'success_rate': df_sr, 'algo': df_legend}
    # sr_timesteps = pandas.DataFrame(data)
    # data = {'samples': df_timesteps, 'hard_ratio': df_hard_ratio, 'algo': df_legend}
    # hr_timesteps = pandas.DataFrame(data)
    data = {'samples': df_timesteps, 'success_rate': df_sr, 'algo': df_legend}
    eval_sr_timesteps = pandas.DataFrame(data)

    wspace = .3
    bottom = .3
    margin = .1
    # left = .08
    left = .15
    width = 1.4 / ((1. - left) / (2 + wspace + margin / 2))
    height = 1.5 / ((1. - bottom) / (1 + margin / 2))

    plt.style.use("ggplot")
    print(plt.rcParams['font.family'])
    # plt.rcParams.update({'legend.fontsize': 14})
    p = sns.color_palette()
    sns.set_palette([p[i] for i in range(len(n_objs))])
    f, axes = plt.subplots(1, 1, figsize=(width, height))
    # sns.lineplot(x='samples', y='success_rate', hue='algo', ax=axes[0], data=sr_timesteps)
    # axes[0].set_xlabel('samples')
    # axes[0].set_ylabel('train succ. rate')
    # axes[0].get_legend().remove()
    # sns.lineplot(x='samples', y='hard_ratio', hue='algo', ax=axes[1], data=hr_timesteps)
    sns.lineplot(x='samples', y='success_rate', hue='algo', ax=axes, data=eval_sr_timesteps)
    axes.set_xlabel('samples')
    axes.set_ylabel('succ. rate')
    axes.get_legend().remove()
    handles, labels = axes.get_legend_handles_labels()
    f.legend(handles[:], ['Pretrain, finetune', 'Mixed'], loc="upper left", ncol=1, bbox_to_anchor=(0.15, 0.97), title='')
    # f.legend(handles[:], ['PPG w/ CL', 'PPG w/o CL'], loc="lower right", ncol=1, bbox_to_anchor=(0.99, 0.22), title='')
    f.subplots_adjust(top=1. - margin / height, bottom=0.21, wspace=wspace, left=left, right=1. - margin / width)
    plt.savefig(os.path.join(folder_name, 'finetune.pdf'))
    plt.show()
