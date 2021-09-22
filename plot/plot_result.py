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
    env_name = sys.argv[2]
    assert env_name in ['FetchBridge7Blocks-v1', 'FetchBridge7Blocks-v2']
    # assert mode in ['train', 'hard', 'iteration']
    max_timesteps = {'FetchBridge7Blocks-v1': 2e7, 'FetchBridge7Blocks-v2': 3e7}
    # df_timesteps, df_sr, df_hard_ratio, df_legend, df_eval_timesteps, df_eval_sr, df_eval_legend = [], [], [], [], [], [], []
    df1_eval_timesteps, df1_eval_sr, df1_eval_legend, df2_eval_timesteps, df2_eval_sr, df2_eval_legend = [], [], [], [], [], []
    subfolders1 = ['ppg_shared', 'ppo_shared', 'ppg_dual', 'ppo_dual']
    subfolders2 = ['ppg_shared', 'ppg_shared_nocl']
    # subfolders = ['ppg_shared', 'ppg_dual', 'ppo_shared', 'ppo_dual', 'ppg_nocl']
    # subfolders = ['ppo_attention_new', 'ppo_attention_sir_new', 'ppo_attention_sil_new']

    for folder_idx, subfolder in enumerate(subfolders1 + subfolders2):
        last_sr = []
        for i in range(3):
            if not os.path.exists(os.path.join(folder_name, subfolder, str(i), 'progress.csv')):
                continue
            progress_file = os.path.join(folder_name, subfolder, str(i), 'progress.csv')
            eval_file = os.path.join(folder_name, subfolder, str(i), 'eval.csv')
            # raw_success_rate = get_item(progress_file, 'success_rate')
            # raw_total_timesteps = get_item(progress_file, 'total_timesteps')
            # raw_hard_ratio = get_item(progress_file, 'hard_ratio')
            eval_timesteps = get_item(eval_file, 'model_idx') * 327680
            eval_sr = get_item(eval_file, 'mean_eval_reward')
            print(eval_timesteps.shape, eval_sr.shape)
            # sr_f = interpolate.interp1d(raw_total_timesteps, raw_success_rate, fill_value="extrapolate")
            # hard_f = interpolate.interp1d(raw_total_timesteps, raw_hard_ratio, fill_value="extrapolate")
            # timesteps = np.arange(0, max_timesteps[env_name], max_timesteps[env_name] // 200)
            # print(timesteps[0], timesteps[-1], raw_total_timesteps[0], raw_total_timesteps[-1])
            # success_rate = sr_f(timesteps)
            # hard_ratio = hard_f(timesteps)
            # timesteps = smooth(timesteps, 5)
            # success_rate = smooth(success_rate, 5)
            # hard_ratio = smooth(hard_ratio, 5)
            eval_timesteps = smooth(eval_timesteps, 2)
            eval_sr = smooth(eval_sr, 2)
            # df_timesteps.append(timesteps)
            # df_sr.append(success_rate)
            # df_hard_ratio.append(hard_ratio)
            if folder_idx < len(subfolders1):
                df1_eval_timesteps.append(eval_timesteps)
                df1_eval_sr.append(eval_sr)
                # last_sr.append(success_rate[-1])
                # df_legend.append(np.array([subfolder.upper()] * len(timesteps)))
                df1_eval_legend.append(np.array([subfolder.upper()] * len(eval_timesteps)))
            else:
                df2_eval_timesteps.append(eval_timesteps)
                df2_eval_sr.append(eval_sr)
                df2_eval_legend.append(np.array([subfolder.upper()] * len(eval_timesteps)))
        print(subfolder, 'sr', np.mean(last_sr))
    # df_timesteps = np.concatenate(df_timesteps, axis=0).tolist()
    # df_sr = np.concatenate(df_sr, axis=0).tolist()
    # df_hard_ratio = np.concatenate(df_hard_ratio, axis=0).tolist()
    # df_legend = np.concatenate(df_legend, axis=0).tolist()
    df1_eval_timesteps = np.concatenate(df1_eval_timesteps, axis=0).tolist()
    df1_eval_sr = np.concatenate(df1_eval_sr, axis=0).tolist()
    df1_eval_legend = np.concatenate(df1_eval_legend, axis=0).tolist()
    df2_eval_timesteps = np.concatenate(df2_eval_timesteps, axis=0).tolist()
    df2_eval_sr = np.concatenate(df2_eval_sr, axis=0).tolist()
    df2_eval_legend = np.concatenate(df2_eval_legend, axis=0).tolist()

    # data = {'samples': df_timesteps, 'success_rate': df_sr, 'algo': df_legend}
    # sr_timesteps = pandas.DataFrame(data)
    # data = {'samples': df_timesteps, 'hard_ratio': df_hard_ratio, 'algo': df_legend}
    # hr_timesteps = pandas.DataFrame(data)
    data = {'samples': df1_eval_timesteps, 'success_rate': df1_eval_sr, 'algo': df1_eval_legend}
    eval_sr_timesteps = pandas.DataFrame(data)
    data = {'samples': df2_eval_timesteps, 'success_rate': df2_eval_sr, 'algo': df2_eval_legend}
    eval_sr_timesteps2 = pandas.DataFrame(data)

    wspace = .3
    bottom = .3
    margin = .1
    # left = .08
    left = .1
    width = 2.15 / ((1. - left) / (2 + wspace + margin / 2))
    height = 1.5 / ((1. - bottom) / (1 + margin / 2))

    plt.style.use("ggplot")
    print(plt.rcParams['font.family'])
    exit()
    # plt.rcParams.update({'legend.fontsize': 14})
    p = sns.color_palette()
    sns.set_palette([p[i] for i in range(len(subfolders1))])
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
    f.legend(handles[:], ['PPG shared', 'PPO shared', 'PPG dual', 'PPO dual'], loc="lower right", ncol=1, bbox_to_anchor=(0.49, 0.22), title='')
    sns.lineplot(x='samples', y='success_rate', hue='algo', ax=axes[1], data=eval_sr_timesteps2)
    axes[1].set_xlabel('samples')
    axes[1].set_ylabel('')
    axes[1].get_legend().remove()

    handles, labels = axes[1].get_legend_handles_labels()
    f.legend(handles[:], ['PPG w/ CL', 'PPG w/o CL'], loc="lower right", ncol=1, bbox_to_anchor=(0.99, 0.22), title='')
    # f.legend(handles[:], ['PPG w/ CL', 'PPG w/o CL'], loc="lower right", ncol=1, bbox_to_anchor=(0.99, 0.22), title='')
    f.subplots_adjust(top=1. - margin / height, bottom=0.21, wspace=wspace, left=left, right=1. - margin / width)
    plt.savefig(os.path.join(folder_name, env_name +'.pdf'))
    print(os.path.join(folder_name, env_name + '.pdf'))
    plt.show()
