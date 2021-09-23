import os, shutil
import torch
import numpy as np
from utils.wrapper import VecPyTorch
import matplotlib.pyplot as plt


def evaluate(eval_env: VecPyTorch, policy, device, n_episode, n_obj, render=False):
    num_processes = eval_env.num_envs
    # eval_env.env_method("set_cur_max_objects", eval_env.get_attr("num_blocks")[0])
    eval_env.env_method("set_cur_max_objects", n_obj)
    eval_env.env_method("set_min_num_objects", n_obj)
    eval_env.env_method("set_success_rate", [1.0] * 7)
    obs = eval_env.reset()
    print(obs)
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, policy.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)
    eval_episode_rewards = []
    eval_success_encounter = [0]
    step_count = 0
    construction_reward = 0
    position_shift = 0
    rotation_shift = 0

    if os.path.exists("tmp"):
        shutil.rmtree("tmp")
    os.makedirs("tmp", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    text_x, text_y = 150, 30

    low_level_paths = []
    meta_info = dict(block_size=[obs[0][17 * i + 12: 17 * i + 15] for i in range(n_obj)],
                     cliff0_center=obs[0][17 * eval_env.get_attr("num_blocks")[0] + 1],
                     cliff1_center=obs[0][17 * eval_env.get_attr("num_blocks")[0] + 18])
    while len(eval_episode_rewards) < n_episode:
        with torch.no_grad():
            value = policy.get_value(torch.from_numpy(obs).float().to(device), eval_recurrent_hidden_states, eval_masks)
            _, action, _, eval_recurrent_hidden_states = policy.act(
                torch.from_numpy(obs).float().to(device),
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=False)

        print(eval_env.get_attr("step_counter")[0], action.numpy()[0])
        img = eval_env.get_images()[0]
        ax.cla()
        ax.text(text_x, text_y, "Episode %d, Step %d, Value %.3f" % (len(eval_episode_rewards),
                                                                     eval_env.get_attr("step_counter")[0],
                                                                     value.squeeze(dim=0)[0].numpy()),
                horizontalalignment='center', verticalalignment='center')
        ax.imshow(img)
        if render:
            plt.pause(0.3)
            plt.imsave("tmp/tmpimg%d.png" % step_count, img)
            # plt.savefig("tmp/tmpimg%d.png" % step_count, bbox_inches='tight', pad_inches=0)
        else:
            plt.axis("off")
            # plt.imsave("tmp/tmpimg%d.png" % step_count, img)
            plt.savefig("tmp/tmpimg%d.png" % step_count, bbox_inches='tight', pad_inches=0)
        # Obser reward and next obs
        obs, reward, done, infos = eval_env.step(action.cpu().numpy())
        # print(obs[0])
        # debug_buffer['reward'].append(reward[0].numpy())
        step_count += 1

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            eval_success_encounter[-1] += info['is_success']
            construction_reward += info['construction']
            print('low level', info.get("low_level_result", -1), 'construction', info['construction'], 'is_success', info['is_success'],
                  "smooth bonus", info["smooth_bonus"], "ratio_used", info["ratio_used_blocks"],
                  )
            position_shift += info['position_shift']
            rotation_shift += info['rotation_shift']
            low_level_paths.append(info.get("low_level_path", None))
            if 'episode' in info.keys():
                with torch.no_grad():
                    value = policy.get_value(torch.from_numpy(obs).float().to(device), eval_recurrent_hidden_states,
                                             eval_masks)
                img = eval_env.get_images()[0]
                ax.cla()
                ax.text(text_x, text_y,
                        "Episode %d, Step %d, Value %.3f" % (len(eval_episode_rewards),
                                                             eval_env.get_attr("step_counter")[0],
                                                             value.squeeze(dim=0)[0].numpy()),
                        horizontalalignment='center', verticalalignment='center')
                ax.imshow(img)
                if render:
                    # plt.axis("off")
                    plt.pause(0.5)
                    # plt.imsave("tmp/tmpimg%d.png" % step_count, img)
                    plt.savefig("tmp/tmpimg%d.png" % step_count, bbox_inches='tight', pad_inches=0)
                else:
                    plt.axis("off")
                    # plt.imsave("tmp/tmpimg%d.png" % step_count, img)
                    plt.savefig("tmp/tmpimg%d.png" % step_count, bbox_inches='tight', pad_inches=0)
                eval_episode_rewards.append(info['episode']['r'])
                print('success encounter', eval_success_encounter[-1], 'reward', eval_episode_rewards[-1],
                      'construction reward', construction_reward,
                      'position shift', position_shift, 'rotation shift', rotation_shift)
                # TODO: dump low level path if any
                import pickle
                with open("low_level_paths.pkl", "wb") as f:
                    pickle.dump(dict(paths=low_level_paths, meta=meta_info), f)
                step_count += 1
                eval_success_encounter.append(0)
                construction_reward = 0
                position_shift = 0
                rotation_shift = 0
                low_level_paths = []
                obs = eval_env.reset()
                print("reset obs", obs)

    eval_env.close()
    return np.mean(eval_episode_rewards)


def evaluate_fixed_scene(eval_env, initial_positions, object_sizes, cliff0_center, cliff1_center,
                         policy, device):
    n_obj = len(initial_positions)
    eval_env.env_method("set_cur_max_objects", n_obj)
    eval_env.env_method("set_min_num_objects", n_obj)
    eval_env.env_method("set_success_rate", [1.0] * 7)
    force_scale = eval_env.get_attr("force_scale")[0]
    eval_env.env_method("set_force_scale", force_scale)
    print("force_scale", eval_env.get_attr("cur_force_scale")[0])
    eval_env.reset()
    obj_poses = [np.concatenate([position, np.array([0., 0., 0., 1.])]) for position in initial_positions]
    eval_env.env_method("reset_scene", obj_poses, object_sizes, cliff0_center, cliff1_center, None)
    obs = np.array(eval_env.env_method("get_obs"))
    num_processes = eval_env.num_envs
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, policy.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)
    done = [False]
    # low_level_paths = []
    meta_info = dict(block_size=[obs[0][17 * i + 12: 17 * i + 15] for i in range(n_obj)],
                     cliff0_center=obs[0][17 * eval_env.get_attr("num_blocks")[0] + 1],
                     cliff1_center=obs[0][17 * eval_env.get_attr("num_blocks")[0] + 18])
    action_seqs = []
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    text_x, text_y = 150, 30
    step_count = 0

    while not done[0]:
        with torch.no_grad():
            value = policy.get_value(torch.from_numpy(obs).float().to(device), eval_recurrent_hidden_states, eval_masks)
            _, action, _, eval_recurrent_hidden_states = policy.act(
                torch.from_numpy(obs).float().to(device),
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=False)
        print(eval_env.get_attr("step_counter")[0], action.numpy()[0])
        action_seqs.append(action.numpy()[0])
        img = eval_env.get_images()[0]
        ax.cla()
        ax.text(text_x, text_y, "Episode %d, Step %d, Value %.3f" % (0,
                                                                     eval_env.get_attr("step_counter")[0],
                                                                     value.squeeze(dim=0)[0].numpy()),
                horizontalalignment='center', verticalalignment='center')
        ax.imshow(img)
        plt.axis("off")
        plt.pause(0.1)
        plt.imsave("tmp/tmpimg%d.png" % step_count, img)
        # plt.savefig("tmp/tmpimg%d.png" % step_count, bbox_inches='tight', pad_inches=0)
        # Obser reward and next obs
        obs, reward, done, infos = eval_env.step(action.cpu().numpy())
        step_count += 1
        # for info in infos:
        #     low_level_paths.append(info.get("low_level_path", None))
    # import pickle
    # with open("low_level_paths.pkl", "wb") as f:
    #     pickle.dump(dict(paths=low_level_paths, meta=meta_info, actions=action_seqs), f)
    # Take shortcuts
    eval_env.env_method("enable_recording")
    valid_action_seqs = []
    for cur_step in range(1, len(action_seqs)):
        if action_seqs[cur_step][0] == action_seqs[cur_step - 1][0]:
            pass
        else:
            valid_action_seqs.append(action_seqs[cur_step - 1])
    eval_env.reset()
    eval_env.env_method("reset_scene", obj_poses, object_sizes, cliff0_center, cliff1_center, None)
    low_level_paths = []
    for step in range(len(valid_action_seqs)):
        action = valid_action_seqs[step]
        img = eval_env.get_images()[0]
        ax.cla()
        ax.imshow(img)
        plt.axis("off")
        # Obser reward and next obs
        obs, reward, done, infos = eval_env.step(np.expand_dims(action, axis=0))
        for info in infos:
            low_level_paths.append(info.get("low_level_path", None))
    import pickle
    with open("low_level_paths.pkl", "wb") as f:
        pickle.dump(dict(paths=low_level_paths, meta=meta_info, actions=valid_action_seqs), f)


def get_success_rate(eval_env, policy, num_object, n_episode, device, auto_reset=True):
    eval_env.env_method("set_cur_max_objects", num_object)
    eval_env.env_method("set_min_num_objects", num_object)
    eval_env.env_method("set_hard_ratio", 1.0, num_object)
    obs = eval_env.reset()
    total_episodes = 0
    success_count = 0
    # TODO: recurrent support
    num_processes = eval_env.num_envs
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, policy.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)
    while total_episodes < n_episode:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = policy.act(
                torch.from_numpy(obs).float().to(device), eval_recurrent_hidden_states, eval_masks, deterministic=True)
            obs, reward, done, infos = eval_env.step(action.cpu().numpy())
            # del eval_masks
            eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)
            # eval_masks = torch.from_numpy(np.array([[0.0] if done_ else [1.0] for done_ in done])).to(device)
        for idx, _done in enumerate(done):
            if _done:
                total_episodes += 1
                success_count += infos[idx]["is_success"]
                if not auto_reset:
                    obs = eval_env.reset()
    return success_count / total_episodes


def render_all_restart_states(eval_env):
    step_count = 0
    print(len(eval_env.get_attr("state_replay_buffer")[0]))
    if os.path.exists("tmp1"):
        shutil.rmtree("tmp1")
    os.makedirs("tmp1")
    while len(eval_env.get_attr("state_replay_buffer")[0]):
        eval_env.reset()
        img = eval_env.get_images()[0]
        plt.imsave("tmp1/restart%d.png" % step_count, img)
        step_count += 1
        if step_count > 300:
            break


# ffmpeg -r 4 -i tmp/tmpimg%d.png -pix_fmt yuv420p -vf pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2" output.mp4
# python run.py --env_id FetchBridgeBullet7Blocks-v1 --no_adaptive_number --algo ppg --policy_arch shared --exclude_time --random_size --reward_type onestep --num_workers 1 --num_timesteps 2e7 --gamma 0.97 --noptepochs 10 --action_scale 0.6 --bilevel_action --num_bin 16 --rotation_low -0.5 --friction_low 0.5 --friction_high 0.5 --aux_freq 1 --bc_coef 1 --ewma 0.995 --inf_horizon --restart_rate 0.0 --priority_type td --priority_decay 0.0 --noop --manual_filter_state --clip_priority --auxiliary_task inverse_dynamics --auxiliary_coef 0.1 --force_scale 0 --load_path trained_models/FetchBridgeBullet7Blocks-v1_continuous/ppg_multidiscrete_a0.6/uniform_357/no_skyline/priority_td/primitive_rotlow-0.5_pretrain21_f0.5_force10_xarm/model_5.pt --robot xarm --play  --discrete_height
