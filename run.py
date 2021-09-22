import os, time
import torch
from utils.cmd_utils import parse_cmd
from utils.make_env_utils import make_env, get_env_kwargs, configure_logger
from vec_env.subproc_vec_env import SubprocVecEnv
from utils.wrapper import VecPyTorch
from torch_algorithms import PPO, PPO_dev
from torch_algorithms.policies import HybridAttentionPolicy, MultiDiscreteAttentionPolicy
from torch_algorithms import logger
import csv
import multiprocessing


def main(args):
    # Configure logger
    log_dir = args.log_path if args.log_path is not None else "/tmp/stable_baselines_" + time.strftime('%Y-%m-%d-%H-%M-%S')
    configure_logger(log_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Make env
    env_kwargs = get_env_kwargs(args.env_id, args.horizon, args.stable_reward_coef, args.rotation_penalty_coef,
                                args.height_coef, args.reward_type, args.no_adaptive_number, args.random_size,
                                args.exclude_time, args.smooth_coef, args.min_num_blocks, args.cost_coef,
                                args.smooth_max, args.cl_type, args.discrete_height, args.observe_skyline,
                                args.skyline_dim, args.random_mode, args.action_scale, args.center_y, args.cons_coef,
                                args.rotation_low, args.rotation_high, args.restart_rate, args.noop,
                                args.robot, args.friction_low, args.friction_high, args.force_scale,
                                args.adaptive_primitive)
    max_episode_steps = env_kwargs.get("max_episode_steps", None)
    env_kwargs.pop("max_episode_steps", None)
    env_kwargs.update({"need_visual": args.play, "render": args.primitive and args.play, "primitive": args.primitive,
                       "compute_path": args.primitive and args.play,
                       })

    def make_thunk(rank):
        return lambda: make_env(args.env_id, rank, args.seed, max_episode_steps, log_dir,
                                done_when_success=args.done_when_success, use_monitor=True,
                                reward_scale=args.reward_scale, bonus_weight=args.bonus_weight,
                                env_kwargs=env_kwargs)
    env = SubprocVecEnv([make_thunk(i) for i in range(args.num_workers)])
    env = VecPyTorch(env, device)

    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs["restart_rate"] = 0.

    def make_eval_thunk(rank):
        return lambda: make_env(args.env_id, rank, args.seed, max_episode_steps,
                                done_when_success=args.done_when_success, reward_scale=args.reward_scale,
                                bonus_weight=args.bonus_weight, env_kwargs=eval_env_kwargs)
    eval_env = SubprocVecEnv([make_eval_thunk(i) for i in range(args.num_workers)])
    eval_env = VecPyTorch(eval_env, device)

    aux_head = (args.algo == "ppg" and args.policy_arch == "dual")
    if args.policy == "attention":
        policy = HybridAttentionPolicy(env.observation_space.shape, env_kwargs['num_blocks'], env.action_space.shape[0] - 1,
                                       args.hidden_size, args.n_attention_blocks, object_dim=env.get_attr("object_dim")[0],
                                       has_cliff=env.get_attr("has_cliff")[0],
                                       aux_head=aux_head, arch=args.policy_arch, base_kwargs={'n_heads': args.n_heads})
    elif args.policy == "discrete_mlp":
        # Use different bins for different dims: y dim, z dim, rotation
        if "_" in args.num_bin:
            num_bin_list = args.num_bin.split("_")
            args.num_bin = [int(item) for item in num_bin_list]
        else:
            args.num_bin = [int(args.num_bin)] * 3
        # args.num_bin = [args.num_bin, 11, 11]
        # Revert to use same num of bins for different dims
        # num_bin = [num_bin] * 3
        policy = MultiDiscreteAttentionPolicy(env.observation_space.shape, env_kwargs['num_blocks'],
                                              env.action_space.shape[0] - 1, num_bin=args.num_bin,
                                              feature_dim=args.hidden_size, n_attention_blocks=args.n_attention_blocks,
                                              object_dim=env.get_attr("object_dim")[0],
                                              has_cliff=env.get_attr("has_cliff")[0], aux_head=aux_head,
                                              arch=args.policy_arch, base_kwargs={'n_heads': args.n_heads},
                                              noop=args.noop, n_values=args.v_ensemble,
                                              refined_action=args.refined_action, bilevel_action=args.bilevel_action)
        # policy = MultiDiscretePolicy(env.observation_space.shape, env_kwargs['num_blocks'], env.action_space.shape[0] - 1, num_bin=20,
        #                              base_kwargs={'recurrent': False, 'hidden_size': hidden_size})
    else:
        raise NotImplementedError
    policy.to(device)
    policy.train()

    if "lin" in args.learning_rate:
        args.learning_rate = float(args.learning_rate.split("_")[1])
        use_linear_lr_decay = True
    else:
        args.learning_rate = float(args.learning_rate)
        use_linear_lr_decay = False
    if "lin" in args.cliprange:
        args.cliprange = float(args.cliprange.split("_")[1])
        use_linear_clip_decay = True
    else:
        args.cliprange = float(args.cliprange)
        use_linear_clip_decay = False

    if args.algo == "ppo":
        model = PPO(env, policy, device, n_steps=args.n_steps, nminibatches=args.nminibatches, noptepochs=args.noptepochs,
                    gamma=args.gamma, lam=args.lam, learning_rate=args.learning_rate, cliprange=args.cliprange, ent_coef=args.ent_coef,
                    max_grad_norm=args.max_grad_norm, use_linear_lr_decay=use_linear_lr_decay,
                    use_linear_clip_decay=use_linear_clip_decay,
                    )
    elif args.algo == "ppg":
        model = PPO_dev(env, policy, device, n_steps=args.n_steps, nminibatches=args.nminibatches, noptepochs=args.noptepochs,
                        gamma=args.gamma, lam=args.lam, learning_rate=args.learning_rate, cliprange=args.cliprange, ent_coef=args.ent_coef,
                        max_grad_norm=args.max_grad_norm, use_linear_lr_decay=use_linear_lr_decay,
                        use_linear_clip_decay=use_linear_clip_decay, inf_horizon=args.inf_horizon,
                        bc_coef=args.bc_coef, n_vf_rollout=args.aux_freq, nvfepochs=args.nauxepochs, ewma_decay=args.ewma_decay, kl_beta=args.kl_beta,
                        auxiliary_task=args.auxiliary_task, aux_coef=args.auxiliary_coef, exp_update=args.exp_update,
                        eval_env=eval_env, priority_type=args.priority_type, optimizer=args.optimizer,
                        manual_filter_state=args.manual_filter_state,
                        state_replay_size=args.state_replay_size, filter_priority=args.filter_priority,
                        nvfminibatches=args.nvfminibatches, priority_decay=args.priority_decay,
                        clip_priority=args.clip_priority,
                        )
    else:
        raise NotImplementedError

    for name, param in policy.named_parameters():
        print(name, param.shape)

    if not args.play:
        def callback(_locals, _globals):
            cur_update = _locals['j']
            if cur_update % 10 == 0:
                save_path = os.path.join(log_dir, "model_%d.pt" % (cur_update // 10))
                model.save(save_path)
        if args.load_path is not None:
            if args.load_path.startswith("hdfs"):
                os.system("hdfs dfs -get " + args.load_path + " pretrain_model.pt")
                args.load_path = "pretrain_model.pt"
            model.load(args.load_path, eval=False)
            # from test_lowlevel import check_lowlevel
            # n_obj = 5
            # check_lowlevel(env, policy, device, 10, n_obj)
            # raise NotImplementedError
        from utils.curriculum_callback import curriculum_callback, always_hard_callback
        cb = always_hard_callback if args.no_cl else curriculum_callback
        model.learn(int(args.num_timesteps), cb)
        model.save(os.path.join(log_dir, 'final.pt'))
    else:
        from utils.evaluation import evaluate, get_success_rate, render_all_restart_states, evaluate_fixed_scene
        eval_env = SubprocVecEnv([make_thunk(i) for i in range(1)], reset_when_done=False)
        eval_env = VecPyTorch(eval_env, device)
        for i in range(eval_env.get_attr("num_blocks")[0]):
            eval_env.env_method("set_hard_ratio", 1.0, i)
        model_file = torch.load(args.load_path, map_location="cpu")
        if "state_buffer" in model_file and args.restart_rate > 0:
            eval_env.env_method("add_restart_states", model_file["state_buffer"])
            eval_env.set_attr("restart_rate", args.restart_rate)
            render_all_restart_states(eval_env)
            eval_env.env_method("add_restart_states", model_file["state_buffer"])

        def log_eval(num_update, mean_eval_reward, file_name='eval.csv'):
            if not os.path.exists(os.path.join(logger.get_dir(), file_name)):
                with open(os.path.join(logger.get_dir(), file_name), 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
                    title = ['model_idx', 'mean_eval_reward']
                    csvwriter.writerow(title)
            with open(os.path.join(logger.get_dir(), file_name), 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
                data = [num_update, mean_eval_reward]
                csvwriter.writerow(data)

        if not args.load_path.endswith(".pt"):
            for model_idx in range(61):
                cmd = "hdfs dfs -get " + os.path.join(args.load_path, "model_%d.pt" % model_idx) + " " + log_dir
                print(cmd)
                os.system(cmd)
                model_path = os.path.join(log_dir, "model_%d.pt" % model_idx)
                model.load(model_path, eval=True)
                n_eval_episode = 50
                success_rate = get_success_rate(eval_env, policy, eval_env.get_attr("num_blocks")[0], n_eval_episode, device)
                print("model", os.path.join(os.path.basename(args.load_path), "model_%d.pt" % model_idx), "eval success rate", success_rate)
                log_eval(model_idx, success_rate)
            cmd = "hdfs dfs -put -f " + os.path.join(logger.get_dir(), "eval.csv") + " " + args.load_path
            print(cmd)
            os.system(cmd)

        else:
            if args.load_path.startswith("hdfs"):
                os.system("hdfs dfs -get " + args.load_path)
                args.load_path = os.path.basename(args.load_path)
            print("Load path", args.load_path)
            model.load(args.load_path, eval=True)
            # Debug
            if "obs_buffer" in model_file and model_file['obs_buffer'] is not None:
                obs_buffer = torch.from_numpy(model_file['obs_buffer']).float()
                next_obs_buffer = torch.from_numpy(model_file['next_obs_buffer']).float()
                reward_buffer = torch.from_numpy(model_file['reward_buffer'])
                time_buffer = model_file['time_buffer']
                priority_buf = model_file.get('priority_buffer', None)
                encounter_buf = model_file.get("encounter_buffer", None)
                print("encounter buffer", encounter_buf[:20])
                with torch.no_grad():
                    values = policy.get_value(obs_buffer, None, None)
                    next_values = policy.get_value(next_obs_buffer, None, None)
                    print(values.shape)
                    critic = values[:, 0].squeeze(dim=-1)
                    next_critic = next_values[:, 0].squeeze(dim=-1)
                    v_std = torch.std(values, dim=1).squeeze(dim=-1)
                    td_error = (reward_buffer.squeeze(dim=-1) + args.gamma * next_critic - critic).abs()
                print(obs_buffer[1:3])
                if priority_buf is not None:
                    print(values[0], next_values[0], priority_buf[0])
                import matplotlib.pyplot as plt
                plt.plot(v_std, label="v_std")
                plt.plot(critic, label="value")
                plt.plot(next_critic, label="next value")
                plt.plot(reward_buffer, label="reward")
                plt.plot(td_error, label="td error")
                if priority_buf is not None:
                    plt.plot(priority_buf, label="priority")
                plt.legend()
                print(v_std[:10], v_std[-10:])
                plt.show()
                plt.hist(time_buffer)
                plt.show()
            initial_positions = [[0.9, 0.0, 0.025], [1.01655152, -0.02405042, 0.025], [0.9, 0.26, 0.025], [1.05, 0.26, 0.025], [0.9, 0.92, 0.025]]
            object_sizes = [[0.025, 0.1, 0.025], [0.025, 0.1, 0.025], [0.025, 0.07, 0.025], [0.025, 0.12, 0.025], [0.025, 0.12, 0.025]]
            cliff0_center = 0.35018731915555334
            cliff1_center = 0.9999242197307323
            initial_positions = [[0.9, 0.0, 0.025],
                                 [1.05, 0.0, 0.025],
                                 [0.9, 0.26, 0.025],
                                 [1.05, 0.26, 0.025],
                                 [0.9, 0.94, 0.025],
                                 [1.05, 0.94, 0.025],
                                 [0.9, 1.2, 0.025]]
            # object_sizes = [[0.025, 0.1, 0.025],
            #                 [0.025, 0.1, 0.025],
            #                 [0.025, 0.1, 0.025],
            #                 [0.025, 0.07, 0.025],
            #                 [0.025, 0.07, 0.025],
            #                 [0.025, 0.12, 0.025],
            #                 [0.025, 0.12, 0.025], ]
            # cliff0_center = 0.22868999
            # cliff1_center = 1.06864712
            # object_sizes = [[0.025, 0.1, 0.025],
            #                 [0.025, 0.1, 0.025],
            #                 [0.025, 0.1, 0.025],
            #                 [0.025, 0.08, 0.025],
            #                 [0.025, 0.07, 0.025],
            #                 [0.025, 0.12, 0.025],
            #                 [0.025, 0.12, 0.025],]
            # cliff0_center = 0.09569973
            # cliff1_center = 0.93559586
            object_sizes = [[0.025, 0.1, 0.025],
                            [0.025, 0.1, 0.025],
                            [0.025, 0.1, 0.025],
                            [0.025, 0.07, 0.025],
                            [0.025, 0.07, 0.025],
                            [0.025, 0.12, 0.025],
                            [0.025, 0.12, 0.025],]
            cliff0_center = 0.15867721
            cliff1_center = 1.01334966
            #################################
            initial_positions = [[0.9, 0.0, 0.025],
                                 [1.05, 0.0, 0.025],
                                 [0.9, 0.26, 0.025],]
            object_sizes = [[0.025, 0.1, 0.025],
                            [0.025, 0.08, 0.025],
                            [0.025, 0.12, 0.025]]
            cliff0_center = 0.41130526
            cliff1_center = 0.82560142
            # cliff0_center = 0.31
            # cliff1_center = 0.825
            '''
            ##################################
            initial_positions = [[0.9, 0.0, 0.025], [1.05, 0.0, 0.025], [0.9, 0.26, 0.025],
                                 [1.05, 0.26, 0.025], [0.9, 0.92, 0.025]]
            object_sizes = [[0.025, 0.1, 0.025], [0.025, 0.1, 0.025], [0.025, 0.08, 0.025], [0.025, 0.12, 0.025],
                            [0.025, 0.12, 0.025]]
            cliff0_center = 0.33800634
            cliff1_center = 0.95923873
            ###################################
            '''
            initial_positions = [[0.9, 0.0, 0.025], [1.05, 0.0, 0.025],
                                 [0.9, 0.26, 0.025], [1.05, 0.26, 0.025],
                                 [0.9, 0.94, 0.025], [1.05, 0.94, 0.025],
                                 [0.9, 1.2, 0.025]]
            '''
            object_sizes = [[0.025, 0.1, 0.025], [0.025, 0.1, 0.025], [0.025, 0.1, 0.025],
                            [0.025, 0.08, 0.025], [0.025, 0.07, 0.025],
                            [0.025, 0.12, 0.025], [0.025, 0.12, 0.025],]
            cliff0_center = 0.15565677
            cliff1_center = 0.91582905
            
            object_sizes = [[0.025, 0.1, 0.025], [0.025, 0.1, 0.025], [0.025, 0.1, 0.025],
                            [0.025, 0.07, 0.025], [0.025, 0.09, 0.025],
                            [0.025, 0.12, 0.025], [0.025, 0.12, 0.025], ]
            cliff0_center = 0.09955115
            cliff1_center = 0.98973008
            '''
            object_sizes = [[0.025, 0.1, 0.025], [0.025, 0.1, 0.025], [0.025, 0.1, 0.025],
                            [0.025, 0.07702753, 0.025], [0.025, 0.06929047, 0.025],
                            [0.025, 0.11481875, 0.025], [0.025, 0.11098686, 0.025]]
            cliff0_center = 0.11598686
            cliff1_center = 0.98745169

            evaluate_fixed_scene(eval_env, initial_positions, object_sizes, cliff0_center, cliff1_center,
                                 model.policy, device)
            # n_episode = 5
            # mean_eval_reward = evaluate(eval_env, model.policy, device, n_episode, 7, render=True,
            #                             auxtask_name=model.auxiliary_task_name, auxtask=model.auxiliary_task)
            # success_rate = get_success_rate(eval_env, policy, eval_env.get_attr("num_blocks")[0], n_episode, device, auto_reset=False)
            # print("success_rate", success_rate)
            # success_rate = get_success_rate(eval_env, policy, 5, n_episode, device, auto_reset=False)
            # print("success_rate", success_rate)
            # success_rate = get_success_rate(eval_env, policy, 3, n_episode, device, auto_reset=False)
            # print("success_rate", success_rate)
        # for n in range(3, eval_env.get_attr("num_blocks")[0] + 1):
        #     success_rate = get_success_rate(eval_env, policy, device, n, n_episode)
        #     print('success rate is', success_rate, 'for', n, 'objects')
        # print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        #     n_episode, mean_eval_reward))


if __name__ == "__main__":
    args = parse_cmd()
    main(args)
