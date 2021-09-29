import os, time
import torch
from utils.cmd_utils import parse_cmd
from utils.make_env_utils import make_env, get_env_kwargs, configure_logger
from vec_env.subproc_vec_env import SubprocVecEnv
from utils.wrapper import VecPyTorch
from torch_algorithms import PPO, PPO_dev
from torch_algorithms.policies import MultiDiscreteAttentionPolicy
from torch_algorithms import logger
import csv
import multiprocessing


def main(args):
    # Configure logger
    log_dir = args.log_path if args.log_path is not None else "/tmp/stable_baselines_" + time.strftime('%Y-%m-%d-%H-%M-%S')
    configure_logger(log_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Make env
    env_kwargs = get_env_kwargs(args.env_id, args.horizon, args.random_size, args.min_num_blocks, args.discrete_height,
                                args.random_mode, args.action_scale, args.restart_rate, args.noop, args.robot,
                                args.force_scale, args.adaptive_primitive)
    max_episode_steps = env_kwargs.get("max_episode_steps", None)
    env_kwargs.pop("max_episode_steps", None)
    env_kwargs.update({"need_visual": args.play, "render": False, "primitive": args.primitive,
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
    if args.policy == "discrete_mlp":
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
                                              bilevel_action=args.bilevel_action)
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
        model = PPO_dev(env, policy, device, n_steps=args.n_steps, nminibatches=args.nminibatches,
                        noptepochs=args.noptepochs, gamma=args.gamma, lam=args.lam, learning_rate=args.learning_rate,
                        cliprange=args.cliprange, ent_coef=args.ent_coef, max_grad_norm=args.max_grad_norm,
                        use_linear_lr_decay=use_linear_lr_decay, bc_coef=args.bc_coef, n_vf_rollout=args.aux_freq,
                        nvfepochs=args.nauxepochs, ewma_decay=args.ewma_decay,
                        use_linear_clip_decay=use_linear_clip_decay, auxiliary_task=args.auxiliary_task,
                        aux_coef=args.auxiliary_coef, inf_horizon=args.inf_horizon, eval_env=eval_env,
                        priority_type=args.priority_type, manual_filter_state=args.manual_filter_state,
                        state_replay_size=args.state_replay_size, filter_priority=args.filter_priority,
                        nvfminibatches=args.nvfminibatches, priority_decay=args.priority_decay,
                        clip_priority=args.clip_priority)
    else:
        raise NotImplementedError

    for name, param in policy.named_parameters():
        print(name, param.shape)

    if not args.play:
        if args.load_path is not None:
            model.load(args.load_path, eval=False)
        from utils.curriculum_callback import curriculum_callback, always_hard_callback
        cb = always_hard_callback if args.no_cl else curriculum_callback
        model.learn(int(args.num_timesteps), cb)
        model.save(os.path.join(log_dir, 'final.pt'))
    else:
        from utils.evaluation import evaluate, get_success_rate, render_all_restart_states, evaluate_fixed_scene
        env_kwargs["render"] = args.primitive and args.play
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

        print("Load path", args.load_path)
        model.load(args.load_path, eval=True)

        n_episode = 5
        mean_eval_reward = evaluate(eval_env, model.policy, device, n_episode, 7, render=True)


if __name__ == "__main__":
    args = parse_cmd()
    main(args)
