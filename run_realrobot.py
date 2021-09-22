import os, time, math
import torch
from utils.cmd_utils import parse_cmd
from utils.make_env_utils import make_env, get_env_kwargs, configure_logger
from vec_env.subproc_vec_env import SubprocVecEnv
from utils.wrapper import VecPyTorch
from torch_algorithms import PPO_dev
from torch_algorithms.policies import MultiDiscreteAttentionPolicy
from torch_algorithms import logger
import sys
sys.path.append("../xArm-Python-SDK")
import numpy as np
from env.bullet_rotations import quat_rot_vec, quat2mat, euler2quat, mat2quat, gen_noisy_q, euler2mat, rvec2mat
import matplotlib.pyplot as plt

from realrobot_utils.parse_scene import parse_view, convert_arm_to_camera, convert_plane_to_mask, parse_scene
from realrobot_utils.setup_utils import arm_set_up, camera_set_up


TCP_payload = {
    'gripper': [0.82, [0, 0, 48]],
    'gripper+object': [1.46, [0, 0, 79.82]], # 20cm
    }
# intrinsic_mtx = np.array([[611.19006348, 0., 319.11096191],
#                               [0., 610.06945801, 231.36959839],
#                               [0., 0., 1.]])
# intrinsic_coef = np.array([0., 0., 0., 0., 0.])




def test():
    arm = arm_set_up("192.168.1.226", mode=0)
    arm.set_position(x=362, y=-149, z=365, roll=-180, pitch=0, yaw=-90, speed=100, wait=True)
    camera_pipeline, camera_align, camera_mtx, camera_dist_coef = camera_set_up()
    marker_size = 0.04

    samples_color, samples_depth = [], []
    for _ in range(10):
        frames = camera_pipeline.wait_for_frames()
        frames = camera_align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asarray(depth_frame.get_data())
        color_image = np.asarray(color_frame.get_data())

        samples_color.append(color_image)
        samples_depth.append(depth_image)
    depth_image = np.median(np.array(samples_depth), axis=0).astype(np.uint16)
    color_image = samples_color[-1]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(color_image)
    ax[1].imshow(depth_image)
    plt.show()

    obj_poses, sizes, servo_angles = parse_view(arm, camera_mtx, camera_dist_coef, color_image, depth_image,
                                                 marker_size)
    print(obj_poses, sizes, servo_angles)
    exit()
    tip_pose = [300, -250, 430, 180, 20, 0]
    camera_xyz, camera_mtx = convert_arm_to_camera(tip_pose, CALIB_CAM2ARM)
    print("camera xyz", camera_xyz, "camera_mtx", camera_mtx)
    print("camera_quat", mat2quat(camera_mtx))
    plane_height = 0
    intrinsic_mtx = np.array([[611.19006348,   0.,         319.11096191],
                              [0.,         610.06945801, 231.36959839],
                              [0.,           0.,           1.]])
    intrinsic_coef = np.array([0., 0., 0., 0., 0.])
    image_width = 640
    image_height = 480
    depth_mtx = convert_plane_to_mask(plane_height, camera_xyz, camera_mtx, intrinsic_mtx, intrinsic_coef,
                                      image_width, image_height)
    import matplotlib.pyplot as plt
    plt.imshow(depth_mtx)
    plt.colorbar()
    plt.show()


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
                                args.rotation_low, args.rotation_high, args.restart_rate, args.noop, args.rel_obs,
                                args.robot, args.friction_low, args.friction_high, args.force_scale)
    max_episode_steps = env_kwargs.get("max_episode_steps", None)
    env_kwargs.pop("max_episode_steps", None)
    env_kwargs.update({"need_visual": True, "render": False, "primitive": args.primitive,
                       "compute_path": True,
                       "restart_rate": 0.0,
                       })

    def make_thunk(rank):
        return lambda: make_env(args.env_id, rank, args.seed, max_episode_steps, log_dir,
                                done_when_success=args.done_when_success, use_monitor=True,
                                reward_scale=args.reward_scale, bonus_weight=args.bonus_weight,
                                env_kwargs=env_kwargs)
    env = SubprocVecEnv([make_thunk(i) for i in range(1)], reset_when_done=False)
    env = VecPyTorch(env, device)
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
                                              refined_action=args.refined_action, bilevel_action=args.bilevel_action)
    else:
        raise NotImplementedError
    policy.to(device)

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

    if args.algo == "ppg":
        model = PPO_dev(env, policy, device, n_steps=args.n_steps, nminibatches=args.nminibatches, noptepochs=args.noptepochs,
                        gamma=args.gamma, lam=args.lam, learning_rate=args.learning_rate, cliprange=args.cliprange, ent_coef=args.ent_coef,
                        max_grad_norm=args.max_grad_norm, use_linear_lr_decay=use_linear_lr_decay,
                        use_linear_clip_decay=use_linear_clip_decay, inf_horizon=args.inf_horizon,
                        bc_coef=args.bc_coef, n_vf_rollout=args.aux_freq, nvfepochs=args.nauxepochs, ewma_decay=args.ewma_decay, kl_beta=args.kl_beta,
                        auxiliary_task=args.auxiliary_task, aux_coef=args.auxiliary_coef, exp_update=args.exp_update,
                        eval_env=None, priority_type=args.priority_type, optimizer=args.optimizer,
                        manual_filter_state=args.manual_filter_state,
                        state_replay_size=args.state_replay_size, filter_priority=args.filter_priority,
                        nvfminibatches=args.nvfminibatches, priority_decay=args.priority_decay,
                        clip_priority=args.clip_priority,
                        )
    else:
        raise NotImplementedError

    for name, param in policy.named_parameters():
        print(name, param.shape)

    if args.load_path is None:
        ans = input("Warning: No model will be loaded. Continue? [Y|n]")
        if ans != "Y":
            exit()
    # assert args.load_path is not None
    # model_file = torch.load(args.load_path, map_location="cpu")
    if args.load_path is not None:
        model.load(args.load_path, eval=True)

    '''
    camera_pipeline, camera_align, camera_mtx, camera_dist_coef = camera_set_up()
    # TODO: read size and initial poses, set simulator.
    #  Then run policy with primitive, get one step of paths.
    #  Look at the scene again
    marker_size = 0.04
    arm = arm_set_up("192.168.1.226", mode=0)
    marker_ids, obj_poses, sizes, servo_angles = parse_scene(
        arm, camera_pipeline, camera_align, camera_mtx, camera_dist_coef, marker_size, num_objects=5)
    print(marker_ids)
    print(obj_poses)
    print(sizes)
    return
    '''
    from utils.evaluation import evaluate_fixed_scene
    initial_positions = [[0.9, 0.0, 0.025], [1.06655152, -0.02405042, 0.025], [0.9, 0.26, 0.025], [1.05, 0.26, 0.025],
                         [0.9, 0.94, 0.025]]
    object_sizes = [[0.025, 0.1, 0.025], [0.025, 0.1, 0.025], [0.025, 0.07, 0.025], [0.025, 0.12, 0.025],
                    [0.025, 0.12, 0.025]]
    cliff0_center = 0.35018731915555334
    cliff1_center = 0.9999242197307323
    evaluate_fixed_scene(env, initial_positions, object_sizes, cliff0_center, cliff1_center,
                         model.policy, device)
    return
    for i in range(args.horizon):
        # marker_ids, obj_poses, sizes, servo_angles = parse_scene(
        #     arm, camera_pipeline, camera_align, camera_mtx, camera_dist_coef, marker_size, num_objects=3)
        marker_ids = (np.array([6]), np.array([7]), np.array([9]))
        obj_poses = (np.array([[ 0.99813972, -0.03029399,  0.05290901,  0.31975185],
                               [ 0.02474998,  0.99442873,  0.10246432, -0.5914005 ],
                               [-0.05571829, -0.10096421,  0.9933286 ,  0.01849714]]),
                     np.array([[ 0.9435363 , -0.02473726,  0.33034423,  0.30466921],
                               [ 0.0464911 ,  0.99722687, -0.05811326, -0.28326996],
                               [-0.32799058,  0.07019004,  0.94206981,  0.01615268]]),
                     np.array([[ 0.99997519, -0.00524957,  0.0046954 ,  0.47782332],
                               [ 0.00493304,  0.99786934,  0.06505733, -0.57990088],
                               [-0.00502692, -0.06503255,  0.99787048,  0.03318581]])
                    )
        # sizes = (np.array([0.0239235 , 0.09892199, 0.0239235 ]),
        #          np.array([0.02566087, 0.10582984, 0.02566087]),
        #          np.array([0.02646932, 0.07004088, 0.02646932])
        # )
        obj_poses = (np.array([[ 1, 0,  0,  0.29],
                               [ 0,  1,  0, -0.6 ],
                               [0, 0,  1 ,  0.015]]),
                     np.array([[ 0.9435363 , -0.02473726,  0.33034423,  0.30466921],
                               [ 0.0464911 ,  0.99722687, -0.05811326, -0.28326996],
                               [-0.32799058,  0.07019004,  0.94206981,  0.01615268]]),
                     np.array([[ 0.99997519, -0.00524957,  0.0046954 ,  0.47782332],
                               [ 0.00493304,  0.99786934,  0.06505733, -0.57990088],
                               [-0.00502692, -0.06503255,  0.99787048,  0.03318581]])
                    )
        sizes = (np.array([0.025 , 0.1, 0.025 ]),
                 np.array([0.025, 0.11, 0.025]),
                 np.array([0.025, 0.07, 0.025])
        )
        servo_angles = [-18.909956, -7.890202, -14.95162, 83.608752, -2.030161, 91.234247, 56.229963]
        print("marker_ids", marker_ids, "obj poses", obj_poses, "sizes", sizes, "servo angles", servo_angles)
        # Only set poses in the later steps
        if i == 0:
            cliffs_center = (-0.25, 0.25)
            env.reset_scene(obj_poses, sizes, servo_angles, cliffs_center)  # Don't forget to align planner
            env.primitive.align_at_reset()
            img = env.render(mode="rgb_array")
            print(env.get_obs())
            plt.imshow(img)
            plt.show()
        else:
            env.reset_scene(obj_poses, None, servo_angles)
        obs = env.get_obs()
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = policy.act(
                torch.from_numpy(obs).float().to(device).unsqueeze(dim=0),
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)
        action = action.squeeze(dim=0).cpu().numpy()
        print("action", action)
        converted_action = env.convert_action(action)
        print("converted action", converted_action)
        sim_obs, reward, done, info = env.step(action)
        total_path = info.get("low_level_path", None)
        out_of_reach = info.get("out_of_reach")
        print(total_path)
        # if not out_of_reach:
        #     exit()
        if not out_of_reach and total_path is not None:
            for k in ["fetch_object", "close_finger", "change_pose", "release_finger", "lift_up", "move_back"]:
                if k in ["fetch_object", "change_pose", "lift_up", "move_back"]:
                    print(arm.tcp_load)
                    path = total_path[k]
                    if path is None:
                        continue
                    if arm.connected and arm.state != 4:
                        for idx, q in enumerate(path):
                            angles = q[:7]
                            ret = arm.set_servo_angle(angle=angles, speed=20 / 180, is_radian=True,
                                                      wait=(idx == len(path) - 1))
                            print(idx, 'set_servo_angle {}, ret={}'.format(angles, ret))
                            print(arm.get_servo_angle(is_radian=True))
                elif k == "close_finger":
                    code = arm.set_gripper_position(450, wait=True)
                    print('[wait]set gripper pos, code={}'.format(code))
                elif k == "release_finger":
                    code = arm.set_gripper_position(550, wait=True)
                    print('[wait]set gripper pos, code={}'.format(code))
        exit()
    # evaluate(env, model.policy, device, n_episode, 3, render=True,
    #          auxtask_name=model.auxiliary_task_name, auxtask=model.auxiliary_task)

# python run_realrobot.py --env_id FetchBridgeBullet7Blocks-v1 --no_adaptive_number --algo ppg --policy_arch shared --exclude_time --random_size --reward_type onestep --num_workers 1 --num_timesteps 2e7 --gamma 0.97 --noptepochs 10 --action_scale 0.6 --bilevel_action --num_bin 16 --rotation_low -0.5 --friction_low 0.5 --friction_high 0.5 --aux_freq 1 --bc_coef 1 --ewma 0.995 --inf_horizon --restart_rate 0.0 --priority_type td --priority_decay 0.0 --noop --manual_filter_state --clip_priority --auxiliary_task inverse_dynamics --auxiliary_coef 0.1 --force_scale 0 --load_path trained_models/FetchBridgeBullet7Blocks-v1_continuous/ppg_multidiscrete_a0.6/uniform_357/no_skyline/priority_td/primitive_rotlow-0.5_f0.5_pretrain21_force10_xarm/model_9.pt --robot xarm --play  --discrete_height --primitive
if __name__ == "__main__":
    # test()
    args = parse_cmd()
    main(args)
