import pickle
from env.robots import XArm7Robot
from env.bridge_construction_bullet import BulletBridgeConstructionLow
import pybullet as p
import time
import numpy as np
from realrobot_utils.setup_utils import arm_set_up, set_tcp_load


def replay_sim(pkl_name):
    with open(pkl_name, "rb") as f:
        info = pickle.load(f)
    paths = info["paths"]
    meta = info["meta"]
    n_obj = len(meta["block_size"])
    print("meta info", meta)
    low_env = BulletBridgeConstructionLow(n_obj, random_size=True, discrete=True, mode="long", cliff_height=0.1,
                                          render=True, need_visual=True, robot="xarm")

    low_env.p.resetDebugVisualizerCamera(2.0, 0, -45, [1.0, 0.6, 0.0])
    low_env.p.resetBasePositionAndOrientation(low_env.body_cliff0, [1.3, meta["cliff0_center"], 0.1], [0.707, 0., 0., 0.707])
    low_env.p.resetBasePositionAndOrientation(low_env.body_cliff1, [1.3, meta["cliff1_center"], 0.1], [0.707, 0., 0., 0.707])
    for i in range(len(meta["block_size"])):
        low_env._update_block(
            low_env.body_blocks[i], meta["block_size"][i], low_env.block_reset_pos[i], low_env.block_reset_orn[i])

    def set_qpos(qpos):
        for i in range(len(low_env.robot.motorIndices)):
            p.resetJointState(low_env.robot._robot, low_env.robot.motorIndices[i], qpos[i])

    for i in range(len(paths)):
        path = paths[i]
        if path is None:
            continue
        assert isinstance(path, dict)
        for key in path.keys():
            if path[key] is not None:
                for step in range(len(path[key])):
                    set_qpos(path[key][step])
                    time.sleep(0.05)
                if key == "release_finger":
                    time.sleep(1)


def replay_real(pkl_name):
    with open(pkl_name, "rb") as f:
        info = pickle.load(f)
    paths = info["paths"]
    meta = info["meta"]
    n_obj = len(meta["block_size"])
    ################################
    # TODO: record this in pkl file
    meta["sizes_per_step"] = [10, 10, 12, 12, 7]
    ################################
    print("meta info", meta)
    ans = input("Please verify scene configurations. Continue? [Y|n]")
    if ans != "Y":
        return
    
    arm = arm_set_up(ip="192.168.1.226", mode=0)
    set_tcp_load(arm, "gripper")
    # exit()
    
    valid_idx = 0
    for idx in range(0, len(paths)):
        total_path = paths[idx]
        if total_path is None:
            continue
        # ans = input("Continue? [Y|n]")
        # if ans != "Y":
        #     return
        for k in ["fetch_object", "close_finger", "change_pose", "release_finger", "lift_up", "move_back"]:
            if k == "change_pose":
                tcp_name = "gripper+obj" + str(int(meta["sizes_per_step"][valid_idx] * 2))
                set_tcp_load(arm, tcp_name)
            if k == "release_finger":
                set_tcp_load(arm, "gripper")
            if k in ["fetch_object", "change_pose", "lift_up", "move_back"]:
                print(k, arm.tcp_load)
                path = total_path[k]
                if path is None:
                    continue
                if arm.connected and arm.state != 4:
                    for idx, q in enumerate(path):
                        angles = list(q[:7])
                        #####################
                        # TODO: workaround
                        # Compensate for error in servo angle
                        angles[3] += 3 / 180 * np.pi
                        #####################
                        ret = arm.set_servo_angle(angle=angles, speed=20 / 180, is_radian=True,
                                                    wait=(idx == len(path) - 1))
                        print(idx, 'set_servo_angle {}, ret={}'.format(angles, ret))
                        # print(arm.get_servo_angle(is_radian=True))
                # if k == "fetch_object":
                #     ans = input("Continue?[Y|n]")
                #     if ans != "Y":
                #         return
            elif k == "close_finger":
                code = arm.set_gripper_position(450, wait=True)
                print('[wait]set gripper pos, code={}'.format(code))
            elif k == "release_finger":
                code = arm.set_gripper_position(850, wait=True)
                print('[wait]set gripper pos, code={}'.format(code))
        valid_idx += 1
if __name__ == "__main__":
    filename = "low_level_paths_3b_xarm.pkl"
    replay_sim(filename)
    # replay_real(filename)