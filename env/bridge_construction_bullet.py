import gym
from gym.utils import seeding
from gym import spaces

from env.bullet_rotations import quat_rot_vec, quat_mul, quat_diff
import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client as bc
import os, time
from typing import List
from collections import deque


BASIC_COLORS = [[1.0, 0, 0], [1, 1, 0], [0.2, 0.8, 0.8], [0.8, 0.2, 0.8], [0, 0, 0], [0.0, 0.0, 1.0], [0.5, 0.2, 0.0]]


def _out_of_reach(object_pos, cliff0_center, cliff1_center, object_size, cliff_size, cos_theta=0):
    # Out of reach will happen at reset,
    # and is within the range of action space (so the agent can throw a block to out-of-reach state)
    # TODO: after cliff configuration change
    assert len(cliff0_center) == 3
    if object_pos[1] + object_size[1] * cos_theta < cliff0_center[1] + cliff_size[2] or \
            object_pos[1] - object_size[1] * cos_theta > cliff1_center[1] - cliff_size[2]:
        return True
    if object_pos[0] < cliff0_center[0] - cliff_size[0] or object_pos[0] > cliff0_center[0] + cliff_size[0]:
        return True
    return False


def _is_feasible(n_std, n_paired, cliff_distance, rand_size: List, half_height=0.1):
    # print("n_std =", n_std, "n_paired =", n_paired, "rand_size =", rand_size, "cliff_distance =", cliff_distance)
    rand_size = rand_size.copy()
    if len(rand_size) < n_std + n_paired + 1:
        temp = n_std + n_paired + 1 - len(rand_size)
        delta = temp - temp // 2
        # if n_std >= delta:
        rand_size.extend([half_height] * min(delta, n_std))
        n_std -= min(delta, n_std)
        # else:
        #     return False
    n_h = min(n_std + n_paired + 1, len(rand_size))
    rand_size.sort()
    return 2 * (sum(rand_size[-n_h:])) - 0.02 > cliff_distance


def generate_block_length(num_blocks, cliff_distance, half_height=0.1, discrete=False, mode="split"):
    assert half_height > 0.04
    _n_trial = 0
    n_std = 0
    n_paired = 0
    rand_size = []
    max_n_trial = 2000
    while not _is_feasible(n_std, n_paired, cliff_distance, rand_size, half_height) and _n_trial < max_n_trial:
        result = []
        if mode == "split":
            # n_std = np.random.randint(0, num_blocks // 2 + 1)  # [0, num_blocks // 2]
            n_std = np.random.randint(0, min(1, num_blocks // 2 + 1))  # [0,]
            result.extend([half_height] * n_std)
            n_paired = np.random.randint(0, (num_blocks - n_std) // 2 + 1)  # [0, (num_blocks - n_std) // 2]
        else:
            n_std = num_blocks // 2
            result.extend([half_height] * n_std)
            n_paired = 0
        if discrete:
            first_part = np.random.choice(np.arange(int(half_height / 2 / 0.01) * 0.01 - 0.02,
                                                    int(half_height / 2 / 0.01 + 1) * 0.01 + 0.02, 0.01),
                                          size=n_paired)
        else:
            first_part = np.random.uniform(half_height / 2 - 0.02, half_height / 2 + 0.02, size=n_paired)
        second_part = half_height - first_part
        result.extend(first_part.tolist())
        result.extend(second_part.tolist())
        n_rand = num_blocks - n_std - n_paired * 2
        if mode == "split":
            if discrete:
                rand_size = np.random.choice(np.arange(int(half_height / 2 / 0.01) * 0.01,
                                                       int(half_height / 0.01 + 1) * 0.01, 0.01),
                                             size=n_rand).tolist()
            else:
                rand_size = np.random.uniform(half_height / 2, half_height, size=n_rand).tolist()
        else:
            if discrete:
                rand_size = np.random.choice(
                    np.arange(0.5 * half_height, 0.91 * half_height, half_height / 10), size=n_rand // 2
                ).tolist() + np.random.choice(
                    np.arange(1.1 * half_height, 1.21 * half_height, half_height / 10), size=n_rand - n_rand // 2
                ).tolist()
            else:
                # rand_size = np.random.uniform(half_height / 2, half_height, size=n_rand - n_rand // 2).tolist() + \
                #             np.random.uniform(half_height, half_height * 1.3, size=n_rand // 2).tolist()
                rand_size = np.random.uniform(half_height * 0.5, half_height * 0.9, size=n_rand // 2).tolist() + \
                            np.random.uniform(half_height * 1.1, half_height * 1.25, size=n_rand - n_rand // 2).tolist()
        result.extend(rand_size)
        _n_trial += 1
    if _n_trial >= max_n_trial:
        print("Warning: infeasible task")
    # print("n_std", n_std, "n_paired", n_paired)
    return result, (n_std, n_paired, cliff_distance, rand_size, half_height)


class PhysClientWrapper:
    """
    This is used to make sure each BulletRobotEnv has its own physicsClient and
    they do not cross-communicate.
    """
    def __init__(self, other, physics_client_id):
        self.other = other
        self.physicsClientId = physics_client_id

    def __getattr__(self, name):
        if hasattr(self.other, name):
            attr = getattr(self.other, name)
            if callable(attr):
                return lambda *args, **kwargs: self._wrap(attr, args, kwargs)
            return attr
        raise AttributeError(name)

    def _wrap(self, func, args, kwargs):
        kwargs["physicsClientId"] = self.physicsClientId
        return func(*args, **kwargs)


class RobotGymBaseEnv(gym.Env):
    def __init__(self, actionRepeat=10, timestep=1./240, render=False, init_qpos=None,
                 init_end_effector_pos=(1.0, 0.6, 0.4), init_end_effector_orn=(0, -np.pi, np.pi / 2),
                 useNullSpace=True, robot="ur"):
        self.actionRepeat = actionRepeat
        self.timestep = timestep
        self.init_qpos = init_qpos
        self.init_end_effector_pos = init_end_effector_pos
        self.init_end_effector_orn = init_end_effector_orn
        self.useNullSpace = useNullSpace
        self.robot_name = robot
        self._render = render

        self._setup_env()

    def _setup_env(self):
        if self._render:
            physics_client = bc.BulletClient(connection_mode=p.GUI)
        else:
            physics_client = bc.BulletClient(connection_mode=p.DIRECT)
        self.p = physics_client
        self._bullet_dataRoot = pybullet_data.getDataPath()

        self.seed()

        self.p.resetSimulation()
        self.p.setTimeStep(self.timestep)
        self.p.setGravity(0, 0, -10)
        self.p.setPhysicsEngineParameter(enableFileCaching=0)
        if self._render:
            self.p.resetDebugVisualizerCamera(1.5, -60, -25, [1.3, 0.6, 0.1])
        if self.robot_name == "ur":
            from env.robots import UR2f85Robot
            self.robot = UR2f85Robot(self.p, init_qpos=self.init_qpos, init_end_effector_pos=self.init_end_effector_pos,
                                     init_end_effector_orn=self.init_end_effector_orn, useOrientation=True,
                                     useNullSpace=self.useNullSpace)
        elif self.robot_name == "xarm":
            from env.robots import XArm7Robot
            self.robot = XArm7Robot(self.p)
        else:
            raise NotImplementedError
        self.p.stepSimulation()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset_sim(self):
        raise NotImplementedError

    def _get_obs(self):
        raise NotImplementedError

    def step_simulation(self):
        for _ in range(self.actionRepeat):
            self.p.stepSimulation()

    def reset(self):
        self._reset_sim()
        return self._get_obs()

    def step(self, action):
        raise NotImplementedError

    def render(self, mode='human', width=500, height=500):
        if mode == 'rgb_array':
            view_matrix = self.p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[1.3, 0.6, 0.1],
                                                                   distance=1.1,
                                                                   yaw=-60,
                                                                   pitch=-20,
                                                                   roll=0,
                                                                   upAxisIndex=2)
            proj_matrix = self.p.computeProjectionMatrixFOV(fov=60,
                                                            aspect=1.0,
                                                            nearVal=0.1,
                                                            farVal=100.0)
            (_, _, px, _, _) = self.p.getCameraImage(width=width,
                                                     height=height,
                                                     viewMatrix=view_matrix,
                                                     projectionMatrix=proj_matrix,
                                                     )
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (height, width, 4))

            rgb_array = rgb_array[: 450, :, :3]
            return rgb_array

    def __del__(self):
        self.p.disconnect()


class BulletBridgeConstructionLow(RobotGymBaseEnv):
    def __init__(self, num_blocks, random_size, discrete=False, mode="split", block_thickness=0.025,
                 cliff_thickness=0.1, cliff_height=0.05, render=False, need_visual=False, robot="ur"):
        self.num_blocks = num_blocks
        self.cur_num_blocks = num_blocks
        self.random_size = random_size
        self.discrete = discrete
        self.mode = mode
        self._cliff_thickness = cliff_thickness
        self._block_thickness = block_thickness
        self._cliff_height = cliff_height
        self.object_names = ['object{}'.format(i) for i in range(self.num_blocks)]
        self.robot_dim = None
        self.object_dim = None
        self.cliff0_center = -0.05
        self.cliff1_center = 1.25
        self.cliff0_boundary = None
        self.cliff1_boundary = None
        self.block_size = [None] * self.num_blocks
        self.cliff_size = None
        self.need_visual = need_visual
        self.all_collision_shapes = None
        self.all_visual_shapes = None
        self.restitution_range = (0.35, 0.65)
        self.friction_range = (0.5, 0.5)
        super(BulletBridgeConstructionLow, self).__init__(render=render, robot=robot)

    def _create_block(self, halfExtents, pos, orn, mass=0.2, rgba=None, vel=None, vela=None):
        col_id = self.p.createCollisionShape(self.p.GEOM_BOX, halfExtents=halfExtents)
        vis_id = self.p.createVisualShape(self.p.GEOM_BOX, halfExtents=halfExtents)
        body_id = self.p.createMultiBody(mass, col_id, vis_id, pos, orn)
        if rgba is not None:
            self.p.changeVisualShape(body_id, -1, rgbaColor=rgba)
        if vel is None:
            vel = [0, 0, 0]
        if vela is None:
            vela = [0, 0, 0]
        self.p.resetBaseVelocity(body_id, vel, vela)
        restitution = np.random.uniform(*self.restitution_range)
        friction = np.random.uniform(*self.friction_range)
        self.p.changeDynamics(body_id, -1, mass=1, restitution=restitution, lateralFriction=friction,
                              linearDamping=0.005)
        return body_id

    def _update_block(self, body_id, halfExtents, pos, orn, vel=None, vela=None):
        # The scale in creation time
        old_halfextents = np.array([self._block_thickness, self._cliff_height, self._block_thickness])
        scaling = np.array(halfExtents) / old_halfextents
        self.p.unsupportedChangeScaling(body_id, scaling)
        new_halfextents = np.array(self.p.getCollisionShapeData(body_id, -1)[0][3]) / 2
        assert np.linalg.norm(new_halfextents - np.array(halfExtents)) < 1e-3, (
        old_halfextents, new_halfextents, halfExtents, scaling)
        self.p.resetBasePositionAndOrientation(body_id, pos, orn)
        if vel is None:
            vel = [0., 0., 0.]
        if vela is None:
            vela = [0., 0., 0.]
        self.p.resetBaseVelocity(body_id, vel, vela)
        restitution = np.random.uniform(*self.restitution_range)
        friction = np.random.uniform(*self.friction_range)
        self.p.changeDynamics(body_id, -1, mass=1, restitution=restitution, lateralFriction=friction,
                              linearDamping=0.005)
        return body_id

    def _setup_env(self):
        super(BulletBridgeConstructionLow, self)._setup_env()
        planeid = self.p.loadURDF(os.path.join(self._bullet_dataRoot, "plane.urdf"), [0, 0, -0.795])
        tableid = self.p.loadURDF(os.path.join(self._bullet_dataRoot, "table/table.urdf"),
                                  [1.200000, 0.60000, -.625000],
                                  [0.000000, 0.000000, 0.707, 0.707])
        self.body_tables = [tableid]

        # Cliffs
        col_id = self.p.createCollisionShape(self.p.GEOM_BOX, halfExtents=[self._cliff_thickness, self._cliff_height, self._cliff_thickness])
        vis_id = self.p.createVisualShape(self.p.GEOM_BOX, halfExtents=[self._cliff_thickness, self._cliff_height, self._cliff_thickness])
        cliff_mass = 0  # in kg
        self.body_cliff0 = self.p.createMultiBody(cliff_mass, col_id, vis_id, [1.3, 0.3, self._cliff_height], [0.707, 0., 0., 0.707])
        self.body_cliff1 = self.p.createMultiBody(cliff_mass, col_id, vis_id, [1.3, 0.9, self._cliff_height], [0.707, 0., 0., 0.707])
        # Blocks
        self.all_collision_shapes = []
        self.all_visual_shapes = []
        for i in range(self.num_blocks):
            self.all_collision_shapes.append(
                self.p.createCollisionShape(self.p.GEOM_BOX, halfExtents=[
                    self._block_thickness, self._cliff_height, self._block_thickness]))
            self.all_visual_shapes.append(
                self.p.createVisualShape(self.p.GEOM_BOX, halfExtents=[
                    self._block_thickness, self._cliff_height, self._block_thickness
                ])
            )
        block_mass = 0.05
        self.body_blocks = []
        self.block_reset_pos = [np.array([0.9 + 0.15 * i, 0.0, self._block_thickness]) for i in range(2)] + \
                               [np.array([0.9 + 0.15 * (i - 2), 0.26, self._block_thickness]) for i in range(2, 4)] + \
                               [np.array([0.9 + 0.15 * (i - 4), 0.94, self._block_thickness]) for i in range(4 ,6)] + \
                               [np.array([0.9 + 0.15 * (i - 6), 1.2, self._block_thickness]) for i in range(6, 7)]
        self.block_reset_orn = [np.array([0., 0., 0., 1.])] * self.num_blocks
        for i in range(self.num_blocks):
            self.body_blocks.append(
                self.p.createMultiBody(block_mass, self.all_collision_shapes[i], self.all_visual_shapes[i],
                                       self.block_reset_pos[i], self.block_reset_orn[i]))
            self.p.changeVisualShape(self.body_blocks[-1], -1, rgbaColor=BASIC_COLORS[i] + [1.])
        for _ in range(100):
            self.p.stepSimulation()

    def set_cur_num_blocks(self, cur_num_blocks):
        self.cur_num_blocks = cur_num_blocks

    def set_cliff_centers(self, left, right):
        self.cliff0_center = left
        self.cliff1_center = right

    def _reset_sim(self):
        self.robot.reset()
        self.p.resetBasePositionAndOrientation(
            self.body_cliff0, [1.3, self.cliff0_center, self._cliff_height], [0.707, 0., 0., 0.707]
        )
        self.p.resetBasePositionAndOrientation(
            self.body_cliff1, [1.3, self.cliff1_center, self._cliff_height], [0.707, 0., 0., 0.707]
        )
        self.p.stepSimulation()
        self.cliff0_boundary = self.cliff0_center + self._cliff_thickness
        self.cliff1_boundary = self.cliff1_center - self._cliff_thickness
        # print(self.p.getVisualShapeData(self.body_cliff0, -1)[0])
        self.cliff_size = np.array(self.p.getCollisionShapeData(self.body_cliff0, -1)[0][3]) / 2

        if self.random_size:
            block_lengths, _info = generate_block_length(self.cur_num_blocks, self.cliff1_boundary - self.cliff0_boundary,
                                                         half_height=self.cliff_size[1], discrete=self.discrete,
                                                         mode=self.mode)
            n_std, n_paired, cliff_distance, rand_size, _ = _info
        else:
            assert abs(self.cliff_size[1] - self._cliff_height) < 1e-5
            block_lengths = [self.cliff_size[1]] * self.cur_num_blocks
        if len(self.body_blocks) and (not self.need_visual):
            assert len(self.body_blocks) == self.num_blocks
            for i in range(self.cur_num_blocks):
                self._update_block(self.body_blocks[i],
                                   [self._block_thickness, block_lengths[i], self._block_thickness],
                                   self.block_reset_pos[i], self.block_reset_orn[i])
                self.block_size[i] = np.array([self._block_thickness, block_lengths[i], self._block_thickness])
            for i in range(self.cur_num_blocks, self.num_blocks):
                initial_pos = np.array([0., 10.0 + i, 0.1])
                initial_orn = np.array([0., 0., 0., 1.])
                self._update_block(self.body_blocks[i],
                                   [self._block_thickness, self._cliff_height, self._block_thickness],
                                   initial_pos, initial_orn)
        else:
            for i in range(self.cur_num_blocks):
                self.p.removeBody(self.body_blocks[i])
                self.body_blocks[i] = self._create_block(
                    [self._block_thickness, block_lengths[i], self._block_thickness],
                    self.block_reset_pos[i], self.block_reset_orn[i], rgba=BASIC_COLORS[i] + [1.])
                self.block_size[i] = np.array([self._block_thickness, block_lengths[i], self._block_thickness])
            for i in range(self.cur_num_blocks, self.num_blocks):
                initial_pos = np.array([0.0, 10.0 + i, 0.1])
                initial_orn = np.array([0., 0., 0., 1.])
                self.p.removeBody(self.body_blocks[i])
                self.body_blocks[i] = self._create_block(
                    [self._block_thickness, self._cliff_height, self._block_thickness],
                    initial_pos, initial_orn)
        for _ in range(10):
            self.p.stepSimulation()
        return True

    def sync_attr(self):
        # Set env-related attributes
        pos, orn = self.p.getBasePositionAndOrientation(self.body_cliff0)
        self.cliff0_center = pos[1]
        pos, orn = self.p.getBasePositionAndOrientation(self.body_cliff1)
        self.cliff1_center = pos[1]
        self.cliff0_boundary = self.cliff0_center + self._cliff_thickness
        self.cliff1_boundary = self.cliff1_center - self._cliff_thickness
        self.cliff_size = np.array(self.p.getCollisionShapeData(self.body_cliff0, -1)[0][3]) / 2
        for i in range(self.cur_num_blocks):
            self.block_size[i] = np.array(self.p.getCollisionShapeData(self.body_blocks[i], -1)[0][3]) / 2

    def _get_obs(self):
        grip_pos, grip_orn, grip_velp, gripper_state, gripper_vel = self.robot.get_observation()
        dt = self.timestep * self.actionRepeat
        grip_velp = np.array(grip_velp) * dt
        gripper_vel = gripper_vel * dt  # change to a scalar if the gripper is made symmetric

        obs = np.concatenate([
            grip_pos,
            grip_orn,
            gripper_state,
            grip_velp,
            gripper_vel,
        ])
        if self.robot_dim is None:
            self.robot_dim = len(obs)

        cliff0_center, _ = self.p.getBasePositionAndOrientation(self.body_cliff0)
        cliff1_center, _ = self.p.getBasePositionAndOrientation(self.body_cliff1)
        cliff_size = np.array(self.p.getCollisionShapeData(self.body_cliff0, -1)[0][3]) / 2
        # objects
        for i in range(self.cur_num_blocks):
            object_i_pos, object_i_quat = self.p.getBasePositionAndOrientation(self.body_blocks[i])
            object_i_pos = np.array(object_i_pos)
            # rotations
            object_i_rot = self.p.getEulerFromQuaternion(convert_symmetric_quat(object_i_quat))
            object_i_rot = np.array(object_i_rot)
            # velocities
            object_i_velp, object_i_velr = self.p.getBaseVelocity(self.body_blocks[i])
            object_i_velp = np.array(object_i_velp)
            object_i_velr = np.array(object_i_velr)
            object_i_velp *= dt
            object_i_velr *= dt
            object_i_velp -= grip_velp
            object_i_size = np.array(self.p.getCollisionShapeData(self.body_blocks[i], -1)[0][3]) / 2
            cliff_height = np.array([cliff_size[1]])
            object_i_type = np.array([0])
            if _out_of_reach(object_i_pos, cliff0_center, cliff1_center, object_i_size, cliff_size,
                             cos_theta=abs(quat_rot_vec(np.array(object_i_quat), np.array([0., 1., 0.]))[1])):
                object_i_pos = -np.ones(3)
                object_i_rot = object_i_velp = object_i_velr = np.zeros(3)
            obs = np.concatenate([obs, object_i_pos, object_i_rot, object_i_velp, object_i_velr, object_i_size,
                                  cliff_height - object_i_size[1], object_i_type]) \
                if self.random_size else np.concatenate([obs, object_i_pos, object_i_rot, object_i_velp, object_i_velr, object_i_type])
            if self.object_dim is None:
                self.object_dim = len(obs) - self.robot_dim
        for i in range(self.cur_num_blocks, self.num_blocks):
            object_i_pos = object_i_rot = object_i_velp = object_i_velr = np.zeros(3)
            object_i_size = np.zeros(3)
            cliff_height = np.array([0])
            object_i_type = np.array([0])
            obs = np.concatenate([obs, object_i_pos, object_i_rot, object_i_velp, object_i_velr, object_i_size,
                                  cliff_height - object_i_size[1], object_i_type]) \
                if self.random_size else np.concatenate([obs, object_i_pos, object_i_rot, object_i_velp, object_i_velr, object_i_type])
        cliff0_pos = cliff0_center
        # cliff_rot = rotations.quat2euler([1., 0., 0., 0.])
        cliff_rot = np.array(self.p.getEulerFromQuaternion([0.707, 0., 0., 0.707]))
        cliff_velp = cliff_velr = np.zeros(3)
        object_size = cliff_size
        object_type = np.array([1])
        obs = np.concatenate([obs, cliff0_pos, cliff_rot, cliff_velp, cliff_velr, object_size, np.zeros(1), object_type]) \
            if self.random_size else np.concatenate([obs, cliff0_pos, cliff_rot, cliff_velp, cliff_velr, object_type])
        cliff1_pos = cliff1_center
        obs = np.concatenate([obs, cliff1_pos, cliff_rot, cliff_velp, cliff_velr, object_size, np.zeros(1), object_type]) \
            if self.random_size else np.concatenate([obs, cliff1_pos, cliff_rot, cliff_velp, cliff_velr, object_type])
        return obs

    def get_obs(self):
        return self._get_obs()

    def get_body_pos_and_orn(self, body_idx):
        pos, orn = self.p.getBasePositionAndOrientation(self.body_blocks[body_idx])
        return np.array(pos), np.array(orn)


class BulletBridgeConstructionHigh(gym.Env):
    def __init__(self, num_blocks, hard_ratio=0.0, action_2d=True, random_size=False, min_num_blocks=3, discrete=False,
                 random_mode="split", action_scale=0.4, block_thickness=0.025, narrow_z=True, restart_rate=0.,
                 cliff_thickness=0.5, cliff_height=0.05, noop=False, render=False, need_visual=False, primitive=False,
                 compute_path=False, robot="ur", force_scale=0, adaptive_primitive=False):
        self.num_blocks = num_blocks  # This is the max number of objects throughout training.
        self.cur_max_blocks = num_blocks  # This is the adaptive current max number of objects
        self.min_num_blocks = min_num_blocks
        self.hard_ratio = [hard_ratio] * self.num_blocks
        self.action_2d = action_2d
        self.has_cliff = True
        self.skyline_dim = 38
        self.narrow_z = narrow_z
        self.rotation_range = np.array([-0.5 * np.pi, 0.5 * np.pi])
        self.restart_rate = restart_rate
        self.noop = noop
        self.force_scale = force_scale
        self.cur_force_scale = 0
        self.env = self._set_env_low(random_size, discrete, random_mode, block_thickness, cliff_thickness, cliff_height,
                                     render, need_visual, robot)
        self.env.reset()
        self.step_counter = 0
        self.object_dim = None  # will be set in _get_obs() call
        # TODO: 2d case
        if self.action_2d:
            self.action_space = spaces.Box(low=np.array([0, -1., -1., -1.]),
                                           high=np.array([self.num_blocks - 1, 1., 1., 1.]), dtype='float32')
        else:
            self.action_space = spaces.Box(low=np.array([0, -1., -1., -1., -1., -1., -1.]),
                                           high=np.array([self.num_blocks - 1, 1., 1., 1., 1., 1., 1.]), dtype='float32')  # TODO: action[0] is discrete??
        self.action_scale = action_scale
        obs_shape = self._get_obs().shape
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_shape[0],), dtype='float32')
        self.test_threshold = 0.25  # Manually set it according to the height of standard block, will be overwritten
        self.state_replay_buffer = deque(maxlen=10000)
        self.compute_path = compute_path
        self.primitive = None
        if primitive:
            self._set_primitive(force_scale, robot)
        self.adaptive_primitive = adaptive_primitive
        self.detailed_sr = [0.0] * self.num_blocks

    def _set_env_low(self, random_size, discrete, random_mode, block_thickness, cliff_thickness, cliff_height, render,
                     need_visual, robot):
        return BulletBridgeConstructionLow(self.num_blocks, random_size, discrete, random_mode, block_thickness,
                                           cliff_thickness, cliff_height, render, need_visual, robot)

    def _set_primitive(self, force_scale, robot):
        # Create low level env for planning
        self.plan_low = BulletBridgeConstructionLow(self.num_blocks, self.env.random_size, self.env.discrete,
                                                    self.env.mode, self.env._block_thickness, self.env._cliff_thickness,
                                                    self.env._cliff_height, False, False, robot)
        from env.mp_interface_bullet import Planner, Executor, Primitive
        planner = Planner(self.plan_low.robot, self.plan_low.p, smooth_path=self.compute_path)
        executor = Executor(self.env.robot, self.env.p, ctrl_mode="teleport", record=False)
        self.primitive = Primitive(planner, executor, self.env.body_blocks, self.env.body_cliff0, self.env.body_cliff1,
                                   self.plan_low.body_blocks, self.plan_low.body_tables,
                                   teleport_arm=not self.compute_path, force_scale=force_scale)

    def reset(self):
        # with some probability, restart from a replay buffer.
        if len(self.state_replay_buffer) and self.env.np_random.rand() < self.restart_rate:
            self.restart_from_buffer = True
            state_dict = self.state_replay_buffer.popleft()
            self.set_state(state_dict)
        else:
            self.restart_from_buffer = False
            if self.env.random_size and self.env.mode == "long":
                cur_num_blocks = np.random.choice(list(filter(lambda v: v % 2 == 1, np.arange(self.min_num_blocks, self.cur_max_blocks + 1))))
            else:
                cur_num_blocks = np.random.choice(np.arange(self.min_num_blocks, self.cur_max_blocks + 1))
            self.env.set_cur_num_blocks(cur_num_blocks)
            if self.env.random_size:
                if self.env.mode == "split":
                    if cur_num_blocks % 2 == 0:
                        cliff_max_distance = self.env.cliff_size[1] * (cur_num_blocks - 2)
                        if cur_num_blocks == 4:
                            cliff_max_distance = 3.5 * self.env.cliff_size[1]  # A special case.
                    else:
                        cliff_max_distance = self.env.cliff_size[1] * (cur_num_blocks - 1)
                    cliff_min_distance = self.env.cliff_size[1]
                else:
                    assert cur_num_blocks % 2 == 1
                    cliff_max_distance = self.env.cliff_size[1] * (cur_num_blocks + 0.5)
                    if cur_num_blocks == 5:
                        cliff_max_distance += 0.5 * self.env.cliff_size[1]
                    cliff_min_distance = self.env.cliff_size[1] + 0.05
            else:
                if cur_num_blocks % 2 == 0:
                    cliff_max_distance = self.env.cliff_size[1] * (cur_num_blocks - 0.5)
                    cliff_min_distance = self.env.cliff_size[1]
                else:
                    cliff_max_distance = self.env.cliff_size[1] * cur_num_blocks
                    cliff_min_distance = self.env.cliff_size[1]
            gap = 2 * self.env.cliff_size[1]
            # adapt cliff distance based on hard_ratio
            if np.random.uniform() < self.hard_ratio[cur_num_blocks - 1]:
                _cliff_distance = np.random.uniform(cliff_max_distance - gap, cliff_max_distance)
            else:
                _cliff_distance = np.random.uniform(cliff_min_distance, cliff_max_distance)
            _noise = np.random.uniform(-self.env.cliff_size[1], self.env.cliff_size[1])
            cliff0_center = 0.6 - self.env.cliff_size[2] - _cliff_distance / 2 + _noise
            cliff1_center = 0.6 + self.env.cliff_size[2] + _cliff_distance / 2 + _noise

            self.env.set_cliff_centers(cliff0_center, cliff1_center)
            self.env.reset()
        self.step_counter = 0
        obs = self._get_obs()
        if self.primitive is not None:
            self.primitive.align_at_reset()
        return obs

    def _calculate_skyline(self):
        # TODO: we must make sure the robot arm does not conflict with skyline detection
        if self.env.random_size:
            num_ray = int(round((self.env.cliff1_boundary - self.env.cliff0_boundary) / 0.02))
        else:
            num_ray = int(round((self.env.cliff1_boundary - self.env.cliff0_boundary) / 0.05))
        skyline = np.ones(self.skyline_dim)
        assert self.skyline_dim >= num_ray
        ray_dist_buf = []
        collision_id_buf = []
        ray_array = np.array([0., 0., -1.])
        for ray_idx in range(num_ray):
            start_point = np.array([1.3,
                                    self.env.cliff0_boundary - 1e-3 + (
                                                self.env.cliff1_boundary - self.env.cliff0_boundary + 2e-3) / (
                                            num_ray - 1) * ray_idx, 1.0])
            to_point = start_point + ray_array
            res = self.env.p.rayTest(start_point, to_point)[0]
            obj_id, link_id, fraction, hit_pos = res[0], res[1], res[2], res[3]
            ray_dist = fraction * np.linalg.norm(ray_array)
            ray_dist_buf.append(ray_dist)
            collision_id_buf.append(obj_id)
            assert obj_id != self.env.robot._robot, (obj_id, link_id, hit_pos)
            assert ray_dist > 0, (ray_dist, self.env.cliff0_boundary, self.env.cliff1_boundary, start_point,
                                  self.env.p.getBasePositionAndOrientation(obj_id), obj_id, link_id, hit_pos)
        skyline[:num_ray] = np.linalg.norm(ray_array) - np.array(ray_dist_buf)
        return skyline, collision_id_buf

    def get_state(self):
        qpos = [np.concatenate(self.env.p.getBasePositionAndOrientation(self.env.body_blocks[i]))
                for i in range(len(self.env.body_blocks))]
        qpos = np.stack(qpos)
        vel = [np.concatenate(self.env.p.getBaseVelocity(self.env.body_blocks[i]))
               for i in range(len(self.env.body_blocks))]
        vel = np.stack(vel)
        geom_size = [np.array(self.env.p.getCollisionShapeData(self.env.body_blocks[i], -1)[0][3]) / 2
                     for i in range(self.num_blocks)]
        geom_size = np.stack(geom_size)
        mj_state = dict(qpos=qpos.copy(), qvel=vel.copy(), size=geom_size.copy())
        other_state = dict(cliff0_pos=np.array(self.env.p.getBasePositionAndOrientation(self.env.body_cliff0)[0]),
                           cliff0_orn=np.array(self.env.p.getBasePositionAndOrientation(self.env.body_cliff0)[1]),
                           cliff1_pos=np.array(self.env.p.getBasePositionAndOrientation(self.env.body_cliff1)[0]),
                           cliff1_orn=np.array(self.env.p.getBasePositionAndOrientation(self.env.body_cliff1)[1]),
                           cliff0_size=np.array(self.env.p.getCollisionShapeData(self.env.body_cliff0, -1)[0][3]) / 2,
                           cliff1_size=np.array(self.env.p.getCollisionShapeData(self.env.body_cliff1, -1)[0][3]) / 2,
                           cur_num_blocks=self.env.cur_num_blocks, skyline=self._calculate_skyline()[0],
                           )
        other_state.update({'object%d' % i: np.array(self.env.p.getCollisionShapeData(self.env.body_blocks[i], -1)[0][3]) / 2 for i in range(self.num_blocks)})
        return dict(mj_state=mj_state, other_state=other_state)

    def set_state(self, state_dict):
        self.env.p.resetBasePositionAndOrientation(self.env.body_cliff0, state_dict["other_state"]["cliff0_pos"],
                                                   state_dict["other_state"]["cliff0_orn"])
        self.env.p.resetBasePositionAndOrientation(self.env.body_cliff1, state_dict["other_state"]["cliff1_pos"],
                                                   state_dict["other_state"]["cliff1_orn"])
        # Update or create blocks
        if len(self.env.body_blocks) and not self.env.need_visual:
            assert len(self.env.body_blocks) == self.num_blocks
            for i in range(self.num_blocks):
                self.env._update_block(
                    self.env.body_blocks[i], state_dict["mj_state"]["size"][i], state_dict["mj_state"]["qpos"][i][:3],
                    state_dict["mj_state"]["qpos"][i][3:], vel=state_dict["mj_state"]["qvel"][i][:3],
                    vela=state_dict["mj_state"]["qvel"][i][3:])
        else:
            for i in range(self.num_blocks):
                self.env.p.removeBody(self.env.body_blocks[i])
                self.env.body_blocks[i] = self.env._create_block(
                    state_dict["mj_state"]["size"][i], state_dict["mj_state"]["qpos"][i][:3],
                    state_dict["mj_state"]["qpos"][i][3:], rgba=BASIC_COLORS[i] + [1.],
                    vel=state_dict["mj_state"]["qvel"][i][:3], vela=state_dict["mj_state"]["qvel"][i][3:])
        self.env.set_cur_num_blocks(state_dict['other_state']['cur_num_blocks'])
        # sync env attributes
        self.env.sync_attr()

    def add_restart_states(self, state_dict):
        # state_dict is list of dict
        if isinstance(state_dict, np.ndarray):
            state_dict = state_dict.tolist()
        if not isinstance(state_dict, list):
            state_dict = [state_dict]
        self.state_replay_buffer.extend(state_dict)
    
    def clear_state_replay(self):
        self.state_replay_buffer.clear()
    
    def _get_obs(self):
        raw_obs = self.env.get_obs()
        objects_obs = raw_obs[self.env.robot_dim: self.env.robot_dim + self.env.object_dim * self.num_blocks]
        objects_obs = np.concatenate([objects_obs[self.env.object_dim * i: self.env.object_dim * (i + 1)] for i in range(self.num_blocks)])
        if self.object_dim is None:
            self.object_dim = len(objects_obs) // self.num_blocks
        # add cliff as objects
        cliffs_objects_obs = raw_obs[self.env.robot_dim + self.env.object_dim * self.num_blocks:]
        obs = np.concatenate([objects_obs, cliffs_objects_obs])
        return obs
    
    def get_obs(self):
        return self._get_obs()
    
    def set_cur_max_objects(self, cur_max_objects):
        self.cur_max_blocks = cur_max_objects

    def set_min_num_objects(self, min_num_blocks):
        self.min_num_blocks = min_num_blocks
    
    def get_cur_num_objects(self):
        return self.env.cur_num_blocks

    def get_cliff_height(self):
        return self.env.cliff_size[1]

    def get_block_thickness(self):
        return self.env.block_size[0][2]

    def get_cliff_pos(self, index: int):
        if index == 0:
            return np.array([1.3, self.env.cliff0_center, self.env.cliff_size[1]])
        if index == 1:
            return np.array([1.3, self.env.cliff1_center, self.env.cliff_size[1]])
        raise RuntimeError

    def get_block_reset_pos(self, index: int):
        return self.env.block_reset_pos[index].copy()

    def get_block_reset_orn(self, index: int):
        return self.env.block_reset_orn[index].copy()
    
    def set_force_scale(self, scale):
        self.cur_force_scale = scale

    def set_success_rate(self, detailed_sr):
        self.detailed_sr = detailed_sr

    def enable_recording(self):
        assert self.primitive is not None
        import shutil
        if os.path.exists("video_tmp"):
            shutil.rmtree("video_tmp")
        os.makedirs("video_tmp", exist_ok=True)
        self.primitive.executor.record = True

    def reset_scene(self, obj_poses, sizes, cliff0_center, cliff1_center, servo_angles):
        # Set the environment according to sensor results
        if servo_angles is not None:
            assert len(servo_angles) == self.env.robot.ndof
            for j in range(len(servo_angles)):
                self.env.p.resetJointState(self.env.robot._robot, self.env.robot.motorIndices[j], servo_angles[j])
        self.env.p.resetBasePositionAndOrientation(
            self.env.body_cliff0, np.array([1.3, cliff0_center, self.env._cliff_height]),
            np.array([0.707, 0, 0, 0.707])
        )
        self.env.p.resetBasePositionAndOrientation(
            self.env.body_cliff1, np.array([1.3, cliff1_center, self.env._cliff_height]),
            np.array([0.707, 0., 0., 0.707])
        )
        cur_num_blocks = len(obj_poses)
        if sizes is None:
            for i in range(cur_num_blocks):
                self.env.p.resetBasePositionAndOrientation(self.env.body_blocks[i], obj_poses[i][:3], obj_poses[i][3:])
        else:
            for i in range(cur_num_blocks):
                self.env.p.removeBody(self.env.body_blocks[i])
                self.env.body_blocks[i] = self.env._create_block(
                    sizes[i], obj_poses[i][:3], obj_poses[i][3:], rgba=BASIC_COLORS[i] + [1.]
                )
        self.env.set_cur_num_blocks(cur_num_blocks)
        self.env.sync_attr()
        if self.primitive is not None:
            self.primitive.align_at_reset()

    def convert_action(self, action):
        if self.action_2d:
            assert action.shape == (4,)
        else:
            assert action.shape == (7,)
        action = action.copy()
        idx = int(action[0])
        if self.noop:
            assert -1 <= idx < self.env.cur_num_blocks
        else:
            assert 0 <= idx < self.env.cur_num_blocks

        if idx >= 0:
            y_pos = 0.6
            z_scale = self.env._cliff_height if self.narrow_z else self.action_scale
            if self.action_2d:
                target_pos = np.array([1.3, y_pos + action[1] * self.action_scale, 2 * self.env._cliff_height + action[2] * z_scale])
                _theta_div_pi = (self.rotation_range[1] - self.rotation_range[0]) / (2 * np.pi) * action[3] \
                                + (self.rotation_range[0] + self.rotation_range[1]) / (2 * np.pi)
                target_orn = np.concatenate([[0., 0.], [_theta_div_pi]])  # theta/pi
            else:
                target_pos = action[1: 4] * self.action_scale + np.array([1.3, y_pos, 2 * self.env._cliff_height])
                target_pos[0] = 1.3
                target_orn = action[4: 7]  # (alpha/pi, beta/pi, theta/pi)
                target_orn[0: 2] = 0.
            out_of_reach = False
            if _out_of_reach(target_pos, self.get_cliff_pos(0), self.get_cliff_pos(1), self.env.block_size[idx],
                             self.env.cliff_size, cos_theta=abs(np.cos(target_orn[2] * np.pi))):
                target_pos = self.get_block_reset_pos(idx)
                target_orn = np.array([0., 0., 0.])
                out_of_reach = True
            return idx, target_pos, target_orn, out_of_reach
        return None

    def step(self, action, render=False):
        info = {'skyline': self._calculate_skyline()[0], 'restart_from_buffer': self.restart_from_buffer}
        ret = self.convert_action(action)
        start_pos = np.zeros((self.env.cur_num_blocks, 3))
        start_rot = np.tile(np.array([[0., 0., 0., 1.]]), [self.env.cur_num_blocks, 1])
        stable_pos = start_pos
        stable_rot = start_rot
        if ret is not None:
            idx, target_pos, target_orn, out_of_reach = ret
            info['out_of_reach'] = out_of_reach

            alpha, beta, theta = target_orn * np.pi
            obj_quaternion = np.array([np.cos(alpha) * np.cos(beta) * np.sin(theta / 2),
                                       np.cos(alpha) * np.sin(beta) * np.sin(theta / 2),
                                       np.sin(alpha) * np.sin(theta / 2), np.cos(theta / 2)])
            info['action_pos'] = target_pos.copy()
            info['action_quat'] = obj_quaternion.copy()
            if self.primitive:
                info['low_level_result'] = 0
            if self.primitive is None or \
                    (self.adaptive_primitive and np.random.uniform() > self.detailed_sr[self.env.cur_num_blocks - 1]):
                # Teleport object
                info['low_level_path'] = {"teleport_idx": idx, "teleport_pos": target_pos,
                                          "teleport_quat": obj_quaternion}
                self.env.p.resetBasePositionAndOrientation(self.env.body_blocks[idx], target_pos, obj_quaternion)
                start_pos_and_rot = [self.env.p.getBasePositionAndOrientation(self.env.body_blocks[i]) for i in
                                     range(self.env.cur_num_blocks)]
                start_pos, start_rot = zip(*start_pos_and_rot)
                start_pos, start_rot = map(lambda x: np.asarray(x), [start_pos, start_rot])
                # Keep robot fixed
                self.env.robot.reset_qpos()
                if self.cur_force_scale > 0:
                    # inject noise
                    self.env.p.applyExternalForce(self.env.body_blocks[idx], -1,
                                                  forceObj=np.random.normal(0., self.cur_force_scale, size=(3,)),
                                                  posObj=(0, 0, 0), flags=self.env.p.LINK_FRAME)
                self._sim_until_stable(render)
                stable_pos_and_rot = [self.env.p.getBasePositionAndOrientation(self.env.body_blocks[i]) for i in
                                      range(self.env.cur_num_blocks)]
                stable_pos, stable_rot = zip(*stable_pos_and_rot)
                stable_pos, stable_rot = map(lambda x: np.asarray(x), [stable_pos, stable_rot])
            else:
                _state = self.env.p.saveState()
                res, path = self.primitive.move_one_object(idx, target_pos, obj_quaternion)
                info.update({'low_level_result': res, 'low_level_path': path})
                if res == 0:
                    start_pos_and_rot = [self.env.p.getBasePositionAndOrientation(self.env.body_blocks[i]) for i in
                                         range(self.env.cur_num_blocks)]
                    start_pos, start_rot = zip(*start_pos_and_rot)
                    start_pos, start_rot = map(lambda x: np.asarray(x), [start_pos, start_rot])
                    stable_pos, stable_rot = start_pos, start_rot
                else:
                    # If low-level fails, do no-op
                    self.env.p.restoreState(stateId=_state)
                    joint_states = self.env.p.getJointStates(self.env.robot._robot, self.env.robot.motorIndices)
                    servo_angles = [item[0] for item in joint_states]
                    self.env.p.setJointMotorControlArray(self.env.robot._robot, self.env.robot.motorIndices,
                                                         self.env.p.POSITION_CONTROL, servo_angles)
                    self.env.p.stepSimulation()
                self.env.p.removeState(_state)
        else:
            if self.primitive:
                info.update({'low_level_result': 0,
                             "action_pos": np.array([-1, -1, -1]), "action_quat": np.array([-1, -1, -1, -1])
                             })
        info.update({'cur_num_objects': self.get_cur_num_objects()})
        reward = self.compute_reward(info, start_pos, start_rot, stable_pos, stable_rot, render=render)
        done = False
        info.update({'next_skyline': self._calculate_skyline()[0]})
        self.step_counter += 1
        next_obs = self._get_obs()
        return next_obs, reward, done, info

    def _sim_until_stable(self, render):
        count = 0
        for _ in range(100):
            self.env.p.stepSimulation()
            count += 1
            if render:
                self.env.render()
                time.sleep(0.05)
        while count < 500 and np.linalg.norm(
                np.concatenate([self.env.p.getBaseVelocity(self.env.body_blocks[i])[0]
                                for i in range(self.env.cur_num_blocks)])) > 1e-3:
            for _ in range(50):
                self.env.p.stepSimulation()
                count += 1
                if render:
                    self.env.render()
                    time.sleep(0.05)

    def compute_reward(self, info=None, start_pos=None, start_rot=None, stable_pos=None, stable_rot=None, render=False):
        # should not consider those out-of-reach objects
        dummy_mask = np.array([_out_of_reach(start_pos[i], self.get_cliff_pos(0), self.get_cliff_pos(1),
                                             self.env.block_size[i], self.env.cliff_size,
                                             cos_theta=abs(quat_rot_vec(start_rot[i], np.array([0., 1., 0.]))[1]))
                               for i in range(self.env.cur_num_blocks)])
        # Distance between two quaternions
        d_pos = [np.linalg.norm(start_pos[i] - stable_pos[i]) for i in np.where(dummy_mask == 0)[0]]
        d_rot = [2 * np.arccos(np.clip(quat_diff(start_rot[i], stable_rot[i])[-1], -1, 1)) for i in np.where(dummy_mask == 0)[0]]
        mean_dpos = np.mean(d_pos) if len(d_pos) else 0
        mean_drot = np.mean(d_rot) if len(d_rot) else 0
        info['position_shift'] = mean_dpos
        info['rotation_shift'] = mean_drot
        # Run ray collision experiment
        self.test_threshold = 2 * self.env.cliff_size[1] + 2 * self.env.block_size[0][0]
        if self.env.random_size:
            num_ray = int(round((self.env.cliff1_boundary - self.env.cliff0_boundary) / 0.02))
        else:
            num_ray = int(round((self.env.cliff1_boundary - self.env.cliff0_boundary) / 0.05))
        assert num_ray > 1
        building_blocks_id = self.env.body_blocks
        n_construction = 0
        cur_height = 0
        skyline, collision_geoms = self._calculate_skyline()
        assert len(collision_geoms) == num_ray
        for ray_idx in range(len(collision_geoms)):
            if collision_geoms[ray_idx] in building_blocks_id and skyline[ray_idx] > self.test_threshold - 0.01:
                n_construction += 1
            cur_height += min(skyline[ray_idx], self.test_threshold) / len(collision_geoms)

        success = (n_construction == num_ray)
        info['construction'] = n_construction / num_ray
        info['remaining_height'] = cur_height
        info['is_success'] = success
        return 0.0

    def render(self, mode='human'):
        width = height = 500
        return self.env.render(mode, width, height)

    def set_hard_ratio(self, hard_ratio, num_block):
        self.hard_ratio[num_block - 1] = hard_ratio

    def get_hard_ratio(self):
        return self.hard_ratio


def convert_symmetric_quat(quat):
    quat = np.array(quat)
    long_axis = quat_rot_vec(quat, np.array([0, 1, 0]))
    short_axis = quat_rot_vec(quat, np.array([0, 0, 1]))
    multipliers = [np.concatenate([np.sin(np.pi / 2 * i / 2) * long_axis, [np.cos(np.pi / 2 * i / 2)]]) for i in range(4)]
    multipliers += [quat_mul(q, np.concatenate([short_axis, [0.]])) for q in multipliers]
    all_quats = [quat_mul(q, quat) for q in multipliers]
    all_quats = [-q if q[-1] < 0 else q for q in all_quats]
    # TODO: which criterion? should return unique result for any quat in the symmetries
    cos_half = [q[-1] for q in all_quats]
    return all_quats[np.argmax(cos_half)]


def test_raycast():
    client = p.connect(p.GUI)
    my_p = PhysClientWrapper(p, client)
    _bullet_dataRoot = pybullet_data.getDataPath()
    planeid = my_p.loadURDF(os.path.join(_bullet_dataRoot, "plane.urdf"), [0., 0., -1.])
    col_id = my_p.createCollisionShape(my_p.GEOM_BOX, halfExtents=[0.1, 0.2, 0.1])
    vis_id = my_p.createVisualShape(my_p.GEOM_BOX, halfExtents=[0.1, 0.2, 0.1])
    mass = 0.0
    body_box1 = my_p.createMultiBody(mass, col_id, vis_id, [-0.5, 0.3, 0.1], [0.707, 0., 0., 0.707])
    body_box2 = my_p.createMultiBody(mass, col_id, vis_id, [0.5, -0.3, 0.1], [0.707, 0., 0., 0.707])
    # my_p.resetBasePositionAndOrientation(body_box1, [-1.0, 0.3, 0.1], [0.707, 0., 0., 0.707])
    print("plane id", planeid, "body id", body_box1, body_box2)
    print(my_p.getBasePositionAndOrientation(body_box1))
    for _ in range(10):
        my_p.stepSimulation()
    res = my_p.rayTest([0.5, 0.3, 1.0], [0.5, 0.3, -0.05])
    print(res)
    res = my_p.rayTest([-0.5, 0.3, 1.0], [-0.5, 0.3, -0.05])
    print(res)
    time.sleep(5)
