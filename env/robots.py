import os, time
import math
import numpy as np
from env.bullet_rotations import quat2euler, quat_mul, euler2quat, quat2mat, is_rotation_mat, mat2quat


KINOVA_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'kinova_description', 'urdf')
UR_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'ur_description', 'urdf')
XARM_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'xarm_description', 'urdf')


class ArmRobot(object):
    '''
    Common base for 6 DoF Arm
    '''
    def __init__(self, physics_client, filename="", urdfrootpath=None, init_qpos=None, base_pos=None, base_orn=None,
                 init_end_effector_pos=None, init_end_effector_orn=None, end_effector_index=None, reset_finger_joints=None,
                 useOrientation=True, useNullSpace=True, topdown_euler=np.array([0., 0., 0.]),
                 init_gripper_axis=np.array([0., 0., 1.]), init_gripper_quat=np.array([0., 0., 0., 1.])):
        self.p = physics_client
        self.urdfrootpath = urdfrootpath
        self.endEffectorPos = init_end_effector_pos
        self.endEffectorOrn = init_end_effector_orn
        self.base_pos = base_pos
        self._robot = self.p.loadURDF(os.path.join(self.urdfrootpath, filename), base_pos, base_orn, useFixedBase=1)
        self.init_qpos = init_qpos
        self.end_effector_index = end_effector_index
        self.motorNames = []
        self.motorIndices = []
        self.maxVelocity = []
        self.maxForce = []
        self.jointDamping = []
        self.joint_ll = []  # Only movable joints in the arm
        self.joint_ul = []  # Only movable joints in the arm
        self.reset_finger_joints = reset_finger_joints
        self.num_joints = self.p.getNumJoints(self._robot)
        for j_idx in range(self.p.getNumJoints(self._robot)):
            joint_info = self.p.getJointInfo(self._robot, j_idx)
            qIndex = joint_info[3]
            if qIndex > -1 and j_idx < self.end_effector_index:
                self.motorNames.append(str(joint_info[1]))
                self.motorIndices.append(j_idx)
                self.joint_ll.append(joint_info[8])
                self.joint_ul.append(joint_info[9])
            self.maxVelocity.append(joint_info[11])
            self.maxForce.append(joint_info[10])
            self.jointDamping.append(joint_info[6])
        self.ndof = len(self.joint_ll)
        self._post_gripper()
        self.IKInfo = None
        self.useOrientation = useOrientation
        self.useNullSpace = useNullSpace
        self.topdown_euler = topdown_euler
        self.init_gripper_axis = init_gripper_axis
        self.init_gripper_quat = init_gripper_quat
        self.collision_pairs = set()

        self.compute_ik_information()
        self.reset()

    def _post_gripper(self):
        pass

    def compute_ik_information(self):
        pass

    def reset(self):
        for j_idx in range(self.p.getNumJoints(self._robot)):
            self.p.resetJointState(self._robot, j_idx, self.init_qpos[j_idx])
        # print("initial qpos, eef orn", self.get_end_effector_orn())
        # print(self.motorNames, self.motorIndices)
        # print('before:', p.getLinkState(self._kinova, self.end_effector_index)[0], p.getLinkState(self._kinova, self.end_effector_index)[1])
        endEffectorOrn = self.p.getQuaternionFromEuler(self.endEffectorOrn)
        target_endpos = np.asarray(self.endEffectorPos) # + np.concatenate([np.random.uniform(-0.15, 0.15, size=2), [0.]])
        # print('target endpos', target_endpos)
        jointPoses = self.run_ik(target_endpos, endEffectorOrn)[0]
        if len(np.array(jointPoses).shape) > 1:
            jointPoses = jointPoses[np.argmin([np.linalg.norm(pos - np.array(self.init_qpos)[self.motorIndices[:self.ndof]]) for pos in jointPoses])]
        jointPoses = np.concatenate([jointPoses[:self.ndof], self.reset_finger_joints])
        for i in range(len(self.motorIndices)):
            self.p.resetJointState(self._robot, self.motorIndices[i], jointPoses[i])
        self.p.stepSimulation()
        # print('after reset, endpos', self.get_end_effector_pos(), self.get_end_effector_orn(as_type="quat"), jointPoses)
        # for i in range(self.num_joints):
        #     print(i, self.p.getLinkState(self._robot, i))
        # raise NotImplementedError

    def reset_qpos(self, qpos=None):
        if qpos is None:
            qpos = self.init_qpos
        for j in range(self.num_joints):
            self.p.resetJointState(self._robot, j, qpos[j])
        self.p.stepSimulation()

    def get_observation(self):
        raise NotImplementedError

    def reset_base(self, base_pos, base_orn=None):
        old_pos, old_orn = self.p.getBasePositionAndOrientation(self._robot)
        new_pos = old_pos
        if base_pos is not None:
            new_pos = base_pos
        new_orn = old_orn
        if base_orn is not None:
            new_orn = base_orn
        self.p.resetBasePositionAndOrientation(self._robot, new_pos, new_orn)
        self.p.stepSimulation()

    def get_base(self):
        pos, orn = self.p.getBasePositionAndOrientation(self._robot)
        return pos, orn

    def get_joint_pos(self, joints):
        states = self.p.getJointStates(self._robot, joints)
        joint_pos, *_ = zip(*states)
        return joint_pos

    def run_ik(self, pos, orn):
        state_before_ik = self.p.saveState()
        counter = 0
        threshold = 20
        joint_poses = [self.p.getJointState(self._robot, self.motorIndices[i])[0] for i in range(len(self.motorIndices))]
        while np.linalg.norm(np.array(self.get_end_effector_pos()) - np.array(pos)) > 1e-2 and counter < threshold:
            if self.useNullSpace:
                if self.useOrientation:
                    joint_poses = self.p.calculateInverseKinematics(
                        self._robot, self.end_effector_index, pos, orn,
                        self.IKInfo["lowerLimits"], self.IKInfo["upperLimits"],
                        self.IKInfo["jointRanges"], self.IKInfo["restPoses"],
                    )
                    joint_poses = np.array(joint_poses)
                    # print("[Before mode]", joint_poses)
                    if np.any(joint_poses > np.pi):
                        joint_poses[np.where(joint_poses > np.pi)] -= 2 * np.pi
                    if np.any(joint_poses < -np.pi):
                        joint_poses[np.where(joint_poses < -np.pi)] += 2 * np.pi
                    # print("[After mode]", joint_poses)
                else:
                    joint_poses = self.p.calculateInverseKinematics(
                        self._robot, self.end_effector_index, pos,
                        lowerLimits=self.IKInfo["lowerLimits"], upperLimits=self.IKInfo["upperLimits"],
                        jointRanges=self.IKInfo["jointRanges"], restPoses=self.IKInfo["restPoses"]
                    )
            else:
                if self.useOrientation:
                    joint_poses = self.p.calculateInverseKinematics(
                        self._robot, self.end_effector_index, pos, orn, jointDamping=self.IKInfo["jointDamping"]
                    )
                else:
                    joint_poses = self.p.calculateInverseKinematics(
                        self._robot, self.end_effector_index, pos
                    )
            counter += 1
            for i in range(len(self.motorIndices)):
                self.p.resetJointState(self._robot, self.motorIndices[i], joint_poses[i])
            self.p.stepSimulation()
        # print("IK counter", counter)
        # width = 500
        # height = 500
        # view_matrix = self.p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[1.3, 0.6, 0.05],
        #                                                        distance=1.5,
        #                                                        yaw=-30,
        #                                                        pitch=-20,
        #                                                        roll=0,
        #                                                        upAxisIndex=2)
        # proj_matrix = self.p.computeProjectionMatrixFOV(fov=60,
        #                                                 aspect=1.0,
        #                                                 nearVal=0.1,
        #                                                 farVal=100.0)
        # (_, _, px, _, _) = self.p.getCameraImage(width=width,
        #                                          height=height,
        #                                          viewMatrix=view_matrix,
        #                                          projectionMatrix=proj_matrix,
        #                                          # renderer=p.ER_BULLET_HARDWARE_OPENGL,
        #                                          )
        # rgb_array = np.array(px, dtype=np.uint8)
        # rgb_array = np.reshape(rgb_array, (width, height, 4))
        #
        # rgb_array = rgb_array[:, :, :3]
        # import matplotlib.pyplot as plt
        # plt.imshow(rgb_array)
        # plt.show()
        cur_pos = np.array(self.get_end_effector_pos())
        pos_error = np.linalg.norm(cur_pos - np.array(pos))
        orn_diff = self.p.getDifferenceQuaternion(self.get_end_effector_orn(as_type="quat"), np.array(orn))
        orn_error = np.arccos(orn_diff[-1]) * 2
        info = {'counter': counter, 'pos_error': pos_error, 'orn_error': orn_error, 'is_success': pos_error < 0.01,
                'target_pos': pos, 'cur_pos': cur_pos}
        self.p.restoreState(state_before_ik)
        self.p.removeState(state_before_ik)
        return joint_poses, info

    def get_end_effector_pos(self):
        state = self.p.getLinkState(self._robot, self.end_effector_index)
        return np.asarray(state[0])

    def get_end_effector_orn(self, as_type="euler"):
        state = self.p.getLinkState(self._robot, self.end_effector_index)
        if as_type == "quat":
            return state[1]
        elif as_type == "euler":
            return np.asarray(self.p.getEulerFromQuaternion(state[1]))

    def gen_gripper_joint_command(self, ctrl):
        '''
        :param ctrl: scalar in range [0, 1], 0: open, 1: close
        :return:
        '''
        raise NotImplementedError

    def position_control(self, tgt_joint_pos, position_gain=None):
        tgt_velocities = [0] * len(tgt_joint_pos)
        kwargs = {}
        if position_gain is not None:
            if isinstance(position_gain, float):
                kwargs['positionGains'] = [position_gain] * len(tgt_joint_pos)
                kwargs['velocityGains'] = [0.1 * position_gain] * len(tgt_joint_pos)
            else:
                assert len(position_gain) == len(tgt_joint_pos)
                kwargs['positionGains'] = np.array(position_gain)
                kwargs['velocityGains'] = 0.1 * np.array(position_gain)
        self.p.setJointMotorControlArray(self._robot, self.motorIndices, self.p.POSITION_CONTROL, tgt_joint_pos,
                                         tgt_velocities, [self.maxForce[j] for j in self.motorIndices],
                                         **kwargs)


class KinovaRobot(ArmRobot):
    def __init__(self, physics_client, urdfrootpath=KINOVA_MODEL_DIR, init_qpos=None,
                 init_end_effector_pos=(0.4, 0., 0.1), init_end_effector_orn=(0, -math.pi, math.pi/2), lock_finger=False,
                 useOrientation=True, useNullSpace=True):

        if init_qpos is None:
            init_qpos = [0., 0., -0.127, 4.234, 1.597, -0.150, 0.585, 2.860, 0.0,
                         0.7505, 0.0, 0.7505, 0.0]
        self.lock_finger = lock_finger
        end_effector_index = 8
        self.finger_index = [9, 11]
        self.finger_tip_index = [10, 12]
        reset_finger_joints = [0.7505, 0., 0.7505, 0.]
        super(KinovaRobot, self).__init__(physics_client, "j2n6s200.urdf", urdfrootpath, init_qpos, [0.8, 0.6, 0.0],
                                          [0., 0., 1., 0.], init_end_effector_pos, init_end_effector_orn,
                                          end_effector_index, reset_finger_joints,
                                          useOrientation, useNullSpace, np.array([0., -np.pi, 0.]),
                                          np.array([0., 0., 1.]))

    def compute_ik_information(self):
        """ Finds the values for the IK solver. """
        joint_information = list(
            map(lambda i: self.p.getJointInfo(self._robot, i),
                self.motorIndices))
        self.IKInfo = {}
        assert all([len(joint_information[i]) == 17 for i in range(len(self.motorIndices))])
        self.IKInfo["solver"] = 0
        self.IKInfo["jointDamping"] = [0.1] * len(self.motorIndices)
        self.IKInfo["lowerLimits"] = [info[8] for info in joint_information]
        self.IKInfo["upperLimits"] = [info[9] for info in joint_information]
        # TODO: tweak jointRange, resetPose?
        self.IKInfo["jointRanges"] = [50] * len(self.motorIndices)
        self.IKInfo["restPoses"] = [-0.127, 4.234, 1.597, -0.150, 0.585, 2.860, 0.7505, 0.0, 0.7505, 0.0]

    def get_observation(self):
        end_effector_state = self.p.getLinkState(self._robot, self.end_effector_index, computeLinkVelocity=1)
        end_effector_pos, end_effector_orn, _, _, _, _, end_effector_vl, end_effector_va = end_effector_state
        end_effector_orn = self.p.getEulerFromQuaternion(end_effector_orn)
        finger1_state, finger2_state = self.p.getJointStates(self._robot, self.finger_index)
        tip1_state, tip2_state = self.p.getJointStates(self._robot, self.finger_tip_index)
        finger1_pos, finger1_vel, *_ = finger1_state
        finger2_pos, finger2_vel, *_ = finger2_state
        tip1_pos, tip1_vel, *_ = tip1_state
        tip2_pos, tip2_vel, *_ = tip2_state
        # return end_effector_pos, end_effector_orn, end_effector_vl, (finger1_pos, finger2_pos), (tip1_pos, tip2_pos), \
        #        (finger1_vel, finger2_vel), (tip1_vel, tip2_vel)
        return np.array(end_effector_pos), np.array(end_effector_orn), np.array(end_effector_vl), \
               np.array([finger1_pos, finger2_pos]), np.array([finger1_vel, finger2_vel])

    def apply_action(self, action):
        action = np.array(action)
        if self.useOrientation:
            assert len(action) == 7
        else:
            assert len(action) == 4
        cur_pos = self.get_end_effector_pos()
        position = cur_pos + action[0:3]  # Now absolute position
        if self.useOrientation:
            # action[3: 6]: delta rotation, orn: target rotation
            # action[3: 6]: roll around X, pitch around Y, yaw around Z
            quat = self.p.getQuaternionFromEuler(action[3: 6])
            mat = np.array(self.p.getMatrixFromQuaternion(quat)).reshape((3, 3))
            cur_quat = self.get_end_effector_orn(as_type="quat")
            cur_mat = np.array(self.p.getMatrixFromQuaternion(cur_quat)).reshape((3, 3))
            tgt_mat = mat @ cur_mat
            from env.bullet_rotations import mat2quat
            orn = mat2quat(tgt_mat)
            # orn = self.p.getQuaternionFromEuler(action[3:6])
        else:
            orn = [0., 0., 0., 1.]
        # fingers and tips
        finger_ctrl = action[-1]
        # fingers = action[6:8]
        # tips = action[8:10]
        # TODO: tweak finger and tip
        fingers = 0.7505 + 0.7505 * np.ones(2) * finger_ctrl
        tips = np.zeros(2)
        fingers_and_tips = np.array([fingers[0], tips[0], fingers[1], tips[1]])
        self._move_end_effector(position, orn, fingers_and_tips)
        self._step_callback()

    def _move_end_effector(self, pos, orn, fingers_and_tips):
        tgt_joint_pos = np.concatenate([self.run_ik(pos, orn)[0][:6], fingers_and_tips])
        # current_joint_states = self.p.getJointStates(self._kinova, self.motorIndices)
        # current_joint_pos = [state[0] for state in current_joint_states]
        for i in range(len(self.motorIndices)):
            self.p.setJointMotorControl2(bodyUniqueId=self._robot,
                                         jointIndex=self.motorIndices[i],
                                         controlMode=self.p.POSITION_CONTROL,
                                         targetPosition=tgt_joint_pos[i],
                                         targetVelocity=0,
                                         force=self.maxForce[self.motorIndices[i]],
                                         # maxVelocity=self.maxVelocity[self.motorIndices[i]],
                                         positionGain=1,
                                         velocityGain=0.1, )
        # for i in range(len(self.motorIndices)):
        #     targetVelocity = 12 * (jointPoses[i] - current_jointPos[i])
        #     targetVelocity = np.clip(targetVelocity, -self.maxVelocity[self.motorIndices[i]], self.maxVelocity[self.motorIndices[i]])
        #     # print(targetVelocity)
        #     targetVelocity = np.clip(targetVelocity, -30, 30)
        #     self.p.setJointMotorControl2(bodyUniqueId=self._kinova,
        #                                  jointIndex=self.motorIndices[i],
        #                                  controlMode=self.p.VELOCITY_CONTROL,
        #                                  targetVelocity=targetVelocity,
        #                                  force=self.maxForce[self.motorIndices[i]],
        #                                  )
        # for i in range(len(self.finger_index)):
        #     self.p.setJointMotorControl2(bodyUniqueId=self._kinova,
        #                                  jointIndex=self.finger_index[i],
        #                                  controlMode=self.p.POSITION_CONTROL,
        #                                  targetPosition=fingers[i],
        #                                  targetVelocity=0,
        #                                  # force=self.maxForce,
        #                                  force=self.maxForce[self.finger_index[i]],
        #                                  maxVelocity=self.maxVelocity[self.finger_index[i]],
        #                                  positionGain=3,
        #                                  velocityGain=0,
        #                                  # velocityGain=1
        #                                  )
        # for i in range(len(self.finger_tip_index)):
        #     self.p.setJointMotorControl2(bodyUniqueId=self._kinova,
        #                                  jointIndex=self.finger_tip_index[i],
        #                                  controlMode=self.p.POSITION_CONTROL,
        #                                  targetPosition=finger_tips[i],
        #                                  targetVelocity=0,
        #                                  # force=self.maxForce,
        #                                  force=self.maxForce[self.finger_tip_index[i]],
        #                                  maxVelocity=self.maxVelocity[self.finger_tip_index[i]],
        #                                  positionGain=3,
        #                                  velocityGain=0,
        #                                  # velocityGain=1,
        #                                  )

    def get_finger_state(self):
        finger1_state, finger2_state = self.p.getJointStates(self._robot, self.finger_index)
        finger1_tip, finger2_tip = self.p.getJointStates(self._robot, self.finger_tip_index)
        finger1_pos, *_ = finger1_state
        finger2_pos, *_ = finger2_state
        finger1_tip_pos, *_ = finger1_tip
        finger2_tip_pos, *_ = finger2_tip
        return finger1_pos, finger2_pos, finger1_tip_pos, finger2_tip_pos

    def _step_callback(self):
        if self.lock_finger:
            for i in self.finger_index:
                self.p.resetJointState(self._robot, i, 1.51)
            for i in self.finger_tip_index:
                self.p.resetJointState(self._robot, i, 0.)

    def gen_gripper_joint_command(self, ctrl):
        return [0.3 + 0.67 * ctrl, 0., 0.3 + 0.67 * ctrl, 0.]


class Kinova2f85Robot(ArmRobot):
    def __init__(self, physics_client, urdfrootpath=KINOVA_MODEL_DIR, init_qpos=None,
                 init_end_effector_pos=(0.4, 0., 0.5), init_end_effector_orn=(0, -math.pi, math.pi/2),
                 useOrientation=True, useNullSpace=True):
        if init_qpos is None:
            init_qpos = [0., 0., -0.127, 4.234, 1.597, -0.150, 0.585, 0., 0.,
                         0., 0., -0., 0., 0.,
                         0., 0., -0., 0., 0.]  # 19d

        end_effector_index = 8
        reset_finger_joints = [0., 0., 0., 0., 0., 0.]  # TODO
        super(Kinova2f85Robot, self).__init__(physics_client, "j2n6robotiq_2f_85.urdf", urdfrootpath, init_qpos,
                                              [0.7, 0.5, 0.0], [0., 0., 1., 0.], init_end_effector_pos,
                                              init_end_effector_orn, end_effector_index, reset_finger_joints,
                                              useOrientation, useNullSpace, np.array([-np.pi, 0., -np.pi / 2]),
                                              np.array([0., 0., 1.]))
        self.gripper_joint_inds = [9, 11, 13, 14, 16, 18]

    def get_observation(self):
        end_effector_state = self.p.getLinkState(self._robot, self.end_effector_index, computeLinkVelocity=1)
        end_effector_pos, end_effector_orn, _, _, _, _, end_effector_vl, end_effector_va = end_effector_state
        end_effector_orn = self.p.getEulerFromQuaternion(end_effector_orn)
        gripper_states = self.p.getJointStates(self._robot, self.gripper_joint_inds)
        gripper_pos, gripper_vel, *_ = zip(*gripper_states)
        return np.array(end_effector_pos), np.array(end_effector_orn), np.array(end_effector_vl), \
               np.array(gripper_pos), np.array(gripper_vel)

    def compute_ik_information(self):
        """ Finds the values for the IK solver. """
        joint_information = list(
            map(lambda i: self.p.getJointInfo(self._robot, i),
                self.motorIndices))
        self.IKInfo = {}
        assert all([len(joint_information[i]) == 17 for i in range(len(self.motorIndices))])
        self.IKInfo["solver"] = 0
        self.IKInfo["jointDamping"] = [0.1] * len(self.motorIndices)
        self.IKInfo["lowerLimits"] = [info[8] for info in joint_information]
        self.IKInfo["upperLimits"] = [info[9] for info in joint_information]
        # TODO: tweak jointRange, resetPose?
        self.IKInfo["jointRanges"] = [50] * len(self.motorIndices)
        self.IKInfo["restPoses"] = [-0.127, 4.234, 1.597, -0.150, 0.585, 2.860, 0., 0., 0., 0., 0., 0.]

    def gen_gripper_joint_command(self, ctrl):
        return [0.4 * ctrl, -0.4 * ctrl, 0.4 * ctrl, 0.4 * ctrl, -0.4 * ctrl, 0.4 * ctrl]

    def position_control(self, tgt_joint_pos, position_gain=None):
        # TODO: not sure if position ctrl works for the gripper with passive joints
        super().position_control(tgt_joint_pos, position_gain)


class UR2f85Robot(ArmRobot):
    def __init__(self, physics_client, urdfrootpath=UR_MODEL_DIR, init_qpos=None,
                 init_end_effector_pos=(1.0, 0.3, 0.6), init_end_effector_orn=(0, -math.pi, math.pi/2),
                 useOrientation=True, useNullSpace=True):
        if init_qpos is None:
            init_qpos = [0, -0.1524, -1.8001, 1.8446, -1.6143, -1.5707, -1.57, 0., 0., 0., 0.,  # Arm
                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,      # Gripper
                         0.]                                              # Base
            init_qpos = [0, -2.596, -1.070, -2.019, -1.628, 1.581, -1.008, 0., 0., 0., 0.,
                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                         0.]

        '''
        end_effector_index = 8  # TODO: 8: wrist_3_link-tool0_fixed_joint. 9: robotiq_coupler_joint. 10: robotiq_85_base_joint
        reset_finger_joints = [0.] * 6
        init_gripper_quat = np.array([0., 0.707, 0.707, 0.])
        topdown = quat2euler(np.array([1, 0, 0., 0.]))
        # [0.6, 0.5, 0.0] works to put horizontal blocks, but not for vertical blocks
        # [0.6, 0.5, -0.1] works to put vertical blocks, but not for horizontal blocks
        # [0.6, 0.55, -0.1] works to put horizontal and vertical blocks
        self.ur5_kin = ikfastpy.PyKinematics()
        '''

        end_effector_index = 7
        reset_finger_joints = [0.] * 6
        import env.ur_kinematics.ikfastpy as ikfastpy
        self.ur5_kin = ikfastpy.PyKinematics()
        init_gripper_quat = mat2quat(np.reshape(self.ur5_kin.forward([0., 0., 0., 0., 0., 0.]), [3, 4])[:, :3])
        print("init gripper quat", init_gripper_quat)
        topdown_quat = quat_mul(np.array([0., 0., -1., 0.]),
                                      quat_mul(np.array([np.sin(-np.pi / 4), 0., 0., np.cos(-np.pi / 4)]),
                                      init_gripper_quat))
        topdown = quat2euler(topdown_quat)  # TODO: has bug in quat2euler
        print("topdown", topdown, euler2quat(topdown),
              topdown_quat, quat2euler(topdown_quat))  # TODO: has bug either in euler2quat or in quat2euler
        super(UR2f85Robot, self).__init__(physics_client, "ur5_robot_with_gripper.urdf", urdfrootpath, init_qpos,
                                          [0.7, 0.6, 0.0], [0., 0., 0., 1.], init_end_effector_pos,
                                          topdown, end_effector_index, reset_finger_joints,
                                          useOrientation, useNullSpace, topdown,
                                          np.array([0., 1., 0.]), init_gripper_quat)
        self.collision_pairs = set()

    def _post_gripper(self):
        self.gripper_joint_inds = [11, 13, 15, 16, 18, 20]
        # self.gripper_joint_inds = [11]
        self.motorIndices.extend(self.gripper_joint_inds)
        # c1 = self.p.createConstraint(self._robot, 11, self._robot, 15, self.p.JOINT_GEAR, [1, 0, 0], [0, 0, 0], [0, 0, 0])
        # self.p.changeConstraint(c1, gearRatio=-1, maxForce=10000, erp=1)
        # c2 = self.p.createConstraint(self._robot, 11, self._robot, 13, self.p.JOINT_GEAR, [1, 0, 0], [0, 0, 0], [0, 0, 0])
        # self.p.changeConstraint(c2, gearRatio=1, maxForce=10000, erp=1)
        # c3 = self.p.createConstraint(self._robot, 11, self._robot, 16, self.p.JOINT_GEAR, [1, 0, 0], [0, 0, 0], [0, 0, 0])
        # self.p.changeConstraint(c3, gearRatio=-1, maxForce=10000, erp=1)
        # c4 = self.p.createConstraint(self._robot, 11, self._robot, 20, self.p.JOINT_GEAR, [1, 0, 0], [0, 0, 0], [0, 0, 0])
        # self.p.changeConstraint(c4, gearRatio=-1, maxForce=10000, erp=1)
        # c5 = self.p.createConstraint(self._robot, 11, self._robot, 18, self.p.JOINT_GEAR, [1, 0, 0], [0, 0, 0], [0, 0, 0])
        # self.p.changeConstraint(c5, gearRatio=1, maxForce=10000, erp=1)

        # print("motor indices", self.motorIndices)  # [1, 2, 3, 4, 5, 6, 11, 13, 15, 16, 18, 20]

    def get_observation(self):
        end_effector_state = self.p.getLinkState(self._robot, self.end_effector_index, computeLinkVelocity=1)
        end_effector_pos, end_effector_orn, _, _, _, _, end_effector_vl, end_effector_va = end_effector_state
        end_effector_orn = self.p.getEulerFromQuaternion(end_effector_orn)
        gripper_states = self.p.getJointStates(self._robot, self.gripper_joint_inds)
        gripper_pos, gripper_vel, *_ = zip(*gripper_states)
        return np.array(end_effector_pos), np.array(end_effector_orn), np.array(end_effector_vl), \
               np.array(gripper_pos), np.array(gripper_vel)

    def compute_ik_information(self):
        """ Finds the values for the IK solver. """
        joint_information = list(
            map(lambda i: self.p.getJointInfo(self._robot, i),
                self.motorIndices))
        self.IKInfo = {}
        assert all([len(joint_information[i]) == 17 for i in range(len(self.motorIndices))])
        self.IKInfo["solver"] = 0
        self.IKInfo["jointDamping"] = [0.1] * len(self.motorIndices)
        self.IKInfo["lowerLimits"] = [info[8] for info in joint_information]
        self.IKInfo["upperLimits"] = [info[9] for info in joint_information]
        # TODO: tweak jointRange, resetPose?
        self.IKInfo["jointRanges"] = [np.pi] * len(self.motorIndices)
        self.IKInfo["restPoses"] = [0., -1.8, 0., 0., 0., 0.]  # TODO

    def gen_gripper_joint_command(self, ctrl):
        # return [0.4 * ctrl]
        return [0.4 * ctrl, -0.4 * ctrl, 0.4 * ctrl, 0.4 * ctrl, -0.4 * ctrl, 0.4 * ctrl]

    def fit_circular(self, joint_pose):
        joint_pose = np.array(joint_pose)
        prime = np.round(joint_pose / (2 * np.pi))
        joint_pose -= prime * (2 * np.pi)
        return joint_pose

    def run_ik(self, pos, orn):
        # ikfastpy
        # rot = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        rot = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        delta_pos = np.reshape(np.asarray(pos) - self.base_pos, (3, 1))
        ik_pos = rot @ delta_pos
        ik_mat = rot @ quat2mat(orn)
        assert is_rotation_mat(ik_mat)
        ee_pose = np.concatenate([ik_mat, ik_pos], axis=-1)
        # print(ee_pose)
        joint_pose = self.ur5_kin.inverse(ee_pose.reshape(-1).tolist())
        joint_pose = self.fit_circular(joint_pose)
        n_solutions = len(joint_pose) // 6
        n = 0

        # def gen_random_rot():
        #     alpha = np.random.uniform(-np.pi, np.pi)
        #     beta = np.random.uniform(-np.pi, np.pi)
        #     theta = np.random.uniform(-0.03, 0.03)
        #     return quat2mat(np.array([np.sin(theta / 2) * np.cos(alpha) * np.cos(beta),
        #                               np.sin(theta / 2) * np.cos(alpha) * np.sin(beta),
        #                               np.sin(theta / 2) * np.sin(alpha), np.cos(theta / 2)]))
        #
        # while n_solutions == 0 and n < 100:
        #     ik_pos = rot @ delta_pos
        #     ik_pos[:2] += np.random.uniform(-0.005, 0.005, size=(2, 1))
        #     ik_mat = rot @ quat2mat(orn)
        #     ik_mat = gen_random_rot() @ ik_mat
        #     ee_pose = np.concatenate([ik_mat, ik_pos], axis=-1)
        #     joint_pose = self.ur5_kin.inverse(ee_pose.reshape(-1).tolist())
        #     joint_pose = self.fit_circular(joint_pose)
        #     n_solutions = len(joint_pose) // 6
        #     n += 1
        joint_pose = np.reshape(joint_pose, (n_solutions, 6))
        info = {'is_success': n_solutions > 0}
        return joint_pose, info


class XArm7Robot(ArmRobot):
    def __init__(self, physics_client, urdfrootpath=XARM_MODEL_DIR, init_qpos=None,
                 init_end_effector_pos=(1.0, 0.3, 0.6),
                 useOrientation=True, useNullSpace=True):
        if init_qpos is None:
            # init_qpos = [0, 0.0, 0.0786759, 0.0, 1.54692674, 0.0, 1.46825087, 0.,
            #              0, 0, 0., 0., 0., 0., 0., 0., 0]
            init_qpos = [0, -np.pi / 2, 0.0, 0.0, 0., 0.0, 0.0, 0.,
                         0, 0, 0., 0., 0., 0., 0., 0., 0]
        end_effector_index = 8
        reset_finger_joints = [0.] * 6
        import env.ikfastpy.ikfastpy_free4 as ikfastpy_free4
        import env.ikfastpy.ikfastpy_free0 as ikfastpy_free0
        self.kin_free4 = ikfastpy_free4.PyKinematics()
        self.kin_free0 = ikfastpy_free0.PyKinematics()
        init_gripper_quat = mat2quat(np.reshape(self.kin_free4.forward([0.] * self.kin_free4.getDOF()), [3, 4])[:, :3])
        topdown_quat = quat_mul(np.array([0., 0., np.sin(np.pi / 4), np.cos(np.pi / 4)]), init_gripper_quat)
        topdown = quat2euler(topdown_quat)
        super(XArm7Robot, self).__init__(physics_client, "xarm7_with_gripper.urdf", urdfrootpath, init_qpos,
                                         [0.7, 0.6, 0.005], [0., 0., 0., 1.], init_end_effector_pos,
                                         topdown, end_effector_index, reset_finger_joints,
                                         useOrientation, useNullSpace, topdown,
                                         np.array([0., 0., -1.]), init_gripper_quat)
        # physics_client.resetBasePositionAndOrientation(self._robot, [0.7, -1.0, 0.005], [0., 0., 0., 1.])
        self.collision_pairs = set()

    def _post_gripper(self):
        self.gripper_joint_inds = [10, 11, 12, 13, 14, 15]
        self.gripper_multipliers = [1, 1, 1, 1, 1, 1]
        self.motorIndices.extend(self.gripper_joint_inds)

    def get_observation(self):
        end_effector_state = self.p.getLinkState(self._robot, self.end_effector_index, computeLinkVelocity=1)
        end_effector_pos, end_effector_orn, _, _, _, _, end_effector_vl, end_effector_va = end_effector_state
        end_effector_orn = self.p.getEulerFromQuaternion(end_effector_orn)
        gripper_states = self.p.getJointStates(self._robot, self.gripper_joint_inds)
        gripper_pos, gripper_vel, *_ = zip(*gripper_states)
        return np.array(end_effector_pos), np.array(end_effector_orn), np.array(end_effector_vl), \
               np.array(gripper_pos), np.array(gripper_vel)

    def gen_gripper_joint_command(self, ctrl):
        return 0.3 * ctrl * np.array(self.gripper_multipliers)

    def run_ik(self, pos, orn):
        ik_pos = np.reshape(np.array(pos) - self.base_pos, (3, 1))
        ik_mat = quat2mat(orn)
        ee_pose = np.concatenate([ik_mat, ik_pos], axis=-1)
        joint_poses = self.kin_free4.inverse(ee_pose.reshape(-1)) + self.kin_free0.inverse(ee_pose.reshape(-1))
        n_solutions = len(joint_poses) // self.kin_free4.getDOF()
        joint_poses = np.array(joint_poses).reshape(n_solutions, self.kin_free4.getDOF()).tolist()
        # if n_solutions == 0:
        #     iterative_ik_joint_poses = self.p.calculateInverseKinematics(
        #         self._robot, self.end_effector_index, pos, orn, maxNumIterations=100)[:self.ndof]
        #     debug_forward_pos = np.array(self.kin.forward(iterative_ik_joint_poses)).reshape(3, 4)[:, -1]
        #     if np.linalg.norm(debug_forward_pos - pos) < 1e-3:
        #         joint_poses = [iterative_ik_joint_poses]
        #         n_solutions += 1
        # Filter out or adjust joint out of range
        if n_solutions > 0:
            joint_poses = list(filter(lambda conf: np.all(conf > np.array(self.joint_ll) - 1e-3) and
                                                   np.all(conf < np.array(self.joint_ul) + 1e-3), joint_poses))
            n_solutions = len(joint_poses)
        return np.array(joint_poses), {'is_success': n_solutions > 0}


if __name__ == "__main__":
    import pybullet as p
    import pybullet_utils.bullet_client as bc
    # import env.ur_kinematics.ikfastpy as ikfastpy
    physics_client = bc.BulletClient(connection_mode=p.GUI)
    physics_client.resetDebugVisualizerCamera(1.5, 90, -30, [0.6, 0.55, 0.])
    # robot = UR2f85Robot(physics_client)
    # print(robot.get_base())
    # jointPoses = np.random.uniform(-np.pi, np.pi, size=6)
    # jointPoses = np.concatenate([jointPoses[:6], robot.reset_finger_joints])
    # for i in range(len(robot.motorIndices)):
    #     robot.p.resetJointState(robot._robot, robot.motorIndices[i], jointPoses[i])
    #     time.sleep(0.1)
    # robot.p.stepSimulation()
    # print('after reset, endpos', robot.get_end_effector_pos(), quat2mat(robot.get_end_effector_orn(as_type="quat")))

    ur5_kin = ikfastpy.PyKinematics()
    # ee_pose = ur5_kin.forward(jointPoses[:6])
    # ee_pose = np.asarray(ee_pose).reshape((3, 4))
    # print("ee pose", ee_pose[:, :3], ee_pose[:, -1])

    ee_pose = ur5_kin.forward([-1.92452457e-01, -2.06325442e+00, 2.01537482e+00, -1.52291262e+00, -1.57102464e+00, -1.76289417e+00])
    ee_pose = np.asarray(ee_pose).reshape((3, 4))
    print(ee_pose)
    joint_poses = ur5_kin.inverse(ee_pose.reshape(-1).tolist())
    print(joint_poses)
