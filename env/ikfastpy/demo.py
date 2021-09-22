import numpy as np
import env.ikfastpy.ikfastpy as ikfastpy
import matplotlib.pyplot as plt
from env.bullet_rotations import quat2mat, mat2quat, quat2euler, is_rotation_mat

# Initialize kinematics for UR5 robot arm
ur5_kin = ikfastpy.PyKinematics()
n_joints = ur5_kin.getDOF()

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
success = 0  # 4897 / 5000
stage2_success = 0
failure_cases = []


for _ in range(5000):
    '''
    # joint_angles = [-3.1,-1.6,1.6,-1.6,-1.6,0.] # in radians
    joint_angles = [-1.92452457e-01, -2.06325442e+00, 2.01537482e+00, -1.52291262e+00,
                    -1.57102464e+00, -1.76289417e+00]
    # joint_angles = np.random.uniform(-np.pi, np.pi, size=n_joints).tolist()

    # Test forward kinematics: get end effector pose from joint angles
    print("\nTesting forward kinematics:\n")
    print("Joint angles:")
    print(joint_angles)
    ee_pose = ur5_kin.forward(joint_angles)
    ee_pose = np.asarray(ee_pose).reshape(3,4) # 3x4 rigid transformation matrix
    print("\nEnd effector pose:")
    print(ee_pose)
    print("\n-----------------------------")
    '''

    # Test inverse kinematics: get joint angles from end effector pose
    print("\nTesting inverse kinematics:\n")
    ee_pose = np.zeros((3, 4))
    # print(ee_pose.reshape(-1).tolist())
    theta_noise = np.random.uniform(-np.pi, np.pi)
    alpha = np.random.uniform(-np.pi, np.pi)
    beta = np.random.uniform(-np.pi, np.pi)
    ee_pose[:, :3] = quat2mat(np.array([np.sin(theta_noise / 2) * np.cos(alpha) * np.cos(beta),
                                        np.sin(theta_noise / 2) * np.cos(alpha) * np.sin(beta),
                                        np.sin(theta_noise / 2) * np.sin(alpha),
                                        np.cos(theta_noise / 2)]))
    assert is_rotation_mat(ee_pose[:, :3]), ee_pose[:, :3]
    ee_pose[:, -1] = np.random.uniform(-1, 1, size=3)
    joint_configs = ur5_kin.inverse(ee_pose.reshape(-1).tolist())
    n_solutions = int(len(joint_configs)/n_joints)
    print("%d solutions found:"%(n_solutions))
    joint_configs = np.asarray(joint_configs).reshape(n_solutions,n_joints)
    for joint_config in joint_configs:
        print(joint_config)

    # Check cycle-consistency of forward and inverse kinematics
    if n_solutions > 0:
    # if n_solutions > 0 and np.any([np.sum(np.abs(joint_config-np.asarray(joint_angles))) < 1e-4 for joint_config in joint_configs]):
        success += 1
        ax.scatter(ee_pose[0][-1], ee_pose[1][-1], ee_pose[2][-1], c='r')
    else:
        new_eepose = ee_pose.copy()
        new_eepose[:, -1] += np.random.uniform(-0.5, 0.5, size=3)
        joint_configs = ur5_kin.inverse(new_eepose.reshape(-1).tolist())
        n_solutions = int(len(joint_configs) / n_joints)
        if n_solutions == 0:
            failure_cases.append((ee_pose, ))
            ax.scatter(new_eepose[0][-1], new_eepose[1][-1], new_eepose[2][-1], c='b')
        else:
            stage2_success += 1
    # assert(np.any([np.sum(np.abs(joint_config-np.asarray(joint_angles))) < 1e-4 for joint_config in joint_configs]))
    # print("\nTest passed!")
print(success)
print(stage2_success)
plt.show()
import pickle
with open("debug_failure_case.pkl", "wb") as f:
    pickle.dump(failure_cases, f)
