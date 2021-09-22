import ikfastpy_free0 as ikfastpy
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1)
    ll = np.array([-6.28318530718, -2.059, -6.28318530718, -0.19198, -6.28318530718, -1.69297, -6.28318530718])
    ul = np.array([6.28318530718, 2.0944, 6.28318530718, 3.927, 6.28318530718, 3.14159265359, 6.28318530718])
    kin = ikfastpy.PyKinematics()
    ee_trans = []
    ee_rot = []
    # Reset pose
    ee_trans.append(np.array([0.3, -0.25, 0.6]).reshape(3, 1))
    ee_rot.append(np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]]))
    # block 1
    # x Lower: -0.3, higher: 0.3. Y = -0.55, Z = 0.2
    ee_trans.append(np.array([0.6, -0.2, 0.4]).reshape(3, 1))
    ee_rot.append(np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]]))
    z = 0.3
    for j in range(5000):
        _trans = np.array([[np.random.uniform(-0.6, 0.6)], [np.random.uniform(-0.6, 0.6)], [z]])
        _rot = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])  # vertical
        # _rot = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
        ee_pose = np.concatenate([_rot, _trans], axis=-1).reshape(-1)
        confs = kin.inverse(ee_pose)
        # assert len(confs), (ee_pose)
        n_success = 0
        for i in range(len(confs) // 7):
            if np.all(confs[7 * i: 7 * (i + 1)] > ll - 1e-3) and np.all(confs[7 * i: 7 * (i + 1)] < ul + 1e-3):
                n_success += 1
                break
        # assert n_success > 0, (ee_pose)
        ax.scatter(_trans[0], _trans[1], c='b' if n_success > 0 else 'r')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("z=%f" % z)
    plt.show()
