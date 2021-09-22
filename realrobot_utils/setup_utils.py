import sys, os, time
sys.path.append("/home/yunfei/projects/xArm-Python-SDK")
try:
    from xarm.wrapper import XArmAPI

    import pyrealsense2 as rs
except:
    XArmAPI = None
    rs = None
import numpy as np


def arm_set_up(ip, mode=0):

    def hangle_err_warn_changed(item):
        print('ErrorCode: {}, WarnCode: {}'.format(item['error_code'], item['warn_code']))
        # TODO：Do different processing according to the error code

    arm = XArmAPI(ip, do_not_open=True)
    arm.register_error_warn_changed_callback(hangle_err_warn_changed)
    arm.connect()

    arm.clean_error()
    # enable motion
    arm.motion_enable(enable=True)
    # set mode: 0: position control, 2: manual
    arm.set_mode(mode)
    arm.set_state(state=0)
    time.sleep(2)

    code = arm.set_gripper_mode(0)
    print('set gripper mode: location mode, code={}'.format(code))

    code = arm.set_gripper_enable(True)
    print('set gripper enable, code={}'.format(code))

    code = arm.set_gripper_speed(2000)
    print('set gripper speed, code={}'.format(code))

    code = arm.set_gripper_position(850, wait=True)
    print('[wait]set gripper pos, code={}'.format(code))

    print(arm.get_position(), arm.get_servo_angle())
    tcp_offset = arm.tcp_offset
    if abs(tcp_offset[2] - 172) > 0.1:
        arm.disconnect()
        raise RuntimeError
    return arm


def camera_set_up():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # width, height,..., framerate
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    color_intrinsics = profile.get_stream(
        rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_mtx = np.array([[color_intrinsics.fx, 0., color_intrinsics.ppx],
                           [0., color_intrinsics.fy, color_intrinsics.ppy],
                           [0., 0., 1.]])
    camera_dist_coef = np.array(color_intrinsics.coeffs)
    return pipeline, align, camera_mtx, camera_dist_coef


TCP_LOAD_CONF = {
    # 'gripper': [0.82, [0.0, 0.0, 48.0]],
    'gripper': [1.1, [0.0, 0.0, 58.46]],
    'gripper+obj10': [1.24, [0.0, 0.0, 85.68]],
    'gripper+obj12': [1.28, [0.0, 0.0, 87.58]],
    'gripper+obj14': [1.33, [0.0, 0.0, 88.46]],
    'gripper+obj16': [1.33, [0.0, 0.0, 96.04]],
    'gripper+obj18': [1.36, [0.0, 0.0, 97.38]],
    'gripper+obj20': [1.41, [5.0, 0.0, 97.98]],
    'gripper+obj22': [1.43, [0.0, 0.0, 102.06]],
    'gripper+obj24': [1.48, [0.0, 0.0, 104.28]]
}


def set_tcp_load(arm, tcp_name):
    assert tcp_name in TCP_LOAD_CONF
    code = arm.set_tcp_load(*TCP_LOAD_CONF[tcp_name])
    time.sleep(1)
    cur_tcp_load = arm.tcp_load
    assert abs(cur_tcp_load[0] - TCP_LOAD_CONF[tcp_name][0]) < 1e-3
    assert abs(cur_tcp_load[1][2] - TCP_LOAD_CONF[tcp_name][1][2]) < 1e-3
    print("Set tcp load {}, code={}".format(tcp_name, code), "Current tcp", arm.tcp_load)
