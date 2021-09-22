try:
    import cv2
except ImportError:
    import sys
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
    sys.path.append("/opt/ros/kinetic/lib/python2.7/dist-packages")
    import cv2
import cv2.aruco as aruco
import numpy as np
from env.bullet_rotations import quat_rot_vec, quat2mat, euler2quat, mat2quat, gen_noisy_q, euler2mat, rvec2mat
import math, time
import matplotlib.pyplot as plt


CALIB_CAM2ARM = np.array(
        [[0.01606431, -0.99982658, 0.00942071, 0.06578977],
         [0.99986186, 0.01602326, -0.00441669, -0.0344229],
         [0.00426497, 0.00949036, 0.99994587, 0.02654628],
         [0., 0., 0., 1.]]
    )


def parse_view(arm, intrinsic_mtx, intrinsic_coef, color_image, depth_image, marker_size, parsed_marker_ids=[]):
    ret = arm.get_servo_angle()
    assert ret[0] == 0
    servo_angles = ret[1]
    code, tip_pose = arm.get_position()
    # TODO
    camera_xyz, camera_mtx = convert_arm_to_camera(tip_pose, CALIB_CAM2ARM)
    print("camera xyz", camera_xyz, "camera_mtx", camera_mtx)
    print("camera_quat", mat2quat(camera_mtx))

    object_poses = []
    new_ids = []
    marker_ret = get_pose(color_image, intrinsic_mtx, intrinsic_coef, arm, marker_size)
    print("marker_ret", marker_ret)
    if marker_ret is not None:
        markers_id, markers_rot, markers_pos = marker_ret
        for i in range(len(markers_id)):
            if markers_id[i] in parsed_marker_ids:
                continue
            new_ids.append(markers_id[i])
            object_pos, object_rot = convert_marker_pose_to_com([0., 0., -0.025], markers_pos[i], markers_rot[i])
            object_pose = np.concatenate([object_rot, np.reshape(object_pos, (3, 1))], axis=-1)
            object_poses.append(object_pose)
    plane_height = 0
    image_width = 640
    image_height = 480
    depth_mtx = convert_plane_to_mask(plane_height, camera_xyz, camera_mtx, intrinsic_mtx, intrinsic_coef,
                                      image_width, image_height)
    tgt_color_low = (0, 0, 0)
    tgt_color_high = (255, 255, 255)
    # TODO: align
    obj_sizes, obj_positions_in_cam = parse_image(color_image, depth_image, depth_mtx, intrinsic_mtx, tgt_color_low, tgt_color_high)
    obj_positions = []
    for i in range(len(obj_positions_in_cam)):
        obj_positions.append(camera2world(obj_positions_in_cam[i], None, tip_pose)[0])
    print("after parsing image", obj_sizes, obj_positions)
    aligned_sizes = []
    for i in range(len(object_poses)):
        idx = np.argmin([np.linalg.norm(pos[:2] - object_poses[i][:2, -1]) for pos in obj_positions])
        if np.linalg.norm(obj_positions[idx][:2] - object_poses[i][:2, -1]) < 0.05:
            aligned_sizes.append(obj_sizes[idx])
        else:
            print("cannot align")
            aligned_sizes.append(None)
    return new_ids, object_poses, aligned_sizes, servo_angles


def parse_scene(arm, camera_pipeline, camera_align, intrinsic_mtx, intrinsic_coef, marker_size, num_objects: int):
    parsed_marker_ids = []
    objects_info = []
    for y in [-450, -250, 250, 450]:
        arm.set_position(x=350, y=y, z=380, roll=-180, pitch=0, yaw=-90, speed=100, wait=True)
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
        new_ids, object_poses, aligned_sizes, servo_angles = parse_view(
            arm, intrinsic_mtx, intrinsic_coef, color_image, depth_image, marker_size, parsed_marker_ids)
        parsed_marker_ids.extend(new_ids)
        # all_object_poses.extend(object_poses)
        # all_sizes.extend(aligned_sizes)
        objects_info += list(zip(new_ids, object_poses, aligned_sizes))
        if len(parsed_marker_ids) >= num_objects:
            break
    objects_info = sorted(objects_info, key=lambda x:x[0])
    parsed_marker_ids, all_object_poses, all_sizes = zip(*objects_info)
    return parsed_marker_ids, all_object_poses, all_sizes, servo_angles


def get_pose(color_image, camera_mtx, camera_dist_coef, arm, marker_size,
             aruco_dict=aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)):
    color_image = color_image.copy()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(color_image, aruco_dict)
    print("id", ids)
    if ids is None:
        np.save("color_image", color_image)
    rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_mtx, camera_dist_coef)
    out_image = aruco.drawDetectedMarkers(color_image, corners, ids)
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    if tvecs is not None:
        tvecs2end = [(CALIB_CAM2ARM @ np.concatenate([tvec.reshape(3, 1), np.array([[1.]])]))[:3] for tvec in tvecs]
        ret, arm_pose = arm.get_position()
        assert ret == 0
        end_x, end_y, end_z, roll, pitch, yaw = arm_pose
        # TODO: rotation not considered in tcp
        tcp_offset = arm.tcp_offset
        rot_mtx = euler2mat([math.radians(roll), math.radians(pitch), math.radians(yaw)])
        tcp_vec = quat_rot_vec(euler2quat([math.radians(roll), math.radians(pitch), math.radians(yaw)]), -np.array(tcp_offset[:3]) / 1000)
        hand_pos = (np.array([end_x, end_y, end_z]) / 1000 + tcp_vec).reshape(3, 1) # correct
        T_end2world = np.concatenate([rot_mtx, hand_pos], axis=-1)
        tvecs2world = [(T_end2world @ np.concatenate([t, np.array([[1.]])]))[:3] for t in tvecs2end]
        rot2camera = [rvec2mat(np.array(rvec).reshape(3)) for rvec in rvecs]
        # rot2world = [T_end2world[:, :3] @ CALIB_CAM2ARM[:3, :3] @ rot for rot in rot2camera]
        # rot2world = [rot @ CALIB_CAM2ARM[:3, :3] @ T_end2world[:, :3] for rot in rot2camera]
        rot2world = [T_end2world[:, :3] @ CALIB_CAM2ARM[:3, :3] @ rot for rot in rot2camera]
        for marker_idx in range(len(tvecs)):
            out_image = aruco.drawAxis(out_image, camera_mtx, camera_dist_coef, rvecs[marker_idx], tvecs[marker_idx], marker_size)
        cv2.imshow('RealSense', out_image)
        cv2.waitKey(1)
        time.sleep(2)
        return ids, np.array(rot2world), np.array(tvecs2world)
    return None

def parse_image(color_image, depth_image, depth_mtx, intrinsic_mtx,
                tgt_color_low=(0, 0, 0), tgt_color_high=(255, 255, 255)):

    # TODO: segment out aligned rgb part according to depth_threshold, shape detection
    #  use marker to for accurate position and orientation detection
    #  return center position, orientation, size
    depth_image = depth_image.copy() / 1000  # in meters
    depth_mask = cv2.inRange(depth_image.astype(np.float64), np.zeros_like(depth_mtx), depth_mtx - 0.03)
    depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv_image, tgt_color_low, tgt_color_high)
    mask = cv2.bitwise_and(depth_mask, color_mask)
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(color_image)
    ax[1].imshow(mask)
    ax[2].imshow(depth_image - depth_mtx)
    plt.show()

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    sizes = []
    positions = []
    # TODO: not robust
    for c in cnts:
        M = cv2.moments(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            # print("com", M)
            # TODO: rectangle with orientation shift
            # (x, y, w, h) = cv2.boundingRect(approx)
            # rect = cv2.minAreaRect(approx)
            # box = cv2.boxPoints(rect)
            # box = np.array(box, dtype=np.int)
            box = np.array(approx, dtype=np.int).reshape(4, 2)
            width_vec = np.concatenate([(box[0] + box[3]) / 2 - (box[1] + box[2]) / 2, [0]])
            height_vec = np.concatenate([(box[0] + box[1]) / 2 - (box[2] + box[3]) / 2, [0]])
            w = np.linalg.norm(width_vec)
            h = np.linalg.norm(height_vec)
            center = (box[0] + box[1] + box[2] + box[3]) / 4
            if w * h > 400:
                print(box)
                depth = np.mean(depth_image[int(center[1]) - 1: int(center[1]) + 2,
                                            int(center[0]) - 1: int(center[0]) + 2])
                # TODO: assume same depth for all parts in the rectangle
                # world_width_vec = np.linalg.inv(intrinsic_mtx) @ np.transpose(np.array([w, 0, 0]) * depth)
                world_width_vec = np.linalg.inv(intrinsic_mtx) @ np.transpose(width_vec * depth)
                # world_height_vec = np.linalg.inv(intrinsic_mtx) @ np.transpose(np.array([0, h, 0]) * depth)
                world_height_vec = np.linalg.inv(intrinsic_mtx) @ np.transpose(height_vec * depth)
                # position in camera coordination
                position = np.linalg.inv(intrinsic_mtx) @ np.transpose(np.array([int(center[0]), int(center[1]), 1]) * depth)
                world_width = np.linalg.norm(world_width_vec)
                world_height = np.linalg.norm(world_height_vec)
                if world_width > world_height:
                    world_width, world_height = world_height, world_width
                _size = np.array([world_width, world_height, world_width]) / 2
                sizes.append(_size)
                positions.append(position)
                # cv2.drawContours(hsv_image, [approx], -1, (0, 255, 0), 2)
                # print("min area rect", box)
                for j in range(4):
                    cv2.line(hsv_image, (box[j][0], box[j][1]), (box[(j + 1) % 4][0], box[(j + 1) % 4][1]), (0, 255, 0), 2)
                # cv2.rectangle(hsv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Image", hsv_image)
            cv2.waitKey(1)
    cv2.imwrite("temp.png", hsv_image)
    return sizes, positions


def convert_marker_pose_to_com(com2marker_t, marker_xyz, marker_rot):
    # com2marker_t is the coordinate of com wrt the marker coordinate
    # marker_xyz, marker_quat is the pose in world coordinate.
    T_com2marker = np.concatenate([np.concatenate([np.eye(3), np.reshape(com2marker_t, (3, 1))], axis=-1),
                                   [[0., 0., 0., 1.]]], axis=0)
    T_marker2world = np.concatenate([np.concatenate([marker_rot, np.reshape(marker_xyz, (3, 1))], axis=-1),
                                     [[0., 0., 0., 1.]]], axis=0)
    pose = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    com_pose = T_marker2world @ T_com2marker @ pose
    com_xyz = com_pose[:3, -1]
    com_rot = com_pose[:3, :3]
    return com_xyz, com_rot


def convert_plane_to_mask(plane_height, camera_xyz, camera_mtx, intrinsic_mtx, intrinsic_coef, image_width, image_height):
    # camera_mtx @ axis_in_camera + trans_cam2world.T @ [1, 1, 1] = eye(3)
    world_axis_and_origin = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=-1)
    # axis and origin in camera coordination
    camera_axis_and_origin = \
        np.linalg.inv(camera_mtx) @ (world_axis_and_origin - camera_xyz.reshape(3, 1) @ np.ones((1, 4)))
    # The plane in camera coordination is origin + all linear combination of (axis 0 - origin) and (axis 1 - origin)
    unit_x = camera_axis_and_origin[:, 0] - camera_axis_and_origin[:, 3]
    unit_y = camera_axis_and_origin[:, 1] - camera_axis_and_origin[:, 3]
    unit_z = camera_axis_and_origin[:, 2] - camera_axis_and_origin[:, 3]
    origin = camera_axis_and_origin[:, 3]
    # (0, -0.9, height) in world
    corner1 = origin + (-0.9) * unit_y + plane_height * unit_z
    # (0.9, -0.9, height) in world
    corner2 = origin + 0.9 * unit_x - 0.9 * unit_y + plane_height * unit_z
    # (0.9, 0.9, height)
    corner3 = origin + 0.9 * unit_x + 0.9 * unit_y + plane_height * unit_z
    # For test
    test_point = origin + camera_xyz[0] * unit_x + camera_xyz[1] * unit_y + plane_height * unit_z
    # intrinsic_mtx @ world_point = [image u, image v, 1].T
    uv1 = intrinsic_mtx @ corner1
    uv2 = intrinsic_mtx @ corner2
    uv3 = intrinsic_mtx @ corner3
    test_uv = intrinsic_mtx @ test_point
    print("uv1", uv1, "uv2", uv2, "uv3", uv3)
    print("test uv", test_uv)
    # interpolate the mask
    # [(uv2 - uv1).T; (uv3 - uv1).T] @ coeff = pixels
    u = np.arange(image_width)
    v = np.arange(image_height)
    uu, vv = np.meshgrid(u, v)
    # [[0, 1, ..., W-1,] H repeats, [0, 0, ..., 0, 1, 1, ..., 1, ..., ,H-1]]
    pixels = np.stack([np.ravel(uu), np.ravel(vv)], axis=0)
    # pixels = np.array([[598], [365]])
    base_vec1 = uv2 - uv1
    base_vec2 = uv3 - uv1
    coeff_mtx = np.linalg.inv(np.stack([base_vec1[:2], base_vec2[:2]], axis=-1)) @ (pixels - uv1[:2].reshape(2, 1))
    depth = np.array([[base_vec1[2], base_vec2[2]]]) @ coeff_mtx + uv1[2]
    # print("should be uv2", np.array([[base_vec1[2], base_vec2[2]]]) @ np.array([[1], [0]]) + uv1[2])
    # print("depth", depth.shape, depth[:, 1000])
    depth_mtx = np.reshape(depth, (image_height, image_width))

    return depth_mtx


def convert_arm_to_camera(tip_pose, T_cam2end):
    '''
    Get camera pose in world coordinate given current arm configuration
    '''
    T_end2tip = np.concatenate([np.concatenate([np.eye(3), [[0.], [0.], [-0.172]]], axis=-1),
                                np.array([[0., 0., 0., 1.]])], axis=0)
    x, y, z, roll, pitch, yaw = tip_pose
    tip_quat = euler2quat(np.array([roll, pitch, yaw]) / 180 * np.pi)
    tip_xyz = np.array([x, y, z]) / 1000
    T_tip2world = np.concatenate([np.concatenate([quat2mat(tip_quat), tip_xyz.reshape(3, 1)], axis=-1),
                                  np.array([[0., 0., 0., 1.]])], axis=0)
    camera_pose = T_tip2world @ T_end2tip @ T_cam2end @ np.eye(4)
    camera_xyz = camera_pose[:3, -1]
    camera_mtx = camera_pose[:3, :3]
    return camera_xyz, camera_mtx


def camera2world(xyz_in_cam, rot_in_cam, tip_pose):
    '''
    Get pose in world coordinate given pose in camera coordinate
    '''
    if rot_in_cam is None:
        rot_in_cam = np.eye(3)
    pose_in_cam = np.concatenate(
        [np.concatenate([rot_in_cam.reshape(3, 3), xyz_in_cam.reshape(3, 1)], axis=-1), 
         np.array([[0., 0., 0., 1.]])], axis=0)
    x, y, z, roll, pitch, yaw = tip_pose
    tip_xyz = np.array([x, y, z]) / 1000
    tip_mat = euler2mat(np.array([roll, pitch, yaw]) / 180 * np.pi)
    T_tip2world = np.concatenate([np.concatenate([tip_mat, tip_xyz.reshape(3, 1)], axis=-1), 
                                  np.array([[0., 0., 0., 1.]])], axis=0)
    T_end2tip = np.concatenate([np.concatenate([np.eye(3), [[0.], [0.], [-0.172]]], axis=-1),
                                np.array([[0., 0., 0., 1.]])], axis=0)
    T_cam2end = CALIB_CAM2ARM
    pose_in_world = T_tip2world @ T_end2tip @ T_cam2end @ pose_in_cam
    xyz_in_world = pose_in_world[:3, -1]
    mtx_in_world = pose_in_world[:3, :3]
    return xyz_in_world, mtx_in_world

