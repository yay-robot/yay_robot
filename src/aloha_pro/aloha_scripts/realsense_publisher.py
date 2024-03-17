#!/usr/bin/env python3

import sys
import rospy
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API
from aloha.msg import RGBGrayscaleImage
from cv_bridge import CvBridge
import time
import numpy as np
from collections import defaultdict

TIMEOUT_MS = 100
FPS = 60

# import IPython
# e = IPython.embed

cam_idx = int(sys.argv[1])
time.sleep(2 + 0.5 * cam_idx)


cv_bridge = CvBridge()
rospy.init_node("realsense_publisher")

camera_names = ["cam_left_wrist", "cam_high", "cam_right_wrist", "cam_low"]
camera_sns = ["218622270323", "128422270492", "128422271425", "128422272271"]

camera_names = [camera_names[cam_idx]]
camera_sns = [camera_sns[cam_idx]]

print(f"\n\nSTARTING REALSENSE PUBLISHER FOR CAMERA {cam_idx}\n\n")

ctx = rs.context()
devices = ctx.query_devices()
# [print(device) for device in devices]

for dev in devices:
    dev.hardware_reset()

print(camera_names, camera_sns)

pipes = [rs.pipeline() for _ in range(len(camera_sns))]
cfgs = [rs.config() for _ in range(len(camera_sns))]
profiles = []
depth_scales = []

mean_intensity_set_point_config = (
    {  # NOTE these numbers are specific to your lighting setup
        "cam_left_wrist": 500,
        "cam_high": 500,
        "cam_right_wrist": 500,
        "cam_low": 1000,
    }
)

for cam_name, sn, pipe, cfg in zip(camera_names, camera_sns, pipes, cfgs):
    print(sn)
    cfg.enable_device(sn)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, FPS)
    profile = pipe.start(cfg)
    device = profile.get_device()

    # https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.option.html # not used here
    # https://dev.intelrealsense.com/docs/high-dynamic-range-with-stereoscopic-depth-cameras#section-2-4-manual-vs-auto-exposure
    advnc_mode = rs.rs400_advanced_mode(device)
    intensity_set_point = mean_intensity_set_point_config[cam_name]
    min_loop = 5
    max_loop = 100
    for loop_id in range(max_loop):
        # Read-modify-write of the AE control table
        ae_ctrl = advnc_mode.get_ae_control()
        if ae_ctrl.meanIntensitySetPoint == intensity_set_point and loop_id > min_loop:
            print("setting meanIntensitySetPoint SUCCESS\n\n")
            break
        else:
            ae_ctrl.meanIntensitySetPoint = intensity_set_point
            advnc_mode.set_ae_control(ae_ctrl)
            print("attempted setting meanIntensitySetPoint")
            time.sleep(0.5)

    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    profiles.append(profile)
    depth_scales.append(depth_scale)

print("\nWAITING FOR FRAMES\n")
for _ in range(3):
    for pipe, cam_name in zip(pipes, camera_names):
        t = time.time()
        try:
            pipe.wait_for_frames()
            print(f"{cam_name} waited {time.time() - t}s")
        except:
            print(f"{cam_name} waited too long: {time.time() - t}s\n\n")
            raise Exception

publishers = [
    rospy.Publisher(cam_name, RGBGrayscaleImage, queue_size=1)
    for cam_name in camera_names
]

# i = 0

print("\n\nREALSENSE PUBLISHER RUNNING\n\n")
no_error = True
t = time.time()
while not rospy.is_shutdown():
    t0 = time.time()
    rgb_imgs = []
    depth_imgs = []
    msgs = []

    for cam_name, pipe, depth_scale in zip(camera_names, pipes, depth_scales):
        try:
            frameset = pipe.wait_for_frames(timeout_ms=TIMEOUT_MS)
        except:
            print("\n\n", cam_name, "failed\n")
            no_error = False
            [pipe.stop() for pipe in pipes]
            break

        color_frame = np.array(frameset.get_color_frame().get_data())
        depth_frame = np.array(frameset.get_depth_frame().get_data())

        rgb_imgs.append(color_frame)
        depth_imgs.append(depth_frame)

        msg = RGBGrayscaleImage()
        msg.header.stamp = rospy.Time.now()
        msg.images.append(cv_bridge.cv2_to_imgmsg(color_frame, encoding="bgr8"))
        msg.images.append(cv_bridge.cv2_to_imgmsg(depth_frame, encoding="mono16"))
        msgs.append(msg)

        h, w = depth_frame.shape
        half_block_size = 5

    if not no_error:
        break

        # if i % 15 == 0:
        # pixel_depth = depth_frame[h//2-half_block_size:h//2+half_block_size, w//2-half_block_size:w//2+half_block_size].mean()
        # pixel_depth = depth_frame.flatten()[depth_frame.flatten().argsort()[-30000:]].mean()

        # nonzero_depth = depth_frame[depth_frame != 0].flatten()
        # pixel_depth = nonzero_depth[nonzero_depth.argsort()[:5000]]
        # pixel_depth_mean = pixel_depth.mean()
        # pixel_depth_min = pixel_depth.min()
        # print(f"{cam_name} depth mean (m):", pixel_depth_mean * depth_scale)
        # print(f"{cam_name} depth min (m):", pixel_depth_min * depth_scale, '\n')

    # print([[mins / scale, maxs / scale] for mins, maxs, scale in zip([0.07, 0.28, 0.07], [0.7, 1., 0.7], depth_scales)])
    # print(np.array(depth_frame.get_data()).shape)
    # if cam_name == camera_names[1]:
    #     print(((np.array(depth_frame.get_data()) * depth_scale) == 0).sum() / (640 * 480))

    # if i % 15 == 0:
    #     print('')
    for pub, msg in zip(publishers, msgs):
        pub.publish(msg)

    # print(f"realsense_publisher {cam_idx} {time.time() - t0:.4f}s")
    # i += 1
