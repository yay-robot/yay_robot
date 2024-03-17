from calendar import c
import sys
import numpy as np

# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API
import cv2

# Setup:
# ctx = rs.context()
# devices = ctx.devices

sns = ["242522072494", "127122270146", "128422270679"]
pipes = [rs.pipeline() for _ in range(len(sns))]
cfgs = [rs.config() for _ in range(len(sns))]

for sn, pipe, cfg in zip(sns, pipes, cfgs):
    print(sn)
    cfg.enable_device(sn)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    profile = pipe.start(cfg)

    for _ in range(5):
        pipe.wait_for_frames()

while True:
    frames = []

    for pipe in pipes:
        frameset = pipe.wait_for_frames()
        color_frame = frameset.get_color_frame()
        frames.append(color_frame.get_data())

    cv2.imshow("", np.concatenate(frames, axis=1))
    cv2.waitKey(1)


# # import pdb; pdb.set_trace()
# side_cam_sn = '127122270146'
# top_cam_sn = '242522072494'
# cfg.enable_device(side_cam_sn)
# cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
# profile = pipe.start(cfg)

# # Skip 5 first frames to give the Auto-Exposure time to adjust
# for x in range(30):
#   pipe.wait_for_frames()

# # Store next frameset for later processing:

# while True:
#   frameset = pipe.wait_for_frames()
#   color_frame = frameset.get_color_frame()
#   # import pdb; pdb.set_trace()

#   cv2.imshow('', np.array(color_frame.get_data()))
#   cv2.waitKey(1)
