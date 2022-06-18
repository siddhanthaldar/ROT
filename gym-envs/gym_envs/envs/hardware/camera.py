import numpy as np
import cv2
import matplotlib.pyplot as plt 
from xarm.wrapper import XArmAPI
import imageio
import glob
import os
import pickle
import pyrealsense2 as rs
from scipy.ndimage import gaussian_filter

class Camera:
	def __init__(self, width=640, height=480, view="side"):
		# initialize camera
		self._connect_cam()
		self._width = width
		self._height = height
		self.resize = True
		self.crop = True
		self.view = view
		
	def _resize(self, img):
		return cv2.resize(img, (self._height, self._width), interpolation=cv2.INTER_AREA)

	def crop_image_input(self, img, depth):
		if self.view == "side":
			crop_center = [240,325]
			crop_size = [450,450]
			h1 = crop_center[0] - int(crop_size[0]/2)
			h2 = crop_center[0] + int(crop_size[0]/2)
			w1 = crop_center[1] - int(crop_size[1]/2)
			w2 = crop_center[1] + int(crop_size[1]/2)
			return img[h1:h2, w1:w2] , depth[h1:h2, w1:w2]
		elif self.view == "front":
			crop_center = [295,325]
			crop_size = [420,400]
			h1 = crop_center[0] - int(crop_size[0]/2)
			h2 = crop_center[0] + int(crop_size[0]/2)
			w1 = crop_center[1] - int(crop_size[1]/2)
			w2 = crop_center[1] + int(crop_size[1]/2)
			return img[h1:h2, w1:w2] , depth[h1:h2, w1:w2]

	def _connect_cam(self):
		self.pipeline = rs.pipeline()

		# Configure streams
		config = rs.config()
		config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
		config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

		# Start streaming and create frame aligner
		self.pipeline.start(config)
		align_to = rs.stream.color
		self.align = rs.align(align_to)

		for _ in range(100):
			frames = self.pipeline.wait_for_frames()

		self.hole_filling = rs.hole_filling_filter()
		self.colorizer = rs.colorizer()
		self.decimate = rs.decimation_filter()
		self.decimate.set_option(rs.option.filter_magnitude, 2 ** 1)

	def scale_to_255(self, frame):
		scaled_frame = (frame - np.min(frame)) * 255.0 / (np.max(frame)-np.min(frame))
		scaled_frame = 255 - scaled_frame
		return scaled_frame

	def get_frame(self):
		"""
		Img is in uint
		Depth is in milimeters
		"""
		frames = self.pipeline.wait_for_frames()
		aligned_frames = self.align.process(frames)

		aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
		aligned_color_frame = aligned_frames.get_color_frame()
		if not aligned_depth_frame or not aligned_color_frame:
			print("ERROR: no new images receieved !")
			return
		
		aligned_depth_frame = self.hole_filling.process(aligned_depth_frame)
		
		depth_image = np.asanyarray(aligned_depth_frame.get_data())
		
		color_image = np.asanyarray(aligned_color_frame.get_data())
		
		if self.crop:
			color_image , depth_image = self.crop_image_input(color_image,depth_image)

		if self.resize:
			color_image = self._resize(color_image)
			depth_image = self._resize(depth_image)
		
		depth_image = self.scale_to_255(depth_image)
		
		image = np.concatenate((color_image, depth_image[:,:, np.newaxis]), axis=2)
		return image    # Returns image as (height,width,channels)

	def stop(self):
		self.pipeline.stop()
