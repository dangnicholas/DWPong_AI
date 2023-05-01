from ast import Subscript
import sys
from tabnanny import check
from turtle import pos
from exhibit.motion import motion_subscriber
from exhibit.shared.config import Config
from exhibit.motion.motion_subscriber import MotionSubscriber
import time
import numpy as np

import base64
import pyrealsense2 as rs
import cv2
import threading
from exhibit.shared.utils import Timer

import paho.mqtt.client as mqtt
import numpy as np
import json

from queue import Queue
"""
This is a class to provide motion input using the depth feature of a Realsense D435 depth camera.
The motion data is retrieved by finding the center of mass of the largest depth "blob" and then calculating it's position along the horizontal axis
and sending the value over mqtt
"""

class MotionDriver:

    def configure_pipeline(self):
        # decimation_filter = rs.decimation_filter()
        self.decimation_filter.set_option(rs.option.filter_magnitude, 6)

        print("starting with crop w at {}".format(self.crop_percentage_w * 100))
        print("starting with crop h at {}".format(self.crop_percentage_h * 100))


        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        profile = self.pipeline.start(rs_config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)

        # We will be removing the background of objects more than clipping_distance away
        self.clipping_distance = self.clipping_distance_in_meters / depth_scale
        print(f'the Clipping Distance is : {self.clipping_distance}')

    def get_human(self):
        #try to get the frame 50 times
        #return 0.5
        for i in range(50):
            #print(f"trials: {i}")
            #Timer.start("wait_frame")
            try:
                frames = self.pipeline.wait_for_frames()
            except Exception:
                continue
            depth = frames.get_depth_frame()
            if not depth: continue

            # filtering the image to make it less noisy and inconsistent
            depth_filtered = self.decimation_filter.process(depth)
            depth_image = np.asanyarray(depth_filtered.get_data())
            
            # cropping the image based on a width and height percentage
            w,h = depth_image.shape
            ws, we = int(w/2 - (w * self.crop_percentage_w)/2), int(w/2 + (w * self.crop_percentage_w)/2)
            hs, he = int(h/2 - (h * self.crop_percentage_h)/2), int(h/2 + (h * self.crop_percentage_h)/2)
            depth_cropped = depth_image[ws:we, hs:he]

            # cut off values farther away than clipping_distance
            cutoffImage = np.where((depth_cropped < self.clipping_distance) & (depth_cropped > 0.1), True, False)

            #get the islands of items in depth zone
            avg_x = 0
            avg_x_array = np.array([])
            countB = 0
            for a in range(np.size(cutoffImage,0)):
                for b in range(np.size(cutoffImage,1)):
                    if cutoffImage[a,b] :
                        avg_x += b
                        #print(b)
                        avg_x_array = np.append(avg_x_array,b)
                        countB = countB+1
            
            # if we got no pixels in depth, return dumb value
            if countB <= 40: 
                return 0.5
            avg_x_array.sort()
            islands = []
            i_min = 0
            i_max = 0
            p = avg_x_array[0]
            for index in range(np.size(avg_x_array,0)) :
                n = avg_x_array[index]
                if n > p+1 and not i_min == i_max : # if the island is done
                    islands.append(avg_x_array[i_min:i_max])
                    i_min = index
                i_max = index
                p = n
            if not i_min == i_max: islands.append(avg_x_array[i_min:i_max])
            
            #compare_islands for largest
            bigIsland = np.array([])
            for array in islands:
                if np.size(array,0) > np.size(bigIsland,0): bigIsland = array
            
            #get center of big island
            m = (np.median(bigIsland))


            aligned_frames = self.align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            grey_color2 = 40

            depth_cropped_3d_actual = np.dstack((depth_cropped,depth_cropped,depth_cropped))
            depth_cropped_3d_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_cropped_3d_actual, alpha=0.03), cv2.COLORMAP_RAINBOW)
            # draw line for viz where center is
            depth_cropped_3d_colormap = np.where((depth_cropped_3d_actual < self.clipping_distance) & (depth_cropped_3d_actual > 0.1), depth_cropped_3d_colormap, grey_color2 )
            
            depth_cropped_3d_colormap = cv2.line(depth_cropped_3d_colormap, (int(m),h), (int(m),0), (255,255,255), 1)
            
            buffer = cv2.imencode('.jpg', depth_cropped_3d_colormap)[1].tostring()
            self.depth_feed = base64.b64encode(buffer).decode()

            # emit visual over mqtt
            self.subscriber.emit_depth_feed(self.depth_feed)
            

            # *****************************************************************************************
            # we multiply by 1.4 and subtract -0.2 so that the player can reach the edges of the self game.
            # In other words, we shrunk the frame so that the edges of the self game can be reached without leaving the camera frame
            return (m/(np.size(cutoffImage,1)) * 1.4) -0.2
        print("depth failed")
        return 0.5 # dummy value if we can't successfully get a good one

    def motion_loop(self):
        position = 0

        while True:
            # we need to publish if player is present and then position data?
            if(self.subscriber.game_state == 0): # waiting for confirmation of player
                arrayVals = np.array([]) # empty numpy array to store values to check if person is still
                has_bad_values = False
                print("starting human blob detection")

                for counter in range(0,50):
                    c_value = self.get_human()
                    if c_value == 0.5:
                        # c_value is our dummy value for not seeing a human blob
                        # continue loop back at waiting for interaction
                        has_bad_values = True
                    arrayVals = np.append(arrayVals, c_value)

                # if standard deviation of values from check_for_still_player was too variable (person walking by) or there was no person:
                if np.std(arrayVals) > 0.06 or has_bad_values:
                    # either there wasn't a person blob large enough, or the player wasn't still. Don't start game
                    print("          No still player.        ")
                    self.subscriber.publish("motion/presence", False)
                    continue
                else:
                    print("detected human blob")
                    self.subscriber.publish("motion/presence", True)
            # let's run this about 60 times a second to approximately keep up with frame rate
            # else: # if ready or running
            position = self.get_human()
            self.subscriber.publish("motion/position", position)
            print("Published to motion/position |", position)
            time.sleep(0.016) # wait so we're not spamming as fast as the system can - approx 60 per second is more than enough for a max

    def __init__(self, config=Config.instance(), in_q = Queue(), pipeline = rs.pipeline(), decimation_filter = rs.decimation_filter(), crop_percentage_w = 1.0, crop_percentage_h = 1.0, clipping_distance_in_meters = 1):
        self.q = in_q

        # Realsense configuration
        self.pipeline = pipeline 
        self.decimation_filter = decimation_filter
        
        self.clipping_distance_in_meters = clipping_distance_in_meters
        self.clipping_distance = clipping_distance_in_meters

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        self.config = config

        self.crop_percentage_w = config.CROP_PERCENTAGE_W
        self.crop_percentage_h = config.CROP_PERCENTAGE_H

        self.configure_pipeline() # set up the pipeline for depth retrieval

        self.subscriber = MotionSubscriber()

        self.motion_thread = threading.Thread(target=self.motion_loop)
        self.motion_thread.start()

        self.subscriber.start() # loop the subscriber forever


def main(in_q):
    # main is separated out so that we can call it and pass in the queue from GUI
    config = Config.instance()
    instance = MotionDriver(config = config, in_q = in_q)

if __name__ == "__main__":
    main("")