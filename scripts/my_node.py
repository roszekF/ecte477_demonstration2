#!/usr/bin/env python
"""
    my_node.py

    A ROS node that repeats the map and odometry topic to the correct ecte477 
    namespace topics for map and path.

    Subscribed: map/, odom/
    Publishes: ecte477/map/, ecte477/path/
    Services: explore/explore_service
    Created: 2021/04/08
    Author: Brendan Halloran
"""

import rospy
import cv2
import imutils
import numpy as np
import transformations as trans
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped, Quaternion, Vector3, Pose, Point
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge, CvBridgeError
from ecte477_demonstration2.msg import Beacon

class my_node:
    def __init__(self):
        # setup for image processing
        self.bridge = CvBridge()
        self.colour_frame = None
        self.depth_frame = None
        self.mask_frame = None
        self.ranges = [
            {'upper': (119, 241, 92), 'lower': (122, 255, 245), 'name': 'blue', 'RGBA': (0,0,1,1)},
            {'upper': (59, 241, 92), 'lower': (65, 255, 245), 'name': 'green', 'RGBA': (0,1,0,1)},
            {'upper': (0, 241, 92), 'lower': (8, 255, 245), 'name': 'red', 'RGBA': (1,0,0,1)},
            {'upper': (27, 241, 92), 'lower': (33, 255, 245), 'name': 'yellow', 'RGBA': (0.5,0.5,0,1)}]
        self.K = None
        self.transform_cam_to_world = None

        # reading beacons parameters 
        self.beacons = rospy.get_param("~beacons")
        # for storing beacon IDs
        self.discovered_beacons = []

        # Object for storing Markers
        self.marker_array = MarkerArray()

        # Object for storing path
        self.path = Path()
        self.path.header.frame_id = "odom"

        # Subs and pubs
        self.subscriber_map = rospy.Subscriber('/map', OccupancyGrid, self.callback_map)
        self.subscriber_odom = rospy.Subscriber('/odom', Odometry, self.callback_odom)
        # for Gazebo8, use normal topic and 'Image' data type instead of compressed
        self.subscriber_colour = rospy.Subscriber('/camera/rgb/image_raw', Image, self.callback_colour)
        self.subscriber_depth = rospy.Subscriber('/camera/depth/image_raw', Image, self.callback_depth)
        self.subscriber_camera_info = rospy.Subscriber('camera/rgb/camera_info', CameraInfo, self.callback_camera_info)
        self.publisher_map = rospy.Publisher('/ecte477/map', OccupancyGrid, queue_size=1)
        self.publisher_path = rospy.Publisher('/ecte477/path', Path, queue_size=1)
        self.publisher_markers = rospy.Publisher('/ecte477/markers', MarkerArray, queue_size=10)
        self.publisher_beacon = rospy.Publisher('/ecte477/beacons', Beacon, queue_size=10)

        # wait for other nodes to start
        rospy.sleep(5)
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.loop()
            r.sleep()

    def find_colors(self):
        if self.colour_frame == None:
            return
        self.mask_frame = self.colour_frame

        colour_contours = []

        frame = self.colour_frame
        # prepare the frame for tresholding
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # loop through the color treshold ranges
        for limit in self.ranges:
            local_mask = cv2.inRange(hsv, limit['upper'], limit['lower'])
            local_mask = cv2.erode(local_mask, None, iterations=2)
            local_mask = cv2.dilate(local_mask, None, iterations=2)
            
            contours = cv2.findContours(local_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)

            # check if there are any contours
            if len(contours) != 0:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                # for accurate location only continue if the contour is big enough
                if cv2.contourArea(largest_contour) < 10000:
                    return
                # don't process contours close to the edge
                if (x+w/2) < 100 or (x+w/2) > (1920-100):
                    return
                

                # append the largest ('nearest') contour to the array
                # along with it's Y coordinate, color name, and center point and RGBA color value
                colour_contours.append({'contour':largest_contour, 'height': y, 'name': limit['name'], 'centre': (x+w/2, y+h/2), 'RGBA': limit['RGBA']})
                
        # check if only one beacon (two contours) is in front
        if len(colour_contours) != 2:
            return
        # sort the contours array by the Y coordinate (top contour first)
        colour_contours.sort(key=lambda x:x['height'])

        colour_masks = self.colour_frame
        for colour in colour_contours:
            x, y, w, h = cv2.boundingRect(colour['contour'])
            # draw the colour contour
            colour_masks = cv2.rectangle(colour_masks, (x, y), (x+w, y+h), (0,0,255), 2)
            # draw a circle at the center
            colour_masks = cv2.circle(colour_masks, colour['centre'], radius=10, color=(0,255,0), thickness=-1)
        self.mask_frame = colour_masks

        # check if the centres of the contours are on top of each other (the same x coordinate) wihin tolerance
        if (abs(colour_contours[0]['centre'][0] - colour_contours[1]['centre'][0]) > 20):
            return

        for beacon in self.beacons:
            if beacon['top'] == colour_contours[0]['name'] and beacon['bottom'] == colour_contours[1]['name']:
                # Check if the beacon was already discovered 
                if beacon['id'] in self.discovered_beacons:
                    return
                # otherwise add to the list
                self.discovered_beacons.append(beacon['id'])
                beacon_id = beacon['id']
                print "MATCH!!"

        print "(x:{}, y:{}) Colours from the top: {}, {}".format(colour_contours[0]['centre'][0], colour_contours[0]['centre'][1], colour_contours[0]['name'], colour_contours[1]['name'])

        if self.K == None or self.transform_cam_to_world == None or self.depth_frame == None:
            return
        
        # x, y of the top colour contour
        x, y = colour_contours[0]['centre']

        # calculate position of the beacon
        depth = self.depth_frame[y, x] + 0.1
        p_h = np.array([[x], [y], [1]])
        p3d = depth * np.matmul(np.linalg.inv(self.K), p_h)
        p3d_h = np.array([[p3d[2][0]], [-p3d[0][0]], [-p3d[1][0]], [1]])
        p3d_w_h = np.matmul(self.transform_cam_to_world, p3d_h)
        # result is an array of [x, y, z]
        p3d_w = np.array([[p3d_w_h[0][0]/p3d_w_h[3][0]], [p3d_w_h[1][0]/p3d_w_h[3][0]], [p3d_w_h[2][0]/p3d_w_h[3][0]]])

        # Construct the Marker
        marker = Marker()
        marker.header.seq= marker.id = beacon_id
        marker.header.frame_id = 'map'
        marker.header.stamp = rospy.Time.now()
        marker.pose = Pose(Point(p3d_w[0], p3d_w[1], 1), Quaternion(0.0, 1.0, 0.0, 1.0))
        marker.scale = Vector3(0.2, 0.2, 0.2)
        marker.color = ColorRGBA(*colour_contours[0]['RGBA'])

        # add to the MarkerArray and publish
        self.marker_array.markers.append(marker)
        self.publisher_markers.publish(self.marker_array)

        beacon = Beacon()
        beacon.header.seq= marker.id = beacon_id
        beacon.header.frame_id = 'map'
        beacon.header.stamp = rospy.Time.now()
        beacon.position = Point(p3d_w[0], p3d_w[1], 1)
        beacon.top = colour_contours[0]['name']
        beacon.bottom = colour_contours[1]['name']

        self.publisher_beacon.publish(beacon)

    def callback_colour(self, colour_image):
        # rospy.loginfo('[Image Processing] callback_colour')
        # convert the image to cv2 format
        try:
            self.colour_frame = self.bridge.imgmsg_to_cv2(colour_image, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_depth(self, depth_image):
        # rospy.loginfo('[Image Processing] callback_depth')
        try:
            self.depth_frame = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

    def callback_camera_info(self, camera_info):
        self.K = np.array(camera_info.K).reshape([3, 3])

    # Simply repeact the map data to the correct topic
    def callback_map(self, data):
        self.publisher_map.publish(data)
        
    # Turn the odometry info into a path and repeat it to the correct topic
    def callback_odom(self, data):
        pose = PoseStamped()
        pose.pose = data.pose.pose
        self.path.poses.append(pose)
        self.publisher_path.publish(self.path)
        # for image precessing 
        self.transform_cam_to_world = trans.msg_to_se3(data.pose.pose)

    def loop(self):
        # run the image precessing function
        self.find_colors()
        # display the results
        if self.mask_frame != None and self.depth_frame != None:
            # cv2.imshow('Colour Image', self.colour_frame)
            # cv2.imshow('Depth Image', self.depth_frame)

            scale_percent = 50 # percent of original size
            width = int(self.mask_frame.shape[1] * scale_percent / 100)
            height = int(self.mask_frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            
            # resize image
            resized = cv2.resize(self.mask_frame, dim, interpolation = cv2.INTER_AREA)
            
            cv2.imshow('Masked Image', resized)
            resp = cv2.waitKey(80)
            if resp == ord('c'):
                rospy.signal_shutdown('C key was pressed!')
	
	
	
# Main function
if __name__ == '__main__':
    rospy.init_node('my_node', anonymous=True)
    rospy.loginfo("Starting My Node!")
    mn = my_node()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting Down My Node!")
