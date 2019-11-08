from lgsensing.lidar import Lidar
from lgsensing.camera import Camera
from lgsensing.proxdetector import ProxDetector
import argparse
import cv2
import json

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--lidar", required=True,
                help="serial port name of lidar device")
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--classes", required=True,
                help="path to classes json file")
ap.add_argument("-i", "--interestedObjects", nargs="+", required=False,
                help="object(s) to detect")
ap.add_argument("-r", "--range", required=False,
                help="range to flag proximity between interested objects")
args = vars(ap.parse_args())

# read arguments
lidar_port = args["lidar"]
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
classes = json.load(open(args["classes"]))
interested_objects = args["interestedObjects"]
range = int(args["range"])

# instantiate the lidar class by passing the serial port name of lidar device
# lidar = Lidar(SERIAL_PORT)

# # instantiate the camera class
# camera = Camera(net=net, classes=classes, interested_object="bottle")
# # seems the pan-tilt hat is a bit offset
# camera.orient(pan_angle=-5, tilt_angle=-90)
# camera.serve_detected_feed()

pd = ProxDetector(net=net, classes=classes,
                  interested_objects=interested_objects, lidar_port=lidar_port)
pd.orient(pan_angle=-5, tilt_angle=-90)
pd.serve_fused_feed(frame_iterator=pd.get_proximity(range))

