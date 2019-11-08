import pantilthat
import time
import numpy as np
import cv2
from imutils.video import VideoStream
from lgsensing.utils import centerpoint, midpoint, serve_feed

class Camera:
    """
    This is the camera class, based on RPi Camera module.
    """
    def __init__(self, **kwargs):
        self.ip_address = '192.168.1.98'
        self.vs = VideoStream(usePiCamera=1).start()
        time.sleep(2.0)
        self.interested_objects = None
        for kw, arg in kwargs.items():
            if kw == 'net':
                self.net = arg
            if kw == 'classes':
                self.classes = arg
            if kw == 'interested_objects':
                self.interested_objects = arg
        try:
            self.detector = self.Detector(self.capture, self.net,
                                          self.classes, self.interested_objects)
        except: print('detector not instantiated.')

    class Detector:
        """ classify and detect objects """
        def __init__(self, capture, net, classes, interested_objects):
            self.net = net
            self.classes = classes
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            self.capture = capture
            self.interested_objects = interested_objects

        def find_angle(self, x_coord, frame_width):
            """ find the angle from the camera to the center of a boundary box """
            angle = x_coord / frame_width * 180
            return angle

        def generate_detections(self, frame, interested_objects):
            """ returns classifications and detections of the given frame  """

            # prepare net
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                         0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()

            # delete non-interested detections
            if interested_objects:
                obj_ids = [self.classes.index(obj) for obj in interested_objects]
                tf_array = np.asarray([1 if e in obj_ids else 0 for e in detections[0, 0, :, 1]])
                detections = np.compress(tf_array, detections, axis=2)

            # obtain information about each detection
            frame_detections = {}
            for e, i in enumerate(np.arange(0, detections.shape[2])):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.2:
                    frame_detections[e] = {}

                    object_id = int(detections[0, 0, i, 1])
                    box_coords = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    box_coords = box_coords.astype('int')
                    midX, midY = centerpoint(box_coords)
                    angle = self.find_angle(midX, midY)

                    frame_detections[e]['object_id'] = object_id
                    frame_detections[e]['box_coords'] = box_coords
                    frame_detections[e]['midpoint'] = (midX, midY)
                    frame_detections[e]['confidence'] = confidence
                    frame_detections[e]['angle'] = angle

            return frame_detections

        def display_detections(self, frame, frame_detections):
            """ display detection on the frame """
            font = cv2.FONT_HERSHEY_SIMPLEX

            # draw bounding box and label for each detection
            for detection in frame_detections.values():

                object_id = detection['object_id']
                (startX, startY, endX, endY) = detection['box_coords']
                midX, midY = detection['midpoint']
                confidence = detection['confidence']

                name = self.classes[object_id]
                color = self.colors[object_id]
                label = "{}: {}%".format(name, int(confidence * 100))

                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), font, 0.5, color, 2)

                if 'distance' in detection.keys():
                    distance = detection['distance']
                    label = "depth: {} mm".format(int(distance))
                    cv2.putText(frame, label, (startX, y-15), font, 0.5, color, 2)

                if 'proximity' in detection.keys():
                    for item in detection['proximity']:
                        guest_obj = item[0]
                        prox = item[1]
                        warning = "In proximity: {} mm".format(int(prox))
                        # TODO: check if guest_obj will give a generic 'bottle' or 'that' 'bottle'
                        guestX, guestY = frame_detections[guest_obj]['midpoint']
                        if guestX < midX:
                            X = guestX
                            Y = guestY
                        else: X, Y = midX, midY
                        cv2.line(frame, (midX, midY), (guestX, guestY), color)
                        cv2.putText(frame, warning, (X, Y), font, 0.5, color, 2)

            return frame

        def iterate_detections(self):
            """ return detected frame iterator """
            while True:
                raw_frame = self.capture()
                frame_detections = self.generate_detections(
                    raw_frame, self.interested_objects)
                frame = self.display_detections(raw_frame, frame_detections)
                yield frame

    def orient(self, pan_angle=0, tilt_angle=0):
        """  pan-tilt hat accessory adjusts the direction of the camera """
        pantilthat.pan(pan_angle)
        pantilthat.tilt(tilt_angle)

    def capture(self):
        """ capture camera data as frame """
        frame = self.vs.read()
        frame = cv2.flip(frame,-1)
        return frame

    def serve_unprocessed_feed(self):
        """ serve raw camera feed """
        def iterate_raw_frame():
            while True:
                frame = self.capture()
                yield frame

        frame_iterator = iterate_raw_frame()
        serve_feed(self.ip_address, port=2000, frame_iterator=frame_iterator)

    def serve_detected_feed(self):
        """ serve camera feed with detections """
        frame_iterator = self.detector.iterate_detections()
        serve_feed(self.ip_address, port=2100, frame_iterator=frame_iterator)
