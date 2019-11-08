from lgsensing.camera import Camera
from lgsensing.lidar import Lidar
from lgsensing.utils import serve_feed, proximity
from itertools import combinations

class ProxDetector(Camera, Lidar):
    def __init__(self, net, classes, interested_objects, lidar_port):
        Camera.__init__(self, net=net, classes=classes,
                        interested_objects=interested_objects)
        Lidar.__init__(self, lidar_port)

    def get_depth(self):
        """ fuse detections from camera detector with depth data from lidar and
        return frame iterator """
        while True:
            raw_frame = self.capture()
            frame_detections = self.detector.generate_detections(
                raw_frame, self.interested_objects)
            for detection in frame_detections.values():
                angle = detection['angle']
                lidar_datum = self.query(angle)
                distance = lidar_datum[2]
                detection['distance'] = distance
            frame = self.detector.display_detections(raw_frame, frame_detections)
            yield frame

    def get_proximity(self, range):
        """ calculates proximity of interested objects and flags if the range is
        exceeded """
        while True:
            raw_frame = self.capture()
            frame_detections = self.detector.generate_detections(
                raw_frame, self.interested_objects)
            for detection in frame_detections.values():
                angle = detection['angle']
                lidar_datum = self.query(angle)
                distance = lidar_datum[2]
                detection['distance'] = distance
                detection['proximity'] = []
            for obj1, obj2 in combinations(frame_detections.keys(), 2):
                depth1 = frame_detections[obj1]['distance']
                depth2 = frame_detections[obj2]['distance']
                angle = abs(frame_detections[obj1]['angle'] -
                            frame_detections[obj2]['angle'])
                prox = proximity(depth1, depth2, angle) / 10
                if prox < range:
                    frame_detections[obj1]['proximity'].append([obj2, prox])
            frame = self.detector.display_detections(raw_frame, frame_detections)
            yield frame

    def serve_fused_feed(self, frame_iterator):
        """ serve feed """
        serve_feed(self.ip_address, port=2200, frame_iterator=frame_iterator)
