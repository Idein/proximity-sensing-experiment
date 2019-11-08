from scipy.spatial import distance as dist
from itertools import product
from flask import Flask, Response
import datetime
import cv2
import math


def midpoint(ptA, ptB):
    return (int((ptA[0] + ptB[0]) * 0.5), int((ptA[1] + ptB[1]) * 0.5))

def centerpoint(box_coords):
    (startX, startY, endX, endY) = box_coords
    midX, midY = midpoint((startX, startY), (endX, endY))
    return midX, midY

def proximity(depth1, depth2, angle):
    proximity = math.sqrt(depth1 ** 2 + depth2 ** 2 -
                          2 * depth1 * depth2 * math.cos(angle))
    return proximity



def min_distance(box1, box2):
    """
    Given the vertices coordinates of two bounding boxes, compute the shortest
    distance between vertices from the two boxes
    """
    (sX1, sY1, eX1, eY1) = box1
    (sX2, sY2, eX2, eY2) = box2
    b1_vertices = [(sX1, sY1), (sX1, eY1), (eX1, sY1), (eX1, eY1)]
    b2_vertices = [(sX2, sY2), (sX2, eY2), (eX2, sY2), (eX2, eY2)]
    pairs = list(product(b1_vertices, b2_vertices))
    distances = [dist.euclidean(*pair) for pair in pairs]
    (min_D, index) = min((v, i) for i, v in enumerate(distances))
    return min_D, pairs[index][0], pairs[index][1]

def serve_feed(ip_address, port, frame_iterator):
    """ serve feed via Flask """
    app = Flask(__name__)

    def generate():
        while True:
            frame = next(frame_iterator)
            timestamp = datetime.datetime.now()
            cv2.putText(frame, timestamp.strftime('%A %d %B %Y %I:%M:%S%p'),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 0, 255), 1)

            (flag, jpg) = cv2.imencode('.jpg', frame)

            if not flag:
                continue

            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(jpg) + b'\r\n')

    @app.route('/video_feed')
    def feed():
        """ return the response generated along with the specific media
        type (mime type)"""
        return Response(generate(),
                        mimetype='multipart/x-mixed-replace; '
                                 'boundary=frame')

    app.run(host=ip_address, port=port, debug=True, use_reloader=False)
