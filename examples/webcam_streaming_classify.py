from http import server
from threading import Condition
import base64
import io
import logging
import os
import socketserver

import numpy as np
import cv2
from PIL import Image
import argparse
import time

from edgetpu.detection.engine import DetectionEngine


# Parameters
AUTH_USERNAME = os.environ.get('AUTH_USERNAME', 'pi')
AUTH_PASSWORD = os.environ.get('AUTH_PASSWORD', 'picamera')
AUTH_BASE64 = base64.b64encode('{}:{}'.format(AUTH_USERNAME, AUTH_PASSWORD).encode('utf-8'))
BASIC_AUTH = 'Basic {}'.format(AUTH_BASE64.decode('utf-8'))
RESOLUTION = os.environ.get('RESOLUTION', '800x600').split('x')
RESOLUTION_X = int(RESOLUTION[0])
RESOLUTION_Y = int(RESOLUTION[1])
FRAMERATE = int(os.environ.get('FRAMERATE', '30'))
ROTATION = int(os.environ.get('ROTATE', 0))
HFLIP = os.environ.get('HFLIP', 'false').lower() == 'true'
VFLIP = os.environ.get('VFLIP', 'false').lower() == 'true'
USBCAMNO = os.environ.get('USBCAMNO',0)
QUALITY = os.environ.get('QUALITY', 50)

PAGE = """\
<html>
<head>
<title>edgeTPU object identification</title>
</head>
<body>
<h1>edgeTPU object identification</h1>
<img src="stream.mjpg" width="{}" height="{}" />
</body>
</html>
""".format(RESOLUTION_X, RESOLUTION_Y)


class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()
        self.engine = None

    def set_engine(self, engine):
        self.engine = engine

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get('Authorization') is None:
            self.do_AUTHHEAD()
            self.wfile.write(b'no auth header received')
        elif self.headers.get('Authorization') == BASIC_AUTH:
            self.authorized_get()
        else:
            self.do_AUTHHEAD()
            self.wfile.write(b'not authenticated')

    def do_AUTHHEAD(self):
        self.send_response(401)
        self.send_header('WWW-Authenticate', 'Basic realm=\"picamera\"')
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def authorized_get(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                stream_video = io.BytesIO()
                stream_tpu = io.BytesIO()
                _, width, height, channels = engine.get_input_tensor_shape()
                
                while True:
                    ret, color_image = cap.read()
                    if not ret:
                        break

                    prepimg = color_image[:, :, ::-1].copy()
                    prepimg = Image.fromarray(prepimg)

                    start_ms = time.time()
                    results = engine.ClassifyWithInputTensor(input, top_k=1)
                    elapsed_ms = time.time() - start_ms

                    if results:
                        print("%s %.2f\n%.2fms" % (
                            labels[results[0][0]], results[0][1], elapsed_ms*1000.0))


                    ret, img = cap.read()
                    if not ret:
                        break
                    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    jpg = Image.fromarray(imgRGB)
                    jpg.save(stream_video,'JPEG')

                    stream_video.truncate()
                    stream_video.seek(0)
 
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(stream_video.getvalue()))
                    self.end_headers()
                    self.wfile.write(stream_video.getvalue())
                    self.wfile.write(b'\r\n')

            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
    parser.add_argument(
      '--label', help='File path of label file.', required=True)

    args = parser.parse_args()
    res = '{}x{}'.format(RESOLUTION_X, RESOLUTION_Y)

    with open(args.label, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)

    engine = DetectionEngine(args.model)
    cap = cv2.VideoCapture(USBCAMNO)
    cap.set(cv2.CAP_PROP_FPS, FRAMERATE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION_X)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_Y)
 
    try:
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        server.serve_forever()
    finally:
        cap.release()
