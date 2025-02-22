#!/usr/bin/python
'''
	Author: Igor Maculan - n3wtron@gmail.com
	A Simple mjpg stream http server
'''
import cv2
from PIL import Image
import threading
from http.server import BaseHTTPRequestHandler,HTTPServer
from socketserver import ThreadingMixIn
from io import BytesIO
import time
from datetime import datetime
capture=None

class CamHandler(BaseHTTPRequestHandler):
	def do_GET(self):
		if self.path.endswith('.mjpg'):
			self.send_response(200)
			self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
			self.end_headers()
			stream_video = BytesIO()
			datetime_format = "{0:%Y-%m-%d %H:%M:%S}"
			font = cv2.FONT_HERSHEY_SIMPLEX
			while True:
				try:
					rc,img = capture.read()
					if not rc:
						continue
					imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
					cv2.putText(imgRGB, datetime_format.format(datetime.now()), (0, 50), font, 1,(255,255,255),2,cv2.LINE_AA)
					jpg = Image.fromarray(imgRGB)
					jpg.save(stream_video,'JPEG')
					stream_video.truncate()
					stream_video.seek(0)
					self.wfile.write(b"--jpgboundary")
					self.send_header('Content-type','image/jpeg')
					self.send_header('Content-length',len(stream_video.getvalue()))
					self.end_headers()
					self.wfile.write(stream_video.getvalue())
					self.wfile.write(b'\r\n')

				except KeyboardInterrupt:
					break
			return
		if self.path.endswith('.html'):
			self.send_response(200)
			self.send_header('Content-type','text/html')
			self.end_headers()
			self.wfile.write(b'<html><head></head><body>')
			self.wfile.write(b'<img src="./cam.mjpg"/>')
			self.wfile.write(b'</body></html>')
			return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
	"""Handle requests in a separate thread."""

def main():
	global capture
	capture = cv2.VideoCapture(0)
	capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320); 
	capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240);
	capture.set(cv2.CAP_PROP_SATURATION,0.2);
	global img
	try:
		server = ThreadedHTTPServer(('', 8080), CamHandler)
		print("server started")
		server.serve_forever()
	except KeyboardInterrupt:
		capture.release()
		server.socket.close()

if __name__ == '__main__':
	main()

