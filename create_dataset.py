# USAGE
# python create_dataset.py --input videos/real.mov --output Dataset/Real
# python create_dataset.py --input videos/fake.mp4 --output Dataset/Fake

import numpy as np
import argparse
from imutils.video import VideoStream
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=False,
	help="path to input video")
ap.add_argument("-o", "--output", type=str, required=True,
	help="path to output directory")
args = vars(ap.parse_args())


protoPath = os.path.sep.join(["FaceDetector", "deploy.prototxt"])
modelPath = os.path.sep.join(["FaceDetector",
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


if args["input"] is None:
	vs = VideoStream(src=0).start()
else:
	vs = cv2.VideoCapture(args["input"])

if not os.path.exists(args['output']):
	os.makedirs(args['output'])

read = 0
saved = 0


while True:

	if args["input"] is None:
		frame = vs.read()
	else:
		(grabbed, frame) = vs.read()
		if not grabbed:
			break

	read += 1

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	net.setInput(blob)
	detections = net.forward()

	if len(detections) > 0:

		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		if confidence > 0.5:
			if frame != []:

				p = os.path.sep.join([args["output"],"{}.png".format(saved)])
				cv2.imwrite(p, frame)
				saved += 1
				print("[INFO] saved {} to disk".format(p))

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("z"):
		break

cv2.destroyAllWindows()

if args["input"] is None:
	vs.stop()
else:
	vs.release()