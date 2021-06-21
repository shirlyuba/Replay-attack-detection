# USAGE
# python liveness_demo.py --model face_final_model.pth --input input_video.mp4 --output output_video.mp4


from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import os
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
from src.matrix_diff import F_feature
from src.model import liveness_model


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=False,
	help="path to input video")
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained model")
ap.add_argument("-o", "--output", type=str, required=False,
	help="path to output video")
args = vars(ap.parse_args())


protoPath = os.path.sep.join(["FaceDetector", "deploy.prototxt"])
modelPath = os.path.sep.join(["FaceDetector","res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

if args["input"] is None:
	vs = VideoStream(src=0).start()
else:
	vs = cv2.VideoCapture(args["input"])


if args["output"] is not None:
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(args["output"], fourcc, 24.0, (640,480))

transform_for_train_and_val = transforms.Compose(
	[
		transforms.ToTensor(),
	]
)

lv_model = liveness_model()
lv_model.load_state_dict(torch.load(args["model"], map_location=torch.device('cpu'))['state_dict'])
lv_model.eval()


classes = {0:'fake', 1:'real'}
prev_face = [0]

while True:

	if args["input"] is None:
		frame = vs.read()
	else:
		(grabbed, frame) = vs.read()
		if not grabbed:
			break

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if len(prev_face) > 1:
			map = F_feature(prev_face, frame).astype('float32')

			if confidence > 0.5:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				startX = max(0, startX)
				startY = max(0, startY)
				endX = min(w, endX)
				endY = min(h, endY)

				face_map = map[startY:endY, startX:endX]
				face_map = cv2.resize(face_map, (256, 256))
				img = Image.fromarray(face_map, "RGB")
				img = transform_for_train_and_val(img)
				im = Variable(img, requires_grad=True).unsqueeze(0)
				preds = F.softmax(lv_model(im))[0][1].cpu().detach().numpy()
				if preds > 0.79:
					j = 1
				else:
					j = 0
				label = classes[j]

				if label == 'real':
					cv2.rectangle(frame, (startX, startY), (endX, endY),
								  (0, 255, 0), 2)
					label = "{}: {:.4f}".format(label, preds)
					cv2.putText(frame, label, (startX, startY - 10),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				else:
					cv2.rectangle(frame, (startX, startY), (endX, endY),
									(0, 0, 255), 2)
					label = "{}: {:.4f}".format(label, preds)
					cv2.putText(frame, label, (startX, startY - 10),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		prev_face = np.copy(frame)

	cv2.imshow("Frame", frame)
	if args["output"] is not None:
		out.write(frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("z"):
		break

cv2.destroyAllWindows()

if args["input"] is None:
	vs.stop()
else:
	vs.release()