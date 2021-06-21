# USAGE
# python make_video.py --output videos --name video.mp4


import argparse
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, required=True,
                help="path to output directory with videos")
ap.add_argument('-n', '--name', type=str, required=True,
                help="name of video")
args = vars(ap.parse_args())


if not os.path.exists(args['output']):
    os.makedirs(args['output'])

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(os.path.join(args["output"], args['name']), fourcc, 24.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('z'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()