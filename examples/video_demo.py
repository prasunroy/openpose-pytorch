import sys
sys.path.append('../')

import cv2
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints


estimator = BodyPoseEstimator(pretrained=True)
videoclip = cv2.VideoCapture('media/example.mp4')

while videoclip.isOpened():
    flag, frame = videoclip.read()
    if not flag:
        break
    keypoints = estimator(frame)
    frame = draw_body_connections(frame, keypoints, thickness=2, alpha=0.7)
    frame = draw_keypoints(frame, keypoints, radius=4, alpha=0.8)
    
    cv2.imshow('Video Demo', frame)
    if cv2.waitKey(20) & 0xff == 27: # exit if pressed `ESC`
        break
videoclip.release()
cv2.destroyAllWindows()
