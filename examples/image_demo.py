import sys
sys.path.append('../')

import cv2
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints


estimator = BodyPoseEstimator(pretrained=True)
image_src = cv2.imread('media/example.jpg')
keypoints = estimator(image_src)
image_dst = draw_body_connections(image_src, keypoints, thickness=4, alpha=0.7)
image_dst = draw_keypoints(image_dst, keypoints, radius=5, alpha=0.8)

while True:
    cv2.imshow('Image Demo', image_dst)
    if cv2.waitKey(1) & 0xff == 27: # exit if pressed `ESC`
        break
cv2.destroyAllWindows()
