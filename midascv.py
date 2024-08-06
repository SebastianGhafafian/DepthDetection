import cv2
import torch
import matplotlib.pyplot as plt

# Load the Midas model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS", 'MiDaS_small')
midas.to('cpu')
midas.eval()

#transforms image to perform depth estimation (pipeline)
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

# OpenCV
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('CV2frame', frame)
    # give cv2 time to update frame
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows() 