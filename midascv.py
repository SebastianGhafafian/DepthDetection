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

    #Transform input from midas
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to('cpu')

    # Perform inference
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        output = prediction.cpu().numpy()
        # print(output)

    plt.imshow(output)
    cv2.imshow('CV2frame', frame)
    plt.pause(0.0001)
    # give cv2 time to update frame
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows() 

plt.show()

        