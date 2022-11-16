#!/usr/bin/env python3

import torch
import matplotlib.pyplot as plt
import numpy as np
import copy
import cv2
from custom_resnet18 import TorchVisionResNet18
from torchvision import transforms
from torchvision.utils import save_image
import PIL

dataColor = (0, 255, 0)
font = cv2.FONT_HERSHEY_SIMPLEX
fx, fy, fh = 10, 50, 45
takingData = 0
count = 0
showMask = 0

classes = ['FIVE', 'FOUR', 'NONE', 'ONE', 'THREE', 'TWO']

def binaryMask(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7, 7), 3)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    ret, new = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return new


def main():
    global font, size, fx, fy, fh
    global dataColor
    global className, count
    global showMask

    model = TorchVisionResNet18()
    model.load_state_dict(torch.load("resnet18.pth", map_location=torch.device("cpu")))
    model.eval()

    x0, y0, width = 800, 170, 300

    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)

    while True:
        # Get camera frame
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)  # mirror

        window = copy.deepcopy(frame)
        cv2.rectangle(window, (x0, y0), (x0 + width - 1, y0 + width - 1), dataColor, 12)

        # get region of interest
        roi = frame[y0 : y0 + width, x0 : x0 + width]

        roi = binaryMask(roi)

        # apply processed roi in frame
        if showMask:
            window[y0 : y0 + width, x0 : x0 + width] = cv2.cvtColor(
                roi, cv2.COLOR_GRAY2BGR
            )


        # transform and predict
        trans = transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
        
        roi_img = PIL.Image.fromarray(roi)
        tensor = trans(roi_img).unsqueeze(0)
        
        img_real_time = tensor[0]
        save_image(img_real_time, './img_real_time.png')

        pred = torch.argmax(model(tensor), dim=1).cpu().numpy()[0]
        print("PRED", f"Index: {pred} - Class: {classes[pred]}")

        # show prediction
        cv2.putText(window, 'Prediction: %s' % (classes[pred]), (x0,y0-25), font, 1.0, (245, 110, 65), 2, 1)

        # show the window
        cv2.imshow("Original", window)

        # Keyboard inputs
        key = cv2.waitKey(10) & 0xFF

        # use q key to close the program
        if key == ord("q"):
            break
        
        # adjust the position of window
        elif key == ord("i"):
            y0 = max((y0 - 5, 0))
        elif key == ord("k"):
            y0 = min((y0 + 5, window.shape[0] - width))
        elif key == ord("j"):
            x0 = max((x0 - 5, 0))
        elif key == ord("l"):
            x0 = min((x0 + 5, window.shape[1] - width))

    cam.release()


if __name__ == "__main__":
    main()
