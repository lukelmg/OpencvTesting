import cv2
import numpy as np
from matplotlib import pyplot as plt
from threading import Thread
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)  # set new dimensionns to cam object (not cap)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

def cvstuff():
    _, frame = cap.read()
    _, box = cap.read()
    _, toblur = cap.read()

    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    low_yellow = np.array([15, 50, 140])
    high_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, low_yellow, high_yellow)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        cv2.drawContours(frame, contour, -1, (255, 0, 0), 3)

    if len(contours) != 0:
        for mask_contour in contours:
            if cv2.contourArea(mask_contour) > 500:
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(box, (x, y), (x + w, y + h), (0, 0, 255), 3) #drawing rectangle


    blur = cv2.blur(toblur, (30,30))
    hsv2 = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv2, low_yellow, high_yellow)

    contours2, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour2 in contours2:
        cv2.drawContours(blur, contour2, -1, (152, 252, 3), 3)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Outline", box)
    cv2.imshow("Blur", blur)


t1 = Thread(target=cvstuff)
t1.start()
        
cap.release()
cv2.destroyAllWindows()