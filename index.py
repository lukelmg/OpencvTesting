import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    width  = cap.get(3)  # float `width`
    height = cap.get(4)  # float `height`

    B,G,R = frame[int(height/2), int(width/2)]
    print(B,G,R)

    # Red color
    low_yellow = np.array([15, 0, 160])
    high_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)

    #cv2.rectangle(frame, (320, 240), (50, 50), (int(B), int(G), int(R)), 3)
    cv2.circle(frame, (320, 240), 5, (int(B), int(G), int(R)), 4, 3)

    mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finding contours in mask image

    # Finding position of all contours
    if len(mask_contours) != 0:
        for mask_contour in mask_contours:
            if cv2.contourArea(mask_contour) > 500:
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3) #drawing rectangle

    cv2.imshow("Frame", frame)
    cv2.imshow("Yellow Mask", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break