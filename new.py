import cv2
import numpy as np
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    _, frame = cap.read()
    _, box = cap.read()
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    low_yellow = np.array([15, 0, 140])
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


    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Outline", box)
    key = cv2.waitKey(1)
    if key == 27:
        break
        
cap.release()
cv2.destroyAllWindows()