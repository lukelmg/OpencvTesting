import cv2
import numpy as np
from matplotlib import pyplot as plt
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)  # set new dimensionns to cam object (not cap)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

while True:
    _, frame = cap.read()
    _, box = cap.read()
    _, toblur = cap.read()
    _, blurbox = cap.read()

    blurbox = cv2.blur(blurbox, (30,30))

    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)


    #Duck

    low_yellow = np.array([15, 50, 140])
    high_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, low_yellow, high_yellow)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        cv2.drawContours(frame, contour, -1, (98, 3, 252), 3)

    if len(contours) != 0:
        for mask_contour in contours:
            if cv2.contourArea(mask_contour) > 500:
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(box, (x, y), (x + w, y + h), (98, 3, 252), 3)


    blur = cv2.blur(toblur, (30,30))
    duck_blur_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    duck_mask_blur = cv2.inRange(duck_blur_hsv, low_yellow, high_yellow)

    duck_blur_contours, _ = cv2.findContours(duck_mask_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in duck_blur_contours:
        cv2.drawContours(blur, contour, -1, (98, 3, 252), 3)


    if len(duck_blur_contours) != 0:
        for mask_contour in duck_blur_contours:
            if cv2.contourArea(mask_contour) > 1000:
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(blurbox, (x, y), (x + w, y + h), (98, 3, 252), 3) #draw rectangle around contours in normal video [marshmallow]

    #Marshmallow
    low_white = np.array([0, 0, 180])
    high_white = np.array([255, 150, 255])
    mashmallow_mask = cv2.inRange(hsv, low_white, high_white)

    marshmallow_contours, _ = cv2.findContours(mashmallow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #find contours in normal video [marshmallow]

    for contour in marshmallow_contours:
        cv2.drawContours(frame, contour, -1, (227, 252, 3), 3) #draw contours in normal video [marshmallow]

    if len(marshmallow_contours) != 0:
        for mask_contour in marshmallow_contours:
            if cv2.contourArea(mask_contour) > 1000:
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(box, (x, y), (x + w, y + h), (227, 252, 3), 3) #draw rectangle around contours in normal video [marshmallow]


    marshmallow_blur_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    marshmallow_mask_blur = cv2.inRange(marshmallow_blur_hsv, low_white, high_white)

    marshmallow_blur_contours, _ = cv2.findContours(marshmallow_mask_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in marshmallow_blur_contours:
        cv2.drawContours(blur, contour, -1, (227, 252, 3), 3)

    if len(marshmallow_blur_contours) != 0:
        for mask_contour in marshmallow_blur_contours:
            if cv2.contourArea(mask_contour) > 1000:
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(blurbox, (x, y), (x + w, y + h), (227, 252, 3), 3) #draw rectangle around contours in normal video [marshmallow]

    cv2.imshow("Original Frame", frame)
    cv2.imshow("Duck Mask", mask)
    cv2.imshow("Marshmallow Mask", mashmallow_mask)
    cv2.imshow("Boxes", box)
    cv2.imshow("Blur", blur)
    cv2.imshow("Blur Boxes", blurbox)
    key = cv2.waitKey(1)
    if key == 27:
        break
        
cap.release()
cv2.destroyAllWindows()