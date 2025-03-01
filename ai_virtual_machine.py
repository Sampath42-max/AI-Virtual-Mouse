import cv2 as cv
import mediapipe as mp
import proj as ht
import numpy as np
import pyautogui as pg
import time

pg.FAILSAFE = False

plocX, plocY = 0, 0
clocX, clocY = 0, 0
smoothening = 2  # Lower value for more responsive movement

Wcam, Hcam = 1920, 1080
ptime = 0
frameR = 100  # Reduced frame reduction for better coverage
detector = ht.handDetector(maxHands=1)
wscr, hscr = pg.size()

cap = cv.VideoCapture(0)
cap.set(3, Wcam)
cap.set(4, Hcam)
cap.set(cv.CAP_PROP_FPS, 60)  # Attempt to increase FPS

def safe_click():
    time.sleep(0.1)
    try:
        pg.mouseDown()
        time.sleep(0.05)
        pg.mouseUp()
    except Exception as e:
        print("Click failed:", e)

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to capture image")
        break

    img = detector.findHands(img, draw=False)  # Reduce processing time
    hand_info = detector.findPosition(img)

    if hand_info:
        lmlist, bbox = hand_info
    else:
        lmlist, bbox = [], None  # Avoid errors if no hands detected

    fingers = [0, 0, 0, 0, 0]
    x3, y3 = wscr // 2, hscr // 2  # Default mouse position

    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]  # Index finger
        x2, y2 = lmlist[12][1:]  # Middle finger
        fingers = detector.fingersUp()

        if fingers[1] == 1 and fingers[2] == 0:
            cv.rectangle(img, (frameR, frameR), (Wcam - frameR, Hcam - frameR), (255, 0, 2), 2)

            x3 = np.interp(x1, (0, Wcam), (0, wscr))
            y3 = np.interp(y1, (0, Hcam), (0, hscr))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            pg.moveTo(wscr - clocX, clocY, duration=0.02)
            cv.circle(img, (x1, y1), 10, (2, 0, 2), cv.FILLED)

        if fingers[1] == 1 and fingers[2] == 1:
            result = detector.findDistance(8, 12, img)

            if result:
                length, img, lineInfo = result
                if length < 30:  # More accurate clicking
                    cv.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv.FILLED)
                    safe_click()

    ctime = time.time()
    fps = 1 / (ctime - ptime) if (ctime - ptime) > 0 else 0
    ptime = ctime
    cv.putText(img, str(int(fps)), (20, 50), cv.FONT_HERSHEY_COMPLEX, 3, (255, 0, 155), 3)

    cv.imshow('image', img)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
