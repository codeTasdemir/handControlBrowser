from selenium import webdriver
import cv2
import mediapipe
import pyttsx3
import pyautogui

camera = cv2.VideoCapture(0)
mpHands = mediapipe.solutions.hands
hands = mpHands.Hands()
mpDraw = mediapipe.solutions.drawing_utils
checkThumbsUp = False
engine = pyttsx3.init()

while True:
    success, img = camera.read()

    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hlms = hands.process(imgRgb)

    height, width, channel = img.shape

    if hlms.multi_hand_landmarks:
        for handlandmarks in hlms.multi_hand_landmarks:

            for fingerNum , landmark in enumerate(handlandmarks.landmark):
                positionX , positionY = int(landmark.x * width), int(landmark.y * height)
                cv2.putText(img, str(fingerNum), (positionX , positionY) , cv2.FONT_HERSHEY_SIMPLEX, 0.5 ,(255,255,0,) ,2)
                finger8x, finger8y = handlandmarks.landmark[8].x, handlandmarks.landmark[8].y
                finger12x, finger12y = handlandmarks.landmark[12].x, handlandmarks.landmark[12].y

                if fingerNum == 8 and landmark.y > (finger8y and finger12y):
                    pyautogui.scroll(30)

                if fingerNum == 8 and landmark.y < (finger8y and finger12y):
                    pyautogui.scroll(-30)

            mpDraw.draw_landmarks(img, handlandmarks, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Camera", img)


    if cv2.waitKey(1) & 0xFF == ord("q"):
       break


