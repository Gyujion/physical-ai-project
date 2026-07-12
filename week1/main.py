import sys
import numpy as np
import matplotlib
import cv2
import time
from collections import deque

def OpenCam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠 연결 실패")
        sys.exit()
    
    print("웹캠 연결 성공")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    return cap

def PlayWebCam(cap):
    prev_frame_time = 0
    FPS = 30
    fps_collector = deque(maxlen=30)

    while True:
        ret,frame = cap.read()
        frame = cv2.flip(frame,1)
        if not ret:
            break
        fps, prev_frame_time = calculate_fps(time.time(),prev_frame_time)
        
        fps_collector.append(fps)
        
        avg_fps = sum(fps_collector) / len(fps_collector)

        gray_frame, blur_frame, edges_frame = preprocess(frame)

        draw_fps(frame,avg_fps)

        cv2.imshow('Result',combine_frames(frame,edges_frame))

        inputKey = cv2.waitKey(1)

        if inputKey == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def calculate_fps(new_frame_time,prev_frame_time):
    fps = 1/(new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    return fps,prev_frame_time

def draw_fps(frame,avg_fps):
    fps_str = "FPS: %0.1f"%avg_fps
    cv2.putText(frame, fps_str,(0,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))

def preprocess(frame):
    gray = to_grayscale(frame)
    blur = apply_blur(gray)
    edges = detect_edges(blur)
    return gray,blur,edges

def to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_blur(gray):
    return cv2.GaussianBlur(gray,(5,5),0)

def detect_edges(blur):
    return cv2.Canny(blur,50,100)

def combine_frames(frame,edges):
    return np.hstack([frame,cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)])

def main():
    cap = OpenCam()
    PlayWebCam(cap)

if __name__ == '__main__':
    main()