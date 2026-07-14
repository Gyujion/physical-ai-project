import sys
import numpy as np
import matplotlib
import cv2
import time
from collections import deque
RED_LOWER1 = np.array([0, 120, 70])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170,120,70])
RED_UPPER2 = np.array([179,255,255])

BLUE_LOWER = np.array([100, 150, 50])
BLUE_UPPER = np.array([130, 255, 255])

GREEN_LOWER = np.array([40, 70, 70])
GREEN_UPPER = np.array([80, 255, 255])

YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([40, 255, 255])

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

        mask = create_mask(frame)
        objects = find_objects(mask)
        print(objects)  #튜플리스트 잘 생성됐는지 확인용 디버깅코드

        draw_fps(frame,avg_fps)
        cv2.imshow('Result',combine_frames(frame,mask))

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

# HSV 변환 및 색 마스킹
def create_mask(frame):
    src_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) # 원본프레임을 HSV로 변환
    target_color = "green"
    final_mask = None
    if target_color == "red":
        mask1 = cv2.inRange(src_hsv,RED_LOWER1,RED_UPPER1)
        mask2 = cv2.inRange(src_hsv,RED_LOWER2,RED_UPPER2)
        final_mask = cv2.bitwise_or(mask1,mask2)
    elif target_color == "blue":
        final_mask = cv2.inRange(src_hsv,BLUE_LOWER,BLUE_UPPER)
    elif target_color == "green":
        final_mask = cv2.inRange(src_hsv,GREEN_LOWER,GREEN_UPPER)
    elif target_color == "yellow":
        final_mask = cv2.inRange(src_hsv,YELLOW_LOWER,YELLOW_UPPER)

    return final_mask

# Contour 검출 및 중심 좌표
def find_objects(mask):
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for contour in contours:

        area = cv2.contourArea(contour) #영역이 감싸는 면적 반환

        if area < 500:
            continue    # 만약 면적이 500 이하라면 무시

        x,y,w,h = cv2.boundingRect(contour)    #주어진 점을 감싸는 최소 크기 사각형 반환
        # 중심 구하기
        cx = x + w//2   
        cy = y + h//2

        tuple_data = (cx,cy,area,(x,y,w,h))
        objects.append(tuple_data)

    return objects

def main():
    cap = OpenCam()
    PlayWebCam(cap)

if __name__ == '__main__':
    main()