import transform
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3, Point
from collections import deque
from transform import rotation_matrix_2d, scale_matrix_2d, translate_2d

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

class PublishNode(Node):

    def __init__(self):
        super().__init__('turtlesim_pub')
        self.twist_pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.point_pub = self.create_publisher(Point, 'target_point', 10)

    def publish_coordi(self,wx,wy):
        point_msg = Point()
        point_msg.x = wx
        point_msg.y = wy
        point_msg.z = 0.0
        self.point_pub.publish(point_msg)

        twist_msg = Twist()
        twist_msg.linear.x = wx
        twist_msg.angular.z = wy
        self.twist_pub.publish(twist_msg)


def webcam_connect():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠 연결 실패")
        sys.exit()
    print("웹캠 연결 성공")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    return cap

def open_webCam(cap,ros_node):
    prev_frame_time = 0
    FPS = 30
    fps_collector = deque(maxlen=30)

    plt.ion()

    pixel_history_x = deque(maxlen=100)
    pixel_history_y = deque(maxlen=100)
    world_history_x = deque(maxlen=100)
    world_history_y = deque(maxlen=100)

    frame_count = 0 # 몇 프레임마다 한번씩만 갱신 부하 줄이기 변수

    scale = 0.01
    angle = 0

    while True:

        sense_prev_time = time.time()
        ret,frame = cap.read()
        sense_new_time = time.time()

        sense_time = (sense_new_time - sense_prev_time) * 1000

        frame = cv2.flip(frame,1)
        if not ret:
            break
        fps, prev_frame_time = calculate_fps(time.time(),prev_frame_time)
        
        fps_collector.append(fps)
        
        avg_fps = sum(fps_collector) / len(fps_collector)

        compute_prev_time = time.time()
        mask,coordi = detect(frame,scale,angle)
        compute_new_time = time.time()

        compute_time = (compute_new_time - compute_prev_time) * 1000

        for (cx,cy,wx,wy) in coordi:
            pixel_history_x.append(cx)
            pixel_history_y.append(cy)
            world_history_x.append(wx)
            world_history_y.append(wy)

            ros_node.publish_coordi(wx,wy)
        
        frame_count+=1

        act_prev_time = time.time()
        # 10프레임마다 그래프 갱신
        if frame_count % 10 == 0:
            plt.cla()
            plt.scatter(pixel_history_x,pixel_history_y, color = 'blue',alpha=0.7)
            plt.scatter(world_history_x,world_history_y, color = 'red',alpha=0.7)
            plt.pause(0.001)

        draw_fps(frame,avg_fps,scale,angle)
        cv2.imshow('Result',combine_frames(frame,mask))

        inputKey = cv2.waitKey(1)

        act_new_time = time.time()

        act_time = (act_new_time - act_prev_time) * 1000

        rclpy.spin_once(ros_node, timeout_sec=0.001)

        print(f"Sense: {sense_time} | Compute: {compute_time} | Act: {act_time}")

        if inputKey == ord('q'):
            break
        elif inputKey == ord('a'):
            angle += 5
        elif inputKey == ord('d'):
            angle -= 5
        elif inputKey == ord('+'):
            scale = 1.1
        elif inputKey == ord('-'):
            scale = 0.9
    
    cap.release()
    cv2.destroyAllWindows()

    plt.ioff()
    plt.close()

def calculate_fps(new_frame_time,prev_frame_time):
    fps = 1/(new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    return fps,prev_frame_time

def draw_fps(frame,avg_fps,scale,angle):
    fps_str = "FPS: %0.1f"%avg_fps
    ang_scl = f"Scale: {scale}\nangle: {angle}"
    cv2.putText(frame, fps_str,(0,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
    cv2.putText(frame,ang_scl,(450,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))

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
    #contours는 윤곽선의 점들의 리스트를 가진다.
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    objects = []

    if not contours:
        return objects
    # 이런 식으로 코드를 작성하면 contour의 면적을 반환하고
    # 가장 큰 영역을 반환하기 위해 max 함수를 쓰는데
    # 그걸 key=cv2.contourArea lambda 함수에 넣어서 면적을 기준으로
    # 가장 큰 contour를 반환해준다
    # key=cv2.contourArea는 key=lambda c: cv2.contourArea와 같다 
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    # 면적이 500보다 클 때만 사각형 그림
    if area >= 500:
        x,y,w,h = cv2.boundingRect(largest_contour)    #주어진 점을 감싸는 최소 크기 사각형 반환
        # 중심 구하기
        cx = x + w//2   
        cy = y + h//2

        tuple_data = (cx,cy,area,(x,y,w,h))
        objects.append(tuple_data)

    return objects

#바운딩 박스, 중심 좌표 표시
def draw_detection(frame,objects,scale,angle):
    #화면의 너비와 높이를 구한다.
    #frame.shape는 height,width,channels를 반환하기 때문에 슬라이싱 한다.
    height,width = frame.shape[:2]
    coordi = [] # 좌표이력 전달용
    for obj in objects:
        # 튜플 리스트의 데이터를 언패킹해서 각각의 변수에 저장
        cx,cy,area,(x,y,w,h) = obj

        wx,wy = pixel_to_world(cx,cy,frame,scale,angle)
        # 블로그에서 frame 다음에 (x,y,w,h)를 넣어도 된다는걸 발견했다.
        cv2.rectangle(frame,(x,y,w,h),(0,255,0),3)
        cv2.circle(frame,(cx,cy),1,(0,0,255),-1)

        str_center = f"pixel:({cx},{cy}) | world:({wx},{wy})"

        cv2.putText(frame,str_center,(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
        print(f"[픽셀중심: ({cx},{cy}) | [실세계중심: ({wx},{wy})]")
        coordi.append((cx,cy,wx,wy))    # 좌표이력 저장

    return coordi

def detect(frame,scale,angle):
    mask = create_mask(frame)
    objects = find_objects(mask)
    coordi = draw_detection(frame,objects,scale,angle)

    return mask,coordi

def pixel_to_world(cx,cy,frame,scale,angle):
    H,W = frame.shape[:2]
    tx = -W/2
    ty = -H/2
    sx = scale
    sy = scale
    point = np.array([cx,cy])
    translate = translate_2d(point,tx,ty)
    scale = scale_matrix_2d(sx,sy)
    rot = rotation_matrix_2d(angle)

    wx,wy = rot @ scale @ translate

    return wx,wy

def combine_frames(frame,mask):
    return np.hstack([frame,cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)])

def main(args=None):
    rclpy.init(args=args)
    ros_node = PublishNode()
    cap = webcam_connect()
    open_webCam(cap, ros_node)
    ros_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()    