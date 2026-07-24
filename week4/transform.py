import numpy as np

#각도(도)를 라디안으로 바꿔 2×2 회전 행렬을 반환합니다
def rotation_matrix_2d(theta_deg):
    rad = theta_deg * np.pi / 180
    rot_mat = np.array([[np.cos(rad), -1*np.sin(rad)],
                       [np.sin(rad), np.cos(rad)]])
    #소수점 10자리 정도에서 반올림해서 미세한 오차 제거
    result = np.round(rot_mat, decimals=10)
    
    return result
#2×2 스케일 행렬을 반환합니다
def scale_matrix_2d(sx,sy):
    return np.array([[sx, 0],
                     [0, sy]])
#점에 (tx, ty)를 더해 반환합니다
def translate_2d(point, tx, ty):
    translate_vector = np.array([tx,ty])

    return point + translate_vector
