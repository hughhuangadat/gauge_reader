#!/usr/bin/env python3

import cv2
import os
import time
import numpy as np
import configparser
import matplotlib.pyplot as plt
import DarknetFunc as DFUNC
import YoloObj 

class circle_gauge():
    def get_circle(self, img):
        assert isinstance(img, np.ndarray)
        img = img.astype(dtype='uint8')

        if len(img.shape)>2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #
        else:
            gray = img
        height, width = gray.shape
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([])
                                   , 100, 50, int(height*0.35), int(height*0.48))
        # average found circles, found it to be more accurate than trying to tune HoughCircles parameters to get just the right one
        a, b, c = circles.shape
        xyr = self.avg_circles(circles, b)
        return xyr
    
    def avg_circles(self, circles, b):
        avg_x=0
        avg_y=0
        avg_r=0
        for i in range(b):
            #optional - average for multiple circles (can happen when a gauge is at a slight angle)
            avg_x = avg_x + circles[0][i][0]
            avg_y = avg_y + circles[0][i][1]
            avg_r = avg_r + circles[0][i][2]
        avg_x = int(avg_x/(b))
        avg_y = int(avg_y/(b))
        avg_r = int(avg_r/(b))
        return avg_x, avg_y, avg_r

class gauge_reader(circle_gauge):
    def __init__(self, src_path):
        super().__init__()
        self.src_path = src_path
        img = self.read_image(src_path)
        
        self.img = img
        self.gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.xyr = self.get_circle(self.gray)


    
    def get_deg(self, last_deg=None, show=True):
        
        pointer_size=5
        
        gray = cv2.adaptiveThreshold(self.gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,pointer_size,2) # 11 define scan windows width
        if last_deg:
            # assert False
            print('last_deg = {last_deg}:'.format(last_deg=last_deg))
            rg = (last_deg-45, last_deg+45)
        else:
            rg = (0,360)
       

        v = []
        for angle in range(rg[0], rg[1]):
            mask, _ = self.line_masking(gray, self.xyr, angle, pointer_size)
            tmp = [self.crop_img_from_center(i, self.xyr) for i in (gray, mask)]
            v0 = tmp[0]*tmp[1]
            v.append(sum(v0.flatten()))

        deg = np.where(v == np.min(v))
        deg = deg[0].item()+rg[0]

        # print(ith, self.src_path, 'degree:',deg)
        if show:
            mask, end_pt = self.line_masking(gray, self.xyr, deg, pointer_size)
            tmp = [self.crop_img_from_center(i, self.xyr) for i in (gray, mask)]

            new_img, _ = self.line_masking(self.img, self.xyr, deg, 5, (255,0,0))

            new_img = cv2.addWeighted(self.img, 1, new_img, 0.5, 0)
            
            plt.imshow(new_img)
            plt.show()
            return new_img, deg
        else:
            return None, deg
    

    def crop_img_from_center(self, img, xyr):
        x,y,r = xyr
        return img[y-r:y+r,
                   x-r:x+r,
                  ]

    # def mask_middle(img, xyr, ratio=0.5):
    #     x,y,r = xyr
    #     ratio = ratio

    #     return cv2.circle(img, (x, y), int(ratio*r), (255, 255, 255), -1, cv2.LINE_AA)  # draw center of circle

    def line_masking(self, gray, xyr, angle_degree, width, color=None):
        x,y,r = xyr
        end_w = int(r*np.cos(angle_degree/180*np.pi))
        end_h = int(r*-1*np.sin(angle_degree/180*np.pi) )
        end_pt = (x+end_w, y+end_h)

        arr=np.zeros(gray.shape, dtype='uint8')

        if not color:
            cv2.line(arr, (x,y), end_pt, (1), width)
        else:
            cv2.line(arr, (x,y), end_pt, color, width)
        # print(xyr[:-1], end_pt)
        # cv2.line(影像, 開始座標, 結束座標, 顏色, 線條寬度)

        return arr, end_pt
    
    def read_image(self, img):
        if isinstance(img, np.ndarray):
            pass
        elif isinstance(img, str):
            img = cv2.imread(img)
            
        # pre-process
        img = img.astype(dtype='uint8')
        if img.shape[0]>512:
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        return img


def readYoloConfig(cfg_file: str):
    config = configparser.RawConfigParser()
    config.read(cfg_file)

    darknet_model_dir = config['DARKNET']['MODEL_DIR']
    files = sorted(os.listdir(darknet_model_dir), key=lambda x: x[::-1])

    for f in files:
        if f.endswith('.data'):
            darknet_data = os.path.join(darknet_model_dir, f)

        elif f.endswith('.weights'):
            darknet_weights = os.path.join(darknet_model_dir, f)

        elif f.endswith('.cfg'):
            darknet_cfg = os.path.join(darknet_model_dir, f)

        elif f.endswith('.names'):
            darknet_names = os.path.join(darknet_model_dir, f)

            with open(darknet_data, 'r') as ff:
                lines = ff.readlines()
    
            with open(darknet_data, 'w') as ff:
                for line in lines:
                    if not line.startswith('names'):
                        ff.write(line)

                ff.write(f'names = {darknet_names}')
                
    print(darknet_data, darknet_weights, darknet_cfg)

    net = DFUNC.load_net(bytes(darknet_cfg, 'utf-8'), 
                     bytes(darknet_weights, 'utf-8'), 0)

    meta = DFUNC.load_meta(bytes(darknet_data, 'utf-8'))

    return net, meta

def rotate(img: np.ndarray, center: np.ndarray, init: np.ndarray, final: np.ndarray):
    (h, w, d) = img.shape

    v_if = final - init
    v_x = np.array([1, 0, 0])

    norm_if = np.linalg.norm(v_if)
    norm_x = np.linalg.norm(v_x)
    
    delta_theta = np.arccos( v_if.dot(v_x) / (norm_if * norm_x) )
    rotation = np.cross(v_if, v_x)[2] / np.abs(np.cross(v_if, v_x)[2]) * -1

    M = cv2.getRotationMatrix2D((int(center[0]), int(center[1])), 
                                np.degrees(delta_theta) * rotation, 
                                1.0)

    rotate_img = cv2.warpAffine(img, M, (w, h))

    return delta_theta, rotation, rotate_img

def findInitFianlAngle2(center: np.ndarray, init: np.ndarray, final: np.ndarray, angle):
    theta = np.radians(angle)

    v_ci = init - center
    v_cf = final - center

    c, s = np.cos(theta), np.sin(theta)

    rm = np.array([[c, -s],
                  [s, c]])

    rotate_v_ci = rm @ v_ci
    rotate_v_cf = rm @ v_cf

def findMinMaxAngle(center: np.ndarray, init: np.ndarray, final: np.ndarray, delta_theta: int):
    v_ci = init - center
    v_cf = final - center
    v_x = np.array([1, 0, 0])

    print('v_ci, v_cf, v_x: ', v_ci, v_cf, v_x)

    norm_ci = np.linalg.norm(v_ci)
    norm_cf = np.linalg.norm(v_cf)
    norm_x = np.linalg.norm(v_x)

    angle_ci = np.arccos( v_ci.dot(v_x) / (norm_ci * norm_x) )
    angle_cf = np.arccos( v_cf.dot(v_x) / (norm_cf * norm_x) )
    print('angle_ci, angle_cf: ', np.degrees(angle_ci+np.pi), np.degrees(angle_cf))

    rotated_angle_ci = angle_ci + np.pi + delta_theta
    rotated_angle_cf = angle_cf + delta_theta

    return rotated_angle_ci , rotated_angle_cf

def findGaugeValue(results, img, min_value, max_value, min_angle_shift):
    objs = []
    for result in results:
        obj = YoloObj.DetectedObj(result)
        objs.append(obj)

    # img_bbox = YoloObj.DrawBBox(objs, img, show=False, save=False)

    print(results)
    results_dict = {}
    for result in results:
        results_dict[result[0]] = list(map(int, result[2]))

    print(results_dict)

    center = np.append(np.array(results_dict[b'center'][:2]), 0)
    min_angle = np.append(np.array(results_dict[b'min_angle'][:2]), 0)
    max_angle = np.append(np.array(results_dict[b'max_angle'][:2]), 0)
    print(center)
    delta_theta, rotation, rotate_img = rotate(img, center, min_angle, max_angle)
    min_angle, max_angle = findMinMaxAngle(center, min_angle, max_angle, delta_theta)
    min_angle = np.degrees(min_angle)
    max_angle = np.degrees(max_angle)

    print('min_angle, max_angle: ', min_angle, max_angle)
    print(np.degrees(delta_theta), rotation)

    value_per_degree = (max_value - min_value) / (max_angle - min_angle - min_angle_shift)
    print('value_per_degree: ', value_per_degree)

    # cvshow(rotate_img)
    # cv2.imwrite('test5.jpg', rotate_img)

    last_deg = None
    a = gauge_reader(rotate_img)
    _, deg = a.get_deg(last_deg, show=False)
    if 0 <= deg < 270:
        last_deg = deg + 90
    else:
        last_deg = deg -270

    gauge_value = (last_deg - min_angle - min_angle_shift) * value_per_degree
    print('deg: ', deg, 'last deg: ', last_deg, 'max_angle: ', max_angle, 
        'min_angle_shift: ', min_angle_shift, 'gauge_value: ', gauge_value)

    return gauge_value

if __name__ == '__main__':
    min_value = 0
    min_angle_shift = -20
    max_value = 160
    # init_angle = 200
    # final_angle = -42

    plt.figure(figsize=(2, 3))

    for idx, img_f in enumerate(['030.jpg', '070.jpg', '150.jpg', 'frame_00000_rot5.jpg', 'frame_00000_rot-5.jpg']):
    # for idx, img_f in enumerate(['frame_00000_rot5.jpg']):
        img = cv2.imread(img_f)
        
        net, meta = readYoloConfig('config.txt')
        results = DFUNC.detect(net, meta, img, thresh=0.7)

        gauge_value = findGaugeValue(results, img, min_value, max_value, min_angle_shift)

        plt.subplot(2, 3, idx+1)
        plt.axis('off')
        plt.title('Gauge_value: '+str(gauge_value), fontsize=12)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.show()
