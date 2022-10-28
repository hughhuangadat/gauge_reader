#!/usr/bin/env python3

import cv2
import os
import time
import numpy as np
import configparser
import matplotlib.pyplot as plt
import DarknetFunc as DFUNC
import YoloObj 

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

def findMinMaxAngle(center: np.ndarray, init: np.ndarray, final: np.ndarray, tail: np.ndarray, delta_theta: int):
    v_ci = init - center
    v_cf = final - center
    v_ctail = tail - center
    v_x = np.array([1, 0, 0])
    v_y = np.array([0, 1, 0])

    print('v_ci, v_cf, v_x: ', v_ci, v_cf, v_x)

    norm_ci = np.linalg.norm(v_ci)
    norm_cf = np.linalg.norm(v_cf)
    norm_ctail = np.linalg.norm(v_ctail)
    norm_x = np.linalg.norm(v_x)
    norm_y = np.linalg.norm(v_y)

    angle_ci = np.arccos( v_ci.dot(v_x) / (norm_ci * norm_x) )
    angle_cf = np.arccos( v_cf.dot(v_x) / (norm_cf * norm_x) )
    angle_ctail = np.arccos( v_ctail.dot(v_y) / (norm_ctail * norm_y) )

    print('angle_ci, angle_cf, angle_ctail: ', np.degrees(angle_ci+np.pi), np.degrees(angle_cf), np.degrees(angle_ctail))

    rotated_angle_ci = angle_ci + np.pi + delta_theta
    rotated_angle_cf = angle_cf + delta_theta
    direction_ctail = np.cross(v_ctail, v_y)[2] / np.abs(np.cross(v_ctail, v_y)[2]) * -1
    if direction_ctail >= 0:
        rotated_angle_ctail = 2*np.pi - angle_ctail + delta_theta
        rotated_angle_chead = rotated_angle_ctail - np.pi
    else:
        rotated_angle_ctail = angle_ctail + delta_theta
        rotated_angle_chead = rotated_angle_ctail + np.pi

    print('rotated_angle_ci, rotated_angle_cf, rotated_angle_chead: ', 
          rotated_angle_ci, rotated_angle_cf, rotated_angle_chead)
    
    return rotated_angle_ci, rotated_angle_cf, rotated_angle_chead

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
    tail_angle = np.append(np.array(results_dict[b'tail'][:2]), 0)
    print(center)
    delta_theta, rotation, rotate_img = rotate(img, center, min_angle, max_angle)
    min_angle, max_angle, pointer_angle = findMinMaxAngle(center, min_angle, max_angle, tail_angle, delta_theta)
    min_angle = np.degrees(min_angle)
    max_angle = np.degrees(max_angle)
    pointer_angle = np.degrees(pointer_angle)

    print('min_angle, max_angle: ', min_angle, max_angle)
    print(np.degrees(delta_theta), rotation)

    value_per_degree = (max_value - min_value) / (max_angle - min_angle - min_angle_shift)
    print('value_per_degree: ', value_per_degree)

    # cvshow(rotate_img)
    # cv2.imwrite('test5.jpg', rotate_img)

    # last_deg = None
    # a = gauge_reader(rotate_img)
    # _, deg = a.get_deg(last_deg, show=False)
    # if 0 <= deg < 270:
    #     last_deg = deg + 90
    # else:
    #     last_deg = deg -270

    last_deg = pointer_angle
    gauge_value = (last_deg - min_angle - min_angle_shift) * value_per_degree
    print('last deg: ', last_deg, 'max_angle: ', max_angle, 
        'min_angle_shift: ', min_angle_shift, 'gauge_value: ', gauge_value)

    return gauge_value

if __name__ == '__main__':
    min_value = 0
    min_angle_shift = -20
    max_value = 160
    # init_angle = 200
    # final_angle = -42

    plt.figure(figsize=(2, 3))
    
    init_t0 = time.time()
    net, meta = readYoloConfig('config.txt')
    time0 = time.time()-init_t0

    # for idx, img_f in enumerate(['030.jpg', '070.jpg', '150.jpg', 'frame_00000_rot5.jpg', 'frame_00000_rot-5.jpg']):
    for idx, img_f in enumerate(['000.jpg', '030.jpg', '070.jpg', '110.jpg', '150.jpg']):
    # for idx, img_f in enumerate(['150.jpg', '150.jpg', '150.jpg']):
        init_t1 = time.time()
        img = cv2.imread(img_f)
        time1 = time.time()-init_t1

        init_t2 = time.time()
        results = DFUNC.detect(net, meta, img, thresh=0.7)
        time2 = time.time()-init_t2

        init_t3 = time.time()
        gauge_value = findGaugeValue(results, img, min_value, max_value, min_angle_shift)
        time3 = time.time()-init_t3

        plt.subplot(2, 3, idx+1)
        plt.axis('off')
        plt.title('Gauge_value: '+str(gauge_value), fontsize=12)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        print('------------------------------------------------------------------------------------')
        print('time: ', time0, time1, time2, time3)
        print('fps: ', 1/time0, 1/time1, 1/time2, 1/time3)
    plt.show()
