import numpy as np
class Constant:
    width, height = 640, 480
    # width, height = 1280, 720
    depth_element_byte = 2
    rgb_element_byte = 1
    depth_frame_length_in_byte = width * height * depth_element_byte
    rgb_frame_length_in_byte = width * height * 3 * rgb_element_byte
    depth_type = np.uint16
    rgb_type = np.uint8
    far_range = 2000.0
    near_range = 200.0


    average_f = 614.0

    depth_scale = 0.001
    #depth_scale = 0.000124987

    crop_size_3d = (270, 270, 350)# 10.23
    #crop_size_3d = (300, 300, 350)  # original

    #crop_size_3d = (300, 300, 300)# original
    crop_size_2d = (64, 64)# original


    # hand detection configurations
    detect_img_ratio = 0.25
    detect_width = int(width * detect_img_ratio)#160
    detect_height = int(height * detect_img_ratio)#120