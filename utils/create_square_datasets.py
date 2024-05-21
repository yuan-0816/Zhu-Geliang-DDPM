"""
Yuan @ 2024.05.21
This is the code for creating square datasets for training DDIM.
Zhu Geliang is alive!

     #######  ###                          ####             ###       ##                                           ##
     #   ##    ##                         ##  ##             ##                                                   ####
        ##     ##      ##  ##            ##        ####      ##      ###      ####    #####     ### ##            ####
       ##      #####   ##  ##            ##       ##  ##     ##       ##         ##   ##  ##   ##  ##              ##
      ##       ##  ##  ##  ##            ##  ###  ######     ##       ##      #####   ##  ##   ##  ##              ##
     ##    #   ##  ##  ##  ##             ##  ##  ##         ##       ##     ##  ##   ##  ##    #####
     #######  ###  ##   ######             #####   #####    ####     ####     #####   ##  ##       ##              ##
                                                                                               #####
"""


import cv2
import numpy as np


def square_image(input_path, output_path):
    
    image = cv2.imread(input_path)
    height, width = image.shape[:2]
    max_dimension = max(height, width)

    # 創建一個空白的正方形圖片
    squared_image = np.zeros((max_dimension, max_dimension, 3), dtype=np.uint8)
    squared_image.fill(255)  # 填充為白色

    # 計算將原始圖片貼上的位置
    paste_x = (max_dimension - width) // 2
    paste_y = (max_dimension - height) // 2

    # 複製原始圖片到正方形圖片的對應位置
    squared_image[paste_y:paste_y+height, paste_x:paste_x+width] = image

    # 儲存結果
    cv2.imwrite(output_path, squared_image)

    print("已生成正方形圖片：", output_path)



if __name__ == '__main__':
    pass