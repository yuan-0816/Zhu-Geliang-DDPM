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

import argparse
import cv2
import numpy as np
import time
import os 
from tqdm import tqdm
import concurrent.futures

from utils.tools import get_images_path_list, Split_file_lists, PrintInfo



def Square_Amd_Resize_Image(datasets_path:list, output_folder:str, size:int=64) -> None:

    for image_path in tqdm(datasets_path, desc="Processing images..."):
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        max_dimension = max(height, width) 
    
        squared_image = np.zeros((max_dimension, max_dimension, 3), dtype=np.uint8)
        squared_image.fill(0)   # fill with black color

        paste_x = (max_dimension - width) // 2
        paste_y = (max_dimension - height) // 2

        squared_image[paste_y:paste_y+height, paste_x:paste_x+width] = image
        squared_image = cv2.resize(squared_image, (size, size))

        output_path = os.path.join(output_folder, "square_"+os.path.basename(image_path))

        cv2.imwrite(output_path, squared_image)


def process_files_threaded(file_lists, output_folder, size, num_threads):
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(Square_Amd_Resize_Image, file_list, output_folder, size) for file_list in file_lists]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")



if __name__ == '__main__':
    start_time = time.time()  # start time
    PrintInfo()
    parser = argparse.ArgumentParser(prog="create_square_datasets.py")
    parser.add_argument("--datasets-path", type=str, default="data/zhugeliang_face", help="datasets path")
    parser.add_argument("--output-path", type=str, default="data/zhugeliang_face_square", help="output path")
    parser.add_argument("--size", type=int, default=64, help="resize size")
    parser.add_argument("--num-threads", type=int, default=16, help="number of threads")

    args = parser.parse_args()

    threads_lists = Split_file_lists(
        get_images_path_list(args.datasets_path), args.num_threads
    )


    process_files_threaded(
        threads_lists,
        args.output_path,
        args.size,
        args.num_threads
    )


    # print run time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"run time: {elapsed_time} seconds")