"""
Yuan @ 2024.04.22
This is a script to find Zhu Geliang face in a folder of images or videos.
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
import os
import argparse
from tqdm import tqdm
import concurrent.futures
import time

from InfoZhuGeliangDDPM import PrintEverything




def Split_file_lists(input_list, num_groups) -> list:
    if len(input_list) < num_groups:
        num_groups = len(input_list)
    avg = len(input_list) // num_groups
    remainder = len(input_list) % num_groups
    result = []
    start = 0
    for i in range(num_groups):
        size = avg + 1 if i < remainder else avg
        result.append(input_list[start : start + size])
        start += size
    return result


def check_file_type(file_path: str) -> str:
    img_formats = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp"]
    vid_formats = ["mov", "avi", "mp4", "mpg", "mpeg", "m4v", "wmv", "mkv"]

    file_extension = file_path.split(".")[-1].lower()

    if file_extension in img_formats:
        return "image"
    elif file_extension in vid_formats:
        return "video"
    else:
        return "unknown"


def Train_Zhu_Geliang_Finder(image_folder_path: str, model_save_path: str) -> None:
    detector = cv2.CascadeClassifier(
        "Find_Zhu_Geliang_model/haarcascade_frontalface_default.xml"
    )
    recog = cv2.face.LBPHFaceRecognizer_create()  # 啟用訓練人臉模型方法
    faces = []  # 儲存人臉位置大小的串列
    ids = []  # 記錄該人臉 id 的串列

    # 取得資料夾中的所有檔案
    file_list = os.listdir(image_folder_path)

    for file_name in file_list:
        # 檢查檔案是否為 JPG 格式
        if file_name.endswith(".jpg") or file_name.endswith(".JPG"):
            # 組合完整的檔案路徑
            file_path = os.path.join(image_folder_path, file_name)

            # 使用 OpenCV 讀取圖片
            img = cv2.imread(file_path)

            # 檢查圖片是否成功讀取
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 色彩轉換成黑白
                img_np = np.array(gray, "uint8")  # 轉換成指定編碼的 numpy 陣列
                face = detector.detectMultiScale(gray)  # 擷取人臉區域
                for x, y, w, h in face:
                    faces.append(img_np[y : y + h, x : x + w])
                    ids.append(1)
            else:
                print(f"無法讀取檔案: {file_path}")
    print("training...")  # 提示開始訓練
    recog.train(faces, np.array(ids))  # 開始訓練
    recog.save(model_save_path + "/" + "Zhu_Geliang_face.yml")
    print("ok!")


def Zhu_Geliang_Finder(
    folder_path: str, file_list: list, OutPut_path: str, Show_img: bool
) -> None:
    detector = cv2.CascadeClassifier(
        "Find_Zhu_Geliang_model/haarcascade_frontalface_default.xml"
    )
    recog = cv2.face.LBPHFaceRecognizer_create()
    # recog.read("Find_Zhu_Geliang_model/Zhu_Geliang_face.yml")
    recog.read("../Zhu_Geliang_face.yml")

    for file_name in tqdm(file_list, desc="Detect Zhu Geliang..."):
        file_path = os.path.join(folder_path, file_name)

        if check_file_type(file_path) == "image":
            img = cv2.imread(file_path)
            if img is not None:
                show_img = img.copy()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 轉換成黑白
                faces = detector.detectMultiScale(
                    gray
                )  # 追蹤人臉 ( 目的在於標記出外框 )

                face_index = 0
                for x, y, w, h in faces:
                    cv2.rectangle(
                        show_img, (x, y), (x + w, y + h), (0, 255, 0), 2
                    )  # 標記人臉外框
                    idnum, confidence = recog.predict(
                        gray[y : y + h, x : x + w]
                    )  # 取出 id 號碼以及信心指數 confidence
                    if confidence < 70:
                        text = "Zhu Geliang"
                        crop_save_path = (
                            OutPut_path + "/" + f"{file_name[:-4]}_{face_index}.jpg"
                        )
                        cv2.imwrite(str(crop_save_path), img)
                        face_index += 1
                    else:
                        text = "???"
                    # 在人臉外框旁加上名字
                    cv2.putText(
                        show_img,
                        text,
                        (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                if Show_img:
                    cv2.imshow("oxxostudio", show_img)
                    if cv2.waitKey(0) == ord("q"):
                        break
        else:
            continue


def process_files_threaded(file_lists, folder_path, output_path, show_img, num_threads):
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        # 將每個文件列表提交給執行器處理
        futures = [executor.submit(Zhu_Geliang_Finder, folder_path, file_list, output_path, show_img) for file_list in file_lists]
        
        # 等待所有任務完成
        for future in concurrent.futures.as_completed(futures):
            # 確認任務是否成功完成
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")



if __name__ == "__main__":
    start_time = time.time()  # start time
    PrintEverything()
    parser = argparse.ArgumentParser(prog="Find_Zhu_Geliang.py")
    parser.add_argument("--is-train", type=bool, default=False, help="是否訓練模型")
    parser.add_argument("--view-img", type=bool, default=False, help="是否顯示圖片")
    parser.add_argument("--source-folder", type=str, default="Zhu_Geliang_datasets/Face_image/exp3", help="目標資料夾路徑")
    parser.add_argument("--output-path", type=str, default="Zhu_Geliang_datasets/Zhu_Geliang_face/Episode_1_Zhu_Geliang_face", help="輸出資料夾路徑")
    parser.add_argument("--model-save-path", type=str, default="Find_Zhu_Geliang_model", help="模型儲存路徑")
    parser.add_argument("--train-image-path", type=str, default="Zhu_Geliang_datasets/Zhu_Geliang_face/test3", help="訓練模型資料夾路徑")
    parser.add_argument("--num-threads", type=int, default=16, help="處理的執行續數量")
    args = parser.parse_args()

    if args.is_train:
        Train_Zhu_Geliang_Finder(args.train_image_path, args.model_save_path)
    else:
        threads_lists = Split_file_lists(
            os.listdir(args.source_folder), args.num_threads
        )

        process_files_threaded(
            threads_lists,
            args.source_folder,
            args.output_path,
            args.view_img,
            args.num_threads
        )

        # print run time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"run time: {elapsed_time} seconds")
