"""
Yuan @ 2024.04.22
This is a script to find Zhu Geliang.
Zhu Geliang is alive!
　　    　　 ＿＿＿
　　　　　／＞　　  フ
　　　　　|  　_　 _|
　 　　　／` ミ＿xノ
　　 　 /　　　 　 |
　　　 /　 ヽ　　 ﾉ
　 　 │　　|　|　|
　／￣|　　 |　|　|
　| (￣ヽ＿_ヽ_)__)
　＼二つ
"""

# TODO: 用其他更精準的人臉辨識模型 ex:DeepFace https://github.com/serengil/deepface
# TODO: Thread 加快速度

import cv2
import numpy as np
import os
from deepface import DeepFace


def Train_Zhu_Geliang_Finder(image_folder_path: str, model_save_path: str) -> None:
    detector = cv2.CascadeClassifier(
        "Zhu_Geliang_datasets/Find_Zhu_Geliang_model/haarcascade_frontalface_default.xml"
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
                    faces.append(
                        img_np[y : y + h, x : x + w]
                    )  # 記錄蔡英文人臉的位置和大小內像素的數值
                    ids.append(1)
            else:
                print(f"無法讀取檔案: {file_path}")
    print("training...")  # 提示開始訓練
    recog.train(faces, np.array(ids))  # 開始訓練
    recog.save(model_save_path + "/" + "Zhu_Geliang_face.yml")
    print("ok!")


def Zhu_Geliang_Finder(image_folder_path: str, OutPut_path: str) -> None:
    detector = cv2.CascadeClassifier(
        "Zhu_Geliang_datasets/Find_Zhu_Geliang_model/haarcascade_frontalface_default.xml"
    )
    recog = cv2.face.LBPHFaceRecognizer_create()  # 啟用訓練人臉模型方法
    recog.read(
        "Zhu_Geliang_datasets/Find_Zhu_Geliang_model/Zhu_Geliang_face.yml"
    )  # 讀取訓練好的模型

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
                show_img = img.copy()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 轉換成黑白
                faces = detector.detectMultiScale(
                    gray
                )  # 追蹤人臉 ( 目的在於標記出外框 )

                face_index = 0
                for x, y, w, h in faces:
                    # cv2.rectangle(show_img, (x,y), (x+w,y+h), (0,255,0), 2)            # 標記人臉外框
                    idnum, confidence = recog.predict(
                        gray[y : y + h, x : x + w]
                    )  # 取出 id 號碼以及信心指數 confidence
                    if confidence < 60:
                        text = "Zhu Geliang"
                        crop_save_path = (
                            OutPut_path + "/" + f"{file_name[:-4]}_{face_index}.jpg"
                        )  # 如果信心指數小於 60，取得對應的名字
                        cv2.imwrite(str(crop_save_path), img)
                        face_index += 1
                    else:
                        text = "???"
                    # 在人臉外框旁加上名字
                    # cv2.putText(show_img, text, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

                # cv2.imshow('oxxostudio', show_img)
                # if cv2.waitKey(0) == ord('q'):
                #     break




def UseDeepFace():
    img_path = "Zhu_Geliang_datasets/Face_image/exp/test_frame_258_head_5_0.jpg"
    db_path = "Zhu_Geliang_datasets/Zhu_Geliang_face/test"


    try:
        dfs = DeepFace.find(img_path = img_path, db_path = db_path)
        print(dfs)
    except:
        pass

    # # 取得資料夾中的所有檔案
    # file_list = os.listdir(img_path)

    # for file_name in file_list:
    #     # 檢查檔案是否為 JPG 格式
    #     if file_name.endswith(".jpg") or file_name.endswith(".JPG"):
            
    #         file_path = os.path.join(image_folder_path, file_name)
    #         try:
    #             dfs = DeepFace.find(img_path = file_path, db_path = db_path)
    #             print(dfs)
    #         except:
    #             pass



if __name__ == "__main__":
    model_save_path = "Zhu_Geliang_datasets/Find_Zhu_Geliang_model"
    image_folder_path = "Zhu_Geliang_datasets/Face_image/exp"
    OutPut_path = "Zhu_Geliang_datasets/Zhu_Geliang_face/test"
    
    # Train_Zhu_Geliang_Finder(image_folder_path, model_save_path)
    # Zhu_Geliang_Finder(image_folder_path, OutPut_path)

    # TODO: 改用 DeepFace 來辨識
    UseDeepFace()
