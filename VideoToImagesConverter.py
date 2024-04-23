"""
Yuan @ 2024.04.22
This is a simple script to convert a video to a folder of images.
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

import os

import cv2
from tqdm import tqdm

class VideoToImagesConverter:
    def __init__(self, video_path:str, output_folder:str) -> None:
        self.video_path = video_path
        self.output_folder = output_folder

    def convert(self) -> None:

        file_list = os.listdir(self.video_path)

        for file_name in file_list:
            print(f'running at {file_name}')
            if file_name.endswith(".mp4"):
                # 組合完整的檔案路徑
                file_path = os.path.join(self.video_path, file_name)

                cap = cv2.VideoCapture(file_path)
                
                if not cap.isOpened():
                    print("Error: Failed to open video file.")
                    return

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                progress_bar = tqdm(total=total_frames, desc='Processing frames')

                count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        cv2.imwrite(os.path.join(self.output_folder, f"{file_name[:-4]}_{count}.jpg"), frame)
                        count += 1
                        progress_bar.update(1)
                    else:
                        break
                progress_bar.close()
                cap.release()

if __name__ == "__main__":
    # Example usage
    video_path = 'Zhu_Geliang_datasets/Zhu_Geliang_video/test_video'
    output_folder = 'Zhu_Geliang_datasets/Video_To_Image'
    video_converter = VideoToImagesConverter(video_path=video_path, output_folder=output_folder)
    video_converter.convert()
