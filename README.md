```
git clone https://github.com/yuan-0816/Zhu-Geliang-DDPM.git
cd Zhu-Geliang-DDPM
```

We use this yolo repository to make the Zhu-Geliang datasets:
```
git clone https://github.com/deepakcrk/yolov5-crowdhuman.git
cd yolov5-crowdhuman
pip install -r requirements.txt
```

Download the pre-trained weights of YOLOv5:
https://github.com/deepakcrk/yolov5-crowdhuman   
put this pre-trained weights to "yolov5-crowdhuman" folder

Move this file to "yolov5-crowdhuman" folder:
```
cd ..
mv FaceDetectionAndCrop.py yolov5-crowdhuman/
```


