### This repostory is inspired by / based on the following repositories:
   - [diffusion-DDIM-pytorch](https://github.com/Alokia/diffusion-DDIM-pytorch)
   - [yolov5-crowdhuman](https://github.com/deepakcrk/yolov5-crowdhuman)

---
```

     #######  ###                          ####             ###       ##                                           ##
     #   ##    ##                         ##  ##             ##                                                   ####
        ##     ##      ##  ##            ##        ####      ##      ###      ####    #####     ### ##            ####
       ##      #####   ##  ##            ##       ##  ##     ##       ##         ##   ##  ##   ##  ##              ##
      ##       ##  ##  ##  ##            ##  ###  ######     ##       ##      #####   ##  ##   ##  ##              ##
     ##    #   ##  ##  ##  ##             ##  ##  ##         ##       ##     ##  ##   ##  ##    #####
     #######  ###  ##   ######             #####   #####    ####     ####     #####   ##  ##       ##              ##
                                                                                               #####
```

![images](https://github.com/yuan-0816/Zhu-Geliang-DDPM/blob/main/doc/GenerateProcess.png)


## Zhu-Geliang face datasets   
I use [yolov5-crowdhuman](https://github.com/deepakcrk/yolov5-crowdhuman) to make the Zhu-Geliang datasets, and I use the opencv ```cv2.face.LBPHFaceRecognizer``` to find the face of Zhu-Geliang.  
If you want to see the code, please go to the branch [ZhuGeliangRecognizer](https://github.com/yuan-0816/Zhu-Geliang-DDPM/tree/ZhuGeliangRecognizer), and you can find the code in the following files:
[FaceDetectionAndCrop.py](https://github.com/yuan-0816/Zhu-Geliang-DDPM/blob/ZhuGeliangRecognizer/yolov5_crowdhuman/FaceDetectionAndCrop.py)
and 
[Find_Zhu_Geliang.py](https://github.com/yuan-0816/Zhu-Geliang-DDPM/blob/ZhuGeliangRecognizer/Find_Zhu_Geliang.py)   

### I upload the datasets to the Kaggle, you can download it from [here](https://www.kaggle.com/datasets/yuanyuan0816/zhugeliang-face) derectly.

![images](https://github.com/yuan-0816/Zhu-Geliang-DDPM/blob/main/doc/ShowZhuGeliangFace.png)

put this all datasets to [zhugeliang_face](https://github.com/yuan-0816/Zhu-Geliang-DDPM/tree/main/data/zhugeliang_face) folder

## Make square image
![images](https://github.com/yuan-0816/Zhu-Geliang-DDPM/blob/main/doc/DataPreprocessing.png)

```
python Create_Square_Datasets.py --size 64 --num-threads 16
```
These are the parameters of ```Create_Square_Datasets.py``` :
   - ```--datasets-path``` : the path of the datasets. default is```data/zhugeliang_face```
   - ```--output-path``` : the path of the output datasets. default is```data/zhugeliang_face_square```
   - ```--size``` : the size of the output image. default is 64, if you want to train another size image, you can change it.
   - ```--num-threads``` : the number of threads. default is 16, if you have more than 16 cores, you can change it.






## Gernarate Zhu-Geliang face


## Train




