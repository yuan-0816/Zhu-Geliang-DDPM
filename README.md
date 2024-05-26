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

---
I use [yolov5-crowdhuman](https://github.com/deepakcrk/yolov5-crowdhuman) to make the Zhu-Geliang datasets, and I use the opencv "face.LBPHFaceRecognizer" to find the face of Zhu-Geliang. 

### Find the face of Zhu-Geliang:
```
git clone https://github.com/yuan-0816/Zhu-Geliang-DDPM.git -b ZhuGeliangRecognizer
```

We use [this](https://github.com/deepakcrk/yolov5-crowdhuman) yolo repository to make the Zhu-Geliang datasets:
```
cd Zhu-Geliang-DDPM && mkdir yolov5_crowdhuman && cd yolov5_crowdhuman
git clone https://github.com/deepakcrk/yolov5-crowdhuman.git/.
```
Follow his tutorial to install the [requirements.txt](https://github.com/deepakcrk/yolov5-crowdhuman/blob/master/requirements.txt)   
And put the pre-trained weights to the [weights](https://github.com/deepakcrk/yolov5-crowdhuman/tree/master/weights) folder.
After completing the installation, copy my head extraction program into the project folder.

### Copy my script to the project folder:
```
cd .. && cp FaceDetectionAndCrop.py yolov5_crowdhuman/
```

### Put the Zhu-Geliang video into the dataset [folder](https://github.com/yuan-0816/Zhu-Geliang-DDPM/tree/ZhuGeliangRecognizer/Zhu_Geliang_datasets/Zhu_Geliang_video):

### Run
```
python FaceDetectionAndCrop.py
```
the result will be saved in the [folder](https://github.com/yuan-0816/Zhu-Geliang-DDPM/tree/ZhuGeliangRecognizer/Zhu_Geliang_datasets/Face_image).

### Filter out non-Zhu-Geliang faces.
I use the opencv "face.LBPHFaceRecognizer" to find the face of Zhu-Geliang.
```
python Find_Zhu_Geliang.py --source-folder <path to the Face_image folder> --output-path <Zhu_Geliang_datasets/Zhu_Geliang_face>  --num-threads <number of threads, default is 16>
```
The result will be saved in the [folder](https://github.com/yuan-0816/Zhu-Geliang-DDPM/tree/ZhuGeliangRecognizer/Zhu_Geliang_datasets/Zhu_Geliang_face).

You can change the number of threads to speed up the process. And also you can train a new model to recognize the faces of Zhu-Geliang.

These are the parameters of ```Find_Zhu_Geliang.py```:
   - ```--is-train``` : whether to train a new model to recognize the faces of Zhu-Geliang. default is ```False```       
   - ```--view-img``` : whether to show the processed images. default is ```False```      
   - ```--source-folder``` : the path to the folder containing the face images. default is ```Zhu_Geliang_datasets/Face_image/exp```   
   - ```--output-path``` : the path to the folder where the result will be saved. default is ```Zhu_Geliang_datasets/Zhu_Geliang_face```   
   - ```--model-save-path``` : the path to the model file. default is ```Find_Zhu_Geliang_model```    
   - ```--train-image-path``` : the path to the folder containing the training images. default is ```None```   
   - ```--num-threads``` : the number of threads to use for parallel processing. default is ```16```   


### The results after recognition may contain misidentifications and will require manual review, but this will save a significant amount of time.

I upload the datasets to the Kaggle, you can download it from [here](https://www.kaggle.com/datasets/yuanyuan0816/zhugeliang-face) derectly.


```
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

By yuan:p
```