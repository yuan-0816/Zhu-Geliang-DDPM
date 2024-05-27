### This repostory is inspired by / based on the following repositories:
   - [diffusion-DDIM-pytorch](https://github.com/Alokia/diffusion-DDIM-pytorch)
   - [yolov5-crowdhuman](https://github.com/deepakcrk/yolov5-crowdhuman)

---


![images](https://github.com/yuan-0816/Zhu-Geliang-DDPM/blob/main/doc/GenerateProcess.png)

# Generate Zhu-Geliang face by DDIM
you can download the pre-trained model from [here](https://drive.google.com/file/d/15oFGT2HXYGjdnGyWpIp4KNCxXb35Pogh/view?usp=drive_link).   
put ```zhu_geliang.pth``` to the [checkpoints](https://github.com/yuan-0816/Zhu-Geliang-DDPM/tree/main/checkpoint) folder.   
After that you can run the following command to generate Zhu-Geliang face:
```
python .\generate.py -cp ./checkpoint/zhu_geliang.pth -bs 1 --interval 10 --show -sp "result/zhugeliang.png" --sampler "ddim" --steps 200

```
These are the parameters of ```generate.py``` :
   - ```-cp``` : the path of checkpoint.
   - ```--device``` : the device used. 'cuda' (default) or 'cpu'.
   - ```--sampler``` : the sampler method, can be 'ddpm'(default) or 'ddim'.
   - ```-bs``` : how many images to generate at once. Default 16.
   - ```--result_only``` : whether to output only the generated results. Default False.
   - ```--interval``` : extract an image every how many steps. Only valid without the result_only parameter. Default 50.
   - ```--eta``` : ddim parameter, in the paper. Default 0.0.
   - ```--steps``` : ddim sampling steps. Default 100.
   - ```--method``` : ddim sampling method. can be 'linear'(default) or 'quadratic'.
   - ```--nrow``` : how many images are displayed in a row. Only valid with the result_only parameter. Default 4.
   - ```--show``` : whether to display the result image. Default False.
   - ```-sp``` : save path of the result image. Default None.
   - ```--to_grayscale``` : convert images to grayscale. Default False.

This check point will generate 64x64 image, If you want to generate custom size images, you can train the model with different ```image_size``` parameter in [config.yml](https://github.com/yuan-0816/Zhu-Geliang-DDPM/blob/main/config.yml) file.


# Train your own model
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

## Train
Almost all the parameters that can be modified are listed in the [config.yml](https://github.com/yuan-0816/Zhu-Geliang-DDPM/blob/main/config.yml) file. You can modify the relevant parameters as needed, and then run the train.py file to start training.
```
python train.py
```

## Result
![images](https://github.com/yuan-0816/Zhu-Geliang-DDPM/blob/main/doc/Result_Light.png)

My Training Loss 

![images](https://github.com/yuan-0816/Zhu-Geliang-DDPM/blob/main/doc/TrainingLoss.png)



