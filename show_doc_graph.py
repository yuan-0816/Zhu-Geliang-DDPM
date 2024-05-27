"""
Yuan @ 2024.05.21
This is the code for showing some graph.
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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import os
import random
import matplotlib.image as mpimg
import cv2
import re


def load_image(path, size):
    image = Image.open(path).convert('RGB')
    image = image.resize((size, size))
    image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    return image


def get_beta_schedule(T, start=0.001, end=0.05):
    return np.linspace(start, end, T)

def forward_process(image, T):
    betas = get_beta_schedule(T)
    alphas = 1 - betas
    alphas_cumprod = np.cumprod(alphas)
    
    images = []
    noisy_image = image.copy()
    
    for t in range(T):
        noise = np.random.normal(0, 1, image.shape)
        noisy_image = np.sqrt(alphas_cumprod[t]) * image + np.sqrt(1 - alphas_cumprod[t]) * noise
        if t % (T // 8) == 0:
            images.append(noisy_image)
    
    return images

def plot_images(images, steps=10):
    fig, axes = plt.subplots(1, len(images), figsize=(20, 2), facecolor='#4B6079')
    for i, ax in enumerate(axes):
        ax.axis('off')
        ax.imshow(np.clip(images[i], 0, 1))
    
    plt.show()


def forward_process(image_path, size, T):
    image = load_image(image_path, size)
    images = forward_process(image, T)
    plot_images(images)



def generate_gaussian_noise_image(size, scale_factor):
    
    noise = np.random.normal(scale=64, size=(size, size)).astype(np.uint8)
    noise = np.clip(noise, 0, 255)
    img = Image.fromarray(noise, 'L')
    img = img.resize((size * scale_factor, size * scale_factor), Image.NEAREST)

    return img




def generate_pixel_space(size=1000, num_points=1500):
    r = np.arange(10, 30, 0.1)
    theta = 2 * np.pi * r

    x = np.random.randint(1, size, num_points)
    y = np.random.randint(1, size, num_points)


    plt.figure(figsize=(9, 9), facecolor='#4B6079', edgecolor='#4B6079')
    
    ax = plt.axes()

    plt.plot(theta, r, '.', color="yellow")

    ax.set_facecolor('#4B6079')
    ax.tick_params(axis='x', colors='#ECECEC')
    ax.tick_params(axis='y', colors='#ECECEC')
    ax.spines['bottom'].set_color('#ECECEC')
    ax.spines['top'].set_color('#ECECEC')
    ax.spines['left'].set_color('#ECECEC')
    ax.spines['right'].set_color('#ECECEC')
    mpl.rcParams['grid.color'] = 'white'
    plt.grid()

    plt.plot(x, y, '.', color='#ECECEC', alpha=1, markersize=8)
    plt.show()


def show_head():


    path = "data/zhugeliang_face"


    file_list = os.listdir(path)


    file_list = [file for file in file_list if file.endswith('.jpg')]

    # random sample 16 images
    file_list = random.sample(file_list, 16)
    file_list = [os.path.join(path, file) for file in file_list]
    print(file_list)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10), facecolor='#4B6079')

    for i, ax in enumerate(axes.flat):
        # ax.axis('off')
        img = mpimg.imread(file_list[i])
        ax.imshow(img)
        ax.tick_params(axis='x', colors='#ECECEC')
        ax.tick_params(axis='y', colors='#ECECEC')
        ax.spines['bottom'].set_color('#ECECEC')
        ax.spines['top'].set_color('#ECECEC')
        ax.spines['left'].set_color('#ECECEC')
        ax.spines['right'].set_color('#ECECEC')

    plt.show()


def show_dataset_process():
    path = "data/zhugeliang_face_square/Episode_2_3_frame_31_head_1_0.jpg"
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    max_dimension = max(height, width) 

    squared_image = np.zeros((max_dimension, max_dimension, 3), dtype=np.uint8)
    squared_image.fill(0)   # fill with black color

    paste_x = (max_dimension - width) // 2
    paste_y = (max_dimension - height) // 2

    squared_image[paste_y:paste_y+height, paste_x:paste_x+width] = image

    small_image = cv2.resize(squared_image, (64, 64))

    fig, axes = plt.subplots(1, 2,  facecolor='#4B6079')

    list_img = [small_image, small_image]

    for i, ax in enumerate(axes.flat):
        # ax.axis('off')
        ax.imshow(list_img[i])
        ax.tick_params(axis='x', colors='#ECECEC')
        ax.tick_params(axis='y', colors='#ECECEC')
        ax.spines['bottom'].set_color('#ECECEC')
        ax.spines['top'].set_color('#ECECEC')
        ax.spines['left'].set_color('#ECECEC')
        ax.spines['right'].set_color('#ECECEC')
    plt.show()


def show_log():

    log_file = 'result/log.txt'

    with open(log_file, 'r', encoding='utf-8') as file:
        log_data = file.readlines()
        
    # find max epoch 
    max_epoch = 0
    for line in log_data:
        match = re.search(r'Epoch: (\d+)', line)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch

    epoch_losses = [None] * max_epoch

    for line in log_data:
        match = re.search(r'Epoch: (\d+).*train_loss=(\d+)', line)
        if match:
            epoch = int(match.group(1))
            loss = int(match.group(2))
            epoch_losses[epoch - 1] = loss
    

    filtered_epochs = [epoch for epoch in range(max_epoch) if epoch_losses[epoch] is not None]
    filtered_losses = [loss for loss in epoch_losses if loss is not None]


    plt.figure(figsize=(9, 9), facecolor='#4B6079', edgecolor='#4B6079')

    ax = plt.axes()
    ax.set_facecolor('#4B6079')
    ax.tick_params(axis='x', colors='#ECECEC')
    ax.tick_params(axis='y', colors='#ECECEC')
    ax.spines['bottom'].set_color('#ECECEC')
    ax.spines['top'].set_color('#ECECEC')
    ax.spines['left'].set_color('#ECECEC')
    ax.spines['right'].set_color('#ECECEC')
    mpl.rcParams['grid.color'] = 'white'


    plt.plot(filtered_epochs, filtered_losses, color="#ECECEC", linestyle='-')
    title = plt.title('Training Loss per Epoch')
    xlabel = plt.xlabel('Epoch')
    ylabel = plt.ylabel('Loss')

    title.set_color('#ECECEC')
    xlabel.set_color('#ECECEC')
    ylabel.set_color('#ECECEC')

    plt.grid(True)
    plt.show()






if __name__ == "__main__":

    # --------------------------- show forward process --------------------------- #
    # image_path = "C:/Users/wj582/Desktop/square_Episode.jpg"  # 替換為你的圖像路徑
    # size = 64  # 圖像大小（正方形）
    # T = 300   # 時間步長
    # forward_process(image_path, size, T)


    # --------------------------------- 8x8 的高斯噪聲 -------------------------------- #
    # size = 8
    # scale_factor = 100
    # img = generate_gaussian_noise_image(size, scale_factor)
    # img.show()


    # ---------------------------------- 產生像素空間圖 --------------------------------- #
    # generate_pixel_space()

    # ------------------------------- 取樣yolo辨識後資料集 ------------------------------- #
    # show_head()

    # --------------------------------- 資料集前處理過程 --------------------------------- #
    # show_dataset_process()


    # ---------------------------- show 訓練 loss/epoch ---------------------------- #
    show_log()


    pass