"""
Yuan @ 2024.05.21

Zhu Geliang is alive!

     #######  ###                          ####             ###       ##                                           ##
     #   ##    ##                         ##  ##             ##                                                   ####
        ##     ##      ##  ##            ##        ####      ##      ###      ####    #####     ### ##            ####
       ##      #####   ##  ##            ##       ##  ##     ##       ##         ##   ##  ##   ##  ##              ##
      ##       ##  ##  ##  ##            ##  ###  ######     ##       ##      #####   ##  ##   ##  ##              ##
     ##    #   ##  ##  ##  ##             ##  ##  ##         ##       ##     ##  ##   ##  ##    #####
     #######  ###  ##   ######             #####   #####    ####     ####     #####   ##  ##       ##              ##


"""

import os
from typing import Optional, Union
import torch
from tqdm import tqdm
from torchvision.utils import make_grid
from PIL import Image
from pathlib2 import Path
import yaml
import matplotlib.pyplot as plt
import json

def plot_losses(losses, filepath):
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig(filepath)
    plt.show()

def save_losses(losses, filepath):
    with open(filepath, 'w') as f:
        json.dump(losses, f)

def load_losses(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def load_yaml(yml_path: Union[Path, str], encoding="utf-8"):
    if isinstance(yml_path, str):
        yml_path = Path(yml_path)
    with yml_path.open('r', encoding=encoding) as f:
        cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return cfg


def train_one_epoch(trainer, loader, optimizer, device, epoch):
    trainer.train()
    total_loss, total_num = 0., 0

    with tqdm(loader, dynamic_ncols=True, colour="#ff924a") as data:
        for images, _ in data:
            optimizer.zero_grad()

            x_0 = images.to(device)
            loss = trainer(x_0)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_num += x_0.shape[0]

            data.set_description(f"Epoch: {epoch}")
            data.set_postfix(ordered_dict={
                "train_loss": total_loss / total_num,
            })

    return total_loss / total_num


def save_image(images: torch.Tensor, nrow: int = 8, show: bool = True, path: Optional[str] = None,
               format: Optional[str] = None, to_grayscale: bool = False, **kwargs):
    """
    concat all image into a picture.

    Parameters:
        images: a tensor with shape (batch_size, channels, height, width).
        nrow: decide how many images per row. Default `8`.
        show: whether to display the image after stitching. Default `True`.
        path: the path to save the image. if None (default), will not save image.
        format: image format. You can print the set of available formats by running `python3 -m PIL`.
        to_grayscale: convert PIL image to grayscale version of image. Default `False`.
        **kwargs: other arguments for `torchvision.utils.make_grid`.

    Returns:
        concat image, a tensor with shape (height, width, channels).
    """
    images = images * 0.5 + 0.5
    grid = make_grid(images, nrow=nrow, **kwargs)  # (channels, height, width)
    #  (height, width, channels)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    im = Image.fromarray(grid)
    if to_grayscale:
        im = im.convert(mode="L")
    if path is not None:
        im.save(path, format=format)
    if show:
        im.show()
    return grid


def save_sample_image(images: torch.Tensor, show: bool = True, path: Optional[str] = None,
                      format: Optional[str] = None, to_grayscale: bool = False, **kwargs):
    """
    concat all image including intermediate process into a picture.

    Parameters:
        images: images including intermediate process,
            a tensor with shape (batch_size, sample, channels, height, width).
        show: whether to display the image after stitching. Default `True`.
        path: the path to save the image. if None (default), will not save image.
        format: image format. You can print the set of available formats by running `python3 -m PIL`.
        to_grayscale: convert PIL image to grayscale version of image. Default `False`.
        **kwargs: other arguments for `torchvision.utils.make_grid`.

    Returns:
        concat image, a tensor with shape (height, width, channels).
    """
    images = images * 0.5 + 0.5

    grid = []
    for i in range(images.shape[0]):
        # for each sample in batch, concat all intermediate process images in a row
        t = make_grid(images[i], nrow=images.shape[1], **kwargs)  # (channels, height, width)
        grid.append(t)
    # stack all merged images to a tensor
    grid = torch.stack(grid, dim=0)  # (batch_size, channels, height, width)
    grid = make_grid(grid, nrow=1, **kwargs)  # concat all batch images in a different row, (channels, height, width)
    #  (height, width, channels)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    im = Image.fromarray(grid)
    if to_grayscale:
        im = im.convert(mode="L")
    if path is not None:
        im.save(path, format=format)
    if show:
        im.show()
    return grid



def get_images_path_list(folder_path: str) -> list:
    file_list = os.listdir(folder_path)
    images_list = []

    for file_name in file_list:
        if check_file_type(file_name) == "image":

            file_path = os.path.join(folder_path, file_name)
            images_list.append(file_path)
    
    return images_list




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


def PrintInfo():
    print('------------------------------------------------------------------------')
    print('This program is made by Yuan')
    print('It is a generative model implemented using DDIM to gernerate Zhu Geliang')
    print('Zhu Geliang is a living being!')
    print('------------------------------------------------------------------------')
    artwork = """
     #######  ###                          ####             ###       ##                                           ##
     #   ##    ##                         ##  ##             ##                                                   ####
        ##     ##      ##  ##            ##        ####      ##      ###      ####    #####     ### ##            ####
       ##      #####   ##  ##            ##       ##  ##     ##       ##         ##   ##  ##   ##  ##              ##
      ##       ##  ##  ##  ##            ##  ###  ######     ##       ##      #####   ##  ##   ##  ##              ##
     ##    #   ##  ##  ##  ##             ##  ##  ##         ##       ##     ##  ##   ##  ##    #####
     #######  ###  ##   ######             #####   #####    ####     ####     #####   ##  ##       ##              ##
                                                                                               #####
    """
    print(artwork)


if __name__ == '__main__':
    pass
 