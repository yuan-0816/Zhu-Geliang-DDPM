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

