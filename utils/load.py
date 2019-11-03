#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os
from PIL import Image
from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    # get rid of data format (for example: abandon.jpg)
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    """
        id is name of image, i is pos of image 
        (`pos` be used to define left or right square in function `get_square`, 
         pos == 0 is left,
         pos == 1 is right)
    """
    return ((id, i) for id in ids for i in range(n))
    # return ((id, i) for i in range(n) for id in ids) # this order is wrong


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    # params suffix: is data format
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        yield get_square(im, pos)


def get_imgs_and_masks(ids, dir_img, dir_mask, scale, dataset):
    """Return all the couples (img, mask)"""
    if dataset == 'CASIA':
        format = 'jpg'
    elif dataset == 'COLUMB':
        format = 'jpg'

    imgs = to_cropped_imgs(ids, dir_img, '.{}'.format(format), scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '_mask.png', scale)

    return zip(imgs_normalized, masks)


# def get_full_img_and_mask(id, dir_img, dir_mask):
#     im = Image.open(dir_img + id + '.jpg')
#     mask = Image.open(dir_mask + id + '_mask.gif')
#     return np.array(im), np.array(mask)