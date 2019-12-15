import torch
import utils as ut
import torchvision.transforms.functional as FT
from skimage.io import imread, imsave
from torchvision import transforms
from models import model_dict
import numpy as np


def cvt_multi_class_to_one_mask(blobs_mask):
    bg_mask = np.zeros((1,) + blobs_mask.shape[1:], int)
    blobs_mask = np.concatenate((bg_mask, blobs_mask), axis=0)
    one_mask = np.argmax(blobs_mask, axis=0)
    return one_mask


def apply(image_path, model_name, model_path):
    transformer = ut.ComposeJoint([
        [transforms.ToTensor(), None],
        [transforms.Normalize(*ut.mean_std), None],
        [None, ut.ToLong()]
    ])

    # Load best model

    # init model with n_classes
    n_classes = 21 if 'pascal' in model_path else 2
    model = model_dict[model_name](n_classes=n_classes).cuda()
    model.load_state_dict(torch.load(model_path))
    print('load done!')

    # Read Image
    image_raw = imread(image_path)

    collection = list(map(FT.to_pil_image, [image_raw, image_raw]))
    image, _ = transformer(collection)

    batch = {"images": image[None]}

    # Make predictions
    pred_blobs = model.predict(batch, method="blobs").squeeze()  # (20, 375, 500), need cvt to 2D
    # for 1-class, squeeze -> 2D
    # for 20-class, can't
    if len(pred_blobs.shape) == 3:
        pred_blobs = cvt_multi_class_to_one_mask(pred_blobs)

    # pred_counts = int(model.predict(batch, method="counts").ravel()[0])
    pred_counts = model.predict(batch, method="counts").squeeze().astype(int)

    # Save Output
    save_path = image_path + "_blobs_count:{}.png".format(sum(pred_counts))
    imsave(save_path, ut.combine_image_blobs(image_raw, pred_blobs))

    print("| Counts: {}\n| Output saved in: {}".format(pred_counts, save_path))
