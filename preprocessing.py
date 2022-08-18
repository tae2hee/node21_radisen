import torch
import numpy as np
from torchvision import transforms as transforms


def make_any_patch(img, x, y, patch_size):

    patch = img[x - int(patch_size / 2) : x + int(patch_size / 2),
            y - int(patch_size / 2) : y + int(patch_size / 2)]

    return patch

def patch_slicing(_img):
    patch_size = 112
    scaler = 3
    image_size = 1024
    step = 16

    patch_size1 = (1 + scaler) * patch_size
    patch_size2 = int((1 + (scaler / 2)) * patch_size)
    patch_size3 = patch_size

    trans = transforms.Compose([transforms.Resize(size=(patch_size, patch_size)),
                                     transforms.Normalize((0.5,), (0.5,))])

    counts = int(image_size / step)

    _img_t = torch.zeros((counts * counts, 3, patch_size, patch_size))

    _img1 = np.pad(_img, (int(patch_size1 / 2), int(patch_size1 / 2)), constant_values=0)

    pidx = 0
    for i in range(counts):
        for j in range(counts):
            x = int(patch_size1 / 2) + step * i
            y = int(patch_size1 / 2) + step * j

            patch1 = make_any_patch(_img1, x, y, patch_size1)
            patch2 = make_any_patch(_img1, x, y, patch_size2)
            patch3 = make_any_patch(_img1, x, y, patch_size3)
            patch1 = torch.from_numpy(patch1).type(torch.float)
            patch2 = torch.from_numpy(patch2).type(torch.float)
            patch3 = torch.from_numpy(patch3).type(torch.float)

            _img_t[pidx, 0, :, :] = trans(patch1.unsqueeze(0))
            _img_t[pidx, 1, :, :] = trans(patch2.unsqueeze(0))
            _img_t[pidx, 2, :, :] = trans(patch3.unsqueeze(0))
            pidx += 1


    return _img_t

