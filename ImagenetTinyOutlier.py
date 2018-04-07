import os
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader, is_image_file
from make_outlier import read_table


def make_dataset(dir, class_to_idx, map):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    l_outlier = map[fname][1]
                    item = (path, class_to_idx[l_outlier])
                    images.append(item)

    return images


class ImagenetTinyOutlier(datasets.ImageFolder):
    def __init__(self, root, ratio=0.1, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImagenetTinyOutlier, self).__init__(root, transform, target_transform, loader)
        self.ratio = ratio
        self.root = root
        map = read_table(root, ratio=self.ratio)
        self.samples = make_dataset(root, self.class_to_idx, map)
        self.imgs = self.samples

dir = './tiny-imagenet-200/train/'