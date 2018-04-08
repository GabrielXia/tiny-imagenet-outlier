import os
import csv
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


def make_val(dir, class_to_index):
    images = []
    dir = os.path.expanduser(dir)
    dir_images = os.path.join(dir, "images")
    annotations_dir =  os.path.join(dir, "val_annotations.txt")
    dic_val = {}
    with open(annotations_dir, 'rb') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            dic_val[row[0]] = row[1]
    for im in os.listdir(dir_images):
        if is_image_file(im):
            path = os.path.join(dir_images, im)
            item = (path, class_to_index[dic_val[im]])
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


class ImagenetTinyVal(datasets.ImageFolder):
    def __init__(self, root, class_to_index, transform=None, target_transform=None,
                 loader=default_loader):
        self.root = root
        self.samples = make_val(root, class_to_index)
        self.imgs = self.samples
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.class_to_idx = class_to_index

train_dir = './tiny-imagenet-200/train/'
val_dir = './tiny-imagenet-200/val/'

a = ImagenetTinyOutlier(train_dir, ratio=0)
class_to_index = a.class_to_idx
b = ImagenetTinyVal(val_dir, class_to_index)

print(b[0])