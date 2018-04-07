import os
import numpy as np
import csv

from torchvision.datasets.folder import is_image_file


# name, is_outlier, outlier_type, true_type
def write_outlier(image_dir, ratio=0.1):
    type_list = [i for i in os.listdir(image_dir) if os.path.isdir(image_dir + i + '/images/')]
    for type in type_list:
        if os.path.isdir(image_dir + type + '/images/'):
            images = [i for i in os.listdir(image_dir + type + '/images/') if is_image_file(i)]
            size_list = len(images)
            outlier_num = int(size_list * ratio)
            outlier_index = np.random.choice(size_list, outlier_num, replace=False)
            outlier_label = np.random.choice([i for i in type_list if i != type], outlier_num, replace=True)
            with open(image_dir + type + '/' + type + '_outlier_' + str(float(ratio)) + '.csv', 'wb') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                count = 0
                for i, im in enumerate(images):
                    if i not in outlier_index:
                        spamwriter.writerow([im, 0, type, type])
                    else:
                        spamwriter.writerow([im, 1, outlier_label[count], type])
                        count += 1

def read_table(image_dir, ratio=0.1):
    map = {}
    r = float(ratio)
    for type in os.listdir(image_dir):
        with open(image_dir + type + '/' + type + '_outlier_' + str(r) + '.csv', 'rb') as csvfile:
            spamwriter = csv.reader(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in spamwriter:
                map[row[0]] = (bool(row[1]), row[2], row[3])
    return map


def main():
    dir = './tiny-imagenet-200/train/'
    for i in [0, 0.1, 0.2, 0.3]:
        write_outlier(dir, i)

main()
