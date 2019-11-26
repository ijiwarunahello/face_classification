import torch
import os
import glob
import cv2
from PIL import Image


class Mydataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.data = []
        self.targets = []
        member_dir = glob.glob(self.root + "/*")
        print(len(member_dir))
        print(member_dir)
        for member in range(len(member_dir)):
            image_list = os.listdir(member_dir[member])
            print(member_dir[member])
            for i in range(len(image_list)):
                if 'DS' in image_list[i]:
                    print("DS file: {}".format(image_list[i]))
            for num in range(len(image_list)):
                if "DS" not in image_list[i]:
                    IMAGE_PATH = os.path.join(member_dir[member] + "/" + image_list[num])
                    self.data.append(cv2.imread(IMAGE_PATH))
                    self.targets.append(member)
                    if self.data[num] is None:
                        print("image_path: {}".format(IMAGE_PATH))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
