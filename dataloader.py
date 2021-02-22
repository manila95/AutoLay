
from __future__ import absolute_import, division, print_function

import os
import cv2
import random
import torch

import PIL.Image as pil

import numpy as np

import torch.utils.data as data

from torchvision import transforms


def pil_loader(path):
    with open(path, 'rb') as f:
        with pil.open(f) as img:
            return img.convert('RGB')

def process_lane(topview, size):
    topview = np.array(topview)
    color_map = {10: 1, 9: 2, 8: 3, 7: 4, 11: 5, 12: 6, 13: 7, 25: 8, 30: 9}
    topview_n = np.zeros(topview.shape)
    for class_ in color_map.keys():
        topview_n[topview==class_] = color_map[class_]
    topview_n = cv2.resize(topview_n, (size, size), cv2.INTER_NEAREST)
    return topview_n



def process_topview(topview, size):
    topview = topview.convert("1")
    topview = topview.resize((size, size), pil.NEAREST)
    topview = topview.convert("L")
    topview = np.array(topview)
    topview_n = np.zeros(topview.shape)
    topview_n[topview == 255] = 1  # [1.,0.]
    return topview_n


def resize_topview(topview, size):
    #topview = topview.convert("1")
    topview = topview.resize((size, size), pil.NEAREST)
    topview = topview.convert("L")
    topview = np.array(topview)
    return topview


def process_discr(topview, size, num_ch=2):
    #print(np.max(topview))
    topview = resize_topview(topview, size).ravel()
    #topview_n = np.zeros((topview.size, num_ch))
    #topview_n[np.arange(topview.size), topview] = 1.
    #topview_n[topview == 255, 1] = 1.
    #topview_n[topview == 0, 0] = 1.
    #topview_n = torch.nn.functional.one_hot(topview)
    print(np.max(topview))
    topview_n = np.eye(num_ch)[topview.ravel()]
    print(topview_n.shape)
    return topview_n


class MonoDataset(data.Dataset):    
    def __init__(self, opt, filenames, channels=2, is_train=True):
        super(MonoDataset, self).__init__()

        self.opt = opt
        self.data_path = self.opt.data_path
        self.filenames = filenames
        self.is_train = is_train
        self.height = self.opt.height
        self.width = self.opt.width
        self.interp = pil.ANTIALIAS
        self.loader = pil_loader
        self.loader_dict = {"road":    self.get_road,
                            "vehicle": self.get_vehicle,
                            "lane":    self.get_lane}

        self.out_ch = channels
        self.to_tensor = transforms.ToTensor()

        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = transforms.Resize(
            (self.height, self.width), interpolation=self.interp)

    def preprocess(self, inputs, color_aug):

        for key in inputs.keys():
            if key != "color":
                inputs[key] = self.to_tensor(inputs[key])
            if "discr" in key:
                inputs[key] = torch.squeeze(inputs[key])
                inputs[key] = torch.transpose(torch.transpose(torch.nn.functional.one_hot(inputs[key].to(torch.int64), self.out_ch), 0, 2), 1, 2)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        frame_index = self.filenames[index]  # .split()
        # check this part from original code if the dataset is changed
        folder = self.opt.data_path

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.layout_loader = self.loader_dict[self.opt.seg_class]

        if self.opt.model_name == "videolayout":
            inputs["color"] = torch.empty(self.opt.seq_len, 3, self.opt.width, self.opt.height)
            for i in range(len(frame_index)):
                inputs["color"][i, :]  = self.to_tensor(color_aug(self.resize(self.get_color(folder, frame_index[i], do_flip))))

            inputs[self.opt.seg_class] = self.layout_loader(folder, frame_index[-1], do_flip)
        elif "pseudolidar" in self.opt.model_name:
            do_color_aug = 0 
            do_flip = 0
            inputs["color"] = self.get_pseudolidar(folder, frame_index, do_flip)
            inputs["color"] = np.transpose(inputs["color"], (2, 0, 1))
            inputs[self.opt.seg_class] = self.layout_loader(folder, frame_index, do_flip)
        else:
            inputs["color"] = self.to_tensor(color_aug(self.resize(self.get_color(folder, frame_index, do_flip))))
            inputs[self.opt.seg_class] = self.layout_loader(folder, frame_index, do_flip)
        inputs["%s_discr"%self.opt.seg_class] = inputs[self.opt.seg_class]

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        return inputs

    def get_color(self, folder, frame_index, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_pseudolidar(self, folder, frame_index, do_flip):
        pseudolidar = np.load(self.get_pseudolidar_path(folder, frame_index))

        return pseudolidar


    def get_road(self, folder, frame_index, do_flip):
        tv = self.loader(self.get_road_path(folder, frame_index))

        if do_flip:
            tv = tv.transpose(pil.FLIP_LEFT_RIGHT)

        return process_topview(tv.convert('L'), self.opt.occ_map_size)

    def get_vehicle(self, folder, frame_index, do_flip):
        tv = self.loader(self.get_vehicle_path(folder, frame_index))

        if do_flip:
            tv = tv.transpose(pil.FLIP_LEFT_RIGHT)

        return process_topview(tv.convert('L'), self.opt.occ_map_size)

    def get_osm(self, root_dir, do_flip):
        osm = self.loader(self.get_osm_path(root_dir))
        return osm

    def get_static_gt(self, folder, frame_index, do_flip):
        tv = self.loader(self.get_static_gt_path(folder, frame_index))
        return tv.convert('L')

    def get_dynamic_gt(self, folder, frame_index, do_flip):
        tv = self.loader(self.get_dynamic_gt_path(folder, frame_index))
        return tv.convert('L')

    def get_lane(self, folder, frame_index, do_flip):
        tv = np.load(self.get_lane_path(folder, frame_index))
        return process_lane(tv, self.opt.occ_map_size)




class KITTIObject(MonoDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """

    def __init__(self, *args, **kwargs):
        super(KITTIObject, self).__init__(*args, **kwargs)
        self.root_dir = "./data/object"

    def get_image_path(self, root_dir, frame_index):
        image_dir = os.path.join(root_dir, 'image_2')
        img_path = os.path.join(image_dir, "%06d.png" % int(frame_index))
        return img_path

    def get_dynamic_path(self, root_dir, frame_index):
        tv_dir = os.path.join(root_dir, 'vehicle_256')
        tv_path = os.path.join(tv_dir, "%06d.png" % int(frame_index))
        return tv_path

    def get_dynamic_gt_path(self, root_dir, frame_index):
        return self.get_dynamic_path(root_dir, frame_index)

    def get_static_gt_path(self, root_dir, frame_index):
        pass


class KITTIOdometry(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(KITTIOdometry, self).__init__(*args, **kwargs)
        self.root_dir = "./data/odometry/sequences/"

    def get_image_path(self, root_dir, frame_index):
        file_name = frame_index.replace("road_dense128", "image_2")
        img_path = os.path.join(root_dir, file_name)
        return img_path

    def get_static_path(self, root_dir, frame_index):
        path = os.path.join(root_dir, frame_index)
        return path

    def get_osm_path(self, root_dir):
        osm_file = np.random.choice(os.listdir(root_dir))
        osm_path = os.path.join(root_dir, osm_file)

        return osm_path

    def get_static_gt_path(self, root_dir, frame_index):
        return self.get_static_path(self, root_dir, frame_index)

    def get_dynamic_gt_path(self, root_dir, frame_index):
        pass


class AutoLay(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(AutoLay, self).__init__(*args, **kwargs)
        self.root_dir = "./data/raw/"

    def get_image_path(self, root_dir, frame_index):
        img_path = os.path.join(root_dir, frame_index)
        return img_path

    def get_pseudolidar_path(self, root_dir, frame_index):
        pseudolidar_path = os.path.join(root_dir, frame_index.replace("image_02/data", "image_02/pseudo_lidar_monodepth2_256"))
        return pseudolidar_path.replace("png", "npy")

    def get_road_path(self, root_dir, frame_index):
        path = os.path.join(
            root_dir, frame_index.replace(
                "image_02/data", "road_bev_gt"))
        return path

    def get_osm_path(self, root_dir):
        osm_file = np.random.choice(os.listdir(root_dir))
        osm_path = os.path.join(root_dir, osm_file)

        return osm_path

    def get_static_gt_path(self, root_dir, frame_index):
        path = os.path.join(
            root_dir, frame_index.replace(
                "image_02/data", "road_bev_gt"))
        return path

    def get_dynamic_gt_path(self, root_dir, frame_index):
        path = os.path.join(root_dir, frame_index.replace("image_02/data", "car_bev_txt"))
        return path

    def get_vehicle_path(self, root_dir, frame_index):
        path = os.path.join(root_dir, frame_index.replace("image_02/data", "car_bev_txt"))
        return path

    def get_lane_path(self, root_dir, frame_index):
        path = os.path.join(root_dir, frame_index.replace("image_02/data", "numbered_bev"))
        return path.replace("png", "npy")

class Argoverse(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(Argoverse, self).__init__(*args, **kwargs)
        self.root_dir = "./data/argo"

    def get_image_path(self, root_dir, frame_index):
        file_name = frame_index.replace(
            "road_gt", "stereo_front_left").replace(
            "png", "jpg")
        img_path = os.path.join(root_dir, file_name)
        return img_path

    def get_road_path(self, root_dir, frame_index):
        path = os.path.join(root_dir, frame_index)
        return path

    def get_vehicle_path(self, root_dir, frame_index):
        file_name = frame_index.replace(
            "road_gt", "car_bev_gt").replace(
            "png", "jpg")
        path = os.path.join(root_dir, file_name)
        return path

    def get_static_gt_path(self, root_dir, frame_index):
        path = os.path.join(
            root_dir,
            frame_index).replace(
            "road_bev",
            "road_gt")
        return path

    def get_dynamic_gt_path(self, root_dir, frame_index):
        return self.get_dynamic_path(self, root_dir, frame_index)

    def get_lane_path(self, root_dir, frame_index):
        return os.path.join(root_dir, frame_index).replace("road_gt", "numbered_lanes")
    
