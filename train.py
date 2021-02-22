
import argparse
import os

from models import MonoLayout, MonoOccupancy, PseudoLidar_UNet, PseudoLidar_ENet, VideoLayout 
from dataloader import AutoLay, Argoverse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import tqdm

from utils import mean_IU, mean_precision
from eval import evaluate_layout

def get_args():
    parser = argparse.ArgumentParser(description="MonoLayout options")
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to the root data directory")
    parser.add_argument("--save_path", type=str, default="./models/",
                        help="Path to save models")
    parser.add_argument("--load_weights_folder", type=str, default="",
                        help="Path to a pretrained model used for initialization")
    parser.add_argument("--model_name", type=str, default="monolayout",
                        help="Model Name with specifications")
    parser.add_argument("--split", type=str,
                        choices=["argo", "AutoLay"],
                        help="Data split for training/validation")
    parser.add_argument("--ext", type=str, default="png",
                        help="File extension of the images")
    parser.add_argument("--height", type=int, default=512,
                        help="Image height")
    parser.add_argument("--width", type=int, default=512,
                        help="Image width")
    parser.add_argument("--seg_class", type=str,
                        choices=["road", "vehicle", "lane"],
                        help="Type of model being trained")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Mini-Batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="learning rate")
    parser.add_argument("--lr_D", type=float, default=1e-5,
                        help="discriminator learning rate")
    parser.add_argument("--scheduler_step_size", type=int, default=5,
                        help="step size for the both schedulers")
    parser.add_argument("--static_weight", type=float, default=5.,
                        help="static weight for calculating loss")
    parser.add_argument("--dynamic_weight", type=float, default=15.,
                        help="dynamic weight for calculating loss")
    parser.add_argument("--occ_map_size", type=int, default=256,
                        help="size of topview occupancy map")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Max number of training epochs")
    parser.add_argument("--log_frequency", type=int, default=5,
                        help="Log files every x epochs")
    parser.add_argument("--num_workers", type=int, default=12,
                        help="Number of cpu workers for dataloaders")
    parser.add_argument("--lambda_D", type=float, default=0.01,
                        help="tradeoff weight for discriminator loss")
    parser.add_argument("--discr_train_epoch", type=int, default=5,
                        help="epoch to start training discriminator")
    parser.add_argument("--seq_len", type=int, default=8,
                        help="number of frames in an input")
    parser.add_argument("--iou_thresh", type=float, default=0.5, 
                        help="IOU threshold for lane detections")
    return parser.parse_args()



class Trainer:
    def __init__(self):
        self.opt = get_args()
        self.models = {}
        self.device = "cuda"
        self.criterion_d = nn.BCEWithLogitsLoss()
        self.parameters_to_train = []
        self.parameters_to_train_D = []

        # Output Channels 
        ch_dict = {"road": 2,
                   "vehicle": 2,
                   "lane": 10}

        # Models
        model_dict = {"monolayout":       MonoLayout,
                      "monooccupancy":    MonoOccupancy,
                      "pseudolidar-unet": PseudoLidar_UNet,
                      "pseudolidar-enet": PseudoLidar_ENet,
                      "videolayout":      VideoLayout}

        # Data Loaders
        dataset_dict = {"AutoLay": AutoLay,
                        "argo": Argoverse}
        Model = model_dict[self.opt.model_name]
        self.model = Model(self.opt, ch_dict[self.opt.seg_class]).cuda()
        self.dataset = dataset_dict[self.opt.split]
        fpath = os.path.join(
            os.path.dirname(__file__),
            "splits",
            self.opt.split,
            "{}_files.txt")

        if self.opt.model_name == "videolayout":
            readlines_fn = self.temporal_readlines
            train_file = "train_temporal"
            val_file = "val_temporal"
        else:
            readlines_fn = self.readlines
            train_file = "train"
            val_file = "val"

        train_filenames = readlines_fn(fpath.format(train_file))
        val_filenames = readlines_fn(fpath.format(val_file))
        self.val_filenames = val_filenames
        self.train_filenames = train_filenames

        train_dataset = self.dataset(self.opt, train_filenames, channels=ch_dict[self.opt.seg_class])
        val_dataset = self.dataset(self.opt, val_filenames, channels=ch_dict[self.opt.seg_class], is_train=False)

        self.train_loader = DataLoader(
            train_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)
        self.val_loader = DataLoader(
            val_dataset,
            1,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)

        if self.opt.load_weights_folder != "":
            self.load_model()

        # Cross Entropy weights (can be tuned further!!!)
        self.weight_dict = {"road":    [1., 5.],
                            "vehicle": [1., 15.],
                            "lane":    [1., 3., 5., 7., 10., 5., 7., 10., 20., 15.]}

        print("Using split:\n  ", self.opt.split)
        print(
            "There are {:d} training items and {:d} validation items\n".format(
                len(train_dataset),
                len(val_dataset)))

    def readlines(self, filename):
        """Read all the lines in a text file and return as a list
        """
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
        return lines

    def temporal_readlines(self, filename):
        f = open(filename, "r")
        files = [k.split("\n")[:-1] for k in f.read().split(",")[:-1]]
        temporal_files = []
        for seq_files in files:
            seq_files = [seq_files[0]]*self.opt.seq_len + seq_files
            for i in range(self.opt.seq_len, len(seq_files)):
                temporal_files.append(seq_files[i-self.opt.seq_len:i])
        return temporal_files

    def train(self):

        for self.epoch in range(self.opt.num_epochs):
            evaluate_layout(self.opt, self.model, self.val_loader)
            loss = self.run_epoch()
            print("Epoch: %d | Loss: %.4f | Discriminator Loss: %.4f" %
                  (self.epoch, loss["cross-entropy"], loss["discr"]))

            if self.epoch % self.opt.log_frequency == 0:
                evaluate_layout(self.opt, self.model, self.val_loader)
                self.save_model()

    def process_batch(self, inputs, validation=False):
        outputs = {}
        for key, inpt in inputs.items():
            inputs[key] = inpt.to(self.device)

        outputs = self.model(inputs["color"])

        if validation:
            return outputs
        losses = self.compute_losses(inputs, outputs)
        losses["loss_discr"] = torch.zeros(1)

        return outputs, losses

    def run_epoch(self):
        loss = {}
        loss["cross-entropy"], loss["discr"], loss["adv"] = 0.0, 0.0, 0.0
        for batch_idx, inputs in tqdm.tqdm(enumerate(self.train_loader)):
            outputs, losses = self.process_batch(inputs)
            losses = self.model.step(inputs, outputs, losses, self.epoch)

            loss["cross-entropy"] += losses["loss"].item()
            if self.opt.model_name in ["monolayout", "videolayout"]:
                loss["discr"] += losses["discr"].item()
                loss["adv"] += losses["adv"].item()
        loss["cross-entropy"] /= len(self.train_loader) 
        return loss

    def validation(self):
        iou, mAP = np.array([0., 0.]), np.array([0., 0.])
        for batch_idx, inputs in tqdm.tqdm(enumerate(self.val_loader)):
            with torch.no_grad():
                outputs = self.process_batch(inputs, True)
            pred = np.squeeze(
                torch.argmax(
                    outputs["topview"].detach(),
                    1).cpu().numpy())
            true = np.squeeze(
                inputs[self.opt.seg_class + "_gt"].detach().cpu().numpy())
            iou += mean_IU(pred, true)
            mAP += mean_precision(pred, true)
        iou /= len(self.val_loader)
        mAP /= len(self.val_loader)
        print(
            "Epoch: %d | Validation: mIOU: %.4f mAP: %.4f" %
            (self.epoch, iou[1], mAP[1]))

    def compute_losses(self, inputs, outputs):
        losses = {}
        losses["loss"] = self.compute_topview_loss(
                                    outputs["topview"],
                                    inputs[self.opt.seg_class])

        return losses

    def compute_topview_loss(self, outputs, true_top_view):

        generated_top_view = outputs
        true_top_view = torch.squeeze(true_top_view.long())
        # if self.opt.seg_class == "lane":
        #     loss = nn.CrossEntropyLoss(weight=torch.Tensor([1., 3., 5., 7., 10., 5., 7., 10., 20., 15.]).to(self.device))
        # else:
        loss = nn.CrossEntropyLoss(weight=torch.Tensor(self.weight_dict[self.opt.seg_class]).to(self.device))
        output = loss(generated_top_view, true_top_view)
        return output.mean()

    def save_model(self):
        save_path = os.path.join(
            self.opt.save_path,
            self.opt.model_name,
            self.opt.split,
            self.opt.seg_class,
            "weights_{}".format(
                self.epoch))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model_dict = self.model.state_dict()
        model_dict["height"] = self.opt.height
        model_dict["width"] = self.opt.width
        model_path = os.path.join(save_path, "{}.pth".format("model"))
        torch.save(model_dict, model_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(
            self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print(
            "loading model from folder {}".format(
                self.opt.load_weights_folder))

        for key in self.models.keys():
            print("Loading {} weights...".format(key))
            path = os.path.join(
                self.opt.load_weights_folder,
                "{}.pth".format(key))
            model_dict = self.models[key].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[key].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(
            self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
