import monai
import numpy as np
import torch
from torchvision import transforms
import pickle

class NeedleSegModel:
    def __init__(self, hparams_path, weights_path):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if hparams_path is not None:
            with open(hparams_path, "rb") as f:
                self.hparams = pickle.load(open(hparams_path, "rb"))
            self.model = monai.networks.nets.UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=4,
                channels=tuple([2**x for x in range(self.hparams["num_channels"][0], self.hparams["num_channels"][1])]),
                strides=tuple([2] * (self.hparams["num_channels"][1] - self.hparams["num_channels"][0] - 1)),
                kernel_size=3,
                num_res_units=self.hparams["num_res_units"],
            ).to(self.device)
        else:
            self.model = monai.networks.nets.UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=4,
                channels=(16, 32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2, 2),
                kernel_size=3,
            ).to(self.device)

        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()
        print("Model loaded successfully")

    def preprocess_volume(self, oct_volume):
        # oct_volume = oct_volume.transpose(1, 0, 2)
        # oct_volume = np.rot90(oct_volume, axes=(1, 2))
        oct_volume = oct_volume.astype(np.float32)
        oct_volume = torch.tensor(oct_volume).to(self.device)
        oct_volume = transforms.Pad((12, 0, 12, 0))(oct_volume)
        oct_volume = oct_volume.unsqueeze(1)
        return oct_volume

    def postprocess_volume(self, seg_volume):
        seg_volume = torch.argmax(torch.softmax(seg_volume, dim=1), dim=1)
        seg_volume = seg_volume.cpu().numpy()
        seg_volume = seg_volume[:, :, 12:-12].astype(np.uint8) # convert to uint8 for performance
        return seg_volume

    def segment_volume(self, oct_volume):
        with torch.no_grad():
            seg_volume = self.model(oct_volume)
        return seg_volume
