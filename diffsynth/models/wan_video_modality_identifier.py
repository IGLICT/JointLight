import torch
import torch.nn as nn
from .wan_video_dit import sinusoidal_embedding_1d


class WanModalityIdentifierModel(torch.nn.Module):
    def __init__(self, freq_dim=256, dim=3072):
        super().__init__()
        self.freq_dim = freq_dim
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(freq_dim, dim),
            torch.nn.SiLU(),
            torch.nn.Linear(dim, dim),
            torch.nn.SiLU(),
            torch.nn.Linear(dim, dim * 6),
        )

    def forward(self, motion_bucket_id):
        emb = sinusoidal_embedding_1d(self.freq_dim, motion_bucket_id * 10)
        emb = self.linear(emb)
        return emb

    def init(self):
        state_dict = self.linear[-1].state_dict()
        state_dict = {i: state_dict[i] * 0 for i in state_dict}
        self.linear[-1].load_state_dict(state_dict)

    @staticmethod
    def state_dict_converter():
        return WanMotionControllerModelDictConverter()


class WanMotionControllerModelDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        return state_dict

    def from_civitai(self, state_dict):
        return state_dict
