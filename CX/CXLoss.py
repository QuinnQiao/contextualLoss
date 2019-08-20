import torch
import numpy as np

class TensorAxis:
    N = 0
    C = 1
    H = 2
    W = 3


class CSFlow:
    def __init__(self, sigma=float(0.1), b=float(1.0)):
        self.b = b
        self.sigma = sigma

    def __calculate_CS(self, scaled_distances, axis_for_normalization=TensorAxis.C):
        self.scaled_distances = scaled_distances
        self.cs_weights_before_normalization = torch.exp((self.b - scaled_distances) / self.sigma)
        self.cs_NHWC = CSFlow.sum_normalize(self.cs_weights_before_normalization, axis_for_normalization)

    # --
    @staticmethod
    def create_using_dotP(I_features, T_features, sigma=float(0.5), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        # prepare feature before calculating cosine distance
        T_features, I_features = cs_flow.center_by_T(T_features, I_features)
        T_features = CSFlow.l2_normalize_channelwise(T_features)
        I_features = CSFlow.l2_normalize_channelwise(I_features)

        # work seperatly for each example in dim 1
        cosine_dist_l = []
        N = T_features.size()[0]
        for i in range(N):
            T_features_i = T_features[i, :, :, :].unsqueeze_(0)
            I_features_i = I_features[i, :, :, :].unsqueeze_(0)
            patches_PC11_i = cs_flow.patch_decomposition(T_features_i)  # 1CHW --> PC11 with P=H*W
            cosine_dist_i = torch.nn.functional.conv2d(I_features_i, patches_PC11_i)   # 1PHW
            cosine_dist_l.append(cosine_dist_i)

        cs_flow.cosine_dist = torch.cat(cosine_dist_l, dim=0)

        cs_flow.raw_distances = (1 - cs_flow.cosine_dist) / 2

        relative_dist = cs_flow.calc_relative_distances()
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    def calc_relative_distances(self, axis=TensorAxis.C):
        epsilon = 1e-5
        div = torch.min(self.raw_distances, dim=axis, keepdim=True)[0]
        relative_dist = self.raw_distances / (div + epsilon)
        return relative_dist

    @staticmethod
    def sum_normalize(cs, axis=TensorAxis.C):
        reduce_sum = torch.sum(cs, dim=axis, keepdim=True)
        cs_normalize = torch.div(cs, reduce_sum)
        return cs_normalize

    def center_by_T(self, T_features, I_features):
        # assuming both input are of the same size
        # calculate stas over [batch, height, width], expecting 1x1xDepth tensor
        # Tensor of pytorch is [NCHW]
        if T_features.size(0) == 1:
            n_channel = T_features.size(1)
            self.meanT = T_features.view(n_channel, -1).mean(1).view(1, n_channel, 1, 1)
        else:
            n_channel = T_features.size(1)
            self.meanT = T_features.permute(dims=(1,0,2,3)).view(n_channel, -1).mean(1).view(1, n_channel, 1, 1)
        # self.meanT = T_features.mean(0, keepdim=True).mean(1, keepdim=True).mean(2, keepdim=True)
        self.T_features_centered = T_features - self.meanT
        self.I_features_centered = I_features - self.meanT

        return self.T_features_centered, self.I_features_centered

    @staticmethod
    def l2_normalize_channelwise(features):
        norms = features.norm(p=2, dim=TensorAxis.C, keepdim=True)
        features = features.div(norms)
        return features

    def patch_decomposition(self, T_features):
        # 1CHW --> PC11, with P=H*W
        n_channel = T_features.size(1)
        patches_PC11 = T_features.view(n_channel, -1).permute(dims=(1,0)).view(-1, n_channel, 1, 1)
        return patches_PC11


# ---------------------------
#           CX loss
# ---------------------------

def CX_loss(T_features, I_features, sigma=1.0):

    cs_flow = CSFlow.create_using_dotP(I_features, T_features, sigma=sigma)
    cs = cs_flow.cs_NHWC

    # reduce_max X and Y dims
    k_max_NC = torch.max(cs.view(cs.size(0), cs.size(1), -1), dim=2)[0]
    # reduce mean over C dim
    CS = torch.mean(k_max_NC, dim=1)
    score = -torch.log(CS)
    # reduce mean over N dim
    return torch.mean(score)
