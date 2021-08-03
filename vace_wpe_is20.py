"""
author: Joon-Young Yang (E-mail: dreadbird06@gmail.com)
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_custom.torch_utils import shape, to_gpu, to_arr
from torch_custom.custom_layers import CustomModel, update_dict

from torch_custom.stft_helper import StftHelper
import torch_custom.spectral_ops as spo
from torch_custom.spectral_ops import MelFeatureExtractor

from torch_custom.wpe_th_utils import wpe_mb_torch_ri


# class VACEWPE(nn.Module):
class VACEWPE(CustomModel):
  def __init__(self, stft_opts, lpsnet=None, vacenet=None, mfcc_opts={}):
    super(VACEWPE, self).__init__()
    assert lpsnet is not None and isinstance(lpsnet, nn.Module)
    assert vacenet is not None and isinstance(vacenet, nn.Module)

    assert len(stft_opts) >= 5
    self.stft_helper = StftHelper(**stft_opts)

    self.lpsnet = lpsnet
    self.vacenet = vacenet
    if self.vacenet.training:
      assert self.lpsnet.training

    self.weights = vacenet.weights
    self.weights_list = vacenet.weights_list
    self.weights_name = vacenet.weights_name

    ## Loss functions for training this model
    self.loss_mse = nn.MSELoss(reduction='mean')
    self.loss_mae = nn.L1Loss(reduction='mean')

    ## MFCC loss
    if len(mfcc_opts):
      self.melfeatext = MelFeatureExtractor(**mfcc_opts)

  def train(self):
    self.vacenet.train()

  def eval(self):
    self.vacenet.eval()
    # self.lpsnet.eval()

  # def to(self, device):
  #   self.vacenet.to(device)
  #   self.lpsnet.to(device)

  @staticmethod
  def parse_batch(data, target, device):
    data, target = to_gpu(data, device), to_gpu(target, device)
    return data, target

  def forward(self, sig_x, delay=3, taps=10, drop=0.0):
    """ sig_x is batched single-channel time-domain waveforms 
        shape: (B, T) == (batch, time)
    """
    ## Convert the time-domain signal to the STFT coefficients
    nb, nt = sig_x.size() # (B,t)
    stft_x = self.stft_helper.stft(sig_x) # (B,F,T,2)

    ## Compute early PSD using the LPSNet
    lps_x = spo.stft2lps(stft_x) # (B,F,T)
    psd_x = self.lpsnet(lps_x, drop=0.0).exp() # (B,F,T)

    ## Compute virtual signal using the VACENet
    stft_v = self.vacenet(stft_x, drop=drop) # (B,F,T,2)

    ## Compute early PSD using the LPSNet
    lps_v = spo.stft2lps(stft_v) # (B,F,T)
    psd_v = self.lpsnet(lps_v, drop=0.0).exp() # (B,F,T)

    ## Average PSD from the dual input channels
    psd_xv = 0.5 * (psd_x + psd_v)

    ## Stack the pair of actual and virtual signals
    stft_xv = torch.stack((stft_x, stft_v), dim=1) # (B,C=2,F,T,2)

    ## Batch-mode WPE
    ## >> STFT and PSD must be in shape (B,C,F,T,2) and (B,F,T), respectively.
    nfreq, nfrm = psd_xv.size(1), psd_xv.size(2)
    stft_wpe = wpe_mb_torch_ri(
      stft_xv, psd_xv, taps=taps, delay=delay) # (B,C=2,F,T,2)

    ## Inverse STFT
    stft_wpe_x, stft_wpe_v = stft_wpe[:,0], stft_wpe[:,1] # (B,F,T,2)
    sig_wpe_x = self.stft_helper.istft(stft_wpe_x, length=nt) # (B,t)
    return sig_wpe_x, stft_wpe_x, lps_x, stft_v

  def dereverb(self, sig_x, delay=3, taps=10):
    sig_wpe_x = self.forward(sig_x, delay, taps)[0]
    return to_arr(sig_wpe_x).squeeze()
    return self.forward(sig_x, delay, taps)[0]

  def get_loss(self, sig_x, sig_early, delay, taps, 
               alpha, beta, gamma, drop=0.0, summarize=False):
    """ Both "sig_x" and "sig_early" are batched time-domain waveforms """
    sig_wpe_x, stft_wpe_x, lps_x, stft_v = \
      self.forward(sig_x, delay, taps, drop=drop) # (B,t)
    # stft_wpe_x = self.stft_helper.stft(sig_wpe_x) # (B,F,T,2)
    stft_early = self.stft_helper.stft(sig_early) # (B,F,T,2)
    lms_wpe_x = spo.stft2lms(stft_wpe_x) # (B,F,T)
    lms_early = spo.stft2lms(stft_early) # (B,F,T)

    mse_stft_r_wpe = self.loss_mse(stft_wpe_x[..., 0], stft_early[..., 0])
    mse_stft_i_wpe = self.loss_mse(stft_wpe_x[..., 1], stft_early[..., 1])
    mse_lms_wpe = self.loss_mse(lms_wpe_x, lms_early)
    mae_wav_wpe = self.loss_mae(sig_wpe_x, sig_early)

    raw_loss = alpha*(mse_stft_r_wpe+mse_stft_i_wpe) \
             + beta*mse_lms_wpe + gamma*mae_wav_wpe

    if not summarize:
      return raw_loss
    else:
      loss_dict = {
        "raw_loss":raw_loss.item(), 
        "mse_stft_r_wpe":mse_stft_r_wpe.item(), 
        "mse_stft_i_wpe":mse_stft_i_wpe.item(), 
        "mse_lms_wpe":mse_lms_wpe.item(), 
        "mae_wav_wpe":mae_wav_wpe.item(), 
      }
      return raw_loss, loss_dict, (
        0.5*lps_x[-1],                                            # lms_x
        spo.stft2lms(stft_v[-1]),                                 # lms_v
        lms_wpe_x[-1],                                            # lms_wpe_x
        sig_x[-1],                                                # sig_x
        self.stft_helper.istft(stft_v[-1], length=sig_x.size(1)), # sig_v
        sig_wpe_x[-1])                                            # sig_wpe_x

  def get_loss_mfcc(self, sig_x, sig_early, delay, taps, 
                    alpha, beta, gamma, delta=0.0, power_scale=True, 
                    drop=0.0, summarize=False):
    """ Both "sig_x" and "sig_early" are batched time-domain waveforms """
    sig_wpe_x, stft_wpe_x, lps_x, stft_v = \
      self.forward(sig_x, delay, taps, drop=drop) # (B,t)
    # stft_wpe_x = self.stft_helper.stft(sig_wpe_x) # (B,F,T,2)
    stft_early = self.stft_helper.stft(sig_early) # (B,F,T,2)
    lms_wpe_x = spo.stft2lms(stft_wpe_x) # (B,F,T)
    lms_early = spo.stft2lms(stft_early) # (B,F,T)
    mfcc_wpe_x = self.melfeatext.mfcc(stft_wpe_x, power_scale=power_scale)
    mfcc_early = self.melfeatext.mfcc(stft_early, power_scale=power_scale)

    mse_stft_r_wpe = self.loss_mse(stft_wpe_x[..., 0], stft_early[..., 0])
    mse_stft_i_wpe = self.loss_mse(stft_wpe_x[..., 1], stft_early[..., 1])
    mse_lms_wpe = self.loss_mse(lms_wpe_x, lms_early)
    mae_wav_wpe = self.loss_mae(sig_wpe_x, sig_early)
    mae_mfcc_wpe = self.loss_mae(mfcc_wpe_x, mfcc_early)

    raw_loss = alpha*(mse_stft_r_wpe+mse_stft_i_wpe) \
             + beta*mse_lms_wpe + gamma*mae_wav_wpe + delta*mae_mfcc_wpe

    if not summarize:
      return raw_loss
    else:
      loss_dict = {
        "raw_loss":raw_loss.item(), 
        "mse_stft_r_wpe":mse_stft_r_wpe.item(), 
        "mse_stft_i_wpe":mse_stft_i_wpe.item(), 
        "mse_lms_wpe":mse_lms_wpe.item(), 
        "mae_wav_wpe":mae_wav_wpe.item(), 
        "mae_mfcc_wpe":mae_mfcc_wpe.item(), 
      }
      return raw_loss, loss_dict, (
        0.5*lps_x[-1],                                            # lms_x
        spo.stft2lms(stft_v[-1]),                                 # lms_v
        lms_wpe_x[-1],                                            # lms_wpe_x
        sig_x[-1],                                                # sig_x
        self.stft_helper.istft(stft_v[-1], length=sig_x.size(1)), # sig_v
        sig_wpe_x[-1])                                            # sig_wpe_x

