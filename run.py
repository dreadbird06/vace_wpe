"""
Example codes for speech dereverberation based on the WPE variants.

author: Joon-Young Yang (E-mail: dreadbird06@gmail.com)
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

import numpy as np
import soundfile as sf

import torch
torch.set_printoptions(precision=10)

from torch_custom.torch_utils import load_checkpoint, to_arr
from torch_custom.iterative_wpe import IterativeWPE
from torch_custom.neural_wpe import NeuralWPE


## ----------------------------------------------------- ##
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('device = {}'.format(device))
# device = "cpu"


sample_wav = 'data/AMI_WSJ20-Array1-1_T10c0201.wav'
sample_wav2 = 'data/AMI_WSJ20-Array1-2_T10c0201.wav'
sample_wav_drc = 'data/raw_signal.wav'

stft_opts_torch = dict(
  n_fft=1024, hop_length=256, win_length=1024, win_type='hanning', 
  symmetric=True)
fft_bins = stft_opts_torch['n_fft']//2 + 1

# mfcc_opts_torch = dict(
#   fs=16000, nfft=1024, lowfreq=20., maxfreq=7600., 
#   nlinfilt=0, nlogfilt=40, nceps=40, 
#   lifter_type='sinusoidal', lift=-22.0) # only useful during fine-tuning

delay, taps1, taps2 = 3, 30, 15
## ----------------------------------------------------- ##


def apply_drc(audio, drc_ratio=0.25, n_pop=100, dtype='float32'):
  normalized = (max(audio.max(), abs(audio.min())) <= 1.0)
  normalizer = 1.0 if normalized else float(2**15)
  ## compute MMD
  audio_sorted = np.sort(audio.squeeze(), axis=-1)  # either (N,) or (D, N)
  audio_mmd = audio_sorted[..., -1:-n_pop-1:-1].mean(dtype=dtype) \
            - audio_sorted[..., :n_pop].mean(dtype=dtype)
  drange_gain = 2 * (normalizer/audio_mmd) * drc_ratio
  return (audio * drange_gain).astype(dtype), drange_gain.astype('float32')


def run_vace_wpe(pretrain_opt='late', simp_opt='b'):
  print('Running "VACE-WPE-%s-%s"...' % (simp_opt, pretrain_opt))

  ## Pre-training option
  if pretrain_opt == 'self': # PT-self
    fake_scale = 1.0
  elif pretrain_opt == 'late': # PT-late
    fake_scale = 2.0

  ## VACE-WPE system architecture
  if simp_opt == 'a':
    from vace_wpe_is20 import VACEWPE
  elif simp_opt == 'b':
    from vace_wpe import VACEWPE

  ## Saved checkpoint file
  if pretrain_opt == 'self' and simp_opt == 'a':
    ## VACE-WPE-a-self: PT-self + VACE-WPE (w/o simplification)
    ckpt_file = 'models/20210409-175700/ckpt-ep60'
  elif pretrain_opt == 'late' and simp_opt == 'a':
    ## VACE-WPE-a-late: PT-late + VACE-WPE (w/o simplification)
    ckpt_file = 'models/20210409-174900/ckpt-ep60'
  elif pretrain_opt == 'late' and simp_opt == 'b':
    ## VACE-WPE-b-late: PT-late + simplified VACE-WPE
    ckpt_file = 'models/20210129-140530/ckpt-ep60'

  ## ------------------------------------------------- ##

  ## VACENet
  from gcunet4c_4M4390 import VACENet
  vacenet = VACENet(
    input_dim=fft_bins, stft_opts=stft_opts_torch, 
    input_norm='globalmvn', # loaded from the saved checkpoint
    scope='vace_unet', fake_scale=fake_scale)
  vacenet = vacenet.to(device)
  vacenet.eval()
  # print('VACENet size = {:.2f}M'.format(vacenet.size))
  # vacenet.check_trainable_parameters()

  ## LPSNet
  from bldnn_4M62 import LstmDnnNet as LPSEstimator
  lpsnet = LPSEstimator(
    input_dim=fft_bins, 
    stft_opts=stft_opts_torch, 
    input_norm='globalmvn', # loaded from the saved checkpoint
    scope='ldnn_lpseir_ns')
  lpsnet = lpsnet.to(device)
  lpsnet.eval()
  # lpsnet.freeze() # should be frozen when fine-tuning the VACENet
  # print('LPSNet size = {:.2f}M'.format(lpsnet.size))
  # lpsnet.check_trainable_parameters()

  ## VACE-WPE
  dnn_vwpe = VACEWPE(
    stft_opts=stft_opts_torch, 
    lpsnet=lpsnet, vacenet=vacenet)#, 
    # mfcc_opts=mfcc_opts_torch) # only useful when fine-tuning the VACENet
  dnn_vwpe, *_ = load_checkpoint(dnn_vwpe, checkpoint=ckpt_file, strict=False)
  dnn_vwpe.to(device)
  dnn_vwpe.eval()
  # print('VACE-WPE size = {:.2f}M'.format(dnn_vwpe.size))
  # dnn_vwpe.check_trainable_parameters()

  ## ------------------------------------------------- ##

  ## Load audio and apply DRC
  aud, fs = sf.read(sample_wav) # (t,), 16000
  aud, drc_gain = apply_drc(aud) # (t,), ()
  if not os.path.isfile(sample_wav_drc):
    sf.write(sample_wav_drc, data=aud, samplerate=fs)

  ## Perform dereverberation
  aud = torch.from_numpy(aud)[None] # (batch, samples)
  with torch.no_grad():
    ## The input audio is in shape (batch, samples) (always assume #channels == 1)
    enh = dnn_vwpe.dereverb(
      aud.to(device), delay=delay, taps=taps2) # (t,)
  ## Save
  output_wav_path = 'data/vace_wpe_%s_%s_taps%d.wav' % (
    simp_opt, pretrain_opt, taps2)
  sf.write(output_wav_path, data=enh, samplerate=fs)


def run_neural_wpe(chs='single', dtype=torch.float64):
  print('Running "Neural-WPE-%s"...' % chs)

  ckpt_file = np.random.choice([
    'models/20210409-175700/ckpt-ep60', # VACE-WPE-a-self
    'models/20210409-174900/ckpt-ep60', # VACE-WPE-a-late
    'models/20210129-140530/ckpt-ep60', # VACE-WPE-b-late
  ]) # the VACE-WPE variants are supposed to share the same LPSNet

  ## ------------------------------------------------- ##

  ## LPSNet
  from bldnn_4M62 import LstmDnnNet as LPSEstimator
  lpsnet = LPSEstimator(
    input_dim=fft_bins, 
    stft_opts=stft_opts_torch, 
    input_norm='globalmvn', # loaded from the saved checkpoint
    scope='ldnn_lpseir_ns')
  lpsnet = lpsnet.to(device)
  # lpsnet.freeze() # should be frozen when fine-tuning the VACENet
  lpsnet.eval()
  # print('LPSNet size = {:.2f}M'.format(lpsnet.size))
  # lpsnet.check_trainable_parameters()

  ## Neural WPE
  dnn_wpe = NeuralWPE(
    stft_opts=stft_opts_torch, 
    lpsnet=lpsnet)
  dnn_wpe, *_ = load_checkpoint(dnn_wpe, checkpoint=ckpt_file, strict=False)
  dnn_wpe.to(device)
  dnn_wpe.eval()
  # print('Neural WPE size = {:.2f}M'.format(dnn_wpe.size))
  # dnn_wpe.check_trainable_parameters()

  ## ------------------------------------------------- ##

  ## Load audio and apply DRC
  aud, fs = sf.read(sample_wav) # (t,), 16000
  aud, drc_gain = apply_drc(aud) # (t,), ()
  if not os.path.isfile(sample_wav_drc):
    sf.write(sample_wav_drc, data=aud, samplerate=fs)

  if chs == 'single':
    aud = aud[None] # (channels=1, samples)
    taps = taps1
  if chs == 'dual':
    aud2, fs2 = sf.read(sample_wav2, dtype='float32')
    aud2 = aud2 * drc_gain
    aud = np.stack((aud, aud2), axis=0) # (channels=2, samples)
    taps = taps2

  ## Perform dereverberation
  aud = torch.from_numpy(aud)[None] # (batch, channels, samples)
  with torch.no_grad():
    ## The input audio must be in shape (batch, channels, samples)
    enh = dnn_wpe(
      aud.to(device), delay=delay, taps=taps, dtype=dtype) # (t,)
  enh = to_arr(enh).squeeze() # convert to numpy array and squeeze
  ## Save
  if chs == 'dual':
    enh = enh[0] # only save the first channel
  # print(enh.sum())
  output_wav_path = 'data/nwpe_%s_taps%d.wav' % (chs, taps)
  sf.write(output_wav_path, data=enh, samplerate=fs)


def run_iterative_wpe(chs='single', n_iter=1, dtype=torch.float64):
  print('Running "Iterative-WPE-%s"...' % chs)

  ## IterativeWPE WPE
  iter_wpe = IterativeWPE(
    stft_opts=stft_opts_torch)

  ## Load audio and apply DRC
  aud, fs = sf.read(sample_wav) # (t,), 16000
  aud, drc_gain = apply_drc(aud) # (t,), ()
  if not os.path.isfile(sample_wav_drc):
    sf.write(sample_wav_drc, data=aud, samplerate=fs)

  if chs == 'single':
    aud = aud[None] # (channels=1, samples)
    taps = taps1
  if chs == 'dual':
    aud2, fs2 = sf.read(sample_wav2, dtype='float32')
    aud2 = aud2 * drc_gain
    aud = np.stack((aud, aud2), axis=0) # (channels=2, samples)
    taps = taps2

  ## Perform dereverberation
  aud = torch.from_numpy(aud)[None] # (batch, channels, samples)
  with torch.no_grad():
    ## The input audio must be in shape (batch, channels, samples)
    enh = iter_wpe(
      aud.to(device), delay=delay, taps=taps, dtype=dtype) # (t,)
  enh = to_arr(enh).squeeze() # convert to numpy array and squeeze
  ## Save
  if chs == 'dual':
    enh = enh[0] # only save the first channel
  output_wav_path = 'data/iwpe_%s_taps%d_iter%d.wav' % (chs, taps, n_iter)
  sf.write(output_wav_path, data=enh, samplerate=fs)



if __name__=="__main__":
  dtype = torch.float64

  ## Iterative WPE
  run_iterative_wpe('single', n_iter=1, dtype=dtype)
  run_iterative_wpe('dual', n_iter=1, dtype=dtype)

  ## Neural WPE
  run_neural_wpe('single', dtype=dtype)
  run_neural_wpe('dual', dtype=dtype)

  ## VACE-WPE
  run_vace_wpe(pretrain_opt='self', simp_opt='a')
  run_vace_wpe(pretrain_opt='late', simp_opt='a')
  run_vace_wpe(pretrain_opt='late', simp_opt='b')
