import numpy as np
import os
import pywt
import nolds
from scipy.signal import spectrogram
from scipy.signal import welch
from scipy.signal import hilbert

def calculate_min(root_path):
  fileExists = True
  min_list = []
  for pilot in range(1, 34):
    events_channels_array = np.zeros((63, 20))
    for event in range(1, 21):
      file_path = root_path + f"pilot{pilot}_evnt{event}_session2.npy"
      if os.path.isfile(file_path):
        fileExists = True
        signal = np.load(file_path)
        num_channels, num_samples = signal.shape
        for channel in range(num_channels):
          minimum = np.min(signal[channel])
          #rows = channels and columns = events
          events_channels_array[channel][event-1] = minimum 
      else:
        fileExists = False
    if(fileExists): 
      #for loop events finished, calculate mean value for each channel
      for channel in range(num_channels):
        mean_minimum_same_channel = np.mean(events_channels_array[channel][:])
        min_list.append(mean_minimum_same_channel)
  print(np.array(min_list).shape)
  return np.array(min_list)

def calculate_max(root_path):
  fileExists = True
  max_list = []
  for pilot in range(1, 34):
    events_channels_array = np.zeros((63, 20))
    for event in range(1, 21):
      file_path = root_path + f"pilot{pilot}_evnt{event}_session2.npy"
      if os.path.isfile(file_path):
        fileExists = True
        signal = np.load(file_path)
        num_channels, num_samples = signal.shape
        for channel in range(num_channels):
          maximum = np.max(signal[channel])
          #rows = channels and columns = events
          events_channels_array[channel][event-1] = maximum 
      else:
        fileExists = False
    if(fileExists): 
      #for loop events finished, calculate mean value for each channel
      for channel in range(num_channels):
        mean_minimum_same_channel = np.mean(events_channels_array[channel][:])
        max_list.append(mean_minimum_same_channel)
  print(np.array(max_list).shape)
  return np.array(max_list)

def calculate_mean(root_path):
  fileExists = True
  mean_list = []
  for pilot in range(1, 34):
    events_channels_array = np.zeros((63, 20))
    for event in range(1, 21):
      file_path = root_path + f"pilot{pilot}_evnt{event}_session2.npy"
      if os.path.isfile(file_path):
        fileExists = True
        signal = np.load(file_path)
        num_channels, num_samples = signal.shape
        for channel in range(num_channels):
          mean = np.mean(signal[channel])
          #rows = channels and columns = events
          events_channels_array[channel][event-1] = mean 
      else:
        fileExists = False
    if(fileExists): 
      #for loop events finished, calculate mean value for each channel
      for channel in range(num_channels):
        mean_same_channel = np.mean(events_channels_array[channel][:])
        mean_list.append(mean_same_channel)
  print(np.array(mean_list).shape)
  return np.array(mean_list)

def calculate_rms(root_path):
  fileExists = True
  rms_list = []
  for pilot in range(1, 34):
    events_channels_array = np.zeros((63, 20))
    for event in range(1, 21):
      file_path = root_path + f"pilot{pilot}_evnt{event}_session2.npy"
      if os.path.isfile(file_path):
        fileExists = True
        signal = np.load(file_path)
        num_channels, num_samples = signal.shape
        for channel in range(num_channels):
          rms = np.sqrt(np.mean(signal[channel]**2))
          #rows = channels and columns = events
          events_channels_array[channel][event-1] = rms 
      else:
        fileExists = False
    if(fileExists): 
      #for loop events finished, calculate mean value for each channel
      for channel in range(num_channels):
        mean_same_channel = np.mean(events_channels_array[channel][:])
        rms_list.append(mean_same_channel)
  print(np.array(rms_list).shape)
  return np.array(rms_list)

def calculate_std(root_path):
  fileExists = True
  std_list = []
  for pilot in range(1, 34):
    events_channels_array = np.zeros((63, 20))
    for event in range(1, 21):
      file_path = root_path + f"pilot{pilot}_evnt{event}_session2.npy"
      if os.path.isfile(file_path):
        fileExists = True
        signal = np.load(file_path)
        num_channels, num_samples = signal.shape
        for channel in range(num_channels):
          std = np.std(signal[channel])
          #rows = channels and columns = events
          events_channels_array[channel][event-1] = std 
      else:
        fileExists = False
    if(fileExists): 
      #for loop events finished, calculate mean value for each channel
      for channel in range(num_channels):
        mean_same_channel = np.mean(events_channels_array[channel][:])
        std_list.append(mean_same_channel)
  print(np.array(std_list).shape)
  return np.array(std_list)

def calculate_variance(root_path):
  fileExists = True
  variance_list = []
  for pilot in range(1, 34):
    events_channels_array = np.zeros((63, 20))
    for event in range(1, 21):
      file_path = root_path + f"pilot{pilot}_evnt{event}_session2.npy"
      if os.path.isfile(file_path):
        fileExists = True
        signal = np.load(file_path)
        num_channels, num_samples = signal.shape
        for channel in range(num_channels):
          variance = np.var(signal[channel])
          #rows = channels and columns = events
          events_channels_array[channel][event-1] = variance 
      else:
        fileExists = False
    if(fileExists): 
      #for loop events finished, calculate mean value for each channel
      for channel in range(num_channels):
        mean_same_channel = np.mean(events_channels_array[channel][:])
        variance_list.append(mean_same_channel)
  print(np.array(variance_list).shape)
  return np.array(variance_list)

def calculate_skweness(root_path):
  fileExists = True
  skewness_list = []
  for pilot in range(1, 34):
    events_channels_array = np.zeros((63, 20))
    for event in range(1, 21):
      file_path = root_path + f"pilot{pilot}_evnt{event}_session2.npy"
      if os.path.isfile(file_path):
        fileExists = True
        signal = np.load(file_path)
        num_channels, num_samples = signal.shape
        for channel in range(num_channels):
          mean = np.mean(signal[channel])
          skewness = np.mean((signal[channel] - mean) ** 3) / np.mean((signal[channel] - mean) ** 2) ** (3/2)
          #rows = channels and columns = events
          events_channels_array[channel][event-1] = skewness 
      else:
        fileExists = False
    if(fileExists): 
      #for loop events finished, calculate mean value for each channel
      for channel in range(num_channels):
        mean_same_channel = np.mean(events_channels_array[channel][:])
        skewness_list.append(mean_same_channel)
  print(np.array(skewness_list).shape)
  return np.array(skewness_list)

def calculate_kurtosis(root_path):
  fileExists = True
  kurtosis_list = []
  for pilot in range(1, 34):
    events_channels_array = np.zeros((63, 20))
    for event in range(1, 21):
      file_path = root_path + f"pilot{pilot}_evnt{event}_session2.npy"
      if os.path.isfile(file_path):
        fileExists = True
        signal = np.load(file_path)
        num_channels, num_samples = signal.shape
        for channel in range(num_channels):
          mean = np.mean(signal[channel])
          kurtosis = np.mean((signal[channel] - mean) ** 4) / np.mean((signal[channel] - mean) ** 2) ** 2
          #rows = channels and columns = events
          events_channels_array[channel][event-1] = kurtosis 
      else:
        fileExists = False
    if(fileExists): 
      #for loop events finished, calculate mean value for each channel
      for channel in range(num_channels):
        mean_same_channel = np.mean(events_channels_array[channel][:])
        kurtosis_list.append(mean_same_channel)
  print(np.array(kurtosis_list).shape)
  return np.array(kurtosis_list)


def calculate_psd_delta(root_path):
  fs = 256
  nperseg = 512
  fileExists = True
  delta_psd_list = []
  for pilot in range(1, 34):
    events_channels_array = np.zeros((63, 20))
    for event in range(1, 21):
      file_path = root_path + f"pilot{pilot}_evnt{event}_session2.npy"
      if os.path.isfile(file_path):
        fileExists = True
        signal = np.load(file_path)
        num_channels, num_samples = signal.shape
        for channel in range(num_channels):
          f, psd = welch(signal[channel], fs = fs, nperseg = nperseg)
          delta_indices = np.where((f >= 1) & (f <= 4))[0]
          delta_psd = np.mean(psd[delta_indices])
          #rows = channels and columns = events
          events_channels_array[channel][event-1] = delta_psd 
      else:
        fileExists = False
    if(fileExists): 
      #for loop events finished, calculate mean value for each channel
      for channel in range(num_channels):
        mean_same_channel = np.mean(events_channels_array[channel][:])
        delta_psd_list.append(mean_same_channel)
  print(np.array(delta_psd_list).shape)
  return np.array(delta_psd_list)

def calculate_psd_theta(root_path):
  fs = 256
  nperseg = 512
  fileExists = True
  theta_psd_list = []
  for pilot in range(1, 34):
    events_channels_array = np.zeros((63, 20))
    for event in range(1, 21):
      file_path = root_path + f"pilot{pilot}_evnt{event}_session2.npy"
      if os.path.isfile(file_path):
        fileExists = True
        signal = np.load(file_path)
        num_channels, num_samples = signal.shape
        for channel in range(num_channels):
          f, psd = welch(signal[channel], fs = fs, nperseg = nperseg)
          theta_indices = np.where((f >= 4) & (f <= 8))[0]
          theta_psd = np.mean(psd[theta_indices])
          #rows = channels and columns = events
          events_channels_array[channel][event-1] = theta_psd 
      else:
        fileExists = False
    if(fileExists): 
      #for loop events finished, calculate mean value for each channel
      for channel in range(num_channels):
        mean_same_channel = np.mean(events_channels_array[channel][:])
        theta_psd_list.append(mean_same_channel)
  print(np.array(theta_psd_list).shape)
  return np.array(theta_psd_list)

def calculate_psd_alpha(root_path):
  fs = 256
  nperseg = 512
  fileExists = True
  alpha_psd_list = []
  for pilot in range(1, 34):
    events_channels_array = np.zeros((63, 20))
    for event in range(1, 21):
      file_path = root_path + f"pilot{pilot}_evnt{event}_session2.npy"
      if os.path.isfile(file_path):
        fileExists = True
        signal = np.load(file_path)
        num_channels, num_samples = signal.shape
        for channel in range(num_channels):
          f, psd = welch(signal[channel], fs = fs, nperseg = nperseg)
          alpha_indices = np.where((f >= 8) & (f <= 13))[0]
          alpha_psd = np.mean(psd[alpha_indices])
          #rows = channels and columns = events
          events_channels_array[channel][event-1] = alpha_psd 
      else:
        fileExists = False
    if(fileExists): 
      #for loop events finished, calculate mean value for each channel
      for channel in range(num_channels):
        mean_same_channel = np.mean(events_channels_array[channel][:])
        alpha_psd_list.append(mean_same_channel)
  print(np.array(alpha_psd_list).shape)
  return np.array(alpha_psd_list)

def calculate_psd_beta(root_path):
  fs = 256
  nperseg = 256
  fileExists = True
  beta_psd_list = []
  for pilot in range(1, 34):
    events_channels_array = np.zeros((63, 20))
    for event in range(1, 21):
      file_path = root_path + f"pilot{pilot}_evnt{event}_session2.npy"
      if os.path.isfile(file_path):
        fileExists = True
        signal = np.load(file_path)
        num_channels, num_samples = signal.shape
        for channel in range(num_channels):
          f, psd = welch(signal[channel], fs = fs, nperseg = nperseg)
          beta_indices = np.where((f >= 13) & (f <= 30))[0]
          beta_psd = np.mean(psd[beta_indices])
          #rows = channels and columns = events
          events_channels_array[channel][event-1] = beta_psd 
      else:
        fileExists = False
    if(fileExists): 
      #for loop events finished, calculate mean value for each channel
      for channel in range(num_channels):
        mean_same_channel = np.mean(events_channels_array[channel][:])
        beta_psd_list.append(mean_same_channel)
  print(np.array(beta_psd_list).shape)
  return np.array(beta_psd_list)

def calculate_psd_gamma(root_path):
  fs = 256
  nperseg = 256
  fileExists = True
  gamma_psd_list = []
  for pilot in range(1, 34):
    events_channels_array = np.zeros((63, 20))
    for event in range(1, 21):
      file_path = root_path + f"pilot{pilot}_evnt{event}_session2.npy"
      if os.path.isfile(file_path):
        fileExists = True
        signal = np.load(file_path)
        num_channels, num_samples = signal.shape
        for channel in range(num_channels):
          f, psd = welch(signal[channel], fs = fs, nperseg = nperseg)
          gamma_indices = np.where((f >= 30) & (f <= 80))[0]
          gamma_psd = np.mean(psd[gamma_indices])
          #rows = channels and columns = events
          events_channels_array[channel][event-1] = gamma_psd 
      else:
        fileExists = False
    if(fileExists): 
      #for loop events finished, calculate mean value for each channel
      for channel in range(num_channels):
        mean_same_channel = np.mean(events_channels_array[channel][:])
        gamma_psd_list.append(mean_same_channel)
  print(np.array(gamma_psd_list).shape)
  return np.array(gamma_psd_list)

def calculate_psd_high_gamma(root_path):
  fs = 256
  nperseg = 256
  fileExists = True
  high_gamma_psd_list = []
  for pilot in range(1, 34):
    events_channels_array = np.zeros((63, 20))
    for event in range(1, 21):
      file_path = root_path + f"pilot{pilot}_evnt{event}_session2.npy"
      if os.path.isfile(file_path):
        fileExists = True
        signal = np.load(file_path)
        num_channels, num_samples = signal.shape
        for channel in range(num_channels):
          f, psd = welch(signal[channel], fs = fs, nperseg = nperseg)
          high_gamma_indices = np.where((f >= 80) & (f <= 150))[0]
          high_gamma_psd = np.mean(psd[high_gamma_indices])
          #rows = channels and columns = events
          events_channels_array[channel][event-1] = high_gamma_psd 
      else:
        fileExists = False
    if(fileExists): 
      #for loop events finished, calculate mean value for each channel
      for channel in range(num_channels):
        mean_same_channel = np.mean(events_channels_array[channel][:])
        high_gamma_psd_list.append(mean_same_channel)
  print(np.array(high_gamma_psd_list).shape)
  return np.array(high_gamma_psd_list)


def calculate_wavelet_coefficients(root_path):
  fileExists = True
  wavelet = 'db4'
  wv_list = []
  for pilot in range(1, 34):
    events_channels_array = np.zeros((63, 20))
    for event in range(1, 21):
      file_path = root_path + f"pilot{pilot}_evnt{event}_session2.npy"
      if os.path.isfile(file_path):
        fileExists = True
        signal = np.load(file_path)
        num_channels, num_samples = signal.shape
        for channel in range(num_channels):
          wv_coeffs, _ = pywt.coeffs_to_array(pywt.wavedec(signal[channel], wavelet))
          #rows = channels and columns = events
          events_channels_array[channel][event-1] = np.mean(wv_coeffs) 
      else:
        fileExists = False
    if(fileExists): 
      #for loop events finished, calculate mean value for each channel
      for channel in range(num_channels):
        mean_same_channel = np.mean(events_channels_array[channel][:])
        wv_list.append(mean_same_channel)
  print(np.array(wv_list).shape)
  return np.array(wv_list)

def calculate_hjorth_parameters(root_path):
  fileExists = True
  activity_list = []
  mobility_list = []
  complexity_list = []
  for pilot in range(1, 34):
    events_channels_array = np.zeros((63, 20, 3))
    for event in range(1, 21):
      file_path = root_path + f"pilot{pilot}_evnt{event}_session2.npy"
      if os.path.isfile(file_path):
        fileExists = True
        signal = np.load(file_path)
        num_channels, num_samples = signal.shape
        for channel in range(num_channels):
          diff1 = np.diff(signal[channel], axis=0)
          diff2 = np.diff(diff1, axis=0)
          var_zero = np.var(signal[channel], axis=0)
          var_d1 = np.var(diff1, axis=0)
          var_d2 = np.var(diff2, axis=0)
          activity = var_zero
          mobility = np.sqrt(var_d1 / var_zero)
          complexity = np.sqrt(var_d2 / var_d1) / mobility
          #rows = channels and columns = events
          events_channels_array[channel][event-1][0] = activity
          events_channels_array[channel][event-1][1] = mobility
          events_channels_array[channel][event-1][2] = complexity
      else:
        fileExists = False
    if(fileExists): 
      #for loop events finished, calculate mean value for each channel
      for channel in range(num_channels):
        mean_same_channel_activity = np.mean(events_channels_array[channel][:][0])
        mean_same_channel_mobility = np.mean(events_channels_array[channel][:][1])
        mean_same_channel_complexity = np.mean(events_channels_array[channel][:][2])
        activity_list.append(mean_same_channel_activity)
        mobility_list.append(mean_same_channel_mobility)
        complexity_list.append(mean_same_channel_complexity)
  print(np.array(activity_list).shape, np.array(mobility_list).shape, np.array(complexity_list).shape)

  return np.array(activity_list), np.array(mobility_list), np.array(complexity_list)

def calculate_sfe(root_path):
  fileExists = True
  fs = 256
  sfe_list = []
  for pilot in range(1, 34):
    events_channels_array = np.zeros((63, 20))
    for event in range(1, 21):
      file_path = root_path + f"pilot{pilot}_evnt{event}_session2.npy"
      if os.path.isfile(file_path):
        fileExists = True
        signal = np.load(file_path)
        num_channels, num_samples = signal.shape
        for channel in range(num_channels):
          f, t, spec = spectrogram(signal[channel], fs=fs)
          cdf = np.cumsum(spec, axis=0) / np.sum(spec)
          idx = np.argmin(np.abs(cdf - 0.95), axis=0)
          sfe_value = f[idx]
          #rows = channels and columns = events
          events_channels_array[channel][event-1] = sfe_value 
      else:
        fileExists = False
    if(fileExists): 
      #for loop events finished, calculate mean value for each channel
      for channel in range(num_channels):
        mean_same_channel = np.mean(events_channels_array[channel][:])
        sfe_list.append(mean_same_channel)
  print(np.array(sfe_list).shape)
  return np.array(sfe_list)


def calculate_hurst(root_path):
  fileExists = True
  hurst_list = []
  for pilot in range(1, 34):
    events_channels_array = np.zeros((63, 20))
    for event in range(1, 21):
      file_path = root_path + f"pilot{pilot}_evnt{event}_session2.npy"
      if os.path.isfile(file_path):
        fileExists = True
        signal = np.load(file_path)
        num_channels, num_samples = signal.shape
        for channel in range(num_channels):
          hurst_value = nolds.hurst_rs(signal[channel])
          #rows = channels and columns = events
          events_channels_array[channel][event-1] = np.mean(hurst_value)
      else:
        fileExists = False
    if(fileExists): 
      #for loop events finished, calculate mean value for each channel
      for channel in range(num_channels):
        mean_same_channel = np.mean(events_channels_array[channel][:])
        hurst_list.append(mean_same_channel)
  print(np.array(hurst_list).shape)
  return np.array(hurst_list)

def calculate_hoc(root_path):
  fileExists = True
  hoc_list = []
  for pilot in range(1, 34):
    events_channels_array = np.zeros((63, 20))
    for event in range(1, 21):
      file_path = root_path + f"pilot{pilot}_evnt{event}_session2.npy"
      if os.path.isfile(file_path):
        fileExists = True
        signal = np.load(file_path)
        num_channels, num_samples = signal.shape
        for channel in range(num_channels):
          analytic_signal = hilbert(signal[channel])
          phase = np.unwrap(np.angle(analytic_signal))
          frequency = np.diff(phase)/(2*np.pi)
          hoc_value = np.sum(np.abs(np.diff(np.sign(np.diff(frequency)))))/len(frequency)
          #rows = channels and columns = events
          events_channels_array[channel][event-1] = np.mean(hoc_value)
      else:
        fileExists = False
    if(fileExists): 
      #for loop events finished, calculate mean value for each channel
      for channel in range(num_channels):
        mean_same_channel = np.mean(events_channels_array[channel][:])
        hoc_list.append(mean_same_channel)
  print(np.array(hoc_list).shape)
  return np.array(hoc_list)

def pfd_single_channel(x):
    """Calculate Petrosian Fractal Dimension (PFD) of a single channel signal"""
    # compute first-order difference
    dx = np.diff(x)
    # count number of sign changes in the first-order difference
    N_delta = np.sum(dx[1:] * dx[:-1] < 0)
    # count number of zero crossings in the signal
    N = len(x)
    x_avg = np.mean(x)
    dx = x - x_avg
    N_zero = 1
    for i in range(1, N):
        if dx[i] == 0:
            if dx[i-1] != 0:
                N_zero += 1
        elif dx[i-1] * dx[i] < 0:
            N_zero += 1
    # calculate PFD
    return np.log10(N) / (np.log10(N) + np.log10(N / N + 0.4 * N_zero + 0.6 * N_delta))

def calculate_pfd(root_path):
  fileExists = True
  pfd_list = []
  for pilot in range(1, 34):
    events_channels_array = np.zeros((63, 20))
    for event in range(1, 21):
      file_path = root_path + f"pilot{pilot}_evnt{event}_session2.npy"
      if os.path.isfile(file_path):
        fileExists = True
        signal = np.load(file_path)
        num_channels, num_samples = signal.shape
        for channel in range(num_channels):
          pfd_value = pfd_single_channel(signal[channel])
          #rows = channels and columns = events
          events_channels_array[channel][event-1] = pfd_value
      else:
        fileExists = False
    if(fileExists): 
      #for loop events finished, calculate mean value for each channel
      for channel in range(num_channels):
        mean_same_channel = np.mean(events_channels_array[channel][:])
        pfd_list.append(mean_same_channel)
  print(np.array(pfd_list).shape)
  return np.array(pfd_list)