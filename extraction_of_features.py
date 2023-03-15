import time
from .feature_extraction import*

def extract_feature_class(root_path):
  print("------------------------------")
  start = time.time()
  min_feature = calculate_min(root_path)
  minutes, remaining_seconds = divmod(time.time() - start, 60)
  print("Calculating minimum feature took {} minutes and {} seconds".format(minutes, remaining_seconds))

  print("------------------------------")
  start = time.time()
  max_feature = calculate_max(root_path)
  minutes, remaining_seconds = divmod(time.time() - start, 60)
  print("Calculating maximum feature took {} minutes and {} seconds".format(minutes, remaining_seconds))

  print("------------------------------") 
  start = time.time()
  mean_feature = calculate_mean(root_path)
  minutes, remaining_seconds = divmod(time.time() - start, 60)
  print("Calculating mean feature took {} minutes and {} seconds".format(minutes, remaining_seconds))

  print("------------------------------")
  start = time.time()
  rms_feature = calculate_rms(root_path)
  minutes, remaining_seconds = divmod(time.time() - start, 60)
  print("Calculating rms feature took {} minutes and {} seconds".format(minutes, remaining_seconds))

  print("------------------------------")
  start = time.time()
  std_feature = calculate_std(root_path)
  minutes, remaining_seconds = divmod(time.time() - start, 60)
  print("Calculating std feature took {} minutes and {} seconds".format(minutes, remaining_seconds))

  print("------------------------------")
  start = time.time()
  variance_feature = calculate_variance(root_path)
  minutes, remaining_seconds = divmod(time.time() - start, 60)
  print("Calculating variance feature took {} minutes and {} seconds".format(minutes, remaining_seconds))

  print("------------------------------")
  start = time.time()
  skewness_feature = calculate_skweness(root_path)
  minutes, remaining_seconds = divmod(time.time() - start, 60)
  print("Calculating skewness feature took {} minutes and {} seconds".format(minutes, remaining_seconds))

  print("------------------------------")
  start = time.time()
  kurtosis_feature = calculate_kurtosis(root_path)
  minutes, remaining_seconds = divmod(time.time() - start, 60)
  print("Calculating kurtosis feature took {} minutes and {} seconds".format(minutes, remaining_seconds))

  print("------------------------------")
  start = time.time()
  delta_psd_feature = calculate_psd_delta(root_path)
  minutes, remaining_seconds = divmod(time.time() - start, 60)
  print("Calculating delta_psd feature took {} minutes and {} seconds".format(minutes, remaining_seconds))

  print("------------------------------")
  start = time.time()
  theta_psd_feature = calculate_psd_theta(root_path)
  minutes, remaining_seconds = divmod(time.time() - start, 60)
  print("Calculating theta_psd feature took {} minutes and {} seconds".format(minutes, remaining_seconds))

  print("------------------------------")
  start = time.time()
  alpha_psd_feature = calculate_psd_alpha(root_path)
  minutes, remaining_seconds = divmod(time.time() - start, 60)
  print("Calculating alpha_psd feature took {} minutes and {} seconds".format(minutes, remaining_seconds))

  print("------------------------------")
  start = time.time()
  beta_psd_feature = calculate_psd_beta(root_path)
  minutes, remaining_seconds = divmod(time.time() - start, 60)
  print("Calculating beta_psd feature took {} minutes and {} seconds".format(minutes, remaining_seconds))

  print("------------------------------")
  start = time.time()
  gamma_psd_feature = calculate_psd_gamma(root_path)
  minutes, remaining_seconds = divmod(time.time() - start, 60)
  print("Calculating gamma_delta_psd feature  took {} minutes and {} seconds".format(minutes, remaining_seconds))

  print("------------------------------")
  start = time.time()
  high_gamma_psd_feature = calculate_psd_high_gamma(root_path)
  minutes, remaining_seconds = divmod(time.time() - start, 60)
  print("Calculating high_gamma_psd feature  took {} minutes and {} seconds".format(minutes, remaining_seconds))

  print("------------------------------")
  start = time.time()
  wv_feature = calculate_wavelet_coefficients(root_path)
  minutes, remaining_seconds = divmod(time.time() - start, 60)
  print("Calculating WAVELET feature took {} minutes and {} seconds".format(minutes, remaining_seconds))

  print("------------------------------")
  start = time.time()
  sfe_feature = calculate_sfe(root_path)
  minutes, remaining_seconds = divmod(time.time() - start, 60)
  print("Calculating SFE feature took {} minutes and {} seconds".format(minutes, remaining_seconds))

  print("------------------------------")
  start = time.time()
  HOC_feature = calculate_hoc(root_path)
  minutes, remaining_seconds = divmod(time.time() - start, 60)
  print("Calculating HOC feature took {} minutes and {} seconds".format(minutes, remaining_seconds))

  print("------------------------------")
  start = time.time() 
  activity_feature, mobility_hjorth_feature, complexity_hjorth_feature = calculate_hjorth_parameters(root_path)
  minutes, remaining_seconds = divmod(time.time() - start, 60)
  print("Calculating hjorth feature took {} minutes and {} seconds".format(minutes, remaining_seconds))

  print("------------------------------")
  start = time.time() 
  pfd_feature = calculate_pfd(root_path)
  minutes, remaining_seconds = divmod(time.time() - start, 60)
  print("Calculating PFD feature took {} minutes and {} seconds".format(minutes, remaining_seconds))
  print("------------------------------")


  features = [min_feature,  max_feature, mean_feature, rms_feature, std_feature, variance_feature, skewness_feature, kurtosis_feature,
              activity_feature, mobility_hjorth_feature, complexity_hjorth_feature,
              wv_feature, sfe_feature, HOC_feature, pfd_feature,
              delta_psd_feature, theta_psd_feature, alpha_psd_feature, beta_psd_feature, gamma_psd_feature, high_gamma_psd_feature]

  return features