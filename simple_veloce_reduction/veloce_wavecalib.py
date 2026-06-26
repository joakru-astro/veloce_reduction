from astropy.io import fits
from astropy.constants import c
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import pickle

from scipy import signal
# from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.linalg import lstsq
from scipy.ndimage import median_filter, minimum_filter, gaussian_filter1d
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks
# from csaps import csaps

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, RANSACRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils import check_array

import warnings
from scipy.linalg import LinAlgWarning

from . import veloce_reduction_tools, veloce_diagnostic

arm_nums = {'red': 3, 'green': 2, 'blue': 1}
REPETITION_RATE = 25e9  # Hz
OFFSET_FREQUENCY = 9.56e9  # Hz
SPEED_OF_LIGHT = c.value  # m/s

def pad_array(array, ref_pixel):
    """
    Pad an array to create 2D array with the size matching min and max of reference pixels.

    Parameters:
    - array (list of numpy.ndarray): The list of 1D arrays to be padded.
    - ref_pixel (list of numpy.ndarray): The list of reference pixel arrays.

    Returns:
    - padded_array (numpy.ndarray): The 2D array after padding.
    """
    lower_bound = min([np.nanmin(order) for order in ref_pixel])
    upper_bound = max([np.nanmax(order) for order in ref_pixel])
    # print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
    padded_array = np.array(
        [np.pad(order, (int(np.nanmin(ref_pixel[i])-lower_bound), int(upper_bound-np.nanmax(ref_pixel[i]))), constant_values=np.nan)
         for i, order in enumerate(array)])

    return padded_array

def load_LC_wave_reference(veloce_paths, arm, wave_calib_file=None):
    """
    Load the wavelength calibration for the LC.

    Parameters:
    - veloce_paths (object): An object containing the paths to the data directories.
    - arm (str): The arm of the spectrograph ('red', 'green', or 'blue').

    Returns:
    - ref_orders (numpy.ndarray): The array of reference orders.
    - ref_wave (list of numpy.ndarray): The list of reference wavelengths.
    - ref_intensity (list of numpy.ndarray): The list of reference intensities.
    - ref_pixel (list of numpy.ndarray): The list of reference pixels.
    """
    if wave_calib_file is None:
        lc_wave_calib_file = os.path.join(veloce_paths.wave_dir, f'{arm.upper()}_LC_SPEC-26aug{arm_nums[arm]}0083.txt')
    else:
        lc_wave_calib_file = wave_calib_file

    if not os.path.exists(lc_wave_calib_file):
        raise FileNotFoundError(f"LC wave calibration file not found: {lc_wave_calib_file}")
    
    dtype=[('wave', float), ('flux', float), ('pixel', float), ('order', float)]
    lc_wave_calib = np.loadtxt(lc_wave_calib_file, dtype=dtype)
    # remove NaN values
    lc_wave_calib = np.array([v for v in lc_wave_calib if v == v], dtype=dtype)
    dtype=[('wave', float), ('flux', float), ('pixel', float), ('order', int)]
    lc_wave_calib = np.array([(v['wave'], v['flux'], v['pixel'], int(v['order'])) for v in lc_wave_calib], dtype=dtype)

    ref_orders = np.unique(lc_wave_calib['order'])
    ref_wave = [lc_wave_calib[lc_wave_calib['order'] == order]['wave'] for order in ref_orders]
    ref_intensity = [lc_wave_calib[lc_wave_calib['order'] == order]['flux'] for order in ref_orders]
    ref_pixel = [lc_wave_calib[lc_wave_calib['order'] == order]['pixel'] for order in ref_orders]
    ref_pixel = pad_array(ref_pixel, ref_pixel)
    ref_wave = pad_array(ref_wave, ref_pixel)
    ref_intensity = pad_array(ref_intensity, ref_pixel)

    return ref_orders, ref_wave, ref_intensity, ref_pixel

def load_simultanous_LC(image, veloce_paths, hdr, arm, traces=None, ref_orders=None, ref_pixel=None):
    """
    Load simultaneous laser comb observations.

    Parameters:
    - image (numpy.ndarray): Image (2D array) containing the simultaneous laser comb data.
    - veloce_paths (object): An object containing the paths to the data directories.
    - hdr (astropy.io.fits.Header): The header.
    - arm (str): The arm of the spectrograph ('red', 'green', or 'blue').
    - traces (object, optional): An object containing traces for LC.
    If None, default traces for the arm will be loaded.
    - ref_orders (numpy.ndarray, optional): The array of reference orders.
    - ref_pixel (list of numpy.ndarray, optional): The list of reference pixel arrays.

    Returns:
    - extracted_LC (list of numpy.ndarray): The list of extracted laser comb orders.
    - extracted_pixel (list of numpy.ndarray): The list of extracted pixel arrays.
    - order_slice (slice): The slice for selecting the relevant orders.
    - pixel_slices (numpy.ndarray): The array of slices for selecting the relevant pixels.
    """
    if hdr is not None and (hdr['FREQREF'] != REPETITION_RATE and hdr['FOFFFREQ'] != OFFSET_FREQUENCY):
        raise ValueError("Repetition rate and offset frequency do not match the values of LC solution.")
    if traces is None:
        traces = veloce_reduction_tools.Traces.load_traces(os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_LC_trace.pkl'))
    
    extracted_LC, extracted_LC_uncertainty, extracted_LC_imgs = veloce_reduction_tools.extract_orders_with_trace(image, traces)

    # extracted_pixel = list(range(len(extracted_LC)))
    if ref_orders is not None:
        if len(ref_orders) != len(extracted_LC):
            print("[Warning]: Reference LC and extracted LC do not have the same number of orders.")
            order_slice = slice(int(np.nanmin(ref_orders)-1), int(np.nanmax(ref_orders)))
            extracted_LC = extracted_LC[order_slice]
        else:
            order_slice = slice(None, None, None)
        # for i, order in enumerate(ref_orders):
        #     pixel_slice = slice(int(min(lc_ref[lc_ref['order']==order]['pixel'])-1), int(max(lc_ref[lc_ref['order']==order]['pixel'])))
        #     extracted_LC[i] = extracted_LC[i][pixel_slice]
        #     extracted_pixel[i] = lc_ref[lc_ref['order']==order]['pixel']
        pixel_slices = np.array([slice(int(np.nanmin(ref_pixel[order-1])-1), int(np.nanmax(ref_pixel[order-1]))) for order in ref_orders])
        extracted_LC = [extracted_LC[i][pixel_slices[i]] for i in range(len(extracted_LC))]
        extracted_pixel = [pixel[~np.isnan(pixel)] for pixel in ref_pixel]
        extracted_LC = pad_array(extracted_LC, ref_pixel)
        extracted_pixel = pad_array(extracted_pixel, ref_pixel)
    else:
        order_slice = slice(None, None, None)
        for i in range(len(extracted_LC)):
            extracted_pixel[i] = np.arange(len(extracted_LC[i]), dtype=int)+1
            
            pixel_slices = np.array([slice(0, len(extracted_LC[i])) for i in range(len(extracted_LC))])

    return extracted_LC, extracted_pixel, order_slice, pixel_slices

# def get_lc_order(data, order):
#     order_data = data[data['order'] == order]
#     wave = order_data['wave']
#     intensity = order_data['flux']
#     pixel = order_data['pixel']
#     return wave, intensity, pixel

# def calculate_lc_ccf(pixel, intensity, pixel_ref, intensity_ref):
#     if len(pixel) != len(pixel_ref) or len(intensity) != len(intensity_ref):
#         raise ValueError("Data array length does not match the refereance array length.")
#     pix_shift = np.arange(-len(pixel_ref)+1, len(pixel_ref), 1)
#     ccf = np.correlate(intensity, intensity_ref, mode='full')
#     return pix_shift, np.array(ccf)

def general_gaussian(x, A, mu, sigma, beta, baseline):
    """
    Generalized Gaussian function.
    
    Parameters:
    - x (numpy.ndarray): The independent variable (e.g., pixels).
    - A (float): Amplitude of the peak.
    - mu (float): Mean of the peak.
    - sigma (float): Standard deviation of the peak.
    - beta (float): Shape parameter of the generalized Gaussian.
    - baseline (float): Baseline level.

    Returns:
    - (numpy.ndarray): The values of the generalized Gaussian function at each point in x.
    """
    return A * np.exp(-np.abs(((x - mu)/(np.sqrt(2)*sigma)))**beta) + baseline

def gaussian(x, A, mu, sigma, baseline):
    """
    Gaussian function.
    
    Parameters:
    - x (numpy.ndarray): The independent variable (e.g., pixels).
    - A (float): Amplitude of the peak.
    - mu (float): Mean of the peak.
    - sigma (float): Standard deviation of the peak.
    - baseline (float): Baseline level.

    Returns:
    - (numpy.ndarray): The values of the generalized Gaussian function at each point in x.
    """
    return A * np.exp(-np.abs(((x - mu)/(np.sqrt(2)*sigma)))**2) + baseline

def fit_lc_peak(pix_shift, ccf, fitting_limit=None):
    """
    Fit the peak of the cross-correlation function (CCF) to determine the pixel shift.
    
    Parameters:
    - pix_shift (numpy.ndarray): The pixel axis for the CCF.
    - ccf (numpy.ndarray): The cross-correlation function values.
    - fitting_limit (float, optional): How far to look for the peak. If None, it will be calculated automatically.

    Returns:
    - (float): The fitted pixel shift corresponding to the peak of the CCF.
    - (list): The fitted parameters of the generalized Gaussian function (A, mu, sigma, beta, baseline).
    - (float): Fitting limit.
    """
    ccf_mask = np.isfinite(ccf)
    # if len(pix_shift) == 0 or len(ccf) == 0:
    if np.sum(ccf_mask) < 10:
        return np.nan, [np.nan], np.nan #slice(0,None)
    else:
        pix_shift = pix_shift[ccf_mask]
        ccf = ccf[ccf_mask]
    
    # consider peak near 0 pixel shift
    peaks, _ = find_peaks(ccf)
    if fitting_limit is None:
        fitting_limit = np.ceil(np.mean(np.diff(peaks)))/2+1
        # print(f"[Info] Fitting limit for LC peak fitting set to {fitting_limit:.2f} pixel.")
    # center_peak_shift = np.min(abs(pix_shift[peaks]))
    # center_peak_idx = np.argmin(abs(pix_shift - center_peak_shift))
    ### use highest peak instead of closest to zero
    center_peak_idx = peaks[np.argmax(ccf[peaks])]
    center_peak_shift = pix_shift[center_peak_idx]
    # print(f"[Info] Closest peak to origin at {center_peak_shift} pixel shift (index {center_peak_idx}).")
    if ccf[center_peak_idx] != np.max(ccf):
        print(f"[Warning] Closest peak to origin (at {center_peak_shift}) is not the highest peak (at {pix_shift[np.argmax(ccf)]}).")
    # fitting_slice = slice(max(0, int(center_peak_idx-fitting_limit+0.5)), min(len(ccf)-1, int(center_peak_idx+fitting_limit+1.5)))
    fitting_slice = slice(max(0, int(center_peak_idx-fitting_limit)), min(len(ccf)-1, int(center_peak_idx+fitting_limit+1)))
    _pix_shift = pix_shift[fitting_slice]
    # _pix_shift = pix_shift[abs(pix_shift) <= fitting_limit]
    _ccf = ccf[fitting_slice]
    # _ccf = ccf[abs(pix_shift) <= fitting_limit]
    # _ccf -= np.min(_ccf)

    # fit a generalised gaussian to the peak
    peak_arg = np.argmax(_ccf)
    peak = _ccf[peak_arg]
    peak_position = _pix_shift[peak_arg]
    sigma = 0.8
    beta = 2.0
    baseline = np.min(_ccf)
    try:
        popt, _ = curve_fit(general_gaussian, _pix_shift, _ccf,
                        p0=[peak, peak_position, sigma, beta, baseline],
                        bounds=([0, np.min(_pix_shift), 1e-3, 1e-3, 0], [2*peak, np.max(_pix_shift), 10, 10, peak]),)
        return popt[1], popt, fitting_limit #fitting_slice
    except Exception as e:
        print(f"[Warning] CCF peak fitting failed: {e}")
        return np.nan, [np.nan], np.nan #slice(0,None)

def calculate_offset_map(ref_orders, ref_intensity, ref_pixel, lc_intensity, lc_pixel, number_of_parts=8, mode='LC', plot=False, veloce_paths=None, filename=None):
    """
    Calculate the cross-correlation function (CCF) for each order of the laser comb.

    Parameters:
    - ref_orders (numpy.ndarray): The array of reference orders.
    - ref_intensity (list of numpy.ndarray): The list of reference intensities.
    - ref_pixel (list of numpy.ndarray): The list of reference pixels.
    - lc_intensity (list of numpy.ndarray): The list of observed laser comb intensities.
    - lc_pixel (list of numpy.ndarray): The list of observed laser comb pixels.
    - number_of_parts (int, optional): The number of parts to split the data into for CCF calculation. Default is 8.
    - mode (str, optional): The mode for fitting the peak. Default is 'LC'.
    - plot (bool, optional): Whether to plot the results. Default is False.
    - veloce_paths (list of str, optional): The paths for the Veloce data. Default is None.
    - filename (str, optional): The filename for the plot. Default is None.
    """
    CCF = [
        [
            np.correlate(_intensity, _intensity_ref, mode='full')
            if np.isnan(_intensity).any() == False and np.isnan(_intensity_ref).any() == False
            else np.array([np.nan])
            for _intensity, _intensity_ref in zip(
                np.array_split(intensity, number_of_parts),
                np.array_split(intensity_ref, number_of_parts)
            )
        ]
        for intensity_ref, intensity in zip(ref_intensity, lc_intensity)
    ]
    pixel_shifts =  [
        [
            np.arange(-len(_pixel_ref) + 1, len(_pixel_ref), 1)
            for _pixel_ref in np.array_split(pixel_ref, number_of_parts)
        ]
        for pixel_ref in ref_pixel
    ]
    dispersion_position = np.array([
        [
            np.mean(_pixel)
            for _pixel in np.array_split(pixel, number_of_parts)
        ]
        for pixel in lc_pixel
    ])

    orders_position = np.repeat(np.array(ref_orders).reshape(len(ref_orders), 1), dispersion_position.shape[1], axis=1)
    if mode == 'LC':
        offset_array = np.array([[fit_lc_peak(pixel_shifts[i][j], CCF[i][j])[0] for j in range(number_of_parts)] for i in range(len(ref_orders))])
    elif mode == 'Th':
        offset_array = np.array([[fit_lc_peak(pixel_shifts[i][j], CCF[i][j], fitting_limit=30)[0] for j in range(number_of_parts)] for i in range(len(ref_orders))])
    else:
        raise ValueError("Mode must be 'LC' or 'Th'.")
    
    if plot:
        veloce_diagnostic.plot_ccf(pixel_shifts, CCF, 15, 4, fit_lc_peak, general_gaussian,
                                   veloce_paths=veloce_paths, filename=filename)
        veloce_diagnostic.plot_offset_map(dispersion_position, orders_position, offset_array,
                                          veloce_paths=veloce_paths, filename=filename)

    return dispersion_position, orders_position, offset_array

# def offset_map(orders, CCF, PIX, dispersion_position, plot=False):
#     dispersion_position = np.array(dispersion_position)
#     orders_position = np.repeat(np.array(orders).reshape(len(orders), 1), dispersion_position.shape[1], axis=1)
#     offset_array = np.zeros((len(orders), len(PIX[0])))
#     for i in range(len(orders)):
#         for j in range(len(CCF[i])):
#             offset, popt = fit_lc_peak(PIX[i][j], CCF[i][j])
#             offset_array[i][j] = offset
#             if offset > 2:
#                 print(f"Order {orders[i]}, Chunk {j}, Offset: {offset:.2f}")

#     if plot:
#         fig = plt.figure(figsize=(12, 8))
#         ax = fig.add_subplot(111, projection='3d')
#         points = ax.scatter(dispersion_position.flatten(), orders_position.flatten(), offset_array.flatten(), c=offset_array.flatten(), cmap='viridis', marker='o')
#         ax.set_title('Offset Map')
#         ax.set_xlabel('Dispersion Position')
#         ax.set_ylabel('Orders')
#         ax.set_zlabel('Offset')
#         fig.colorbar(points, shrink=0.5, aspect=10)
#     return dispersion_position, orders_position, offset_array

def fit_surface(dispersion_position, orders_position, offset_array, extracted_pixels, degree=1, plot=False, veloce_paths=None, filename=None):
    """
    Fit a surface to the offset map using least squares.

    Fitting procedure inspired by: https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6

    Parameters:
    - dispersion_position (numpy.ndarray): The array of dispersion positions.
    - orders_position (numpy.ndarray): The array of order positions.
    - offset_array (numpy.ndarray): The array of offsets.
    - extracted_pixels (numpy.ndarray): The array of extracted pixel positions for each order to calculate the surface.
    - degree (int, optional): The degree of the polynomial surface to fit (1 for linear, 2 for quadratic, 3 for cubic). Default is 1.
    - plot (bool, optional): Whether to plot the fitted surface and residuals. Default is False.
    - veloce_paths (list of str, optional): The paths for the Veloce data. Default is None.
    - filename (str, optional): The filename for the plot. Default is None.

    Returns:
    - Z (numpy.ndarray): The fitted surface values at the extracted pixel positions.
    - C (numpy.ndarray): The coefficients of the fitted surface.
    - data (numpy.ndarray): The filtered data points used for fitting.
    - residuals (numpy.ndarray): The residuals of the fitted surface.
    """
    
    data = np.array([(x, y, z) for x, y, z in zip(dispersion_position.flatten(), orders_position.flatten(), offset_array.flatten())])
    
    grid_points = [[(pixel,i+1) for pixel in extracted_pixels[i]] for i in range(len(extracted_pixels))]
    grid_points = np.vstack(grid_points)

    # Perform iterative sigma clipping around the fitted surface
    max_iterations = 10
    sigma_threshold = 3.0
    mask = np.ones(data.shape[0], dtype=bool)
    mask *= ~np.isnan(data[:, 2]) 
    for _ in range(max_iterations):
        if degree == 1:
            A = np.c_[data[mask, 0], data[mask, 1], np.ones(data[mask].shape[0])]
        elif degree == 2:
            A = np.c_[data[mask, 0]**2, data[mask, 1]**2, data[mask, 0]*data[mask, 1], data[mask, 0], data[mask, 1], np.ones(data[mask].shape[0])]
        elif degree == 3:
            A = np.c_[data[mask, 0]**3, data[mask, 1]**3, data[mask, 0]**2 * data[mask, 1], data[mask, 0] * data[mask, 1]**2,
                      data[mask, 0]**2, data[mask, 1]**2, data[mask, 0]*data[mask, 1], data[mask, 0], data[mask, 1], np.ones(data[mask].shape[0])]
        else:
            raise ValueError("Only polynomial orders 1, 2, or 3 are supported.")
        
        C, _, _, _ = lstsq(A, data[mask, 2])

        if degree == 1:
            residuals = data[:, 2] - (C[0] * data[:, 0] + C[1] * data[:, 1] + C[2])
        elif degree == 2:
            residuals = data[:, 2] - (C[0] * data[:, 0]**2 + C[1] * data[:, 1]**2 + C[2] * data[:, 0] * data[:, 1] +
             C[3] * data[:, 0] + C[4] * data[:, 1] + C[5])
        elif degree == 3:
            residuals = data[:, 2] - (C[0] * data[:, 0]**3 + C[1] * data[:, 1]**3 + C[2] * data[:, 0]**2 * data[:, 1] +
             C[3] * data[:, 0] * data[:, 1]**2 + C[4] * data[:, 0]**2 + C[5] * data[:, 1]**2 +
             C[6] * data[:, 0] * data[:, 1] + C[7] * data[:, 0] + C[8] * data[:, 1] + C[9])
            
        std_dev = np.std(residuals[mask])
        new_mask = np.abs(residuals) < sigma_threshold * std_dev
        if np.array_equal(mask, new_mask):
            break

        mask = new_mask

    # Evaluate for each pixel
    if degree == 1:
        z = [C[0]*x + C[1]*y + C[2] for x, y in grid_points]
    elif degree == 2:
        z = [C[0]*x**2 + C[1]*y**2 + C[2]*x*y + C[3]*x + C[4]*y + C[5] for x, y in grid_points]
    elif degree == 3:
        z = [C[0]*x**3 + C[1]*y**3 + C[2]*x**2*y + C[3]*x*y**2 + C[4]*x**2 + C[5]*y**2 +
             C[6]*x*y + C[7]*x + C[8]*y + C[9] for x, y in grid_points]

    Z = np.array(z).reshape(extracted_pixels.shape)

    if plot:
        veloce_diagnostic.plot_surface(np.unique(orders_position), extracted_pixels, Z, data[mask],
                                       veloce_paths=veloce_paths, filename=filename)

    return Z, C, data[mask], residuals[mask]

def interpolate_offsets_optimised(extracted_pixels, offsets, ref_wave, ref_pixel):
    """
    Interpolate wavelenght using pixel offsets.

    Parameters:
    - extracted_pixels (numpy.ndarray): The array of extracted pixel positions for each order.
    - offsets (numpy.ndarray): The array of pixel offsets.
    - ref_wave (numpy.ndarray): The array of reference wavelengths.
    - ref_pixel (numpy.ndarray): The array of reference pixel positions.

    Returns:
    - new_wave (numpy.ndarray): The array of new wavelengths corresponding to the extracted pixel positions after applying the offsets.
    """
    # offset pixels
    new_pixels = extracted_pixels - offsets
    # find wavelengths of the observation
    new_wave = np.array([np.interp(new_pix, ref_pix, ref_w) for new_pix, ref_pix, ref_w in zip(new_pixels, ref_pixel, ref_wave)])
    
    return new_wave

def estimate_calibration_precision(residuals, order, ref_wave):
    """
    Estimate the calibration precision.

    Parameters:
    - residuals (numpy.ndarray): The array of residuals from the surface fitting.
    - order (int): The order for which to estimate the calibration precision.
    - ref_wave (numpy.ndarray): The array of reference wavelengths.

    Returns:
    - calibration_precision (float): The estimated calibration precision in m/s.
    """
    # Calculate the standard deviation of the residuals
    n_points = len(residuals)
    std_dev = np.std(residuals)
    average_step = np.nanmean(np.diff(ref_wave[order-1]))
    average_wave = np.nanmean(ref_wave[order-1])
    
    # Calculate the calibration precision
    calibration_precision = std_dev / np.sqrt(n_points) * average_step / average_wave * SPEED_OF_LIGHT
    # calibration_precision = std_dev * average_step / average_wave * SPEED_OF_LIGHT

    print(f"Calibration Precision estimated at {average_wave:.0f}nm: {calibration_precision:.0f} m/s")
    
    return calibration_precision

def apply_wavelength_shift(wave, arm, veloce_paths):
    """
    Apply the wavelength shift to the spectrum based on the predetermined velocity offsets for each arm.
    Parameters:
    - wave (numpy.ndarray): The array of wavelengths to be shifted.
    - arm (str): The arm of the spectrograph ('red', 'green', or 'blue').
    - veloce_paths (object): An object containing the paths to the data directories.
    
    Returns:
    - wave (numpy.ndarray): The array of wavelengths after applying the shift.
    """
    # Apply the wavelength shift to the spectrum
    shifts = np.load(os.path.join(veloce_paths.wave_dir, f'{arm}_velocity_orders_offsets.npy'))
    if len(wave) != len(shifts):
        raise ValueError(f"Number of orders in wave ({len(wave)}) does not match number of predetermined offsets ({len(shifts)})")
    for i, v in enumerate(shifts):
        # Calculate the convertion factor
        # v is in km/s, c is in m/s, convert will be in nm
        convert = 1 - 1000*v / SPEED_OF_LIGHT
        # Shift the spectrum order
        wave[i] *= convert
    return wave

def calibrate_simLC(extracted_science_orders, veloce_paths, lc_image, hdr, arm, traces=None, plot=False, filename=None, flux_error=None):
    """
    Calibrate the wavelength solution for the simultaneous laser comb observations using CCF.
    
    Parameters:
    - extracted_science_orders (list of numpy.ndarray): The list of extracted science orders to be calibrated.
    - veloce_paths (object): An object containing the paths to the data directories.
    - lc_image (numpy.ndarray): Image (2D array) containing the simultaneous laser comb data.
    - hdr (astropy.io.fits.Header): The header of the LC image.
    - arm (str): The arm of the spectrograph ('red', 'green', or 'blue').
    - traces (object, optional): An object containing traces for LC. If None, default traces for the arm will be loaded.
    - plot (bool, optional): Whether to plot the intermediate results. Default is False.
    - filename (str, optional): The filename for the plots. Default is None.
    - flux_error (list of numpy.ndarray, optional): The list of flux uncertainties. 
    
    Returns:
    - wave (numpy.ndarray): The array of calibrated wavelengths.
    - extracted_science_orders (list of numpy.ndarray): The list of extracted science orders trimmed to calibrated range.
    - flux_error (list of numpy.ndarray, optional): The list of flux uncertainties trimmed to calibrated range. Returned only if flux_error is provided as input.
    """
    if arm == 'blue':
        raise NotImplementedError("Blue arm is not supported for LC calibration.")
        # print("[warning] Blue arm is not supported for LC calibration.")
        # return np.array([None]), np.array([None])
    ref_orders, ref_wave, ref_intensity, ref_pixel = load_LC_wave_reference(veloce_paths, arm)
    lc_intensity, lc_pixel, order_slice, pixel_slices = load_simultanous_LC(lc_image, veloce_paths, hdr, arm, traces=traces, ref_orders=ref_orders, ref_pixel=ref_pixel)
    # align extracted orders with calibrated orders and pixel ranges
    extracted_science_orders = extracted_science_orders[order_slice]
    extracted_science_orders = [order[pixel_slices[i]] for i, order in enumerate(extracted_science_orders)]
    extracted_science_orders = pad_array(extracted_science_orders, ref_pixel)

    if flux_error is not None:
        flux_error = flux_error[order_slice]
        flux_error = [error[pixel_slices[i]] for i, error in enumerate(flux_error)]
        flux_error = pad_array(flux_error, ref_pixel)

    # cross-correlate the observed LC pixel positions with the reference LC pixel positions
    dispersion_position, orders_position, offset_array = calculate_offset_map(ref_orders, ref_intensity, ref_pixel, lc_intensity, lc_pixel,
                                                                              plot=plot, veloce_paths=veloce_paths, filename=filename)
    
    # fit a surface to the offset map
    results = []
    for degree in range(1, 4):
        fit_result = fit_surface(dispersion_position, orders_position, offset_array, lc_pixel, degree=degree,
                                 plot=plot, veloce_paths=veloce_paths, filename=filename)
        results.append(fit_result)

    # Select the result with the smallest standard deviation of residuals
    best_fit = min(results, key=lambda result: np.std(result[3]))
    surface_points, coeffs, filtered_points, residuals = best_fit

    # estimate the calibration precision
    calibration_precision = estimate_calibration_precision(residuals, 18, ref_wave)

    # interpolate wavelength solution to pixel positions
    wave = interpolate_offsets_optimised(lc_pixel, surface_points, ref_wave, ref_pixel)

    shifts = np.load(os.path.join(veloce_paths.wave_dir, f'{arm}_velocity_orders_offsets.npy'))
    shifts = shifts[:len(wave)] # Ensure shifts array matches the length of wave
    wave = [w * (1 + shift/SPEED_OF_LIGHT) for w, shift in zip(wave, shifts)]
    
    # apply shift between calibration fiber and science fibers expressed as rv
    # wave = apply_wavelength_shift(wave, arm, veloce_paths)
    if flux_error is not None:
        return wave, extracted_science_orders, flux_error
    else:
        return wave, extracted_science_orders

def load_wave_calibration_for_interpolation():
    raise NotImplementedError

def interpolate_wave(orders, hdr):
    raise NotImplementedError

def load_static_Th_wavelength_solution(arm, veloce_paths, traces):
    """
    Load the static wavelength solution for the ThAr calibration.
    Parameters:
    - arm (str): The arm of the spectrograph ('red', 'green', or 'blue').
    - veloce_paths (object): An object containing the paths to the data directories.
    - traces (object): A trace object (same as science fibres).
    Returns:
    - wave (numpy.ndarray): The array of wavelengths corresponding to the ThAr calibration.
    """
    wave = pickle.load(open(os.path.join(veloce_paths.wave_dir, f'ThXe_wave_230826_{arm}.pkl'), 'rb'))
    for w, trace_y in zip(wave, traces.y):
        assert len(w) == len(trace_y), "Size missmatch between used trace and static wavelength solution."
    return wave

def load_reference_Th_spectrum(arm, veloce_paths):
    """Load the reference Th spectrum for the ThAr calibration.
    
    Parameters:
    - arm (str): The arm of the spectrograph ('red', 'green', or 'blue').
    - veloce_paths (object): An object containing the paths to the data directories.
    
    Returns:
    - ref_th_spectrum (numpy.ndarray): The array of flux values for the reference Th spectrum.
    - ref_th_header (astropy.io.fits.Header): The header of the reference Th spectrum.
    """
    ref_th_file = os.path.join(veloce_paths.wave_dir, f'Th_reference_spectrum_230828_{arm}.pkl')
    if not os.path.exists(ref_th_file):
        raise FileNotFoundError(f"Reference Th spectrum file not found: {ref_th_file}")
    with fits.open(ref_th_file) as hdul:
        ref_th_spectrum = hdul[0].data
        ref_th_header = hdul[0].header
    return ref_th_spectrum, ref_th_header

# def get_Th_master(obs_list, arm, )
#     master_flat_filename = os.path.join(veloce_paths.master_dir, f'master_flat_{arm}_{date}.fits')
#     if os.path.exists(master_flat_filename):
#         with fits.open(master_flat_filename) as hdul:
#             master_flat = hdul[0].data
#             hdr = hdul[0].header
#     else:
#         master_flat, hdr = veloce_reduction_tools.get_master_mmap(
#             obs_list, veloce_paths.input_dir,
#             date, arm, amplifier_mode)
#         master_flat, hdr = veloce_reduction_tools.normalise_flat(master_flat, hdr)
#         veloce_reduction_tools.save_image_fits(master_flat_filename, master_flat, hdr)

#     return master_flat

def append_column_to_recarray(array, column_name, column_data):
    """
    Append a new column to a structured numpy array.
    
    Parameters:
    - array: The original structured numpy array.
    - column_name: The name of the new column to be added.
    - column_data: The data for the new column.
    
    Returns:
    - A new structured numpy array with the additional column.
    """
    dtype = array.dtype.descr + [(column_name, column_data.dtype)]
    new_array = np.empty(array.shape, dtype=dtype)
    for name in array.dtype.names:
        new_array[name] = array[name]
    new_array[column_name] = column_data
    return new_array

def load_UVES_linelist(file):
    """
    Load the UVES ThAr linelist from a text file and convert it to a structured numpy array.
    
    Parameters:
    - file: The path to the UVES linelist text file.
    
    Returns:
    - data: A structured numpy array containing the linelist data with appropriate field names and types.
    """
    # with field labels matching nist linelist
    types = np.array(['f', 'f', 'f', '<U2', '<U3', '<U1'])
    # columns = np.array(['wavenumber(cm-1)', 'air_wave(nm)', 'log_intens', 'Element', 'Ion', 'Reference'], dtype=str)
    # make relevant labels match ones from nist linelist
    columns = np.array(['wavenumber(cm-1)', 'obs_wl_air(nm)', 'intens', 'element', 'ion', 'line_ref'], dtype=str)  
    dtype = [(col, t) for col, t in zip(columns, types)]
    with open(file, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            values = np.array([value.strip() for value in line.strip().split()], dtype=str)
            data.append(values)

    data = [tuple(row) for row in data]
    dtype = [(col, t) for col, t in zip(columns, types)]
    data = np.array(data, dtype=dtype)
    data['obs_wl_air(nm)'] = data['obs_wl_air(nm)']/10
    ### force all lines meet preset intensity threshold (which is designed for nist linelist)
    data['intens'] = 200+10**data['intens']  # Convert intensity
    ### or just from log intensity 
    # data['intens'] = 10**data['intens']
    # Add columns for compatibility with nist linelist
    data = append_column_to_recarray(data, 'unc_obs_wl', np.zeros_like(data['obs_wl_air(nm)']))
    data = append_column_to_recarray(data, 'intens_flag', np.array(['1' for _ in range(len(data['obs_wl_air(nm)']))], dtype=str))
    return data

def load_Th_linelist(veloce_paths, filename='Default', linelist_type='NIST'):
    """
    Load the ThAr linelist from a text file and convert it to a structured numpy array.
    
    Parameters:
    - veloce_paths: The paths to the veloce data directories.
    - filename: The name of the linelist file to load.
    If 'Default', it will load the default linelist from the veloce_paths, which is from NIST.
    - linelist_type: The type of linelist to load ('NIST' or 'UVES').

    Returns:
    - data: A structured numpy array containing the linelist data with appropriate field names and types.
    """
    if linelist_type == 'NIST':
        if filename == 'Default':
            filename = 'th_linelist_NIST.pickle'
        with open(os.path.join(veloce_paths.wave_dir, filename), 'rb') as f:
            atomic_data_dict = pickle.load(f)
            print(f"Loaded line list\n Notes: {atomic_data_dict['notes']} \n Cite as: \n {atomic_data_dict['cite']}\n")
            return atomic_data_dict['linelist']
    elif linelist_type == 'UVES':
        if filename == 'Default':
            filename = 'thar_UVES_MM090311.dat'
        linelist = load_UVES_linelist(os.path.join(veloce_paths.wave_dir, filename))
        print(f"Loaded UVES line list with {len(linelist)} lines.")
        return linelist
    return False

def normalise_ArcTh_order_with_spline(y, nknots=15, norm_type='continuum', node_distribution='chebyshev', smooth=10, bc_type="clamped", plot=False):
    """
    Normalise using spline fitting with continuum estimation and adaptive knot placement.

    This method uses a combination of local peak detection, continuum estimation, and adaptive knot placement
    to create a blaze correction for ThXe orders.

    Parameters:
    - y: 1D array of flux values for the ThXe order.
    - nknots: Number of knots to use for spline fitting.
    - norm_type: 'minimum' to use local minima, 'continuum' to use continuum estimate.
    - node_distribution: 'linear' for evenly spaced knots, 'chebyshev' for Chebyshev nodes.
    - smooth: Smoothing parameter.
    - bc_type: Boundary condition type for spline fitting.
    - plot: If True, generates plots to visualize the process.

    Returns:
    - (numpy.ndarray): Normalized flux values after spline fitting.
    """
    if sum(np.isfinite(y)) < nknots:
        print("[Warning] Order length is less than number of knots. Normalisation not possible.")
        return y
    _y = y + 1
    ylen = len(y)
    x = np.arange(ylen, dtype=float)
    
    # Initial knot distribution
    if node_distribution == 'linear':
        x_fit = np.linspace(np.nanmin(x), np.nanmax(x), nknots)
    elif node_distribution == 'chebyshev':
        # Generate standard Chebyshev nodes on [-1, 1]
        k = np.arange(1, nknots + 1)
        cheb_nodes = np.cos((2 * k - 1) * np.pi / (2 * nknots))
        
        # Transform from [-1, 1] to [np.nanmin(x), np.nanmax(x)]
        x_fit = 0.5 * (np.nanmax(x) - np.nanmin(x)) * (cheb_nodes + 1) + np.nanmin(x)
        x_fit = np.sort(x_fit)  # Sort in ascending order
    else:
        raise ValueError("Invalid node_distribution. Use 'linear' or 'chebyshev'.")
    intial_knots = x_fit.copy()
    # Local percentile-based peak detection (catches weaker peaks)
    window_size = ylen // (nknots * 2)  # Adaptive window size

    # print(f"Using window size: {window_size}")

    signal_peaks = []
    for i in range(ylen):
        start = max(0, i - window_size//2)
        end = min(ylen, i + window_size//2 + 1)
        local_window = _y[start:end]
        # Find peaks in this window
        peaks, _ = find_peaks(local_window, prominence=(np.median(local_window)-np.min(local_window)))
        # Convert local indices to global indices
        global_peaks = (peaks + start).tolist()
        signal_peaks.extend(global_peaks)
     # Extend signal peaks to the 'floor' level (local minimum to left and right)
    extended_peaks = set(signal_peaks)
    for peak in signal_peaks:
        # Search left
        left = peak
        while left > 0 and _y[left-1] < _y[left]:
            left -= 1
        # Search right
        right = peak
        while right < ylen-1 and _y[right+1] < _y[right]:
            right += 1
        # Add all points from left to right (inclusive)
        extended_peaks.update(range(left+1, right)) # don't use the actual floor points 
    extended_peaks = sorted(extended_peaks)
    
    # all_detected_peaks = np.unique(np.concatenate((percentile_peaks, extended_peaks)))
    all_detected_peaks = np.unique(np.array(extended_peaks))
    # print(f"Detected {len(extended_peaks)} peaks using signal method with window size {window_size}")
    # print(f"Total detected peaks: {len(all_detected_peaks)}")
    
    # Create a mask for all detected peaks
    peak_mask = np.zeros_like(_y, dtype=bool)
    if all_detected_peaks.size > 0:
        peak_mask[all_detected_peaks] = True

    # Interpolate over detected peaks using neighbors that are not rejected
    y_for_min = _y.copy()
    if np.any(peak_mask):
        not_peak = ~peak_mask
        # Use linear interpolation for peak regions
        interp_vals = np.interp(x[peak_mask], x[not_peak], _y[not_peak])
        y_for_min[peak_mask] = interp_vals

    # Apply minimum filter (same size as y)
    min_filtered = minimum_filter(median_filter(y_for_min, size=window_size//4), size=window_size//2)
    continuum_estimate = gaussian_filter1d(min_filtered, sigma=smooth)  # Smooth estimate
    
    # Line contamination score: how far above continuum estimate
    line_contamination = _y - continuum_estimate
    line_contamination[line_contamination < 0] = 0  # make at least 0
    line_contamination /= np.max(line_contamination)
    # line_contamination = gaussian_filter1d(line_contamination, sigma=smooth)  # Smooth line contamination
    
    # Local variance-based detection (high variance = variable region)
    local_variance = np.zeros_like(_y)
    for i in range(ylen):
        start = max(0, i - window_size//10)
        end = min(ylen, i + window_size//10 + 1)
        local_variance[i] = np.var(_y[start:end])
    # Variance penalty: high variance regions are less preferred
    smoothed_variance = gaussian_filter1d(local_variance, sigma=smooth)
    max_variance = np.max(smoothed_variance)
    variance_penalty = smoothed_variance / max_variance if max_variance > 0 else np.zeros_like(smoothed_variance)
    # print(f"Max variance: {max_variance:.2f}, Variance penalty range: {np.min(variance_penalty):.2f} - {np.max(variance_penalty):.2f}")
    
    # Combined continuum preference score (lower = better for knot placement)
    continuum_score = line_contamination + variance_penalty
    # continuum_score /= np.max(continuum_score)  # Normalize to [0, 1]
    # Optionally scale continuum_score to have a running window maximum of 1 before further processing
    if True:  # Set to True to enable normalization to 1 in running window
        window_norm = window_size // 2
        running_max = np.array([np.max(continuum_score[max(0, i-window_norm):min(len(continuum_score)-1, i+window_norm+1)]) for i in range(len(continuum_score))])
        running_max[running_max == 0] = 1  # Avoid division by zero
        continuum_score = continuum_score / running_max
        
    # Move knots away from detected peaks and towards better continuum regions
    for i, knot in enumerate(x_fit[1:-1]):  # Skip first and last knots
        if continuum_score[int(knot)] == 0:
            # If the knot is already in a good continuum region, skip it
            continue
        else: 
            # Define search range around the knot
            search_radius = window_size  # Adaptive search radius
            search_range = slice(max(0, int(knot)-search_radius), min(ylen, int(knot)+search_radius))
            
            local_x = x[search_range]
           
            local_score = continuum_score[search_range]
            min_score = np.min(local_score)
            best_indices = np.where(local_score == min_score)[0]
            if len(best_indices) == 1:
                best_idx = best_indices[0]
            elif len(best_indices) > 1:
                # If multiple, choose the one closest to the original knot
                distances = np.abs(local_x[best_indices] - knot)
                best_idx = best_indices[np.argmin(distances)]
            else:
                # remove knot if no valid position found
                best_idx = None
            if best_idx is not None:
                x_fit[i+1] = local_x[best_idx]
            else:
                x_fit[i+1] = np.nan
            # print(f"Moving knot {i+1} at {knot:.1f} within range {local_np.nanmin(x)}-{local_np.nanmax(x)}, from score {continuum_score[int(knot)]:.3f} to {local_score[best_idx]:.3f} at {local_x[best_idx]:.1f}.")
            # Mask out regions around all detected peaks in this local area
            # mask = np.ones_like(local_score, dtype=bool)
            # for p in all_detected_peaks:
            #     if search_range.start <= p < search_range.stop:
            #         local_peak_idx = p - search_range.start
            #         # Mask out around peaks
            #         # mask_start = max(0, local_peak_idx - 1)
            #         # mask_end = min(len(mask)-1, local_peak_idx + 2)
            #         # mask[mask_start:mask_end] = False
            #         mask[local_peak_idx] = False
            # if np.any(mask):
            #     # Find the position with minimum continuum score (best continuum location)
            #     valid_scores = local_score[mask]
            #     valid_x = local_x[mask]
            # Choose position(s) with lowest continuum score
                # min_score = np.min(valid_scores)
                # best_indices = np.where(valid_scores == min_score)[0]
                # if len(best_indices) == 1:
                #     best_idx = best_indices[0]
                # else:
                #     # If multiple, choose the one closest to the original knot
                #     distances = np.abs(valid_x[best_indices] - knot)
                #     best_idx = best_indices[np.argmin(distances)]
                # x_fit[i] = valid_x[best_idx]
            # else:
            #     print(f"Warning: Could not find good continuum region for knot at {knot:.1f}, moving to nearest non-peak.")
            #     # Move to nearest non-peak in search range
            #     non_peak_indices = np.where(mask)[0]
            #     if len(non_peak_indices) > 0:
            #         nearest = local_x[mask][np.argmin(np.abs(local_x[mask] - knot))]
            #         x_fit[i] = nearest
            #     else:
            #         # As a last resort, remove the knot at its current position
            #         x_fit[i] = np.nan

    # set first and last knots to the edges that are not nan
    x[np.isnan(y)] = np.nan
    x_fit[0] = np.nanmin(x)  # Ensure first knot is at start
    x_fit[-1] = np.nanmax(x)  # Ensure last knot is at end
    continuum_score[int(x_fit[0])] = -1e-6  # Set first knot score to negative
    continuum_score[int(x_fit[-1])] = -1e-6  # Set last knot score to negative
    # print(f"Knots after moving: {x_fit}")

    # Remove any knots marked as np.nan
    if np.any(~np.isfinite(x_fit)):
        # print("Removing knots that could not be placed in good continuum regions.")
        x_fit = x_fit[np.isfinite(x_fit)]
    # Remove duplicate knots
    if len(x_fit) != len(np.unique(x_fit)):
        # print("Removing duplicate knots after placement.")
        x_fit = np.unique(x_fit)
    
    # Ensure knots are still in order and within bounds
    if np.any(x_fit < np.nanmin(x)) or np.any(x_fit > np.nanmax(x)):
        # print("Clipping knots to valid range.")
        x_fit = np.clip(x_fit, np.nanmin(x), np.nanmax(x))
    # x_fit = np.sort(x_fit)

    # Drop knots that are too close together, keeping the one with lower continuum_score
    if node_distribution == 'chebyshev':
        min_dist = 2*smooth
    else:
        min_dist = max(10, window_size // 4)  # Minimum allowed distance between knots
    keep = np.ones(len(x_fit), dtype=bool)
    i = 0
    while i < len(x_fit) - 1:
        if x_fit[i+1] - x_fit[i] < min_dist:
            # print(f"Dropping close knots at {x_fit[i]:.1f} and {x_fit[i+1]:.1f}, distance: {x_fit[i+1] - x_fit[i]:.1f}")
            # Compare continuum_score at both knots
            score_i = continuum_score[int(x_fit[i])]
            score_ip1 = continuum_score[int(x_fit[i+1])]
            # Drop the one with higher score
            if score_i <= score_ip1:
                keep[i+1] = False
            else:
                keep[i] = False
            # After dropping, don't increment i to check new neighbor
            x_fit = x_fit[keep]
            keep = np.ones(len(x_fit), dtype=bool)
            i = 0  # Restart to ensure all pairs checked after removal
        else:
            i += 1
    # x_fit = x_fit[keep]
    # print(f"Final nknot after moving knots and dropping close pairs: {np.sum(keep)} out of {nknots}")
    # print(f"Final knot positions: {x_fit} based on {keep}.")
    
    if norm_type == 'minimum':
        y_fit = [np.min(_y[int(max(0, int(_x - smooth))):int(min(len(x)-1, int(_x + smooth + 1)))+1]) for _x in x_fit]  # Use local minima around knots
    elif norm_type == 'continuum':
        y_fit = continuum_estimate[x_fit.astype(int)]  # Use local continuum estimate for knot
    else:
        raise ValueError("Invalid norm_type. Use 'minimum' or 'continuum'.")
    # Improved boundary conditions using continuum estimate
    # No, index len(x) would be out of bounds for array x (valid indices are 0 to len(x)-1).
    boundary_width = window_size // 2
    y_fit[0] = np.nanmedian(continuum_estimate[int(x_fit[0]):int(x_fit[0]+boundary_width)])
    y_fit[-1] = np.nanmedian(continuum_estimate[int(x_fit[-1]-boundary_width):int(x_fit[-1])])
    
    # spline = make_interp_spline(x_fit, y_fit, k=3, bc_type=([(1, 0.0)], [(1, 0.0)]))
    spline = make_interp_spline(x_fit, y_fit, k=3, bc_type=bc_type)
    baseline = spline(x)
    
    if plot:
        plt.close('all')
        plt.figure(figsize=(16, 12))
        
        # Top panel: Peak detection analysis
        plt.subplot(4, 1, 1)
        plt.plot(x, _y, 'gray', alpha=0.7, label='Original')
        plt.scatter(extended_peaks, _y[extended_peaks], marker='x', c='red', s=15, alpha=0.8, label='Signal peaks', zorder=4)
        plt.plot(x, continuum_estimate, 'blue', alpha=0.8, label='Continuum estimate')
        plt.xlim(np.nanmin(x), np.nanmax(x))
        plt.ylim(0, np.max(_y) * 1.05)
        plt.ylabel('Flux')
        plt.xlabel('Pixel')
        plt.legend()
        plt.title('Peak Detection and Continuum Estimation')
        
        # Second panel: Continuum scoring and selected knots
        plt.subplot(4, 1, 2)
        plt.plot(x, line_contamination, 'orange', alpha=0.6, label='Line contamination')
        plt.plot(x, variance_penalty, 'green', alpha=0.6, label='Variance penalty')
        plt.plot(x, continuum_score, 'purple', alpha=0.8, label='Continuum score')
        plt.scatter(x_fit, continuum_score[x_fit.astype(int)], s=30, c='blue', edgecolor='white', linewidth=0.5, label='Knots', zorder=5)
        for guess_knot in intial_knots:
            plt.axvline(guess_knot, color='blue', ls='--', alpha=0.5, lw=0.5)
            plt.axvline(guess_knot-window_size, color='red', ls='--', alpha=0.5, lw=0.5)
            plt.axvline(guess_knot+window_size, color='red', ls='--', alpha=0.5, lw=0.5)
        plt.xlim(np.nanmin(x), np.nanmax(x))
        plt.ylim(-0.05, 1.05)
        plt.ylabel('Normalised Score')
        plt.xlabel('Pixel')
        plt.legend()
        plt.title('Scoring and Knot Placement')
        
        # Third panel: Knot placement and fitting
        plt.subplot(4, 1, 3)
        plt.plot(x, _y, 'gray', alpha=0.5, label='Original')
        plt.scatter(x[peak_mask], _y[peak_mask], marker='x', c='red', s=1, label='Rejected (emission lines)')
        plt.scatter(x_fit, y_fit, s=30, c='blue', edgecolor='white', linewidth=0.5, label='Knots', zorder=5)
        plt.plot(x, baseline, 'b-', linewidth=2, label='Baseline')
        plt.xlim(np.nanmin(x), np.nanmax(x))
        plt.ylim(0, max(baseline) * 1.2)
        plt.ylabel('Flux')
        plt.xlabel('Pixel')
        plt.legend()
        plt.title('Baseline Fitting')
        
        # Bottom panel: Normalized result
        plt.subplot(4, 1, 4)
        plt.plot(x, _y/baseline, 'b-', label='Normalised')
        plt.axhline(1, color='k', ls='--', alpha=0.5)
        plt.xlim(np.nanmin(x), np.nanmax(x))
        plt.ylim(-0.05, np.max(_y/baseline) * 1.05)
        plt.ylabel('Normalised Flux')
        plt.xlabel('Pixel')
        plt.legend()
        plt.title('Final Normalised Spectrum')
        
        plt.tight_layout()
        plt.show()
    
    return _y/baseline

def get_lines_in_order(wave, linelist, elements=None, intensity_threshold=None, flag=None):
    """
    Get the lines in the order from the linelist.

    Parameters:
    - wave (numpy.ndarray): 1D array of wavelengths in the order (in air).
    - linelist (numpy.ndarray): Structured array containing the linelist with at least 'obs_wl_air(nm)' field.
    - elements (list of str, optional): List of element symbols to filter the linelist (e.g., ['Th', 'Xe']).
    If None, no filtering by element is applied. Default is None.
    - intensity_threshold (int or tuple of int, optional): If int, only lines with intensity greater than or equal to this value are included.
    If tuple (min_intensity, max_intensity), only lines with intensity within this range are included.
    If None, no intensity filtering is applied. Default is None.
    - flag (str or list of str, optional): If str, only lines with this intensity flag are included. If list of str, only lines with any of these flags are included.
    If None, no filtering by intensity flag is applied. Default is None.

    Returns:
    - (numpy.ndarray): Structured array of lines from the linelist that fall within orders wavelength range and meet the specified criteria.
    """
    # Build mask for wavelength range
    mask = (linelist['obs_wl_air(nm)'] >= wave.min()) & (linelist['obs_wl_air(nm)'] <= wave.max())
    # Optional: intensity threshold
    if intensity_threshold is not None:
        # Filter linelist by intensity threshold
        if type(intensity_threshold) == int:
            min_intensity = intensity_threshold
            max_intensity = np.inf
        elif len(intensity_threshold) == 2:
            min_intensity, max_intensity = intensity_threshold
        else:
            raise ValueError("Intensity threshold should be a single value or a tuple of two values.")
        mask &= (linelist['intens'] >= min_intensity) & (linelist['intens'] <= max_intensity)
    # Optional: intensity flag
    if flag is not None:
        if isinstance(flag, str):
            mask &= (linelist['intens_flag'] == flag)
        elif isinstance(flag, list):
            mask &= np.isin(linelist['intens_flag'], flag)
        else:
            raise ValueError("Flag should be a string or a list of strings.")
    # Optional: element condition
    if elements is not None:
        mask &= np.isin(linelist['element'], elements)
    # print(f"Found {np.sum(mask)} lines in the order.")
    return linelist[mask]

# def plot_order_with_lines(wave, thxe_order, linelist, original_solution=None):
#     """
#     Plot the ThAr order with the lines from the linelist and optionally the original solution.
#     Parameters:
#     - wave (numpy.ndarray): 1D array of wavelengths in the order (in air).
#     - thxe_order (numpy.ndarray): 1D array of flux values for the ThAr order.
#     - linelist (numpy.ndarray): Structured array containing the linelist with at least 'obs_wl_air(nm)' field.
#     - original_solution (tuple of arrays, optional): Tuple containing (MATCH_LAM, GUESS_LAM) from the original solution to plot for comparison."""
#
#     Returns:
#     - None: Displays a plot of the ThAr order with lines
#     """
#     plt.close('all')
#
#     plt.plot(wave, thxe_order, label='fibThAr')
#
#     lines = get_lines_in_order(wave, linelist, intensity_threshold=100)
#     # lines = get_lines_in_order(veloce_reduction_tools.vacuum_to_air(wave[order]), nist_linelist, elements=['Th', 'Xe'], intensity_threshold=100)
#     for line in lines:
#         plt.axvline(line['obs_wl_air(nm)'], c='r', ls='--', label="linelist")
#
#     if original_solution is not None:
#         MATCH_LAM, GUESS_LAM = original_solution
#         for match_wave in MATCH_LAM:
#             plt.axvline(veloce_reduction_tools.vacuum_to_air(match_wave), c='g', ls=':', label="match_wave")
#         for guess_wave in GUESS_LAM:
#             plt.axvline(veloce_reduction_tools.vacuum_to_air(guess_wave), c='b', ls=':', label="guess_wave")
#     # Remove duplicate labels in legend
#     handles, labels = plt.gca().get_legend_handles_labels()
#     unique = dict()
#     for h, l in zip(handles, labels):
#         if l not in unique:
#             unique[l] = h
#     plt.legend(unique.values(), unique.keys())
#     plt.show()

def fit_lines_in_order(wavelengths, flux, pixels, linelist, arm, offset=0, plot=False, verbose=False):
    """
    Fit spectral lines in a given order using a provided linelist and return the pixel and wavelength positions
    of successfully fitted lines.
    This function identifies and fits emission lines in a spectrum for a specified spectrograph arm ('green', 'red', or 'blue').
    It uses different selection and fitting criteria depending on the arm. The function attempts to find peaks near the expected
    line positions, applies quality checks (height, asymmetry, position), and fits a general Gaussian profile to each candidate line.
    Optionally, it can plot the fit for visual inspection.
    Parameters
    ----------
    wavelengths : array-like
        Array of wavelength values corresponding to the spectrum.
    flux : array-like
        Array of flux values (intensity) for each pixel in the spectrum.
    pixels : array-like
        Array of pixel indices corresponding to the spectrum.
    linelist : structured array or list of dict
        List or array containing line information, including at least 'obs_wl_air(nm)' for observed air wavelength.
    arm : str
        Spectrograph arm to use for selection criteria. Must be one of 'green', 'red', or 'blue'.
    plot : bool, optional
        If True, plots the fit for each line (default is False).
    Returns
    -------
    lines_pixel_positions : np.ndarray
        Array of pixel positions where lines were successfully fitted.
    lines_wave_positions : np.ndarray
        Array of corresponding wavelengths for the fitted lines.
    lines_passed : np.ndarray
        Array of linelist entries for lines that passed all selection and fitting criteria.
    Raises
    ------
    ValueError
        If an unknown arm is specified.
    Notes
    -----
    - The function applies arm-specific thresholds for peak height, asymmetry, and position.
    - Lines that do not meet the criteria or cannot be fitted are skipped.
    - Duplicate pixel positions (blends) are removed from the output.
    - The function prints diagnostic messages for lines that fail selection or fitting.
    """
    ### need different conditions for each arm
    if arm == 'green':
        lines = get_lines_in_order(wavelengths, linelist, elements=['Th'], intensity_threshold=200, flag=['1'])
        peak_height_threshold = 1.0
        peak_asymmetry_threshold = 0.4
        peak_position_threshold = 3.0
    elif arm == 'red':
        lines = get_lines_in_order(wavelengths, linelist, elements=['Th'], intensity_threshold=100, flag=['1'])
        peak_height_threshold = 0.3
        peak_asymmetry_threshold = 0.5
        peak_position_threshold = 4.0
    elif arm == 'blue':
        # TODO implement blue arm conditions
        # For now, using similar conditions as for green arm
        lines = get_lines_in_order(wavelengths, linelist, elements=['Th'], intensity_threshold=200, flag=['1'])
        peak_height_threshold = 0.7
        peak_asymmetry_threshold = 0.3
        peak_position_threshold = 3.0
    else:
        raise ValueError(f"Unknown arm: {arm}. Supported arms are 'green', 'red', and 'blue'.")
    
    pixels = np.array(pixels, dtype=np.int_)
    min_pixel, max_pixel = np.min(pixels), np.max(pixels)
    passed_mask = []
    lines_pixel_positions = []
    lines_wave_positions = []
    lines_sigma = []
    for line in lines:
        line_wave = line['obs_wl_air(nm)']
        idx = np.argmin(np.abs(wavelengths - line_wave))
        pix_frac_in_wave = (line_wave - wavelengths[idx])
        idx += int(offset) # apply offset to index (i.e. guess pixel position)
        if idx <= 0 or idx >= len(pixels)-1:
            # print(f"Line {line_wave:.3f} nm is out of pixel range after offset.")
            passed_mask.append(False)
            # lines_pixel_positions.append(np.nan)
            # lines_wave_positions.append(np.nan)
            continue
        # line_pixel = min_pixel + idx + (line_wave - wavelengths[idx])/(wavelengths[idx+1] - wavelengths[idx]) \
        #     if line_wave - wavelengths[idx] > 0 \
        #     else min_pixel + idx + (line_wave - wavelengths[idx])/(wavelengths[idx] - wavelengths[idx-1])
        line_pixel = min_pixel + idx + (pix_frac_in_wave)/(wavelengths[idx+1] - wavelengths[idx]) \
            if line_wave - wavelengths[idx] > 0 \
            else min_pixel + idx + (pix_frac_in_wave)/(wavelengths[idx] - wavelengths[idx-1])
        # line_pixel += offset

        fit_range = slice(max(0, idx-10), min(len(pixels)-1, idx+11))
        x_fit = pixels[fit_range]
        y_fit = flux[fit_range]

        peaks, _ = find_peaks(y_fit, prominence=peak_height_threshold/2)
        if len(peaks) == 0:
            # print(f"No peaks found for line {line_wave:.3f} nm.")
            passed_mask.append(False)
            # lines_pixel_positions.append(np.nan)
            # lines_wave_positions.append(np.nan)
            continue
        peak_idx = peaks[np.argmin(np.abs(x_fit[peaks] - line_pixel))]
        center = x_fit[peak_idx]
        if abs(center-line_pixel) > peak_position_threshold:
            # print(f"Peak {center} for line {line_wave:.3f} nm is too far from the guess pixel {line_pixel:.2f}.")
            # if line_wave - wavelengths[idx] > 0:
            #     print((line_wave - wavelengths[idx])/(wavelengths[idx+1] - wavelengths[idx]))
            # else:
            #     print((line_wave - wavelengths[idx])/(wavelengths[idx] - wavelengths[idx-1]))
            passed_mask.append(False)
            # lines_pixel_positions.append(np.nan)
            # lines_wave_positions.append(np.nan)
            continue

        lower_bound = peak_idx-1
        while lower_bound > 1:
            if (y_fit[lower_bound] <= y_fit[lower_bound-1] and y_fit[lower_bound] <= y_fit[lower_bound+1]):
                break
            lower_bound -= 1

        upper_bound = peak_idx + 1
        while upper_bound < len(y_fit)-1:
            if (y_fit[upper_bound] <= y_fit[upper_bound-1] and y_fit[upper_bound] <= y_fit[upper_bound+1]):
                break
            upper_bound += 1

        line_floor = np.min(y_fit[lower_bound:upper_bound+1])
        peak_height = y_fit[peak_idx] - line_floor
        if peak_height < peak_height_threshold:
            # print(f"Peak height for line {line_wave:.3f} nm is too low: {peak_height:.2f}.")
            passed_mask.append(False)
            # lines_pixel_positions.append(np.nan)
            # lines_wave_positions.append(np.nan)
            continue
        if (y_fit[lower_bound]-line_floor)/peak_height > peak_asymmetry_threshold or (y_fit[upper_bound]-line_floor)/peak_height > peak_asymmetry_threshold:
            # print(f"Line {line_wave:.3f} nm is asymmetric - probably a blend")
            passed_mask.append(False)
            # lines_pixel_positions.append(np.nan)
            # lines_wave_positions.append(np.nan)
            continue
        fit_mask = np.zeros_like(y_fit, dtype=bool)
        fit_mask[lower_bound:upper_bound+1] = True
        x_fit_masked = x_fit[fit_mask]
        y_fit_masked = y_fit[fit_mask]
        if len(x_fit_masked) < 5: # Not enough points to fit
            passed_mask.append(False)
            # lines_pixel_positions.append(np.nan)
            # lines_wave_positions.append(np.nan)
            continue

        # print(f"Fitting line {line_wave:.3f} nm at pixel {line_pixel:.2f}, peak height {peak_height:.2f}, floor {line_floor:.2f}")

        p0 = [peak_height, center, 2, 2, line_floor] # use general gaussian for peak position
        _p0 = [peak_height, center, 2, line_floor] # use gaussian for line fwhm
        bounds = ([0.5*peak_height, center-3, 1e-3, 1e-3, 0], [line_floor+2.5*peak_height, center+3, 5, 5, line_floor+0.5*peak_height])
        _bounds = ([0.5*peak_height, center-3, 1e-3, 0], [line_floor+2.5*peak_height, center+3, 5, line_floor+0.5*peak_height])
        try:
            popt, _ = curve_fit(general_gaussian, x_fit_masked, y_fit_masked, p0=p0, bounds=bounds)
            _popt, _ = curve_fit(gaussian, x_fit_masked, y_fit_masked, p0=_p0, bounds=_bounds) # use gaussian for line fwhm
            if plot:
                plt.close('all')
                fig, ax = plt.subplots()
                ax.plot(x_fit, y_fit, 'C0-', label='Data')
                # plt.scatter(x_fit_masked, y_fit_masked, c='k', s=5, label='Line points')
                ax.scatter(x_fit_masked, y_fit_masked, c='C0', s=5)
                x_fine = np.arange(x_fit_masked.min(), x_fit_masked.max()+0.1, 0.1)
                ax.plot(x_fine, general_gaussian(x_fine, *popt), 'r-', label='Fit')
                # plt.axvline(line_pixel, c='orange', ls=':', label='Line guess')
                # plt.axvline(center, c='green', ls=':', label='Closest peak')
                # plt.axvline(popt[1], c='red', ls=':', label='Fit center')
                ax.set_title(f"Line {line_wave:.3f} nm")
                ax.set_xlim(x_fit.min(), x_fit.max())
                ax.set_ylim(min(0.7, y_fit.min()*0.9), general_gaussian(x_fine, *popt).max()*1.1)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_xlabel('Pixel')
                ax.set_ylabel('Flux')
                # plt.legend()
                plt.show()
            passed_mask.append(True)
            lines_pixel_positions.append(popt[1])
            lines_wave_positions.append(line_wave)
            lines_sigma.append(_popt[2])
            # weight = _popt[0]*_popt[2]*np.sqrt(2*np.pi) #- _popt[3]*len(x_fit_masked)
            # lines_weight.append(weight) # use gaussian for line fwhm
        except Exception as e:
            passed_mask.append(False)
            if verbose:
                print(f"Fit failed for line {line_wave:.3f} nm: {e}")
            # lines_pixel_positions.append(np.nan)
            # lines_wave_positions.append(np.nan)
    # Remove duplicate pixel positions and corresponding wavelengths
    lines_pixel_positions = np.array(lines_pixel_positions)
    _, unique_indices = np.unique(lines_pixel_positions, return_index=True)
    lines_pixel_positions = lines_pixel_positions[unique_indices]

    lines_wave_positions = np.array(lines_wave_positions)[unique_indices]
    lines_sigma = np.array(lines_sigma)[unique_indices]

    # print(f"Found {len(np.unique(lines_pixel_positions[~unique_indices]))} blends (two or more wavelengths corresponding to single peak).")
    passed_mask = np.array(passed_mask)
    passed_mask[~unique_indices] = False  # Mark duplicates as not passed
    # Filter out lines that did not pass the selection criteria
    # print(f"Total lines passed: {np.sum(passed_mask)} out of {len(lines)}")
    lines = np.array(lines)[passed_mask]
    # # Keep only unique pixel positions and corresponding wavelengths
    # lines_wave_positions = lines_wave_positions[passed_mask]
    # lines_pixel_positions = lines_pixel_positions[passed_mask]
    # print(f"Total lines fitted: {len(lines_pixel_positions)}")
    
    return lines_pixel_positions, lines_wave_positions, lines, lines_sigma

def fit_all_lines_per_order(wave, norm_extracted_Th, ORDER, traces, linelist, arm, offset=0, veloce_paths=None, plot=False):
    """
    Fits spectral lines for each order and returns their pixel, wavelength, and order positions.
    Parameters
    ----------
    wave : list or np.ndarray
        List or array of wavelength solutions for each order.
    extracted_ThXe : list or np.ndarray
        Extracted ThXe arc lamp spectra for each order.
    ORDER : list or np.ndarray
        List of absolute order numbers to process.
    traces : object
        Object containing trace information, must have attribute `y` with trace positions per order.
    linelist : list or np.ndarray
        List of known spectral lines to fit.
    arm : str
        Spectrograph arm identifier ('blue', 'green', or 'red'), determines spline knot count.
    veloce_paths : object
        Object or structure containing relevant file paths (not used directly in this function).
    plot : bool, optional
        If True, plots the fitted lines for each order. Default is False.
    Returns
    -------
    pixel_positions : np.ndarray
        Concatenated array of pixel positions of fitted lines across all orders.
    wave_positions : np.ndarray
        Concatenated array of wavelength positions of fitted lines across all orders.
    order_positions : np.ndarray
        Concatenated array of order numbers corresponding to each fitted line.
    Notes
    -----
    This function normalizes each order's extracted ThXe spectrum using a spline with a number of knots
    determined by the spectrograph arm. It then fits spectral lines in each order using the provided line list.
    """
    #TODO: save fitted lines to file
    pixel_positions, wave_positions, order_positions, fwhms = [], [], [], []
    # fitted_lines = []
    
    for order, absolute_order in enumerate(ORDER):
        # print(f"Fitting lines in order {absolute_order} ({order+1}/{len(ORDER)})")
        lines_pixel_positions, lines_wave_positions, _fitted_lines, lines_sigmas = fit_lines_in_order(
            wave[order],
            norm_extracted_Th[order],
            traces.y[order],
            linelist, arm, offset=offset)
        lines_order_positions = np.ones_like(lines_pixel_positions) * absolute_order
        pixel_positions.append(lines_pixel_positions)
        wave_positions.append(lines_wave_positions)
        order_positions.append(lines_order_positions)
        fwhms.append(lines_sigmas*2.355)  # Convert sigma to FWHM for each line
        # fitted_lines.append(_fitted_lines)
    pixel_positions = np.concatenate(pixel_positions)
    wave_positions = np.concatenate(wave_positions)
    order_positions = np.concatenate(order_positions)
    fwhms = np.concatenate(fwhms)
    # fitted_lines = np.concatenate(fitted_lines)
    # print(f"Total fitted lines: {len(np.unique(wave_positions))}")

    return pixel_positions, wave_positions, order_positions, fwhms

def get_pixels_for_ArcTh_fit(orders, traces):
    """
    Get full pixel range for all orders to extrapolate the ThAr fit onto.

    Parameters:
    - orders (list or numpy.ndarray): List of orders to generate pixel ranges for.
    - traces (object): A trace object. 

    Returns:
    - full_pixels (numpy.ndarray): 2D array of pixel indices for each order.
    Each row corresponds to an order and contains the pixel indices that are extractable for it.
    """

    max_extracted_pixel = max([max(trace_y) for trace_y in traces.y])
    min_extracted_pixel = min([min(trace_y) for trace_y in traces.y])
    full_pixels = np.array([np.arange(min_extracted_pixel, max_extracted_pixel + 1) for _ in orders])
    return full_pixels

def apply_n_limit_constraint(initial_mask, orders_position, y_fit, X, model, n_limit, all_idx, residuals):
    """
    Apply n_limit constraint to any mask, ensuring taht each order has at least minimum points (n_limit).

    Parameters:
    - initial_mask (numpy.ndarray): Initial boolean mask of inliers.
    - orders_position (numpy.ndarray): Order positions corresponding to each data point.
    - y_fit (numpy.ndarray): Array of fitted wavelength values for each data point.
    - X (numpy.ndarray): Design matrix used for fitting the model.
    - model (sklearn estimator): The fitted model used to predict wavelength values.
    - n_limit (int): Minimum number of points required per order.
    - all_idx (numpy.ndarray): Array of all indices corresponding to the data points.
    - residuals (numpy.ndarray): Array of residuals for each data point.

    Returns:
    - constrained_mask (numpy.ndarray): Boolean mask meeting the n_limit constraint.
    """
    constrained_mask = np.zeros_like(initial_mask, dtype=bool)
    
    # Get residuals for current model
    current_residuals = y_fit - model.predict(X)
    
    for order in np.unique(orders_position):
        in_order = np.array(orders_position == order, dtype=bool)
        order_mask = initial_mask & in_order
        
        if np.sum(order_mask) >= n_limit:
            # Enough points selected by RANSAC for this order
            constrained_mask[order_mask] = True
        elif np.sum(in_order) < n_limit:
            # Not enough total points in order
            print(f"[Warning]: Order {order} has fewer than {n_limit} total points.")
            constrained_mask[in_order] = True
        else:
            # Need to select best n_limit points for this order
            order_residuals = np.abs(current_residuals[in_order])
            order_indices = all_idx[in_order]
            sorted_indices = order_indices[np.argsort(order_residuals)]
            best_indices = sorted_indices[:n_limit]
            constrained_mask[best_indices] = True
            
    return constrained_mask

def fit_surface_sklearn(dispersion_position, orders_position, y_fit, extracted_pixels, degree=7, sigma_clip=3, robust=False, n_limit=0, seed=None, max_iter=1000):
    """
    Fit a bivariate polynomial surface to the wavelength solution using RANSAC for outlier rejection.
    
    Parameters:
    - dispersion_position (numpy.ndarray): Array of pixel positions along the dispersion direction for each fitted line.
    - orders_position (numpy.ndarray): Array of order numbers corresponding to each fitted line.
    - y_fit (numpy.ndarray): Array of wavelengths or resolutions values corresponding to each fitted line.
    - extracted_pixels (numpy.ndarray): 2D array of pixel indices for each order to extrapolate the fit onto.
    - degree (int): Degree of the polynomial surface to fit.
    - sigma_clip (float): Sigma threshold for iterative outlier rejection.
    - robust (bool): If True, use RANSAC for robust fitting. If False, use standard least squares fitting.
    - n_limit (int): Minimum number of points required per order to be considered in the fit. If 0, no minimum is enforced.
    - seed (int or None): Random seed for RANSAC reproducibility. If None, RANSAC will use a random seed.
    - max_iter (int): Maximum number of iterations for the sigma clipping process.
    
    Returns:
    - fitted surface (numpy.ndarray): 2D array of fitted wavelength values for each pixel in the extracted_pixels grid.
    - residuals (numpy.ndarray): Array of residuals between the fitted surface and the input wavelength values for each data point.
    - mask (numpy.ndarray): Boolean array indicating which data points were considered inliers after sigma clipping or RANSAC.
    - model (sklearn estimator): The fitted model used to predict wavelengths
    - converged (bool): Indicates whether the iterative sigma clipping process converged before reaching max_iter.
    """
    
    # Prepare data
    x = dispersion_position
    y = orders_position
    # y_fit = wave_array * orders_position

    grid_points = np.vstack([(pixel, abs_order) for i, abs_order in enumerate(np.unique(orders_position)) for pixel in extracted_pixels[i]])

    # Use scikit-learn PolynomialFeatures + Ridge for surface fitting
    X = np.column_stack([x, y])
    
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    model = make_pipeline(poly, ridge)
    
    model.fit(X, y_fit)
    residuals = y_fit - model.predict(X)
    init_residuals_std = np.std(residuals)
    all_idx = np.arange(len(y_fit))
    
    mask = residuals < sigma_clip * np.std(residuals)
    converged = False
    
    # Iterative sigma clipping with n_limit constraint
    for iteration in range(max_iter):
        model.fit(X[mask], y_fit[mask])
        residuals = y_fit - model.predict(X)
        
        std_dev = np.std(residuals[mask])
        new_mask = np.abs(residuals) < sigma_clip * std_dev
        
        if n_limit > 0: # validate n points per order:
            final_idx = []
            for order in np.unique(orders_position):
                in_order = np.array(orders_position==order, dtype=bool)
                if np.sum(new_mask * in_order) >= n_limit:
                    final_idx.extend(all_idx[new_mask * in_order])
                elif np.sum(in_order) < n_limit:
                    print(f"[Warning]: not enough points in order {order} for {degree} degree polynomial, this order might be poorly conditioned.")
                    final_idx.extend(all_idx[in_order])
                else:
                    # pick best residuals for the order
                    sorted_order_idx = np.argsort(np.abs(residuals[in_order]))
                    final_idx.extend(all_idx[in_order][sorted_order_idx[:n_limit]]) # use n_limit indexes of best residuals
            final_idx = np.array(sorted(final_idx))
            new_mask = np.zeros_like(mask, dtype=bool)
            new_mask[final_idx] = True
            
        if np.array_equal(mask, new_mask):
            print(f"Converged.")
            converged = True
            break
        mask = new_mask
        
    print(f"Std of residuals dropped from {init_residuals_std:.3f} to {std_dev:.3f} after {iteration} iterations.")

    if robust:
        if n_limit <= 0:
            # Standard RANSAC without n_limit constraint
            if seed is None:
                ransac = RANSACRegressor(model, min_samples=0.5, residual_threshold=sigma_clip*std_dev)
            else:
                ransac = RANSACRegressor(model, min_samples=0.5, residual_threshold=sigma_clip*std_dev, random_state=seed)
            ransac.fit(X, y_fit)
            inlier_mask = ransac.inlier_mask_

            print(f"RANSAC inliers: {np.sum(inlier_mask)} vs iterative outlier rejection {np.sum(mask)}")
            if not np.array_equal(mask, inlier_mask):
                print("Warning: Inlier mask from RANSAC does not match the previous mask. This may indicate that RANSAC found a different set of inliers.")
            
            Z = ransac.predict(grid_points).reshape(extracted_pixels.shape)
            residuals = y_fit - ransac.predict(X)
            mask = inlier_mask
            model = ransac
        else:
            # RANSAC with n_limit constraint - need to validate and fix the mask
            if seed is None:
                ransac = RANSACRegressor(model, min_samples=0.5, residual_threshold=sigma_clip*std_dev)
            else:
                ransac = RANSACRegressor(model, min_samples=0.5, residual_threshold=sigma_clip*std_dev, random_state=seed)
            ransac.fit(X, y_fit)
            ransac_mask = ransac.inlier_mask_
            
            # Apply n_limit constraint to RANSAC results
            constrained_mask = apply_n_limit_constraint(
                ransac_mask, orders_position, y_fit, X, model, n_limit, all_idx, residuals
            )
            
            print(f"RANSAC inliers: {np.sum(ransac_mask)} -> constrained: {np.sum(constrained_mask)} vs iterative: {np.sum(mask)}")
            
            # Refit with constrained mask
            model.fit(X[constrained_mask], y_fit[constrained_mask])
            Z = model.predict(grid_points).reshape(extracted_pixels.shape)
            residuals = y_fit - model.predict(X)
            mask = constrained_mask
    else:
        Z = model.predict(grid_points).reshape(extracted_pixels.shape)
        residuals = y_fit - model.predict(X)

    return Z, residuals, mask, model, converged

def get_wave_solution_from_surface(model, traces, ORDER):
    """
    For each order use model to predict wavelengths.

    Parameters:
    - model (sklearn estimator): The fitted model used to predict wavelengths.
    - traces (object): A trace object.
    - ORDER (list or numpy.ndarray): List of absolute order numbers.

    Returns:
    - wave_solution (list of numpy.ndarray): List of wavelength solutions for each order.
    """
    return [model.predict((np.column_stack([trace_y, np.ones_like(trace_y)*absolute_order])))/absolute_order for absolute_order, trace_y in zip(ORDER, traces.y)]

def get_resolution_from_surface(model, traces, ORDER):
    """
    For each order use model to predict resolution.

    Parameters:
    - model (sklearn estimator): The fitted model used to predict resolution.
    - traces (object): A trace object.
    - ORDER (list or numpy.ndarray): List of absolute order numbers.

    Returns:
    - resolution_solution (list of numpy.ndarray): List of resolution solutions for each order.
    """
    return [model.predict((np.column_stack([trace_y, np.ones_like(trace_y)*absolute_order]))) for absolute_order, trace_y in zip(ORDER, traces.y)]

def get_arcTh_master(veloce_paths, arm, date, amplifier_mode, obs_list=None, filename=None):
    """
    Load or compute the master ThAr image for the given arm and date.

    Parameters:
    - veloce_paths: An object containing paths to the data directories.
    - arm: The spectrograph arm ('blue', 'green', or 'red').
    - date: The date of the observations (used to find the relevant files).
    - amplifier_mode: The amplifier mode used for the observations.
    - obs_list: Optional. A dictionary containing lists of observation files.
    - filename: Optional. If provided, the function will load the master ThAr image from this file instead of computing it.
    
    Returns:
    - arcTh_image: The master ThAr image as a numpy array.
    - hdr: The header associated with the master ThAr image.
    """
    if filename is not None:
        arcTh_master_filename = filename
    else:
        arcTh_master_filename = os.path.join(veloce_paths.master_dir, f"master_ARC-ThAr_{arm}_{date}.fits")

    if os.path.exists(arcTh_master_filename):
        with fits.open(arcTh_master_filename) as hdul:
            arcTh_image = hdul[0].data
            hdr = hdul[0].header
    else:
        if obs_list is not None:
            file_list = obs_list[f'ARC-ThAr_{arm}'][date]
            file_list = veloce_reduction_tools.get_longest_consecutive_files(file_list)
            if file_list:
                arcTh_image, hdr = veloce_reduction_tools.get_master_mmap(
                    file_list, veloce_paths.input_dir, date, arm, amplifier_mode)
                veloce_reduction_tools.save_image_fits(arcTh_master_filename, arcTh_image, hdr)
            else:
                raise FileNotFoundError(f"No ARC-ThAr_{arm} files found for date {date}. Cannot create master.")
        else:
            raise ValueError("Either obs_list or filename must be provided to get the arcTh master.")
    return arcTh_image, hdr

def get_resolution_from_fitted_lines(fwhms, wave_positions, pixel_positions, order_positions, wave_solution, all_orders, traces):
    """
    Determine the spectral resolution from the fitted lines.

    Parameters:
    - fwhms: Array of FWHM values for the fitted lines.
    - wave_positions: Array of wavelength positions corresponding to the fitted lines.
    - pixel_positions: Array of pixel positions corresponding to the fitted lines.
    - order_positions: Array of order positions corresponding to the fitted lines.
    - wave_solution: The wavelength solution array.
    - all_orders: List of all orders.
    - traces: Trace object containing pixel positions for each order.

    Returns:
    - resolution: Estimated spectral resolution (R = lambda / delta_lambda) based on the fitted lines.
    """
    # Convert FWHM from pixels to wavelength units using the wavelength solution
    _fwhms = np.zeros_like(fwhms)
    # for i, order in enumerate(np.unique(order_positions)):
    for i, order in enumerate(all_orders):
        order_wave_solution = wave_solution[i]
        order_mask = order_positions == order
        if np.sum(order_mask) == 0:
            continue
        pixel_at_zero = int(traces.y[i][0])
        _fwhms[order_mask] = fwhms[order_mask] * np.gradient(order_wave_solution)[pixel_positions[order_mask].astype(int)-pixel_at_zero]
    resolutions = wave_positions / _fwhms

    return resolutions

# def calibrate_absolute_Th(extracted_science_orders, obs_list, veloce_paths, traces, thxe_image, hdr, arm, plot=False, filename=None):
def calibrate_absolute_Th(traces, veloce_paths, obs_list, date, arm, amplifier_mode, estimate_resolution=False, plot=False, plot_filename=None, th_linelist_filename='Default'):
    """
    Function to perform full wavelength calibration using ThAr.
    If a wavelength solution file already exists for the given arm and date, it will be loaded.
    Otherwise, a new wavelength solution will be computed using Th linelist.

    Parameters:
    - traces (object): Trace object containing pixel positions for each order.
    - veloce_paths (object): Object containing paths to data directories.
    - obs_list (dict): Dictionary containing lists of observation files.
    - date (str): Date of the observation.
    - arm (str): Spectrograph arm ('blue', 'green', or 'red').
    - amplifier_mode (str): Amplifier mode used for the observations.
    - plot (bool): If True, the diagnostic plots are made.
    - plot_filename (str or None): If provided, the diagnostic plot will be saved to this file.
    - th_linelist_filename (str): Filename for the Th linelist to use.
    By 'Default', the NIST linelist will be loaded.

    Returns:
    - wave (list of numpy.ndarray): List of wavelength solutions for each order.
    - resolution (list of numpy.ndarray): List of resolution solutions for each order.
    """
    ### TODO: add header info to wavelength solution file, including params used, save fitted lines to file
    wave_solution_filename = f"arcTh_wave_{arm}_{date}.fits"
    resolution_fit_filename = f"arcTh_resolution_{arm}_{date}.fits"
    if os.path.exists(os.path.join(veloce_paths.wavelength_calibration_dir, wave_solution_filename)):
        print(f"Reading existing wavelength solution file {wave_solution_filename}")
        # wave = pickle.load(open(os.path.join(veloce_paths.wavelength_calibration_dir, wave_solution_filename), 'rb'))
        wave, _, _ = veloce_reduction_tools.load_extracted_spectrum_fits(
            os.path.join(veloce_paths.wavelength_calibration_dir, wave_solution_filename))
    else:
        print(f"Building new wavelength solution based on arc Th lines for {arm} arm on {date}")
        
        arcTh_image, hdr = get_arcTh_master(veloce_paths, arm, date, amplifier_mode, obs_list=obs_list, filename=None)

        ORDER, COEFFS, MATCH_LAM, MATCH_PIX, MATCH_LRES, GUESS_LAM, Y0 = veloce_reduction_tools.load_prefitted_wave(
            arm=arm, wave_path=veloce_paths.wave_dir)
        # static_wave = veloce_reduction_tools.calibrate_orders_to_wave(ORDER, Y0, COEFFS, traces) # vacuum
        # static_wave = [veloce_reduction_tools.vacuum_to_air(static_wave[i]) for i in range(len(static_wave))] # air
        # static_wave = load_static_Th_wavelength_solution(arm, veloce_paths, traces) # air
        static_wave, ref_arcTh, _ = veloce_reduction_tools.load_extracted_spectrum_fits(
            os.path.join(veloce_paths.wave_dir, f"arcTh_wave_{arm}_230828.fits"))
        static_wave = [order_wave[np.isfinite(order_wave)] for order_wave in static_wave]
        ref_arcTh = [th_order[np.isfinite(th_order)] for th_order in ref_arcTh]

        linelist = load_Th_linelist(veloce_paths, filename=th_linelist_filename, linelist_type='NIST')

        extracted_arcTh, extracted_arcTh_uncertainty, hdr_arcTh = veloce_reduction_tools.extract_orders_with_trace(arcTh_image, traces)

        if arm =='blue':
            nknots=13
        elif arm == 'green':
            nknots=15
        elif arm == 'red':
            nknots=17
        extracted_arcTh = [normalise_ArcTh_order_with_spline(extracted_arcTh_order, nknots=nknots) for extracted_arcTh_order in extracted_arcTh]
        ref_arcTh = [normalise_ArcTh_order_with_spline(ref_arcTh_order, nknots=nknots) for ref_arcTh_order in ref_arcTh]

        # TODO: add trace.y (pixels in dispersion) to saved fits to have an information on the extracted pixel positions???
        _, _, offset_array = calculate_offset_map(np.array(ORDER), ref_arcTh, traces.y, extracted_arcTh, traces.y, 8, mode='Th')
        offset = np.nanmedian(offset_array[abs(offset_array-np.nanmedian(offset_array))<=np.nanstd(offset_array)])
        print(f"Median offset between reference and current arcTh: {offset:.2f} [pixel].")
        if np.nanstd(offset_array) > 1.0:
            print(f"[Warning]: Large scatter of offsets found between reference and current arcTh ({np.nanstd(offset_array):.2f} [pixel]).")
        if abs(offset) < 2.0:
            offset = 0
            print("Offset is small, setting to zero.")
        else:
            offset = int(np.round(offset))
            print(f"Applying offset of {offset} pixels to initial guess positions.")

        pixel_positions, wave_positions, order_positions, fwhms = fit_all_lines_per_order(
            static_wave, extracted_arcTh, ORDER, traces, linelist, arm, offset=offset, plot=False)

        if len(np.unique(order_positions)) != len(ORDER):
            print(f"[Warning]: {len(ORDER) - len(np.unique(order_positions))} order(s) don't have fitted lines.")
            missing_orders = [order for order in ORDER if order not in np.unique(order_positions)]
            print(f"Missing orders: {missing_orders}")
            
        full_pixels = get_pixels_for_ArcTh_fit(np.unique(order_positions), traces)

        warnings.filterwarnings("ignore", category=LinAlgWarning)
        if arm == 'blue':
            Z, residuals, mask, model, converged = fit_surface_sklearn(
                pixel_positions, order_positions, wave_positions*order_positions,
                full_pixels, degree=6, sigma_clip=2.3, robust=False, n_limit=8)
        elif arm == 'green':
            Z, residuals, mask, model, converged = fit_surface_sklearn(
                pixel_positions, order_positions, wave_positions*order_positions,
                full_pixels, degree=7, sigma_clip=2.4, robust=False, n_limit=9)
            # Z, residuals, mask, model, converged = fit_surface_sklearn(
            #     pixel_positions, order_positions, wave_positions,
            #     full_pixels, degree=7, sigma_clip=2.2, robust=False, n_limit=9)
        elif arm == 'red':
            Z, residuals, mask, model, converged = fit_surface_sklearn(
                pixel_positions, order_positions, wave_positions*order_positions,
                full_pixels, degree=7, sigma_clip=2.3, robust=False, n_limit=9)
        if not converged:
            print("[Warning]: Wavelength solution fitting did not converge.")
        
        wave = get_wave_solution_from_surface(model, traces, ORDER)

        # precision = estimate_calibration_precision(residuals[mask], int(len(traces.y)/2), wave)

        if plot:
            veloce_diagnostic.plot_ArcTh_surface(Z, pixel_positions, order_positions, wave_positions, full_pixels, veloce_paths, plot_filename)
            veloce_diagnostic.plot_ArcTh_points_positions(pixel_positions, order_positions, mask, veloce_paths, plot_filename)
            veloce_diagnostic.plot_ArcTh_residuals(residuals, order_positions, pixel_positions, wave_positions, mask, veloce_paths, plot_filename, plot_type='wavelength')
        
        if estimate_resolution:
            if os.path.exists(os.path.join(veloce_paths.wavelength_calibration_dir, resolution_fit_filename)):
                print(f"There is an existing resolution fit: {resolution_fit_filename}")
            else:
                resolution_positions = get_resolution_from_fitted_lines(fwhms, wave_positions, pixel_positions, order_positions, wave, ORDER, traces)

                res_Z, res_residuals, res_mask, res_model, converged = fit_surface_sklearn(
                        pixel_positions, order_positions, resolution_positions,
                        full_pixels, degree=1, sigma_clip=2, robust=False, n_limit=8)
                if not converged:
                    print("[Warning]: Resolution solution fitting did not converge.")
                resolution = get_resolution_from_surface(res_model, traces, ORDER)
            
                veloce_reduction_tools.save_extracted_spectrum_fits(
                    os.path.join(veloce_paths.wavelength_calibration_dir, resolution_fit_filename),
                    wave,
                    resolution,
                    hdr)
        else:
            resolution = None
        # with open(os.path.join(veloce_paths.wavelength_calibration_dir, wave_solution_filename), 'wb') as f:
        #     pickle.dump(wave, f)

        veloce_reduction_tools.save_extracted_spectrum_fits(
            os.path.join(veloce_paths.wavelength_calibration_dir, wave_solution_filename),
            wave,
            extracted_arcTh,
            hdr)
        
        
    
    return wave

def get_LC_master(veloce_paths, arm, date, amplifier_mode, obs_list=None, filename=None):
    """
    Load or compute the master LC image for the given arm and date.

    Parameters:
    - veloce_paths (object): An object containing paths to the data directories.
    - arm (str): The spectrograph arm ('blue', 'green', or 'red').
    - date (str): The date of the observations.
    - amplifier_mode (int): The amplifier mode used.
    - obs_list (dict): A list of observation files.
    - filename (str): The filename of the master LC image.

    Returns:
    - LC_image: The master LC image as a numpy array.
    - hdr: The header associated with the master LC image.
    """
    if filename is not None:
        LC_master_filename = filename
    else:
        LC_master_filename = os.path.join(veloce_paths.master_dir, f"master_LC_{arm}_{date}.fits")

    if os.path.exists(LC_master_filename):
        with fits.open(LC_master_filename) as hdul:
            LC_image = hdul[0].data
            hdr = hdul[0].header
    else:
        if obs_list is not None:
            file_list = [file for file in obs_list[f'SimLC'][date] if int(file[-10]) == arm_nums[arm]]
            file_list = veloce_reduction_tools.get_longest_consecutive_files(file_list)
            if file_list:
                LC_image, hdr = veloce_reduction_tools.get_master_mmap(
                    file_list, veloce_paths.input_dir, date, arm, amplifier_mode)
                veloce_reduction_tools.save_image_fits(LC_master_filename, LC_image, hdr)
            else:
                raise FileNotFoundError(f"No SimLC files found for date {date}. Cannot create master.")
        else:
            raise ValueError("Either obs_list or filename must be provided to get the LC master.")
    return LC_image, hdr

def select_lc_lines_in_wave_range(lc_lines, wave):
    """
    Selects Laser Comb lines that fall within given wavelength range.

    Parameters:
    - lc_lines (numpy.ndarray): Array of wavelengths of the Laser Comb lines.
    - wave (list of numpy.ndarray): List of wavelength solutions for each order.

    Returns:
    - (numpy.ndarray): Array of wavelengths of the Laser Comb lines that fall within the specified range.
    """
    wave_min, wave_max = min(wave), max(wave)
    return lc_lines[(lc_lines >= wave_min) & (lc_lines <= wave_max)]

def fit_lc_peaks_in_order(wavelengths, flux, pixels, lc_lines, arm, offset=0, plot=False, verbose=False):
    ### need different conditions for each arm
    lines = select_lc_lines_in_wave_range(lc_lines, wavelengths)
    criteria_use = {'position': 0, 'height': 0, 'asymmetry': 0, 'n_points': 0}
    # peak_position_threshold = max(5.0, int(abs(offset)+1.5))
    peak_position_threshold = 3.0
    peak_position_bound = 1.5
    # peak_asymmetry_threshold = 0.2
    if arm == 'red':
        # peak_height_threshold = 20.0
        peak_height_threshold = 50.0
    elif arm == 'green':
        # peak_height_threshold = 20.0
        peak_height_threshold = 50.0
    elif arm == 'blue':
        raise ValueError("Blue arm doesn't have LC.")
    
    pixels = np.array(pixels, dtype=np.int_)
    min_pixel, max_pixel = np.min(pixels), np.max(pixels)
    lines_pixel_positions = []
    lines_wave_positions = []
    # lines_sigmas = []
    # lines_weights = []
    for iter, line_wave in enumerate(lines):
        discard = False
        idx = int(np.argmin(np.abs(wavelengths - line_wave)) + offset)
        if idx  < max((min_pixel+3), 200)  or (min_pixel + idx) > min((max_pixel - 3), 3900):
            if verbose:
                print(f"Line {line_wave:.3f} nm at index {idx} is too close to ithe edge of the order.")
            # lines_pixel_positions.append(np.nan)
            # lines_wave_positions.append(np.nan)
            # lines_sigmas.append(np.nan)
            # lines_weights.append(np.nan)
            continue
            # discard = True
            # criteria_use['position'] += 1
        line_pixel = min_pixel + idx + (line_wave - wavelengths[idx])/(wavelengths[idx+1] - wavelengths[idx]) \
            if line_wave - wavelengths[idx] > 0 \
            else min_pixel + idx + (line_wave - wavelengths[idx])/(wavelengths[idx] - wavelengths[idx-1])
        # line_pixel+=offset

        # fit_range = slice(max(0, int(idx-peak_position_threshold+offset)), min(len(pixels), int(idx+peak_position_threshold+1+offset)))
        fit_range = slice(max(0, int(idx-peak_position_threshold)), min(len(pixels)-1, int(idx+peak_position_threshold+1)))
        x_fit = pixels[fit_range]
        y_fit = flux[fit_range]

        # peaks, _ = find_peaks(y_fit, prominence=peak_height_threshold/2)
        peaks, _ = find_peaks(y_fit, prominence=peak_height_threshold)
        if len(peaks) == 0:
            if verbose:
                print(f"No peaks found for line {line_wave:.3f} nm.")
            # lines_pixel_positions.append(np.nan)
            # lines_wave_positions.append(np.nan)
            # lines_sigmas.append(np.nan)
            # lines_weights.append(np.nan)
            continue
            # discard = True
        local_peak_idx = peaks[np.argmin(np.abs(x_fit[peaks] - line_pixel))]
        peak_x = x_fit[local_peak_idx]
        peak_y = y_fit[local_peak_idx]
        # update fitting range to be around the found peak
        peak_idx = int(peak_x-min_pixel)
        
        fit_range = slice(max(0, int(peak_idx-peak_position_threshold)), min(len(pixels), int(peak_idx+peak_position_threshold+1)))
        x_fit = pixels[fit_range]
        y_fit = flux[fit_range]
        # print(line_pixel, peak_x, peak_idx, x_fit.min(), x_fit.max())
        
        if abs(peak_x-line_pixel) > peak_position_threshold:
            if verbose:
                print(f"Peak {peak_x} for line {line_wave:.3f} nm is too far from the guess pixel {line_pixel:.2f}.")
            # if line_wave - wavelengths[idx] > 0:
            #     print((line_wave - wavelengths[idx])/(wavelengths[idx+1] - wavelengths[idx]))
            # else:
            #     print((line_wave - wavelengths[idx])/(wavelengths[idx] - wavelengths[idx-1]))
            # lines_pixel_positions.append(np.nan)
            # lines_wave_positions.append(np.nan)
            # lines_sigmas.append(np.nan)
            # lines_weights.append(np.nan)
            continue
            # discard = True
            # criteria_use['position'] += 1

        if peak_y < peak_height_threshold:
            if verbose:
                print(f"Peak height for line {line_wave:.3f} nm is too low: {peak_y:.2f}.")
            # lines_pixel_positions.append(np.nan)
            # lines_wave_positions.append(np.nan)
            # lines_sigmas.append(np.nan)
            # lines_weights.append(np.nan)
            continue
            # discard = True
            # criteria_use['height'] += 1
        
        # get an area between 1 and y_fit points:
         
        # print(f"Fitting line {line_wave:.3f} nm at pixel {line_pixel:.2f}, peak height {peak_height:.2f}, floor {line_floor:.2f}")

        # p0 = [peak_y, peak_x, 2, 2, 1] # general gaussian
        p0 = [peak_y, peak_x, 2, 1] # simple gaussian
        # bounds = ([0.8*peak_y, peak_x-peak_position_bound, 1e-3, 1e-3, 0], [5*peak_y, peak_x+peak_position_bound, 5, 5, 0.5*peak_y]) # general gaussian
        bounds = ([0.8*peak_y, peak_x-peak_position_bound, 1e-3, 0], [2.5*peak_y, peak_x+peak_position_bound, 5, 0.5*peak_y]) # simple gaussian
        
        try:
            # popt, pcov = curve_fit(veloce_wavecalib.general_gaussian, x_fit, y_fit, p0=p0, bounds=bounds)
            popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=p0, bounds=bounds)
            # Calculate uncertainty for popt[1] (the fitted peak position)
            perr = np.sqrt(np.diag(pcov))
            uncertainty = perr[1]
            # if iter == plot:
            if verbose:
                print(f"Fitted line {line_wave:.3f} nm at pixel {popt[1]:.2f} +/- {uncertainty:.3f}")
                # print(f"Parameters of the fit:")
                # print(f"Amplitude={popt[0]:.2f},\nCenter={popt[1]:.2f},\nsigma={popt[2]:.2f},\nBeta={popt[3]:.2f},\nbaseline={popt[4]:.2f}")
            # print(popt)
            if plot:
                plt.close('all')
                plt.plot(x_fit, y_fit, 'b-', label='Data')
                plt.scatter(x_fit, y_fit, c='k', s=5, label='Line points')
                x_fine = np.arange(x_fit.min(), x_fit.max()+0.1, 0.1)
                # plt.plot(x_fine, veloce_wavecalib.general_gaussian(x_fine, *popt), 'r-', label='Fit')
                plt.plot(x_fine, gaussian(x_fine, *popt), 'r-', label='Fit')
                plt.axvline(line_pixel, c='orange', ls=':', label='Line guess')
                plt.axvline(peak_x, c='green', ls=':', label='Closest peak')
                plt.axvline(popt[1], c='red', ls=':', label='Fit center')
                plt.title(f"Line {line_wave:.3f} nm")
                plt.xlim(x_fit.min(), x_fit.max())
                # plt.ylim(min(0.7, y_fit.min()*0.9), veloce_wavecalib.general_gaussian(x_fine, *popt).max()*1.1)
                # plt.ylim(min(0.7, y_fit.min()*0.9), gaussian(x_fine, *popt).max()*1.1)
                plt.xlabel('Pixel')
                plt.ylabel('Flux')
                plt.legend()
                plt.show()
            lines_pixel_positions.append(popt[1])
            lines_wave_positions.append(line_wave)
            # lines_sigmas.append(popt[2])
            # print(1/np.sum(y_fit - popt[-1]), uncertainty**2)
            # if verbose:
            #     print(f"Weight: {1/np.sqrt(1/np.sum(y_fit - popt[-1])+uncertainty**2)}")
            # lines_weights.append(1/np.sqrt(1/np.sum(y_fit - popt[-1])+uncertainty**2))
        except Exception as e:
            lines_pixel_positions.append(np.nan)
            lines_wave_positions.append(np.nan)
            # lines_weights.append(np.nan)
            # lines_sigmas.append(np.nan)
            if verbose:
                print(f"Failed to fit line {line_wave:.3f} nm: {e}")
            continue
    lines_wave_positions = np.array(lines_wave_positions)
    lines_pixel_positions = np.array(lines_pixel_positions)
    # lines_weights = np.array(lines_weights)
    # lines_sigmas = np.array(lines_sigmas)

    return lines_pixel_positions, lines_wave_positions

def remove_lc_background(y, plot=False):
    ### takes lc spectrum, finds background by flipping it and applying find_peaks, fits a spline to the background peaks and subtracts it
    pixel = np.arange(len(y))
    y_flipped = np.max(y) - y
    peaks, _ = find_peaks(y_flipped)
    # fit spline to background peaks
    try:
        spline = make_interp_spline(peaks, y[peaks], k=3)
        background = spline(pixel)
        y_corrected = y - background
    except Exception as e:
        print(f"Failed to fit spline to background because: {e}. Returning original spectrum without background subtraction.")
        return y
    
    if plot:
        plt.close('all')
        plt.figure(figsize=(10,6))
        plt.subplot(2,1,1)
        plt.plot(pixel, y, label='Original spectrum')
        plt.scatter(peaks, y[peaks], color='red', label='Background peaks')
        plt.plot(pixel, background, label='Fitted background')
        plt.subplot(2,1,2)
        plt.plot(pixel, y_corrected, label='Background subtracted spectrum')
        plt.scatter(peaks, y_corrected[peaks], color='red', label='Background peaks')
        plt.legend()
        plt.show()
    return y_corrected

def fit_all_lc_lines_per_order(wave, extracted_spectrum, ORDER, traces, lc_lines, arm, offset=0):
    pixel_positions, wave_positions, order_positions = [], [], []
    for order, absolute_order in enumerate(ORDER):
        # print(f"Fitting lines in order {absolute_order} ({order+1}/{len(ORDER)})")
        lines_pixel_positions, lines_wave_positions  = fit_lc_peaks_in_order(
            wave[order],
            remove_lc_background(extracted_spectrum[order]),
            traces.y[order],
            lc_lines,
            arm,
            offset=offset)
        lines_order_positions = np.ones_like(lines_pixel_positions) * absolute_order
        pixel_positions.append(lines_pixel_positions)
        order_positions.append(lines_order_positions)
        wave_positions.append(lines_wave_positions)
        # fwhms.append(lines_sigmas*2.355)
        # weights.append(lines_weights)
        # fitted_lines.append(_fitted_lines)
        # print(f"Found {np.sum(np.isfinite(lines_pixel_positions))} lines.")
        # print("---")
    pixel_positions = np.concatenate(pixel_positions)
    wave_positions = np.concatenate(wave_positions)
    order_positions = np.concatenate(order_positions)
    # fwhms = np.concatenate(fwhms)
    # weights = np.concatenate(weights)
    # fitted_lines = np.concatenate(fitted_lines)

    return pixel_positions, wave_positions, order_positions

def get_orders_with_lc_lines(ORDER, order_positions, pixel_positions):
    # unique_orders = np.unique(order_positions[np.isfinite(order_positions)])
    orders_with_lines = [order for order in ORDER if np.sum(np.isfinite(pixel_positions[order_positions==order]))>50]
    # orders_with_lines = [order for order in ORDER if order in unique_orders]
    mask = np.array([order in orders_with_lines for order in order_positions]) & np.isfinite(pixel_positions)
    return orders_with_lines, mask

def _scale_X_and_grid(X_raw, grid_points, method='none'):
    """
    Scale input design matrix X and grid points for prediction.

    method: 'standard'|'minmax'|'robust'|None
    """
    if method is None or method == 'none':
        scaler = None
        X = X_raw.copy()
        grid_scaled = grid_points.copy()
    else:
        if method == 'minmax':
            scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        X = scaler.fit_transform(X_raw)
        grid_scaled = scaler.transform(grid_points)
    return scaler, X, grid_scaled

def monomial_powers_2d(degree_x, degree_y):
    """
    Generate list of (i,j) powers for 2D monomials up to degree_x in x and degree_y in y.
    """
    powers = [(i, j) for i in range(degree_x + 1) for j in range(degree_y + 1) if i + j <= max(degree_x, degree_y)]
    return powers

# def monomial_powers_2d(degree):
#     return [(i, j) for i in range(degree + 1) for j in range(degree - i + 1)]

def build_design_matrix(x, y, powers):
    return np.stack([x**i * y**j for i, j in powers], axis=-1)

class PolyXYFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, degree_x=9, degree_y=5):
        self.degree_x = degree_x
        self.degree_y = degree_y
        self.powers_ = monomial_powers_2d(degree_x, degree_y)

    def fit(self, X, y=None):
        X = check_array(X)
        if X.shape[1] != 2:
            raise ValueError("Input must have exactly two columns: x and y.")
        return self
    
    def transform(self, X):
        X = check_array(X)
        x, y = X[:, 0], X[:, 1]
        return build_design_matrix(x, y, self.powers_)

def fit_xy_surface_sklearn(dispersion_position, orders_position, wave_array, extracted_pixels,
                           degree_x=7, degree_y=5, sigma_clip=2.5, max_iter=1000, ridge_alpha=0.01,
                           input_scaling='minmax', target_scaling='standard', fit_type='wavelength'):
    """
    Simplified bivariate polynomial surface fit WITH optional target scaling.

    Fits z = wave_array * order as a 2D polynomial in (pixel, order) with iterative
    sigma-clipping to remove outliers. Returns (Z_grid (nm), residuals (nm), inlier_mask, model, converged).
    target_scaling: 'standard'|'minmax'|'robust'|None
    """
    # Prepare data
    x = np.asarray(dispersion_position, dtype=np.float64)
    orders = np.asarray(orders_position, dtype=np.float64)
    if fit_type == 'wavelength':
        z_fit = np.asarray(wave_array * orders, dtype=np.float64)  # original units (nm * order)
    elif fit_type == 'resolution':
        z_fit = np.asarray(wave_array, dtype=np.float64)
    else:
        raise ValueError(f"Unknown fit_type: {fit_type}")

    # Grid for prediction (expects extracted_pixels shaped (n_orders, n_pix_each))
    grid_points = np.vstack([
        (pixel, abs_order)
        for i, abs_order in enumerate(np.unique(orders))
        for pixel in extracted_pixels[i]
    ]).astype(np.float64)

    # Design matrix input
    X_raw = np.column_stack([x, orders]).astype(np.float64)
    finite = np.isfinite(z_fit)
    if not np.any(finite):
        raise ValueError("No finite target values to fit.")
    
    # scale inputs
    scaler, X, grid_points = _scale_X_and_grid(X_raw, grid_points, method=input_scaling)

    # scale target (z)
    if target_scaling is None or target_scaling == 'none':
        z = z_fit.copy()
        z_scaler = None
    else:
        if target_scaling == 'minmax':
            z_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        elif target_scaling == 'robust':
            z_scaler = RobustScaler()
        else:
            z_scaler = StandardScaler()
        z = z_scaler.fit_transform(z_fit.reshape(-1, 1)).ravel()

    # Pipeline: polynomial features -> ridge regression
    model = Pipeline([
        ('poly', PolyXYFeatures(degree_x=degree_x, degree_y=degree_y)),
        ('ridge', Ridge(alpha=ridge_alpha, fit_intercept=True))
    ])

    # Initial fit using all finite points
    mask = finite.copy()
    powers = monomial_powers_2d(degree_x, degree_y)
    n_params = len(powers)

    converged = False
    init_resid = None

    for iteration in range(max_iter):
        # ensure system is not under-determined
        if np.sum(mask) <= 5 * n_params:
            print(f"[Warning]: Not enough points to fit the model robustly. Stopping iteration {iteration+1}.")
            break

        model.fit(X[mask], z[mask])
        pred_scaled = model.predict(X)
        residuals_scaled = z - pred_scaled

        if init_resid is None:
            init_resid = np.std(residuals_scaled[mask])

        std_dev = np.std(residuals_scaled[mask]) if np.sum(mask) > 1 else init_resid
        if not np.isfinite(std_dev) or std_dev == 0:
            converged = True
            print(f"[Warning]: Standard deviation of residuals is non-finite or zero. Stopping iteration {iteration+1}.")
            break

        new_mask = np.abs(residuals_scaled) < (sigma_clip * std_dev)

        if np.array_equal(mask, new_mask):
            converged = True
            print(f"Converged after {iteration+1} iterations.")
            break

        mask = new_mask

    # final predictions (convert back to original z units)
    final_pred_scaled = model.predict(X)
    if z_scaler is not None:
        final_pred = z_scaler.inverse_transform(final_pred_scaled.reshape(-1, 1)).ravel()
    else:
        final_pred = final_pred_scaled
    residuals = z_fit - final_pred

    # predict on scaled grid and reshape to original extracted_pixels layout (inverse-transform)
    grid_pred_scaled = model.predict(grid_points)
    if z_scaler is not None:
        Z = z_scaler.inverse_transform(grid_pred_scaled.reshape(-1, 1)).ravel().reshape(extracted_pixels.shape)
    else:
        Z = grid_pred_scaled.reshape(extracted_pixels.shape)

    return Z, residuals, mask, model, converged, scaler, z_scaler

def predict_lc_wavelength_from_surface(model, y_traces, orders, input_scaler=None, target_scaler=None):
    """
    Predict wavelength (same units as original wave) from a fitted surface model.

    model: fitted sklearn estimator mapping (pixel, order) -> z where z = wavelength * order
    x: array-like dispersion positions (pixels)
    orders: array-like orders (same shape as x)
    input_scaler: scaler used to transform X before fitting (e.g. returned by _scale_X_and_grid). If None, X is passed raw.
    z_scaler: scaler used on target z during fitting (if any). If provided, inverse_transform will be applied.

    Returns: numpy array of predicted wavelengths (same shape as x)
    """
    wavelengths = []
    for trace_y, order in zip(y_traces, orders):
        order_arr = np.ones_like(trace_y)*order

        X = np.column_stack([trace_y, order_arr])

        if input_scaler is not None:
            X_in = input_scaler.transform(X)
        else:
            X_in = X

        z_pred = np.asarray(model.predict(X_in)).ravel()

        if target_scaler is not None:
            # supports scikit-learn scalers
            z_pred = np.asarray(target_scaler.inverse_transform(z_pred.reshape(-1, 1))).ravel()

        wavelengths.append(z_pred / order_arr)
    return wavelengths

def get_calibrated_pixels(order_list, traces, pixel_positions, order_positions, fitted_orders):
    """
    Trims the edges of the orders based on the fitted LC line positions to get the pixels that are reliably calibrated by the LC lines in each order.

    Parameters:
    - order_list (list): List of all orders.
    - traces (Traces): The traces for the Laser Comb.
    - pixel_positions (numpy.ndarray): The pixel positions of the fitted LC lines.
    - order_positions (numpy.ndarray): The corresponding order positions of the fitted LC lines.
    - fitted_orders (list): List of orders that have fitted LC lines.
    
    Returns:
    - calibrated_pixels_per_order (list of numpy.ndarray): A list where each element is a boolean array. True is calibrated, False - discard.
    """
    calibrated_pixels_per_order = []
    for i, order in enumerate(order_list):
        if order in fitted_orders:
            pixel_in_order = pixel_positions[order_positions == order]
            for j, pixel in enumerate(pixel_in_order):
                if abs(pixel_in_order[j+1] - pixel) > 50:
                    pixel_in_order[j] = np.nan
                else:
                    break
            for j, pixel in enumerate(pixel_in_order[::-1]):
                if abs(pixel_in_order[len(pixel_in_order)-2-j] - pixel) > 50:
                    pixel_in_order[len(pixel_in_order)-1-j] = np.nan
                else:
                    break
            
            calibrated_pixels = (traces.y[i] > np.nanmin(pixel_in_order)) & (traces.y[i] < np.nanmax(pixel_in_order))
            # print(len(calibrated_pixels), len(traces.y[i]), order)
            calibrated_pixels_per_order.append(calibrated_pixels)
        else:
            calibrated_pixels = np.zeros_like(traces.y[i], dtype=bool)
            calibrated_pixels_per_order.append(calibrated_pixels)

    return calibrated_pixels_per_order

def build_LC_wavelength_solution(traces, veloce_paths, date, arm, amplifier_mode, obs_list, estimate_resolution=False, plot=False, filename=None):
    """
    Builds the new wavelength solution for the Laser Comb.

    Parameters:
    - traces (Traces): The traces for the Laser Comb.
    - veloce_paths (object): An object containing paths to the data directories.
    - date (str): The date of the observations.
    - arm (str): The spectrograph arm ('blue', 'green', or 'red').
    - amplifier_mode (int): The amplifier mode used.
    - obs_list (dict): A list of observation files.
    - estimate_resolution (bool): If True, estimate the resolution (but from arcTh).
    - plot (bool): If True, the diagnostic plots are made.

    Returns:
    - wave (numpy.ndarray): The wavelength solution.
    """
    ### TODO: finish this function, line fitting, surface model, predict waves, save to file, plot diagnostics
    wave_solution_filename = f"LC_wave_{arm}_{date}.fits"
    if os.path.exists(os.path.join(veloce_paths.wavelength_calibration_dir, wave_solution_filename)):
        print(f"Reading existing wavelength solution file {wave_solution_filename}")
        wave, _, _ = veloce_reduction_tools.load_extracted_spectrum_fits(
            os.path.join(veloce_paths.wavelength_calibration_dir, wave_solution_filename))
    else:

        lc_image, header = get_LC_master(veloce_paths, arm, date, amplifier_mode, obs_list=obs_list, filename=None)
        print(f"Building new wavelength solution for LC based on {arm} arm Th wavelength solution on {date}")
        lc_traces = veloce_reduction_tools.Traces.load_traces(os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_LC_trace.pkl'))
        ref_traces = veloce_reduction_tools.Traces.load_traces(os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_4amp_sim_calib_trace.pkl'))
        offsets = [np.nanmean(ref_x-x) for ref_x, x in zip(ref_traces.x, traces.x)]
        if np.any(np.abs(np.array(offsets)) > 1.0):
            new_traces_x = [x+offset for x, offset in zip(lc_traces.x, offsets)]
            lc_traces.set_traces_yx(lc_traces.y, new_traces_x)
        extracted_LC, extracted_LC_uncertainty, extracted_LC_imgs = veloce_reduction_tools.extract_orders_with_trace(lc_image, lc_traces)

        ORDER, COEFFS, MATCH_LAM, MATCH_PIX, MATCH_LRES, GUESS_LAM, Y0 = veloce_reduction_tools.load_prefitted_wave(arm=arm, wave_path=veloce_paths.wave_dir)
        arcTh_wave = calibrate_absolute_Th(traces, veloce_paths, obs_list,
                                date, arm, amplifier_mode, estimate_resolution=estimate_resolution,
                                plot=plot, plot_filename=f'arcTh_wavecalib_{arm}_{date}',
                                th_linelist_filename='Default', )
        vacuum_static_wave = [veloce_reduction_tools.air_to_vacuum(th_order[np.isfinite(th_order)]) for th_order in arcTh_wave]

        if header is not None and (header['FREQREF'] != REPETITION_RATE and header['FOFFFREQ'] != OFFSET_FREQUENCY):
            raise ValueError("Repetition rate and offset frequency do not match the values of LC solution.")
        freq_start = SPEED_OF_LIGHT/(950e-9)
        freq_end =  SPEED_OF_LIGHT/(450e-9)
        n = np.arange(np.floor((freq_start - OFFSET_FREQUENCY)/REPETITION_RATE), np.ceil((freq_end - OFFSET_FREQUENCY)/REPETITION_RATE), 1)
        lc_lines = SPEED_OF_LIGHT/(OFFSET_FREQUENCY+n*REPETITION_RATE)*1e9

        _pixel_positions, _wave_positions, _order_positions = fit_all_lc_lines_per_order(vacuum_static_wave, extracted_LC, ORDER, traces, lc_lines, arm, offset=0)

        fitted_orders, order_mask = get_orders_with_lc_lines(ORDER, _order_positions, _pixel_positions)

        pixel_positions, wave_positions, order_positions = _pixel_positions[order_mask], _wave_positions[order_mask], _order_positions[order_mask]

        max_extracted_pixel = max([max(trace_y) for trace_y in traces.y])
        min_extracted_pixel = min([min(trace_y) for trace_y in traces.y])
        full_pixels = np.array([np.arange(min_extracted_pixel, max_extracted_pixel + 1) for _ in range(len(fitted_orders))])

        if arm == 'green':
            Z, residuals, mask, model, converged, input_scaler, target_scaler = fit_xy_surface_sklearn(pixel_positions, order_positions, wave_positions, full_pixels,
                                degree_x=13, degree_y=8, sigma_clip=3, max_iter=1000, ridge_alpha=0.001,
                                input_scaling='minmax', target_scaling='standard')
        elif arm == 'red':
            Z, residuals, mask, model, converged, input_scaler, target_scaler = fit_xy_surface_sklearn(pixel_positions, order_positions, wave_positions, full_pixels,
                                degree_x=9, degree_y=7, sigma_clip=3, max_iter=1000, ridge_alpha=0.001,
                                input_scaling='minmax', target_scaling='standard')
        stdev_vel = np.std(residuals[mask]/order_positions[mask]/wave_positions[mask])*c/np.sqrt(np.sum(mask))
        print(f"SE: {stdev_vel:.2f}, with mean deviation from model: {np.mean(residuals[mask]/order_positions[mask]/wave_positions[mask]):.2e}, based on {np.sum(mask)}/{len(mask)} points.")
    
        calibrated_pixels_per_order = get_calibrated_pixels(ORDER, traces, pixel_positions, order_positions, fitted_orders)
        # order_mask = [i for i, order in enumerate(ORDER) if order in fitted_orders]
        fitted_traces = [traces.y[i][calibrated_pixels] for i, calibrated_pixels in enumerate(calibrated_pixels_per_order) if np.any(calibrated_pixels)]
        wave = predict_lc_wavelength_from_surface(model, fitted_traces, fitted_orders, input_scaler=input_scaler, target_scaler=target_scaler)
        # wave = [veloce_reduction_tools.vacuum_to_air(w) for w in wave]
        # fitted_extracted_LC = [extracted_LC[i] for i, order in enumerate(ORDER) if order in fitted_orders]
        fitted_extracted_LC = [extracted_LC[i][calibrated_pixels] for i, calibrated_pixels in enumerate(calibrated_pixels_per_order) if np.any(calibrated_pixels)]
        fitted_extracted_LC_uncertainty = [extracted_LC_uncertainty[i][calibrated_pixels] for i, calibrated_pixels in enumerate(calibrated_pixels_per_order) if np.any(calibrated_pixels)]

        shifts = np.load(os.path.join(veloce_paths.wave_dir, f'{arm}_velocity_orders_offsets.npy'))
        shifts = [shifts[i] for i, order in enumerate(ORDER) if order in fitted_orders]
        wave = [w * (1 + shift/SPEED_OF_LIGHT) for w, shift in zip(wave, shifts)]

        print(f"Saving new LC wavelength solution to {os.path.join(veloce_paths.wavelength_calibration_dir, wave_solution_filename)}")
        veloce_reduction_tools.save_extracted_spectrum_fits(os.path.join(veloce_paths.wavelength_calibration_dir, wave_solution_filename), wave, fitted_extracted_LC, header, err=fitted_extracted_LC_uncertainty)
    
    return wave, calibrated_pixels_per_order #, extracted_LC

def calibrate_simTh():
    raise NotImplementedError

def add_predetermined_resolution(arm, date, veloce_paths):
    """
    Adds a predetermined resolution fit for the given arm if it doesn't already exist.

    Parameters:
    - arm (str): The spectrograph arm ('blue', 'green', or 'red').
    - date (str): The date of the observations.
    - veloce_paths (object): An object containing paths to the data directories.
    """
    resolution_fit_filename = f"arcTh_resolution_{arm}_{date}.fits"

    if os.path.exists(os.path.join(veloce_paths.wavelength_calibration_dir, resolution_fit_filename)):
        print(f"There is an existing resolution fit: {resolution_fit_filename}")
        return os.path.join(veloce_paths.wavelength_calibration_dir, resolution_fit_filename)
    elif not os.path.exists(os.path.join(veloce_paths.wavelength_calibration_dir, f"arcTh_resolution_{arm}_230828.fits")):
        # copy static resolution fit
        wave, resolution, hdr = veloce_reduction_tools.load_extracted_spectrum_fits(
        os.path.join(veloce_paths.wave_dir, f"arcTh_resolution_{arm}_230828.fits"))
        print(f"Copying predetermined resolution fit: arcTh_resolution_{arm}_230828.fits")
        veloce_reduction_tools.save_extracted_spectrum_fits(
            os.path.join(veloce_paths.wavelength_calibration_dir, f"arcTh_resolution_{arm}_230828.fits"),
            wave,
            resolution,
            hdr)
        return os.path.join(veloce_paths.wavelength_calibration_dir, f"arcTh_resolution_{arm}_230828.fits")
    else:
        return os.path.join(veloce_paths.wavelength_calibration_dir, f"arcTh_resolution_{arm}_230828.fits")