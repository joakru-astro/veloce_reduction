from astropy.io import fits
from astropy.constants import c
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from scipy import signal
# from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.linalg import lstsq
from scipy.ndimage import median_filter, minimum_filter, gaussian_filter1d
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks
# from csaps import csaps

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RANSACRegressor

import warnings
from scipy.linalg import LinAlgWarning

from . import veloce_reduction_tools, veloce_diagnostic

arm_nums = {'red': 3, 'green': 2, 'blue': 1}
REPETITION_RATE = 25e9  # Hz
OFFSET_FREQUENCY = 9.56e9  # Hz

def pad_array(array, ref_pixel):
    """
    Pad an array to create 2D array with the size matching min and max of reference pixels.
    """
    lower_bound = min([np.nanmin(order) for order in ref_pixel])
    upper_bound = max([np.nanmax(order) for order in ref_pixel])
    # print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
    padded_array = np.array(
        [np.pad(order, (int(np.nanmin(ref_pixel[i])-lower_bound), int(upper_bound-np.nanmax(ref_pixel[i]))), constant_values=np.nan)
         for i, order in enumerate(array)])

    return padded_array

def load_LC_wave_reference(veloce_paths, arm):
    """
    Load the wavelength calibration for the LC.
    """
    lc_wave_calib_file = os.path.join(veloce_paths.wave_dir, f'{arm.upper()}_LC_SPEC-26aug{arm_nums[arm]}0083.txt')

    if not os.path.exists(lc_wave_calib_file):
        raise FileNotFoundError(f"LC wave calibration file not found: {lc_wave_calib_file}")
    
    dtype=[('wave', float), ('flux', float), ('pixel', float), ('order', int)]
    lc_wave_calib = np.loadtxt(lc_wave_calib_file, dtype=dtype)
    # remove NaN values
    lc_wave_calib = np.array([v for v in lc_wave_calib if v == v], dtype=dtype)

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
    """
    if hdr is not None and (hdr['FREQREF'] != REPETITION_RATE and hdr['FOFFFREQ'] != OFFSET_FREQUENCY):
        raise ValueError("Repetition rate and offset frequency do not match the values of LC solution.")
    if traces is None:
        traces = veloce_reduction_tools.Traces.load_traces(os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_LC_trace.pkl'))
    
    extracted_LC, extracted_LC_imgs = veloce_reduction_tools.extract_orders_with_trace(image, traces)

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
    return A * np.exp(-np.abs(((x - mu)/(np.sqrt(2)*sigma)))**beta) + baseline

def fit_lc_peak(pix_shift, ccf, fitting_limit=None):
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
        print(f"[Warning] LC peak fitting failed: {e}")
        return np.nan, [np.nan], np.nan #slice(0,None)

def calculate_offset_map(ref_orders, ref_intensity, ref_pixel, lc_intensity, lc_pixel, number_of_parts=8, mode='LC', plot=False, veloce_paths=None, filename=None):
    """
    Calculate the cross-correlation function (CCF) for each order of the laser comb.
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
    """
    # offset pixels
    new_pixels = extracted_pixels - offsets
    # find wavelengths of the observation
    new_wave = np.array([np.interp(new_pix, ref_pix, ref_w) for new_pix, ref_pix, ref_w in zip(new_pixels, ref_pixel, ref_wave)])
    
    return new_wave

def estimate_calibration_precision(residuals, order, ref_wave):
    """
    Estimate the calibration precision.
    """
    # c = 2.99792458e8  # Speed of light in m/s
    # Calculate the standard deviation of the residuals
    n_points = len(residuals)
    std_dev = np.std(residuals)
    average_step = np.nanmean(np.diff(ref_wave[order-1]))
    average_wave = np.nanmean(ref_wave[order-1])
    
    # Calculate the calibration precision
    calibration_precision = std_dev / np.sqrt(n_points) * average_step / average_wave * c.value
    # calibration_precision = std_dev * average_step / average_wave * c.value

    print(f"Calibration Precision estimated at {average_wave:.0f}nm: {calibration_precision:.0f} m/s")
    
    return calibration_precision

def apply_wavelength_shift(wave, arm, veloce_paths):
    # Apply the wavelength shift to the spectrum
    shifts = np.load(os.path.join(veloce_paths.wave_dir, f'{arm}_velocity_orders_offsets.npy'))
    if len(wave) != len(shifts):
        raise ValueError(f"Number of orders in wave ({len(wave)}) does not match number of predetermined offsets ({len(shifts)})")
    for i, v in enumerate(shifts):
        # Calculate the convertion factor
        # v is in km/s, c is in m/s, convert will be in nm
        convert = 1 - 1000*v / c.value
        # Shift the spectrum order
        wave[i] *= convert
    return wave

def calibrate_simLC(extracted_science_orders, veloce_paths, lc_image, hdr, arm, traces=None, plot=False, filename=None):
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
    
    # apply shift between calibration fiber and science fibers expressed as rv
    # wave = apply_wavelength_shift(wave, arm, veloce_paths)

    return wave,  extracted_science_orders

def load_wave_calibration_for_interpolation():
    raise NotImplementedError

def interpolate_wave(orders, hdr):
    raise NotImplementedError

def load_static_Th_wavelength_solution(arm, veloce_paths, traces):
    wave = pickle.load(open(os.path.join(veloce_paths.wave_dir, f'ThXe_wave_230826_{arm}.pkl'), 'rb'))
    for w, trace_y in zip(wave, traces.y):
        assert len(w) == len(trace_y), "Size missmatch between used trace and static wavelength solution."
    return wave

def load_reference_Th_spectrum(arm, veloce_paths):
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
#             obs_list, f"flat_{arm}", veloce_paths.input_dir,
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
    - baseline: Normalized flux values after spline fitting.
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
    Wavelengths in linelist should be in air.
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

def plot_order_with_lines(wave, thxe_order, linelist, original_solution=None):
    plt.close('all')

    plt.plot(wave, thxe_order, label='arc ThXe')

    lines = get_lines_in_order(wave, linelist, intensity_threshold=100)
    # lines = get_lines_in_order(veloce_reduction_tools.vacuum_to_air(wave[order]), nist_linelist, elements=['Th', 'Xe'], intensity_threshold=100)
    for line in lines:
        plt.axvline(line['obs_wl_air(nm)'], c='r', ls='--', label="linelist")

    if original_solution is not None:
        MATCH_LAM, GUESS_LAM = original_solution
        for match_wave in MATCH_LAM:
            plt.axvline(veloce_reduction_tools.vacuum_to_air(match_wave), c='g', ls=':', label="match_wave")
        for guess_wave in GUESS_LAM:
            plt.axvline(veloce_reduction_tools.vacuum_to_air(guess_wave), c='b', ls=':', label="guess_wave")
    # Remove duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict()
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h
    plt.legend(unique.values(), unique.keys())
    plt.show()

def fit_lines_in_order(wavelengths, flux, pixels, linelist, arm, offset=0, plot=False):
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

        p0 = [peak_height, center, 2, 2, line_floor]
        bounds = ([0.9*peak_height, center-3, 1e-3, 1e-3, 0], [2*peak_height, center+3, 10, 10, np.max(y_fit)])
        try:
            popt, _ = curve_fit(general_gaussian, x_fit_masked, y_fit_masked, p0=p0, bounds=bounds)
            if plot:
                plt.close('all')
                plt.plot(x_fit, y_fit, 'b-', label='Data')
                plt.scatter(x_fit_masked, y_fit_masked, c='k', s=5, label='Line points')
                x_fine = np.arange(x_fit_masked.min(), x_fit_masked.max()+0.1, 0.1)
                plt.plot(x_fine, general_gaussian(x_fine, *popt), 'r-', label='Fit')
                plt.axvline(line_pixel, c='orange', ls=':', label='Line guess')
                plt.axvline(center, c='green', ls=':', label='Closest peak')
                plt.axvline(popt[1], c='red', ls=':', label='Fit center')
                plt.title(f"Line {line_wave:.3f} nm")
                plt.xlim(x_fit.min(), x_fit.max())
                plt.ylim(min(0.7, y_fit.min()*0.9), general_gaussian(x_fine, *popt).max()*1.1)
                plt.xlabel('Pixel')
                plt.ylabel('Flux')
                plt.legend()
                plt.show()
            passed_mask.append(True)
            lines_pixel_positions.append(popt[1])
            lines_wave_positions.append(line_wave)
        except Exception as e:
            # print(f"Fit failed for line {line_wave:.3f} nm: {e}")
            passed_mask.append(False)
            # lines_pixel_positions.append(np.nan)
            # lines_wave_positions.append(np.nan)
    # Remove duplicate pixel positions and corresponding wavelengths
    lines_wave_positions = np.array(lines_wave_positions)
    lines_pixel_positions = np.array(lines_pixel_positions)
    _, unique_indices = np.unique(lines_pixel_positions, return_index=True)
    lines_wave_positions = lines_wave_positions[unique_indices]
    lines_pixel_positions = lines_pixel_positions[unique_indices]
    # print(f"Found {len(np.unique(lines_pixel_positions[~unique_indices]))} blends (two or more wavelengths corresponding to single peak).")
    passed_mask = np.array(passed_mask)
    # passed_mask[~unique_indices] = False  # Mark duplicates as not passed
    # Filter out lines that did not pass the selection criteria
    # print(f"Total lines passed: {np.sum(passed_mask)} out of {len(lines)}")
    lines = np.array(lines)
    lines = lines[passed_mask]
    # # Keep only unique pixel positions and corresponding wavelengths
    # lines_wave_positions = lines_wave_positions[passed_mask]
    # lines_pixel_positions = lines_pixel_positions[passed_mask]
    # print(f"Total lines fitted: {len(lines_pixel_positions)}")
    
    return lines_pixel_positions, lines_wave_positions, lines

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
    pixel_positions, wave_positions, order_positions = [], [], []
    # fitted_lines = []
    
    for order, absolute_order in enumerate(ORDER):
        # print(f"Fitting lines in order {absolute_order} ({order+1}/{len(ORDER)})")
        lines_pixel_positions, lines_wave_positions, _fitted_lines = fit_lines_in_order(
            wave[order],
            norm_extracted_Th[order],
            traces.y[order],
            linelist, arm, offset=offset)
        lines_order_positions = np.ones_like(lines_pixel_positions) * absolute_order
        pixel_positions.append(lines_pixel_positions)
        wave_positions.append(lines_wave_positions)
        order_positions.append(lines_order_positions)
        # fitted_lines.append(_fitted_lines)
    pixel_positions = np.concatenate(pixel_positions)
    wave_positions = np.concatenate(wave_positions)
    order_positions = np.concatenate(order_positions)
    # fitted_lines = np.concatenate(fitted_lines)
    # print(f"Total fitted lines: {len(np.unique(wave_positions))}")

    return pixel_positions, wave_positions, order_positions

def get_pixels_for_ArcTh_fit(orders, traces):
    max_extracted_pixel = max([max(trace_y) for trace_y in traces.y])
    min_extracted_pixel = min([min(trace_y) for trace_y in traces.y])
    full_pixels = np.array([np.arange(min_extracted_pixel, max_extracted_pixel + 1) for _ in orders])
    return full_pixels

def apply_n_limit_constraint(initial_mask, orders_position, y_fit, X, model, n_limit, all_idx, residuals):
    """
    Apply n_limit constraint to any mask, ensuring minimum points per order.
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

def fit_surface_sklearn(dispersion_position, orders_position, wave_array, extracted_pixels, degree=7, sigma_clip=3, robust=False, n_limit=0, seed=None, max_iter=1000):
    """
    Fit a bivariate polynomial surface to the wavelength solution using RANSAC for outlier rejection.
    Returns fitted surface, residuals, and inlier mask.
    """
    
    # Prepare data
    x = dispersion_position
    y = orders_position
    y_fit = wave_array * orders_position

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
    return [model.predict((np.column_stack([trace_y, np.ones_like(trace_y)*absolute_order])))/absolute_order for absolute_order, trace_y in zip(ORDER, traces.y)]

def get_arcTh_master(veloce_paths, arm, date, amplifier_mode, obs_list=None, filename=None):
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
                    file_list, f'ARC-ThAr_{arm}_master', veloce_paths.input_dir, date, arm, amplifier_mode)
                veloce_reduction_tools.save_image_fits(arcTh_master_filename, arcTh_image, hdr)
            else:
                raise FileNotFoundError(f"No ARC-ThAr_{arm} files found for date {date}. Cannot create master.")
        else:
            raise ValueError("Either obs_list or filename must be provided to get the arcTh master.")
    return arcTh_image, hdr

# def calibrate_absolute_Th(extracted_science_orders, obs_list, veloce_paths, traces, thxe_image, hdr, arm, plot=False, filename=None):
def calibrate_absolute_Th(traces, veloce_paths, obs_list, date, arm, amplifier_mode, plot=False, plot_filename=None, th_linelist_filename='Default'):
    ### TODO: add header info to wavelength solution file, including params used, save fitted lines to file
    wave_solution_filename = f"arcTh_wave_{arm}_{date}.fits"
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

        extracted_arcTh, _ = veloce_reduction_tools.extract_orders_with_trace(arcTh_image, traces)

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

        pixel_positions, wave_positions, order_positions = fit_all_lines_per_order(
            static_wave, extracted_arcTh, ORDER, traces, linelist, arm, offset=offset, plot=False)
        
        if len(np.unique(order_positions)) != len(ORDER):
            print(f"[Warning]: {len(ORDER) - len(np.unique(order_positions))} order(s) don't have fitted lines.")
            missing_orders = [order for order in ORDER if order not in np.unique(order_positions)]
            print(f"Missing orders: {missing_orders}")
            
        full_pixels = get_pixels_for_ArcTh_fit(np.unique(order_positions), traces)

        warnings.filterwarnings("ignore", category=LinAlgWarning)
        if arm == 'blue':
            Z, residuals, mask, model, converged = fit_surface_sklearn(
                pixel_positions, order_positions, wave_positions,
                full_pixels, degree=6, sigma_clip=2.3, robust=False, n_limit=8)
        elif arm == 'green':
            Z, residuals, mask, model, converged = fit_surface_sklearn(
                pixel_positions, order_positions, wave_positions,
                full_pixels, degree=7, sigma_clip=2.4, robust=False, n_limit=9)
            # Z, residuals, mask, model, converged = fit_surface_sklearn(
            #     pixel_positions, order_positions, wave_positions,
            #     full_pixels, degree=7, sigma_clip=2.2, robust=False, n_limit=9)
        elif arm == 'red':
            Z, residuals, mask, model, converged = fit_surface_sklearn(
                pixel_positions, order_positions, wave_positions,
                full_pixels, degree=7, sigma_clip=2.3, robust=False, n_limit=9)
        if not converged:
            print("[Warning]: Wavelength solution fitting did not converge.")
        
        wave = get_wave_solution_from_surface(model, traces, ORDER)

        # precision = estimate_calibration_precision(residuals[mask], int(len(traces.y)/2), wave)

        if plot:
            veloce_diagnostic.plot_ArcTh_surface(Z, pixel_positions, order_positions, wave_positions, full_pixels, veloce_paths, plot_filename)
            veloce_diagnostic.plot_ArcTh_points_positions(pixel_positions, order_positions, mask, veloce_paths, plot_filename)
            veloce_diagnostic.plot_ArcTh_residuals(residuals, order_positions, pixel_positions, wave_positions, mask, veloce_paths, plot_filename, plot_type='wavelength')
        
        # with open(os.path.join(veloce_paths.wavelength_calibration_dir, wave_solution_filename), 'wb') as f:
        #     pickle.dump(wave, f)

        veloce_reduction_tools.save_extracted_spectrum_fits(
            os.path.join(veloce_paths.wavelength_calibration_dir, wave_solution_filename),
            wave,
            extracted_arcTh,
            hdr)
    
    return wave #, extracted_science_orders

def get_LC_master(veloce_paths, arm, date, amplifier_mode, obs_list=None, filename=None):
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
            file_list = obs_list[f'SimLC'][date]
            file_list = veloce_reduction_tools.get_longest_consecutive_files(file_list)
            if file_list:
                LC_image, hdr = veloce_reduction_tools.get_master_mmap(
                    file_list, f'SimLC_{arm}_master', veloce_paths.input_dir, date, arm, amplifier_mode)
                veloce_reduction_tools.save_image_fits(LC_master_filename, LC_image, hdr)
            else:
                raise FileNotFoundError(f"No SimLC files found for date {date}. Cannot create master.")
        else:
            raise ValueError("Either obs_list or filename must be provided to get the LC master.")
    return LC_image, hdr

def select_lc_lines_in_wave_range(lc_lines, wave):
    wave_min, wave_max = min(wave), max(wave)
    return lc_lines[(lc_lines >= wave_min) & (lc_lines <= wave_max)]

def build_LC_wavelength_solution(traces, veloce_paths, date, arm, amplifier_mode, obs_list, config, plot=False, filename=None):
    wave_solution_filename = f"LC_wave_{arm}_{date}.fits"
    if os.path.exists(os.path.join(veloce_paths.wavelength_calibration_dir, wave_solution_filename)):
        print(f"Reading existing wavelength solution file {wave_solution_filename}")
        wave, _, _ = veloce_reduction_tools.load_extracted_spectrum_fits(
            os.path.join(veloce_paths.wavelength_calibration_dir, wave_solution_filename))
    else:
        lc_image, hdr = get_LC_master(veloce_paths, arm, date, amplifier_mode, obs_list=obs_list, filename=None)
        print(f"Building new wavelength solution for LC based on {arm} arm Th wavelength solution on {date}")
        lc_traces = veloce_reduction_tools.Traces.load_traces(os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_LC_trace.pkl'))
        ref_traces = veloce_reduction_tools.Traces.load_traces(os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_4amp_sim_calib_trace.pkl'))
        offsets = [np.nanmean(ref_x-x) for ref_x, x in zip(ref_traces.x, traces.x)]
        if np.any(np.abs(np.array(offsets)) > 1.0):
            new_traces_x = [x+offset for x, offset in zip(lc_traces.x, offsets)]
            lc_traces.set_traces_yx(lc_traces.y, new_traces_x)
        extracted_LC, extracted_LC_imgs = veloce_reduction_tools.extract_orders_with_trace(lc_image, lc_traces)

        arcTh_wave = calibrate_absolute_Th(traces, veloce_paths, obs_list,
                                date, arm, amplifier_mode,
                                plot=plot, plot_filename=f'arcTh_wavecalib_{arm}_{date}',
                                th_linelist_filename='Default')

        if hdr is not None and (hdr['FREQREF'] != REPETITION_RATE and hdr['FOFFFREQ'] != OFFSET_FREQUENCY):
            raise ValueError("Repetition rate and offset frequency do not match the values of LC solution.")
        freq_start = c/(950e-9)
        freq_end =  c/(430e-9)
        n_start = np.floor((freq_start - OFFSET_FREQUENCY)/REPETITION_RATE)
        n_end = np.ceil((freq_end - OFFSET_FREQUENCY)/REPETITION_RATE)
        n = np.arange(n_start, n_end, 1)
        lc_lines = c/(OFFSET_FREQUENCY+n*REPETITION_RATE)*1e9

    return wave #, extracted_LC

def calibrate_simTh():
    raise NotImplementedError