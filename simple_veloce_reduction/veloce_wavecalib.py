from astropy.io import fits
from astropy.constants import c
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.linalg import lstsq
from scipy import signal
# from csaps import csaps

from . import veloce_reduction_tools

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

def load_simultanous_LC(image, veloce_paths, hdr, arm, ref_orders=None, ref_pixel=None):
    """
    Load simultaneous laser comb observations.
    """
    if hdr is not None and (hdr['FREQREF'] != REPETITION_RATE and hdr['FOFFFREQ'] != OFFSET_FREQUENCY):
        raise ValueError("Repetition rate and offset frequency do not match the values of LC solution.")
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

def fit_lc_peak(pix_shift, ccf):
    # pix_shift = pix_shift[~np.isnan(pix_shift)]
    ccf = ccf[~np.isnan(ccf)]
    
    # if len(pix_shift) == 0 or len(ccf) == 0:
    if len(ccf) == 0:
        return np.nan, [np.nan]
    
    # consider peak near 0 pixel shift
    fitting_limit = np.ceil(np.mean(np.diff(signal.find_peaks(ccf)[0])))/2 + 1
    _pix_shift = pix_shift[abs(pix_shift) <= fitting_limit]
    _ccf = ccf[abs(pix_shift) <= fitting_limit]
    # _ccf -= np.min(_ccf)

    # fit a generalised gaussian to the peak
    peak_arg = np.argmax(_ccf)
    peak = _ccf[peak_arg]
    peak_position = _pix_shift[peak_arg]
    sigma = 0.8
    beta = 2.0
    baseline = np.min(_ccf)

    popt, _ = curve_fit(general_gaussian, _pix_shift, _ccf,
                        p0=[peak, peak_position, sigma, beta, baseline],
                        bounds=([0, np.min(_pix_shift), 1e-3, 1e-3, 0], [2*peak, np.max(_pix_shift), 10, 10, peak]),)

    return popt[1], popt

def plot_ccf(PIX, CCF, order, chunk):
    fitting_limit = np.ceil(np.mean(np.diff(signal.find_peaks(CCF[order-1][chunk])[0])))/2 + 1
    plt.figure(figsize=(10, 6))
    plt.title('Cross-Correlation Function')
    plt.plot(PIX[order-1][chunk], CCF[order-1][chunk], label=f'Order {order}')
    shift, popt = fit_lc_peak(PIX[order-1][chunk], CCF[order-1][chunk])
    print(f"Amplitude: {popt[0]}\n Shift: {popt[1]}\n Sigma: {popt[2]}\n Beta: {popt[3]}\n Baseline: {popt[4]}")
    subpixel = np.arange(np.min(PIX[order-1][chunk]), np.max(PIX[order-1][chunk]), 0.01)
    plt.plot(subpixel, general_gaussian(subpixel, *popt), label='Gaussian Fit', linestyle='--')
    plt.axvline(shift, color='r', linestyle='--', label='Peak Position')
    plt.title(f'Cross-Correlation Function for Order {order}, Shift = {popt[1]:.2f}')
    plt.xlabel('Pixel Shift')
    plt.ylabel('Cross-Correlation')
    plt.xlim(-1*fitting_limit, fitting_limit)
    plt.legend()
    plt.grid()
    plt.show()

def plot_offset_map(dispersion_position, orders_position, offset_array):
    """
    Plot the offset map in 3D.
    """    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    points = ax.scatter(dispersion_position.flatten(), orders_position.flatten(), offset_array.flatten(), c=offset_array.flatten(), cmap='viridis', marker='o')
    ax.set_title('Offset Map')
    ax.set_xlabel('Dispersion Position')
    ax.set_ylabel('Orders')
    ax.set_zlabel('Offset')
    fig.colorbar(points, shrink=0.5, aspect=10)
    plt.show()

def calculate_offset_map(ref_orders, ref_intensity, ref_pixel, lc_intensity, lc_pixel, number_of_parts=8, plot=False):
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
    offset_array = np.array([[fit_lc_peak(pixel_shifts[i][j], CCF[i][j])[0] for j in range(number_of_parts)] for i in range(len(ref_orders))])
    
    if plot:
        plot_ccf(pixel_shifts, CCF, 15, 4)
        plot_offset_map(dispersion_position, orders_position, offset_array)

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

def plot_surface(ref_orders, extracted_pixels, surface_points, filtered_points):
    """
    Plot the offset map in 3D.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    max_pixel = max([np.nanmax(order) for order in extracted_pixels])+1
    min_pixel = min([np.nanmin(order) for order in extracted_pixels])
    X, Y = np.meshgrid(np.arange(min_pixel, max_pixel, 1), ref_orders)
    surf = ax.plot_surface(X, Y, surface_points, vmin=np.min(filtered_points[:,2]), vmax=np.max(filtered_points[:,2]), cmap='viridis', edgecolor='none', alpha=0.5)
    points = ax.scatter(filtered_points[:,0], filtered_points[:,1], filtered_points[:,2], c=filtered_points[:,2], cmap='viridis', marker='o')
    ax.set_title('Offset Map')
    ax.set_xlabel('Dispersion Position')
    ax.set_ylabel('Orders')
    ax.set_zlabel('Offset')
    fig.colorbar(points, shrink=0.5, aspect=10)
    plt.show()

def fit_surface(dispersion_position, orders_position, offset_array, extracted_pixels, degree=1, plot=False):
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
        plot_surface(np.unique(orders_position), extracted_pixels, Z, data[mask])

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
    # calibration_precision = std_dev / np.sqrt(n_points) * average_step / average_wave * c
    calibration_precision = std_dev * average_step / average_wave * c.value

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

def calibrate_simLC(extracted_science_orders, veloce_paths, lc_image, hdr, arm, plot=False):
    if arm == 'blue':
        print("[warning] Blue arm is not supported for LC calibration.")
        return np.array([None]), np.array([None])
    ref_orders, ref_wave, ref_intensity, ref_pixel = load_LC_wave_reference(veloce_paths, arm)
    lc_intensity, lc_pixel, order_slice, pixel_slices = load_simultanous_LC(lc_image, veloce_paths, hdr, arm, ref_orders=ref_orders, ref_pixel=ref_pixel)
    # align extracted orders with calibrated orders and pixel ranges
    extracted_science_orders = extracted_science_orders[order_slice]
    extracted_science_orders = [order[pixel_slices[i]] for i, order in enumerate(extracted_science_orders)]
    extracted_science_orders = pad_array(extracted_science_orders, ref_pixel)

    # cross-correlate the observed LC pixel positions with the reference LC pixel positions
    dispersion_position, orders_position, offset_array = calculate_offset_map(ref_orders, ref_intensity, ref_pixel, lc_intensity, lc_pixel, plot=plot)
    
    # fit a surface to the offset map
    results = []
    for degree in range(1, 4):
        fit_result = fit_surface(dispersion_position, orders_position, offset_array, lc_pixel, degree=degree)
        results.append(fit_result)

    # Select the result with the smallest standard deviation of residuals
    best_fit = min(results, key=lambda result: np.std(result[3]))
    surface_points, coeffs, filtered_points, residuals = best_fit

    # estimate the calibration precision
    calibration_precision = estimate_calibration_precision(residuals, 18, ref_wave)

    # interpolate wavelength solution to pixel positions
    wave = interpolate_offsets_optimised(lc_pixel, surface_points, ref_wave, ref_pixel)
    
    # apply shift between calibration fiber and science fibers expressed as rv
    wave = apply_wavelength_shift(wave, arm, veloce_paths)

    return wave,  extracted_science_orders

def load_wave_calibration_for_interpolation():
    raise NotImplementedError

def interpolate_wave(orders, hdr):
    raise NotImplementedError

def calibrate_simTh():
    raise NotImplementedError