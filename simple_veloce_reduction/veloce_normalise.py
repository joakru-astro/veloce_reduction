import numpy as np
import os, sys

from astropy.io import fits

from scipy.ndimage import median_filter, maximum_filter, minimum_filter, gaussian_filter1d
from scipy.signal import find_peaks
from scipy.interpolate import make_interp_spline, interp1d, splrep, splev

from simple_veloce_reduction import veloce_reduction_tools, veloce_config
from matplotlib import pyplot as plt

from astropy.constants import c
SPEED_OF_LIGHT = c.value  # m/s
ccd_nums = {'red': 3, 'green': 2, 'blue': 1}
MAX_LENGTH = 4112  # hardcoded max length of orders in science spectra

def get_smooth_blaze(y, window_size=100, smooth=15):
    x = np.arange(len(y))

    signal_peaks = []
    for i in range(len(y)):
        start = max(0, i - window_size//2)
        end = min(len(y), i + window_size//2 + 1)
        local_window = y[start:end]
        peaks, _ = find_peaks(local_window, prominence=(np.median(local_window)-np.min(local_window)))
        global_peaks = (peaks + start).tolist()
        signal_peaks.extend(global_peaks)

    extended_peaks = set(signal_peaks)
    for peak in signal_peaks:
        left = peak
        while left > 0 and y[left-1] < y[left]:
            left -= 1
        right = peak
        while right < len(y)-1 and y[right+1] < y[right]:
            right += 1
        extended_peaks.update(range(left+1, right)) # don't use the actual floor points 
    extended_peaks = sorted(extended_peaks)

    signal_dips = []
    for i in range(len(y)):
        start = max(0, i - window_size//2)
        end = min(len(y), i + window_size//2 + 1)
        local_window = -y[start:end]
        dips, _ = find_peaks(local_window, prominence=abs(np.max(local_window)-np.mean(local_window))/2)
        global_dips = (dips + start).tolist()
        signal_dips.extend(global_dips)
    extended_dips = set(signal_dips)
    for dip in signal_dips:
        left = dip
        while left > 1 and y[left-1] > y[left]:
            left -= 1
        right = dip
        while right < len(y)-1 and y[right+1] > y[right]:
            right += 1
        extended_dips.update(range(left+1, right))  # don't use the actual ceiling points
    
    extended_dips = sorted(extended_dips)
    extended_peaks = sorted(set(extended_peaks) | set(extended_dips))

    all_detected_peaks = np.array(list(set(extended_peaks)))

    peak_mask = np.zeros_like(y, dtype=bool)
    if all_detected_peaks.size > 0:
        peak_mask[all_detected_peaks] = True

    smooth_y = y.copy()
    if np.any(peak_mask):
        not_peak = ~peak_mask
        interp_vals = np.interp(x[peak_mask], x[not_peak], y[not_peak])
        smooth_y[peak_mask] = interp_vals

    nan_mask = np.isnan(smooth_y)
    smooth_y[~nan_mask] = gaussian_filter1d(median_filter(smooth_y[~nan_mask], size=51), sigma=smooth) # Smooth estimate

    return smooth_y

def mask_order_by_flux_threshold(flux, percentile=25, median_window=51, gaussian_window=5, plot=False):
    """
    Masks regions in the flux array that are below or above specified sigma thresholds.

    Parameters:
    - flux: 1D numpy array of flux values.
    - low_sigma: float, lower sigma threshold for masking.
    - median_window: int, window size for median_filter.
    - gaussian_window: float, sigma for gaussian_filter1d.

    Returns:
    - mask: 1D boolean numpy array where True indicates valid (unmasked) data points.
    
    Notes:
    - Re-evaluates mask to keep only contiguous False runs that touch the array edges.
      Any False run fully inside the array is converted back to True.
    """
    smooth_flux = gaussian_filter1d(median_filter(flux, median_window), sigma=gaussian_window)
    
    lower_threshold = np.nanpercentile(smooth_flux, percentile)

    mask = smooth_flux >= lower_threshold
    mask[:600] = False
    mask[-300:] = False

    if np.all(mask) or not np.any(mask):
        return mask
    
    false_idx = np.where(~mask)[0]
    runs = np.split(false_idx, np.where(np.diff(false_idx) != 1)[0] + 1)

    n = mask.size
    for run in runs:
        if run.size == 0:
            continue
        if (run[0] != 0) and (run[-1] != n - 1):
            mask[run] = True

    masked_flux = flux.copy()
    masked_flux[~mask] = np.nan 

    return masked_flux

def line_in_wave(lines, wave, flux, margin=0):
    # Vectorised check: any line falls within the valid-wave range (considering margin)
    valid = ~np.isnan(flux)
    if not np.any(valid):
        return False
    wmin = np.nanmin(wave[valid])
    wmax = np.nanmax(wave[valid])
    lines_arr = np.asarray(lines)
    return np.any((lines_arr >= (wmin - margin)) & (lines_arr <= (wmax + margin)))

def renormalise_order(y, err, wave=None, initial_continuum=None, high_sigma_threshold=3.0, low_sigma_threshold=1.0, window_size=100, poly_order=1, max_iter=100, plot=False, save_plot=None, verbose=False):
    """
    Normalise science spectrum while detecting absorption lines for continuum fitting.

    """
    x = np.arange(len(y))
    # Iterative sigma clipping for continuum region detection
    mask = np.ones_like(y, dtype=bool)
    _smooth_y = gaussian_filter1d(median_filter(y, size=3), sigma=3)
    if initial_continuum is None:
        initial_continuum = gaussian_filter1d(maximum_filter(median_filter(_smooth_y, size=window_size//4+1), size=window_size//2+1), sigma=window_size+1)

    continuum = initial_continuum
    for iteration in range(max_iter):
        # std = np.nanstd(y[mask])
        std = np.sqrt(err**2 + np.nanmean((y-_smooth_y)[mask]**2))
        # std = err
        high_threshold = continuum + high_sigma_threshold * std
        low_threshold = continuum - low_sigma_threshold * std
        new_mask = (y < high_threshold) & (y > low_threshold)
        # Stop if mask does not change
        if np.array_equal(new_mask, mask):
            if iteration == 0:
                if verbose:
                    print("No significant outliers detected in first iteration.")
                _continuum_coeffs = np.polyfit(x[mask], y[mask], poly_order)
                continuum = np.polyval(_continuum_coeffs, x)
            elif verbose:
                print(f"Converged after {iteration+1} iterations.")
            
            break
        elif np.sum(new_mask) < 100:
            if verbose:
                print(f"[Warning] Too few points left for continuum fitting ({np.sum(new_mask)}). Stopping iteration at iteration {iteration+1} and using initial mask.")
            _continuum_coeffs = np.polyfit(x[mask], y[mask], poly_order)
            continuum = np.polyval(_continuum_coeffs, x)
            break
        mask = new_mask
        _continuum_coeffs = np.polyfit(x[mask], y[mask], poly_order)
        continuum = np.polyval(_continuum_coeffs, x)

    if poly_order == 1 and verbose:
        print(f"Renormalisation fit: slope={_continuum_coeffs[0]:.3e}, intercept={_continuum_coeffs[1]:.3e}")
    
    end_to_end_diff = continuum[x[mask].min()] - continuum[x[mask].max()]
    if verbose:
        print(f"End-to-end difference: {end_to_end_diff:.3f}")
        print(f"Estimated mean noise level: {np.nanmean(std):.3f} vs original error estimate: {np.nanmean(err):.3f}")

    normalised = y / continuum

    if plot:
        
        plt.close('all')
        fig, axes = plt.subplots(2, 1,figsize=(3.15, 2.5), sharex=True)
        if wave is not None:
            plot_x = wave
        else:
            plot_x = x
        xmin, xmax = plot_x[~np.isnan(y)].min(), plot_x[~np.isnan(y)].max()
        axes[0].text(0.02, 0.15, 'a', transform=axes[0].transAxes, fontsize=8, va='top')
        axes[0].plot(plot_x, y, 'gray', alpha=0.7, lw=0.5, label='Original', zorder=1)
        axes[0].scatter(plot_x[mask], y[mask], alpha=0.6, color='C2', s=2, marker='o', label='Used for continuum fit', zorder=2)
        axes[0].plot(plot_x, continuum, c='k', ls='--', lw=0.5, label='Refitted continuum', zorder=3)
        axes[0].set_xlim(xmin, xmax)
        axes[0].set_ylim(max(0, np.nanmin(y) - (low_sigma_threshold+1)*np.nanmean(std)), 1+(high_sigma_threshold+1)*np.nanmean(std))
        axes[0].set_ylabel('Deblazed Flux', fontsize=6, labelpad=0)

        axes[1].text(0.02, 0.15, 'b', transform=axes[1].transAxes, fontsize=8, va='top')
        axes[1].plot(plot_x, normalised, c='C0', lw=0.5, label='Renormalised spectrum', zorder=1)
        axes[1].axhline(1, color='k', ls='--', lw=0.5, zorder=2)
        axes[1].set_xlim(xmin, xmax)
        axes[1].set_ylim(max(0, np.nanmin(normalised) - (low_sigma_threshold+1)*np.nanmean(std)), 1+(high_sigma_threshold+1)*np.nanmean(std))
        axes[1].set_ylabel('Renormalised Flux', fontsize=6, labelpad=0)
        if wave is not None:
            axes[1].set_xlabel('Wavelength [nm]', fontsize=6, labelpad=0)
        else:
            axes[1].set_xlabel('Pixel', fontsize=6, labelpad=0)
        plt.subplots_adjust(wspace=0, hspace=0)
        if save_plot is not None:
            plt.savefig(save_plot, dpi=600, bbox_inches='tight')
        plt.show()
    
    return normalised, continuum, end_to_end_diff  #, sigma * np.ones_like(y)

def find_cosmics(y, err, wave=None, peak_threshold=5, sigma_clip=2, high_flux_threshold=0.5, plot=False):
    """
    Remove cosmic rays from a 1D spectrum using signal find_peaks.

    return the mask of detected cosmics
    """
    x = np.arange(len(y))
    smooth_y = y.copy()
    smooth_y[~np.isnan(y)] = gaussian_filter1d(median_filter(y[~np.isnan(y)], size=3), sigma=3)
    std = np.nanstd(y - smooth_y)
    smooth_y[smooth_y > 1+sigma_clip*std] = 1
    # smooth_y[~np.isnan(y)] = gaussian_filter1d(y[~np.isnan(y)], sigma=3)
    # # smooth_y[~np.isnan(y)] = median_filter(y[~np.isnan(y)], size=5)
    # _y = y - smooth_y
    # _y[_y < 0] = 0
    
    # height = sigma_clip*np.nanstd(_y)
    # signal_peaks, _ = find_peaks(_y, prominence=peak_threshold*np.nanmedian(err))
    # if high_flux_threshold is not None:
    #     high_flux = np.where(_y > high_flux_threshold)[0]
    # else:
    #     high_flux = np.array([])
    _err = gaussian_filter1d(err, sigma=51)
    
    peaks = np.where((y > 1+peak_threshold*_err))[0]
    # peaks = np.sort(np.unique(np.hstack([signal_peaks, high_flux])))
    # extand peaks until no adjacent pixels are above threshold
    extended_peaks = []
    for p in peaks:
        start = p
        end = p + 1
        while start > 0 and y[start-1] > smooth_y[start-1]+sigma_clip*err[start-1]:
            start -= 1
        while end < len(y) and y[end] > smooth_y[end]+sigma_clip*err[end]:
            end += 1
        extended_peaks.extend(range(start, end))
    extended_peaks = np.unique(extended_peaks).astype(int)
    # print(f"Detected {len(signal_peaks)} cosmic features, extended to {len(extended_peaks)} pixels")
    
    cosmic_mask = np.zeros_like(y, dtype=bool)
    if len(extended_peaks) > 0:
        cosmic_mask[extended_peaks] = True

    interp_func = interp1d(x[~cosmic_mask], y[~cosmic_mask], kind='linear', bounds_error=False, fill_value=np.nan)
    y_fix = interp_func(x)

    if plot:
        plot_x = wave if wave is not None else x
        plt.close('all')
        plt.figure(figsize=(3.15, 2))
        # plt.subplot(1, 1, 1)
        plt.plot(plot_x, y, 'gray', lw=0.5, label='Original Spectrum', zorder=1)
        # plt.plot(x, smooth_y, 'green', alpha=0.7, label='Smoothed Spectrum')
        plt.plot(plot_x, y_fix, 'C0', lw=0.5, label='Corrected Spectrum', zorder=2)
        plt.plot(plot_x[~np.isnan(y)], 1+peak_threshold*_err[~np.isnan(y)], color='gray', ls=':', lw=0.5, label='Detection Threshold', zorder=3)
        plt.scatter(plot_x[extended_peaks], y[extended_peaks], marker='x', c='C3', s=5, label='Corrected Pixels', zorder=4)
        if wave is not None:
            plt.xlabel('Wavelength [nm]', fontsize=8, labelpad=0)
        else:
            plt.xlabel('Pixel', fontsize=8, labelpad=0)
        plt.ylabel('Normalised Flux', fontsize=8, labelpad=0)
        plt.xlim(plot_x[~np.isnan(y)].min(), plot_x[~np.isnan(y)].max())
        # plt.subplot(2, 1, 2)
        # plt.plot(x, _y, 'blue', alpha=0.7, label='High-pass filtered')
        # # plt.scatter(signal_peaks, _y[signal_peaks], marker='x', c='red', s=50, label='Detected Peaks')
        # plt.scatter(extended_peaks, _y[extended_peaks], marker='o', facecolors='none', edgecolors='orange', s=50, label='Extended Cosmic Rays')
        # plt.axhline(peak_threshold*np.nanmedian(err), color='k', ls='--', alpha=0.5, label='Detection Threshold {:.2f}'.format(peak_threshold*np.nanmedian(err)))
        # plt.axhline(sigma_clip*np.nanmedian(err), color='k', ls=':', alpha=0.3, label='Cut-off Threshold {:.2f}'.format(sigma_clip*np.nanmedian(err)))
        # plt.xlabel('Pixel')
        # plt.ylabel('High-pass Flux')
        # plt.legend()
        plt.tight_layout()
        plt.show()

    return y_fix, cosmic_mask

def get_blaze_function(arms, veloce_paths, date, config, obs_list, skip_edge_orders=True):

    extracted_blaze = []
    order_numbers = {} 
    # print()
    for arm in arms:
        flat, header = veloce_reduction_tools.get_flat(veloce_paths, arm, config['amplifier_mode'], date, obs_list)
        norm_flat, _ = veloce_reduction_tools.normalise_flat(flat, header)
        flat, header = veloce_reduction_tools.flat_field_correction(flat, norm_flat, header)
        traces = veloce_reduction_tools.Traces.load_traces(os.path.join(veloce_paths.trace_shift_dir, f'trace_{arm}_{date}.pkl'))
        blaze_traces = veloce_reduction_tools.Traces.load_traces(os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_4amp_no_sim_calib_trace.pkl'))
        blaze_traces.set_traces_yx(blaze_traces.y, traces.x)  # shift blaze traces to trace position on the date
        _extracted_blaze, _uncertainty, _extracted_blaze_imgs = veloce_reduction_tools.extract_orders_with_trace(flat, blaze_traces)
        if skip_edge_orders:
            extracted_blaze.extend(_extracted_blaze[1:-1])  # skip edge orders
            order_numbers[arm] = len(_extracted_blaze[1:-1])
        else:
            extracted_blaze.extend(_extracted_blaze)  # use all orders
            order_numbers[arm] = len(_extracted_blaze)

    signal_mask = []
    blaze_model = []
    for blaze in extracted_blaze:
        mask = np.isfinite(blaze)
        smooth_blaze = get_smooth_blaze(blaze)
        signal_mask.append(mask)
        blaze_model.append(smooth_blaze)

    ### pad arrays to same format as extracted science spectrum
    signal_mask = np.array([np.pad(order, (0, MAX_LENGTH - len(order)), constant_values=False) for order in signal_mask])
    blaze_model = np.array([np.pad(order, (0, MAX_LENGTH - len(order)), constant_values=np.nan) for order in blaze_model])

    np.savez_compressed(os.path.join(veloce_paths.blaze_dir, f'blaze_model_{date}.npz'),
                    blaze_model=blaze_model,
                    signal_mask=signal_mask)

    return blaze_model, signal_mask, order_numbers

def interpolate_spectrum(wave, flux, new_wave):
    """
    Interpolates the input spectrum (wave, flux) onto a new wavelength grid (new_wave).
    Uses linear interpolation and fills out-of-bounds values with the nearest valid flux.

    Parameters:
    - wave: 1D array of original wavelength values.
    - flux: 1D array of original flux values.
    - new_wave: 1D array of new wavelength values to interpolate onto.

    Returns:
    - new_flux: 1D array of flux values interpolated onto new_wave.
    """
    # Create an interpolation function    
    interp_func = interp1d(wave, flux, kind='linear', bounds_error=False, fill_value=np.nan)
    # Interpolate to the new wavelength grid
    new_flux = interp_func(new_wave)
    
    return new_flux

import numpy as np

def log_lambda_grid(wave_min, wave_max, dv):
    """
    Create a log-linear wavelength grid with constant velocity step.

    Parameters:
    - wave_min (float): Minimum wavelength
    - wave_max (float): Maximum wavelength
    - dv (float): Velocity step in km/s

    Returns:
    - wave (ndarray): Log-linear wavelength grid
    """
    dloglam = (1e3 * dv) / SPEED_OF_LIGHT
    logwave_array = np.arange(np.log(wave_min), np.log(wave_max), dloglam)
    wave = np.exp(logwave_array)
    return wave

def merge_orders(wave, flux, weight, step=None, step_type='linear'):
    """
    Merge multiple spectral orders into a single spectrum on a common wavelength grid.

    Parameters:
    - wave: list of 1D arrays of wavelength values for each order
    - flux: list of 1D arrays of flux values for each order
    - weight: list of 1D arrays of weights for each order
    """

    wavelengths = np.sort(np.hstack(wave))
    if step is None:
        # print("Using original wavelength grid for coaddition")
        pass
    else:
        if step_type == 'linear':
            # print(f"Using step size {step} nm for coaddition")
            wavelengths = np.arange(np.nanmin(wavelengths), np.nanmax(wavelengths), step)
        elif step_type == 'log-linear':
            # step is interpreted as km/s, convert to logarithmic wavelength step
            # print(f"Using logarithmic step size corresponding to {step} km/s for coaddition")
            wavelengths = log_lambda_grid(np.nanmin(wavelengths), np.nanmax(wavelengths), dv=step)
    
    resampled_flux = np.array([interpolate_spectrum(_wave, _flux, wavelengths) for _wave, _flux in zip(wave, flux)])
    resampled_weight = np.array([interpolate_spectrum(_wave, _weight, wavelengths) for _wave, _weight in zip(wave, weight)])

    weights = resampled_weight / np.nansum(resampled_weight, axis=0)
    merged_flux = np.nansum(resampled_flux * weights, axis=0)

    return wavelengths, merged_flux, weights

def save_merged_spectrum(filename, wave, flux, err=None):
    """
    Save the merged spectrum to a text file.

    Parameters:
    - filename: str, path to the output file
    - wave: 1D array of wavelength values
    - flux: 1D array of flux values
    - err: 1D array of error values (optional)
    """
    if err is None:
        err = np.zeros_like(flux)
    spectrum = np.array(
        [tuple([nm, f, err]) for nm, f, err in zip(wave, flux, err)],
        dtype=[('waveobs', float),('flux',float),('err',float)])
    np.savetxt(filename, spectrum, fmt=['%1.3f', '%1.3f', '%1.3f'], header='waveobs\tflux\terr',delimiter='\t')

    return spectrum

def merge_veloce_orders(veloce_paths, target_list, arms, config, obs_list, single_file=False, skip_edge_orders=True, verbose=False):
    """
    Merge Veloce orders for a list of targets. No return value, saves merged spectra as txt files in output directory.

    Parameters:
    - veloce_paths: object containing paths to Veloce data directories
    - target_list: dictionary with dates as keys and lists of (target, obs_n) tuples as values
    - arms: list of arms to include
    - config: dictionary containing configuration parameters
    - obs_list: list of observation IDs to include
    - single_file: boolean indicating whether to process a single file
    - skip_edge_orders: boolean indicating whether to skip edge orders
    - verbose: boolean indicating whether to print verbose output

    Returns:
    - None
    """

    balmer_series_waves = [656.28, 486.13, 434.05, 410.17]  # H-alpha, H-beta, H-gamma, H-delta
    paschen_series_waves = [923.2, 901.5, 886.3, 875.0, 866.5]  # paschen series lines in nm
    magnesium_triplet_waves = [516.7, 517.2, 518.3]
    sodium_dublet_waves = [589.00, 589.59]
    calcium_triplet_waves = [849.8, 854.2, 866.2]
    Ca_II_HK_waves = [393.37, 396.85]
    hydrogen_lines = balmer_series_waves + paschen_series_waves
    strong_lines = magnesium_triplet_waves + sodium_dublet_waves + calcium_triplet_waves + Ca_II_HK_waves
    strong_telluric_waves = [687, 725, 760, 823, 932]

    if single_file:
        _date = config['date']
        arms = [config['arm']]
    else:
        _date = list(target_list.keys())[0]

    print(f"Using arms {arms} for merging orders for date {_date}")
    blaze_model, signal_mask, blaze_order_numbers = get_blaze_function(arms, veloce_paths, _date, config, obs_list)

    science_targets = list(set([obs[0] for _date in target_list.keys() for obs in target_list[_date]]))
    for star_n, star in enumerate(science_targets):
        if single_file:
            _data_files = [(single_file[-15:-10], single_file[-9:-5])]
        else:
            _data_files = sorted([(file_name[-15:-10], file_name[-9:-5]) for file_name in sorted(os.listdir(veloce_paths.extracted_spectra_dir)) if (science_targets[star_n] in file_name) and (arms[0] in file_name)])

        order_numbers = {}
        for (night, obs_n) in _data_files:
            _wave, _flux, _err = [], [], []
            try:
                for arm in arms:
                    filename = f"{science_targets[star_n]}_veloce_{arm}_{night}{ccd_nums[arm]}{obs_n}.fits"
                    w, f, e, hdr = veloce_reduction_tools.load_extracted_spectrum_fits(os.path.join(veloce_paths.extracted_spectra_dir, filename))
                    # if arm == 'blue':
                    w = np.array([np.pad(order, (0, MAX_LENGTH - len(order)), constant_values=np.nan) for order in w])
                    f = np.array([np.pad(order, (0, MAX_LENGTH - len(order)), constant_values=np.nan) for order in f])
                    e = np.array([np.pad(order, (0, MAX_LENGTH - len(order)), constant_values=np.nan) for order in e])
                    # order_numbers[arm] = len(f)
                    if skip_edge_orders:
                        order_numbers[arm] = len(f)-2  # when skipping edge orders
                    else:
                        order_numbers[arm] = len(f)  # when using all orders
                    
                    if arm == 'red':
                        f = [mask_order_by_flux_threshold(np.array(order, dtype=float), percentile=30) for order in f]
                    else:
                        f = [mask_order_by_flux_threshold(np.array(order, dtype=float), percentile=40) for order in f]

                    if skip_edge_orders:
                        _wave.extend(w[1:-1])  # skip edge orders
                        _flux.extend(f[1:-1])  # skip edge orders
                        _err.extend(e[1:-1])  # skip edge orders
                    else:
                        _wave.extend(w)
                        _flux.extend(f)
                        _err.extend(e)
            except Exception as e:
                print(f"Warning: Incomplete data for {science_targets[star_n]} on night {night} observation {obs_n}")
                print(f"Error details: {e}")
                continue

            _wave = np.array(_wave, dtype=float)
            _flux = np.array(_flux, dtype=float)
            _err = np.array(_err, dtype=float)
            if verbose:
                print(f"Median SNR before renormalisation for {science_targets[star_n]} on night {night} observation {obs_n}: {np.nanmedian(_flux/_err):.2f} based on median error of {np.nanmedian(_err):.3e}")

            if 'LC' in config['calib_type']:
                _blaze = np.concatenate([blaze_model[:blaze_order_numbers['red']][:order_numbers['red']], blaze_model[blaze_order_numbers['red']:][:order_numbers['green']]])  # only use blaze orders corresponding to the used science orders
            else:
                _blaze = blaze_model.copy()
            
            _weight = _blaze.copy()
            _flux /= _blaze  # apply blaze correction
            _err /= _blaze  # apply blaze correction to errors
            _weight[np.isnan(_flux)] = np.nan

            _err = np.transpose(_err.T/np.nanpercentile(_flux, 95, axis=1))  # normalize to 95th percentile
            _flux = np.transpose(_flux.T/np.nanpercentile(_flux, 95, axis=1))  # normalize to 95th percentile

            norm_flux, norm_err = [], []
            for order_n, f, e, w in zip(range(len(_flux)), _flux, _err, _wave):
                if line_in_wave(hydrogen_lines, w, f, margin=0):
                    norm_f, cont, _ = renormalise_order(f, e, wave=w, initial_continuum=np.ones_like(f), poly_order=0, low_sigma_threshold=1., high_sigma_threshold=3.)
                elif line_in_wave(strong_lines, w, f, margin=0):
                    norm_f, cont, _ = renormalise_order(f, e, wave=w, initial_continuum=np.ones_like(f), poly_order=1, low_sigma_threshold=1., high_sigma_threshold=3.)
                elif line_in_wave(strong_telluric_waves, w, f, margin=0):
                    norm_f, cont, _ = renormalise_order(f, e, wave=w, initial_continuum=np.ones_like(f), poly_order=1, low_sigma_threshold=1, high_sigma_threshold=3.)
                else:
                    norm_f, cont, _ = renormalise_order(f, e, wave=w, poly_order=4, low_sigma_threshold=1, high_sigma_threshold=3.)
                
                e /= cont
                # e[e < 0.005] = 0.005  # cap errors for cosmic detection to avoid flagging too many points in low-snr regions
                try:
                    norm_f, cosmics = find_cosmics(norm_f, e, wave=w, peak_threshold=5, sigma_clip=2)
                except Exception as exeption:
                    print(f"Error occurred while finding cosmics for {science_targets[star_n]} on night {night} observation {obs_n}: {exeption}")
                    # continue
                norm_flux.append(norm_f)
                norm_err.append(e)

            norm_flux = np.array(norm_flux)
            norm_err = np.array(norm_err)
            log_linear_step = SPEED_OF_LIGHT / 1000 / 80000 / 5  # 5 pixels per resolution element at R=80,000
            merged_spectrum_wave, merged_spectrum_flux, merged_spectrum_weights = merge_orders(_wave, norm_flux, weight=_weight, step=log_linear_step, step_type='log-linear')
            merged_spectrum_wave, merged_spectrum_err, merged_spectrum_weights = merge_orders(_wave, norm_err, weight=_weight, step=log_linear_step, step_type='log-linear')
            save_merged_spectrum(os.path.join(veloce_paths.merged_spectra_dir, f"{science_targets[star_n]}_veloce_merged_{night}{obs_n}.txt"),
                                merged_spectrum_wave, merged_spectrum_flux, merged_spectrum_err)
            

if __name__ == '__main__':
    pass