from astropy.io import fits
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from csaps import csaps
from scipy.ndimage import median_filter
# from astropy.stats import sigma_clip
import pickle

from . import veloce_reduction_tools
from . import veloce_path

def  extract_run_no_blaze(obs_list_filename, run, arm, veloce_paths=None, output_path=None):
    """
    Extracts spectral data from Veloce observations without applying blaze correction.

    This function processes observation data for a specific run and spectrograph arm from Veloce.
    It loads wave calibration data, trace data, and modifies summing ranges based on the
    specified arm. The function is designed to work with 'green' and 'red' arms, with a note to add
    support for the 'blue' arm in the future.

    Parameters:
    - obs_list_filename (str): The filename of the observation list to be processed.
    - run (str): The specific run identifier for which the data is to be extracted.
    - arm (str): The spectrograph arm to be processed. Currently supports 'green' and 'red'.

    Raises:
    - ValueError: If an unsupported arm is specified.

    Notes:
    - The function assumes a standard directory structure for storing Veloce data and extracted data.
    - Wave calibration is based on pre-fitted Thorium-Argon (ThAr) lamp observations.
    - Trace data is loaded from a predefined location and may need adjustments for summing ranges,
      especially if the spectrograph setup has been altered.
    - TODOs indicate future work for adding support for the 'blue' arm and revisiting the handling
      of summing ranges to accommodate potential changes in the spectrograph setup.

    Returns:
    None. The function is designed to perform data extraction and processing, with outputs
    saved to files (used in further processing).
    """
    if veloce_paths is None:
        veloce_paths = veloce_path.VelocePaths(run)
    if output_path is None:
        output_path = veloce_paths.extracted_dir
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    # standard place were all veloce runs are kept:
    # data_path = '/home/usqobserver2/VeloceData'
    # pick which arm to reduce 
    ### TODO: add blue arm
    if arm == 'green':
        ccd = 'ccd_2'
    elif arm == 'red':
        ccd = 'ccd_3'
    else:
        raise ValueError('Unsupported arm')
    
    # load wave calibration based on ThAr
    ORDER, COEFFS, MATCH_LAM, MATCH_PIX, MATCH_LRES, GUESS_LAM, Y0 = veloce_reduction_tools.load_prefitted_ThAr(arm=arm, wave_path=veloce_paths.wave_dir)
    # load traces
    trace_data = np.load(os.path.join(veloce_paths.trace_dir, f'veloce_trace_{arm}.npz'))
    traces, summing_ranges = trace_data['traces'], trace_data['summing_ranges']
    # adjust summing ranges 
    ### TODO: take a close look at this summing ranges and prepare files that aren't modified?
    ### remember that traces can change between runs/nights if someone was fiddling with spectrograph
    if arm == 'green':
        summing_ranges_mod = np.array([[25,27] for _ in range(len(summing_ranges))])
    elif arm == 'red':
        summing_ranges_mod = np.array([[31,31] for _ in range(len(summing_ranges))])
    else:
        raise ValueError('Unsupported arm')

    with open(os.path.join(os.path.join(veloce_paths.obs_list_dir, obs_list_filename)), 'rb') as f:
        obs_list = pickle.load(f)
    
    for date in obs_list.keys(): 
        for obs in obs_list[date]:
            target, filename = obs
            print(target, filename)
            spectrum_filename =  os.path.join(veloce_paths.raw_parent_dir, run, date, ccd, filename)
            with fits.open(spectrum_filename) as hdul:
                image_data = hdul[0].data
                hdr = hdul[0].header
                # times.append(hdr['MJD-OBS'])
                image_subtracted_bias = veloce_reduction_tools.remove_overscan_bias(image_data, overscan_range=32)
                extracted_science_orders, extracted_order_imgs = veloce_reduction_tools.extract_orders_with_trace(
                    image_subtracted_bias, traces, summing_ranges_mod, remove_background=False)
                ### green part needs shift by one order - issue with labeling?
                if arm == 'green':
                    final_wave = np.array(veloce_reduction_tools.calibrate_orders_to_wave(
                        extracted_science_orders[1:], Y0[0], COEFFS))
                    final_flux = np.array(extracted_science_orders[1:])
                else:
                    final_wave = np.array(veloce_reduction_tools.calibrate_orders_to_wave(
                        extracted_science_orders, Y0[0], COEFFS))
                    final_flux = np.array(extracted_science_orders)

                np.savez(
                    os.path.join(output_path, f"{target}_veloce_{arm}_{filename.split('.')[0]}"),
                    wave=final_wave, flux=final_flux, mjd=float(hdr['MJD-OBS']))
                
def extract_run_with_blaze(obs_list_filename, run, arm, blaze_path=None, veloce_paths=None, output_path=None):
    """
    Extracts spectral data from Veloce observations with blaze correction applied.

    This function processes observation data for a specific run and spectrograph arm from Veloce.
    It loads wave calibration data, trace data, and modifies summing ranges based on the specified arm.
    The function also applies blaze correction using precomputed blaze functions. The function is designed
    to work with 'green' and 'red' arms, with a note to add support for the 'blue' arm in the future.

    Parameters:
    - obs_list_filename (str): The filename of the observation list to be processed.
    - run (str): The specific run identifier for which the data is to be extracted.
    - arm (str): The spectrograph arm to be processed. Currently supports 'green' and 'red'.
    - blaze_path (str, optional): The path to the blaze function file. If None, a default path is used.
    - output_path (str, optional): The path to save the extracted data. If None, a default path is used.

    Raises:
    - ValueError: If an unsupported arm is specified.

    Notes:
    - The function assumes a standard directory structure for storing Veloce data and extracted data.
    - Wave calibration is based on pre-fitted Thorium-Argon (ThAr) lamp observations.
    - Trace data is loaded from a predefined location and may need adjustments for summing ranges,
      especially if the spectrograph setup has been altered.
    - Blaze correction is applied using precomputed blaze functions loaded from a specified or default path.
    - TODOs indicate future work for adding support for the 'blue' arm and revisiting the handling
      of summing ranges to accommodate potential changes in the spectrograph setup.

    Returns:
    None. The function is designed to perform data extraction and processing, with outputs
    saved to files (used in further processing).
    """
    if veloce_paths is None:
        veloce_paths = veloce_path.VelocePaths(run)
    if output_path is None:
        output_path = veloce_paths.extracted_dir
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    # pick which arm to reduce 
    ### TODO: add blue arm
    if arm == 'green':
        ccd = 'ccd_2'
    elif arm == 'red':
        ccd = 'ccd_3'
    else:
        raise ValueError('Unsupported arm')
    
    # load wave calibration based on ThAr
    ORDER, COEFFS, MATCH_LAM, MATCH_PIX, MATCH_LRES, GUESS_LAM, Y0 = veloce_reduction_tools.load_prefitted_ThAr(arm=arm, wave_path=veloce_paths.wave_dir)
    # load traces
    trace_data = np.load(os.path.join(veloce_paths.trace_dir, f'veloce_trace_{arm}.npz'))
    traces, summing_ranges = trace_data['traces'], trace_data['summing_ranges']
    # adjust summing ranges 
    ### TODO: take a close look at this summing ranges and prepare files that aren't modified?
    ### remember that traces can change between runs/nights if someone was fiddling with spectrograph
    if arm == 'green':
        summing_ranges_mod = np.array([[25,27] for _ in range(len(summing_ranges))])
    elif arm == 'red':
        summing_ranges_mod = np.array([[31,31] for _ in range(len(summing_ranges))])
    else:
        raise ValueError('Unsupported arm')
    ### load blazes
    if blaze_path in None:
        blaze_path = os.path.join(veloce_paths.blaze_dir, f'veloce_blaze_{arm}_pix.npz')
    blazes = np.load(blaze_path)
    blazes = blazes['blazes']

    with open(os.path.join(veloce_paths.obs_list_dir, obs_list_filename), 'rb') as f:
        obs_list = pickle.load(f)
    
    # number of pixels to discard from both sides
    # (so only well wave calibrated high snr part in the middle is left)
    cutoff = 1250
    # pix_max = 4112
    for date in obs_list.keys(): 
        for obs in obs_list[date]:
            target, filename = obs
            print(target, filename)
            spectrum_filename =  os.path.join(veloce_paths.raw_parent_dir, run, date, ccd, filename)
            with fits.open(spectrum_filename) as hdul:
                image_data = hdul[0].data
                hdr = hdul[0].header
                # times.append(hdr['MJD-OBS'])
                image_subtracted_bias = veloce_reduction_tools.remove_overscan_bias(image_data, overscan_range=32)
                extracted_science_orders, extracted_order_imgs = veloce_reduction_tools.extract_orders_with_trace(
                    image_subtracted_bias, traces, summing_ranges_mod, remove_background=False)
                ### green part needs shift by one order issue with labeling?
                if arm == 'green':
                    waves = np.array(veloce_reduction_tools.calibrate_orders_to_wave(
                        extracted_science_orders[1:], Y0[0], COEFFS))
                    fluxs = np.array(extracted_science_orders[1:])
                    blazes = blazes[1:]
                else:
                    waves = np.array(veloce_reduction_tools.calibrate_orders_to_wave(
                        extracted_science_orders, Y0[0], COEFFS))
                    fluxs = np.array(extracted_science_orders)

                final_wave = []
                final_flux = []
                for wave, extracted_science_order, blaze in zip(waves, fluxs, blazes):
                    y = np.array(blaze, dtype=np.float64)
                    ysm = median_filter(y,50)
                    ysm /= max(ysm)
                    flux = extracted_science_order.copy()
                    flux /= ysm
                    flux /= np.median(flux)
                    final_wave.append(wave[cutoff:-cutoff])
                    final_flux.append(flux[cutoff:-cutoff])
                final_wave = np.array(final_wave)
                final_flux = np.array(final_flux)

                np.savez(
                    os.path.join(output_path, f"{target}_veloce_{arm}_{filename.split('.')[0]}"),
                    wave=final_wave, flux=final_flux, mjd=float(hdr['MJD-OBS']))

def extract_blaze(file_name, arm, blaze_path=None, master_path=None, veloce_paths=None,):
    """
    Extracts blaze function from a master flat field file for a specific spectrograph arm.

    This function processes a master flat field file to extract the blaze function for a specific
    spectrograph arm. It loads trace data, adjusts summing ranges, and extracts the blaze function
    from the flat field image. The extracted blaze function is saved to a specified or default path.

    Parameters:
    - file_name (str): The filename of the master flat field file to be processed.
    - arm (str): The spectrograph arm to be processed. Currently supports 'green' and 'red'.
    - blaze_path (str, optional): The path to save the extracted blaze function. If None, a default path is used.

    Raises:
    - ValueError: If an unsupported arm is specified.

    Notes:
    - The function assumes a standard directory structure for storing Veloce data and extracted data.
    - Trace data is loaded from a predefined location and may need adjustments for summing ranges,
      especially if the spectrograph setup has been altered.
    - The blaze function is extracted by summing the flat field image over the adjusted summing ranges
      for each trace.

    Returns:
    None. The function is designed to perform data extraction and processing, with outputs
    saved to files (used in further processing).
    """
    if veloce_paths is None:
        veloce_paths = veloce_path.VelocePaths()
    if blaze_path is None:
        blaze_path = veloce_paths.blaze_dir
    if master_path is None:
        master_path = veloce_paths.master_dir
    # load traces
    trace_data = np.load(os.path.join(veloce_paths.trace_dir, f'veloce_trace_{arm}.npz'))
    traces, summing_ranges = trace_data['traces'], trace_data['summing_ranges']
    # adjust summing ranges 
    ### TODO: take a close look at this summing ranges and prepare files that aren't modified?
    ### remember that traces can change between runs/nights if someone was fiddling with spectrograph
    summing_ranges_mod = np.array([[35,35] for _ in range(len(summing_ranges))])

    with fits.open(os.path.join(master_path, file_name)) as hdul:
        image_data = hdul[0].data
        image_subtracted_bias = veloce_reduction_tools.remove_overscan_bias(image_data, overscan_range=32)
        
        extracted_orders, extracted_order_imgs = veloce_reduction_tools.extract_orders_with_trace(image_subtracted_bias, traces, summing_ranges_mod, remove_background=False)
        if arm == 'green':
            blazes = np.array(extracted_orders[1:])
        else:
            blazes = np.array(extracted_orders)
    np.savez(os.path.join(blaze_path, f"blaze_{file_name.split('.')[0]}"), blazes=blazes)

if __name__ == '__main__':
    obs_list_filename = "/home/usqobserver2/Joachim_veloce/veloce_reduction/Obs_lists/obs_list_test_HD70703.pkl"
    extract_run_no_blaze(obs_list_filename, "23xmasRun", "green")