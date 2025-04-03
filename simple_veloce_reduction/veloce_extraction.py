from astropy.io import fits
import os #, sys
import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from csaps import csaps
# from scipy.ndimage import median_filter
# from astropy.stats import sigma_clip
# import pickle

from . import veloce_reduction_tools
from . import veloce_wavecalib
from . import veloce_diagnostic

data_dirs = {'red': 'ccd_3', 'green': 'ccd_2', 'blue': 'ccd_1'}

# class ExtractedSpectrum():
#     def __init__(self, wave, flux, hdr):
#         self.wave = wave
#         self.flux = flux
#         self.header = hdr

#     def save(self, filename):
#         np.savez(filename, wave=self.wave, flux=self.flux, hdr=self.header)

def load_trace_data(arm, trace_path, sim_calib=False):
    if sim_calib:
        filename = os.path.join(trace_path, f'veloce_{arm}_4amp_sim_calib_trace.pkl')
    else:
        filename = os.path.join(trace_path.trace_dir, f'veloce_{arm}_4amp_no_sim_calib_trace.pkl')
    traces = veloce_reduction_tools.Traces.load_traces(filename)
    return traces

def remove_scattered_light(frame, hdr, traces, diagnostic):
    """
    Remove scattered light from the image.
    """
    background_threshold = 20
    # this models scattered light and subtracts it
    background = veloce_reduction_tools.fit_background(frame, traces)
    # head = f'scattered light corrected\n---\nBackground statistics:\n---'
    # median_str = f'median = {np.median(background)}'
    # max_str = f'max = {np.max(background)}'
    # std_str = f'stdev = {np.std(background)}'
    # print('\n'.join([head, median_str, max_str, std_str]))
    corrected_frame = frame.copy()
    corrected_frame -= background
    corrected_frame[corrected_frame < 0] = 0
    
    if diagnostic:
        veloce_diagnostic.plotted_scattered_light(frame, background, corrected_frame, traces)
    
    ### TODO: add note to reduction log
    hdr['HISTORY'] = 'Scattered light corrected'
    if np.max(background) < background_threshold:
        print(f'[Warning] Scattered light correction: background level is below {background_threshold} ADU')

    return corrected_frame, hdr

def get_flat(config, veloce_paths, date, obs_list):
    norm_flat_filename = os.path.join(veloce_paths.master_dir, f'master_flat_{config['arm']}_{date}_norm.fits')
    if os.path.exists(norm_flat_filename):
        with fits.open(norm_flat_filename) as hdul:
            flat = hdul[0].data
            hdr = hdul[0].header
    else:
        master_flat_filename = os.path.join(veloce_paths.master_dir, f'master_flat_{config['arm']}_{date}.fits')
        if os.path.exists(master_flat_filename):
            with fits.open(master_flat_filename) as hdul:
                master_flat = hdul[0].data
                hdr = hdul[0].header
            ### TODO: swicht all flats to have removed overscan?
            master_flat = veloce_reduction_tools.remove_overscan_bias(
                master_flat, hdr, overscan_range=32, amplifier_mode=config['amplifier_mode'])
        else:
            master_flat, hdr = veloce_reduction_tools.get_master_mmap(
                obs_list, f"flat_{config['arm']}", veloce_paths.input_dir,
                date, config['arm'])
            veloce_reduction_tools.save_image_fits(master_flat_filename, master_flat, hdr)
            master_flat = veloce_reduction_tools.remove_overscan_bias(
                master_flat, hdr, overscan_range=32, amplifier_mode=config['amplifier_mode'])
        flat, hdr = veloce_reduction_tools.get_normalised_master_flat(master_flat, hdr)
        veloce_reduction_tools.save_image_fits(norm_flat_filename, flat, hdr)

    return flat

# def extract_run(obs_list, run, arm, amp_mode, sim_calib=False, remove_background=True, veloce_paths=None, output_path=None):
def extract_run(target_list, config, veloce_paths, obs_list):
    """
    Extracts spectral data from Veloce observations without applying blaze correction.

    This function processes observation data for a specific run and spectrograph arm from Veloce.
    It loads wave calibration data, trace data, and modifies summing ranges based on the
    specified arm. The function is designed to work with 'green' and 'red' arms, with a note to add
    support for the 'blue' arm in the future.

    Parameters:
    - obs_list_filename (str): The filename of the observation list to be processed.
    - run (str): The specific run identifier for which the data is to be extracted.
    - arm (str): The spectrograph arm to be processed.
    - amp_mode (int): The amplifier mode used for the observations.
    - sim_calib (bool, optional): A flag to indicate if observations were performed with simultaneous calibration.
    - remove_background (bool, optional): A flag to indicate if background subtraction is to be performed.
    - veloce_paths (VelocePaths, optional): An object containing paths to Veloce data directories.
    - output_path (str, optional): The path to save the extracted data. If None, a default path is used. [deprecated]

    Raises:
    - ValueError: If an unsupported arm is specified.

    Notes:
    - The function assumes a standard directory structure for storing Veloce data and extracted data.
    - Wave calibration is based on pre-fitted Thorium-Argon (ThAr) lamp observations.
    - Trace data is loaded from a predefined location and may need adjustments for summing ranges,
      especially if the spectrograph setup has been altered.

    Returns:
    None. The function is designed to perform data extraction and processing, with outputs
    saved to files (used in further processing).
    """
    # pick which arm to reduce
    if config['arm'] in data_dirs.keys():
        arms = [config['arm']]
    elif config['arm'] == 'all':
        arms = data_dirs.keys()
    else:
        raise ValueError('Unsupported arm')
    
    for arm in arms:
        ccd = data_dirs[arm]
        ### load traces
        traces = load_trace_data(arm, veloce_paths.trace_dir, sim_calib=config['sim_calib'])
        # if sim_calib:
        #     # trace_data = np.load(os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_4amp_trace.npz'))
        #     filename = os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_4amp_sim_calib_trace.pkl')
        #     traces = veloce_reduction_tools.Traces.load_traces(filename)
        # else:
        #     # trace_data = np.load(os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_4amp_no_sim_calib_trace.npz'))
        #     filename = os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_4amp_no_sim_calib_trace.pkl')
        #     traces = veloce_reduction_tools.Traces.load_traces(filename)

        ### load wave calibration based on ThAr
        ORDER, COEFFS, MATCH_LAM, MATCH_PIX, MATCH_LRES, GUESS_LAM, Y0 = \
            veloce_reduction_tools.load_prefitted_wave(arm=arm, wave_calib_slice=traces.wave_calib_slice,
                                                    wave_path=veloce_paths.wave_dir)
        if config['calib_type'] == 'Static':
            static_wave = veloce_reduction_tools.calibrate_orders_to_wave(None, Y0, COEFFS, traces=traces)
        elif config['calib_type'] == 'Interpolate':
            wave_interp_base = veloce_wavecalib.load_wave_calibration_for_interpolation()
        else:
            pass

        for date in target_list.keys(): 
            if config['flat_field']:
                flat = get_flat(config, veloce_paths, date, obs_list)
                
            for obs in target_list[date]:
                target, filename = obs
                print(target, filename)
                spectrum_filename =  os.path.join(veloce_paths.input_dir, date, ccd, filename)
                # spectrum_filename =  os.path.join(veloce_paths.input_dir, config['run'], date, ccd, filename)
                with fits.open(spectrum_filename) as hdul:
                    image_data = hdul[0].data
                    hdr = hdul[0].header

                    image_subtracted_bias = veloce_reduction_tools.remove_overscan_bias(
                        image_data, hdr, overscan_range=32, amplifier_mode=config['amplifier_mode'])
                    
                    if config['flat_field']:
                        image_subtracted_bias, hdr = veloce_reduction_tools.flat_field_correction(image_subtracted_bias, flat, hdr)
                    
                    if config['scattered_light']:
                        image_subtracted_bias, hdr = remove_scattered_light(image_subtracted_bias, hdr, traces)
                    
                    extracted_science_orders, extracted_order_imgs = veloce_reduction_tools.extract_orders_with_trace(
                        image_subtracted_bias, traces, remove_background=False)
                    
                    if config['calib_type'] == 'Static':
                        final_wave = static_wave
                    elif config['calib_type'] == 'Interpolate':
                        final_wave = veloce_wavecalib.interpolate_wave(
                            extracted_science_orders, hdr)
                    elif config['calib_type'] == 'SimThXe':
                        final_wave = veloce_wavecalib.calibrate_simTh(
                            extracted_science_orders, hdr)
                    elif config['calib_type'] == 'SimLC':
                        final_wave = veloce_wavecalib.calibrate_simLC(
                            extracted_science_orders, hdr)
                                    
                    final_flux = extracted_science_orders

                    if config['plot_diagnostic']:
                        veloce_diagnostic.plot_order_cross_section(
                            image_subtracted_bias, traces, 10, filename,
                            veloce_paths, plot_type='median')
                        if config['flat_field']:
                            veloce_diagnostic.plot_extracted_2D_order(
                                extracted_order_imgs, order=10, traces=traces, filename=filename,
                                veloce_paths=veloce_paths, flatfielded=True, flatfield=flat)
                        else:
                            veloce_diagnostic.plot_extracted_2D_order(
                                extracted_order_imgs, order=10, traces=traces, filename=filename,
                                veloce_paths=veloce_paths)

                    # save extracted spectrum as fits file
                    fits_filename = os.path.join(veloce_paths.output_dir, f"{target}_veloce_{arm}_{filename}")
                    veloce_reduction_tools.save_extracted_spectrum_fits(
                        filename=fits_filename, wave=final_wave, flux=final_flux, hdr=hdr)

# def extract_night(obs_list, run, date, arm, amp_mode, sim_calib=False, remove_background=True, veloce_paths=None, output_path=None):
def extract_night(target_list, config, veloce_paths, obs_list):
    """
    Extracts spectral data from Veloce observations for a specific night.

    This function processes observation data for a specific night and spectrograph arm from Veloce.
    It loads wave calibration data, trace data, and modifies summing ranges based on the specified arm.
    The function is designed to work with 'green' and 'red' arms, with a note to add support for the 'blue' arm in the future.

    Parameters:
    - obs_list_filename (str): The filename of the observation list to be processed.
    - run (str): The specific run identifier for which the data is to be extracted.
    - arm (str): The spectrograph arm to be processed.
    - amp_mode (int): The amplifier mode used for the observations.
    - sim_calib (bool, optional): A flag to indicate if observations were performed with simultaneous calibration.
    - remove_background (bool, optional): A flag to indicate if background subtraction is to be performed.
    - veloce_paths (VelocePaths, optional): An object containing paths to Veloce data directories.
    - output_path (str, optional): The path to save the extracted data. If None, a default path is used. [deprecated]

    Raises:
    - ValueError: If an unsupported arm is specified.

    Notes:
    - The function assumes a standard directory structure for storing Veloce data and extracted data.
    - Wave calibration is based on pre-fitted Thorium-Argon (ThAr) lamp observations.
    - Trace data is loaded from a predefined location and may need adjustments for summing ranges,
      especially if the spectrograph setup has been altered.

    Returns:
    None. The function is designed to perform data extraction and processing, with outputs
    saved to files (used in further processing).
    """
        # pick which arm to reduce
    if config['arm'] in data_dirs.keys():
        arms = [config['arm']]
    elif config['arm'] == 'all':
        arms = data_dirs.keys()
    else:
        raise ValueError('Unsupported arm')
    
    for arm in arms:
        ccd = data_dirs[arm]
        ### load traces
        traces = load_trace_data(arm, veloce_paths.trace_dir, sim_calib=config['sim_calib'])
        # if sim_calib:
        #     # trace_data = np.load(os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_4amp_trace.npz'))
        #     filename = os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_4amp_sim_calib_trace.pkl')
        #     traces = veloce_reduction_tools.Traces.load_traces(filename)
        # else:
        #     # trace_data = np.load(os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_4amp_no_sim_calib_trace.npz'))
        #     filename = os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_4amp_no_sim_calib_trace.pkl')
        #     traces = veloce_reduction_tools.Traces.load_traces(filename)

        # load wave calibration based on ThAr
        ORDER, COEFFS, MATCH_LAM, MATCH_PIX, MATCH_LRES, GUESS_LAM, Y0 = \
            veloce_reduction_tools.load_prefitted_wave(arm=arm, wave_calib_slice=traces.wave_calib_slice,
                                                        wave_path=veloce_paths.wave_dir)
        if config['calib_type'] == 'Static':
            static_wave = veloce_reduction_tools.calibrate_orders_to_wave(None, Y0, COEFFS, traces=traces)
        elif config['calib_type'] == 'Interpolate':
            wave_interp_base = veloce_wavecalib.load_wave_calibration_for_interpolation()
        else:
            pass
        
        # with open(os.path.join(os.path.join(veloce_paths.obs_list_dir, obs_list_filename)), 'rb') as f:
        #     obs_list = pickle.load(f)
        date = config['date']
        if config['flat_field']:
            flat = get_flat(config, veloce_paths, date, obs_list)

        for target, filename in target_list[date]:
            print(target, filename)
            spectrum_filename =  os.path.join(veloce_paths.input_dir, date, ccd, filename)
            # spectrum_filename =  os.path.join(veloce_paths.input_dir, config['run'], date, ccd, filename)
            with fits.open(spectrum_filename) as hdul:
                image_data = hdul[0].data
                hdr = hdul[0].header
                # times.append(hdr['MJD-OBS'])
                image_subtracted_bias = veloce_reduction_tools.remove_overscan_bias(
                    image_data, hdr, overscan_range=32, amplifier_mode=config['amplifier_mode'])
                
                if config['flat_field']:
                    image_subtracted_bias, hdr = veloce_reduction_tools.flat_field_correction(image_subtracted_bias, flat, hdr)

                if config['scattered_light']:
                    image_subtracted_bias, hdr = remove_scattered_light(image_subtracted_bias, hdr, traces)

                extracted_science_orders, extracted_order_imgs = veloce_reduction_tools.extract_orders_with_trace(
                    image_subtracted_bias, traces, remove_background=False)
                
                if config['calib_type'] == 'Static':
                    final_wave = static_wave
                elif config['calib_type'] == 'Interpolate':
                    final_wave = veloce_wavecalib.interpolate_wave(extracted_science_orders, hdr)
                elif config['calib_type'] == 'SimThXe':
                    final_wave = veloce_wavecalib.calibrate_simTh(extracted_science_orders, hdr)
                elif config['calib_type'] == 'SimLC':
                    final_wave = veloce_wavecalib.calibrate_simLC(extracted_science_orders, hdr)
                
                final_flux = extracted_science_orders

                if config['plot_diagnostic']:
                        veloce_diagnostic.plot_order_cross_section(
                            image_subtracted_bias, traces, 10, filename,
                            veloce_paths, plot_type='median')
                        if config['flat_field']:
                            veloce_diagnostic.plot_extracted_2D_order(
                                extracted_order_imgs, order=10, traces=traces, filename=filename,
                                veloce_paths=veloce_paths, flatfielded=True, flatfield=flat)
                        else:
                            veloce_diagnostic.plot_extracted_2D_order(
                                extracted_order_imgs, order=10, traces=traces, filename=filename,
                                veloce_paths=veloce_paths)

                # save extracted spectrum as fits file
                fits_filename = os.path.join(veloce_paths.output_dir, f"{target}_veloce_{arm}_{filename}")
                veloce_reduction_tools.save_extracted_spectrum_fits(
                    filename=fits_filename, wave=final_wave, flux=final_flux, hdr=hdr)

# def extract_single_file(filename, run, date, arm, amp_mode, sim_calib=False, remove_background=True, veloce_paths=None, output_path=None):
def extract_single_file(filename, config, veloce_paths, obs_list):
    # pick which arm to reduce
    if config['arm'] in data_dirs.keys():
        arms = [config['arm']]
    elif config['arm'] == 'all':
        arms = data_dirs.keys()
    else:
        raise ValueError('Unsupported arm')
    
    for arm in arms:
        ccd = data_dirs[arm]
        ### load traces
        traces = load_trace_data(config['arm'], veloce_paths.trace_dir, sim_calib=config['sim_calib'])
        # if sim_calib:
        #     # trace_data = np.load(os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_4amp_trace.npz'))
        #     filename = os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_4amp_sim_calib_trace.pkl')
        #     traces = veloce_reduction_tools.Traces.load_traces(filename)
        # else:
        #     # trace_data = np.load(os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_4amp_no_sim_calib_trace.npz'))
        #     filename = os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_4amp_no_sim_calib_trace.pkl')
        #     traces = veloce_reduction_tools.Traces.load_traces(filename)

        # load wave calibration based on ThAr
        ORDER, COEFFS, MATCH_LAM, MATCH_PIX, MATCH_LRES, GUESS_LAM, Y0 = \
            veloce_reduction_tools.load_prefitted_wave(arm=config['arm'], wave_calib_slice=traces.wave_calib_slice,
                                                        wave_path=veloce_paths.wave_dir)
        if config['calib_type'] == 'Static':
            static_wave = veloce_reduction_tools.calibrate_orders_to_wave(None, Y0, COEFFS, traces=traces)
        elif config['calib_type'] == 'Interpolate':
            wave_interp_base = veloce_wavecalib.load_wave_calibration_for_interpolation()
        else:
            pass

        print(filename)
        spectrum_filename =  os.path.join(veloce_paths.input_dir, config['date'], ccd, filename)
        with fits.open(spectrum_filename) as hdul:
            image_data = hdul[0].data
            hdr = hdul[0].header
            # times.append(hdr['MJD-OBS'])
            image_subtracted_bias = veloce_reduction_tools.remove_overscan_bias(
                image_data, hdr, overscan_range=32, amplifier_mode=config['amplifier_mode'])
            
            if config['flat_field']:
                flat = get_flat(config, veloce_paths, config['date'], obs_list)
                image_subtracted_bias, hdr = veloce_reduction_tools.flat_field_correction(image_subtracted_bias, flat, hdr)
            
            if config['scattered_light']:
                image_subtracted_bias, hdr = remove_scattered_light(image_subtracted_bias, hdr, traces)
            
            extracted_science_orders, extracted_order_imgs = veloce_reduction_tools.extract_orders_with_trace(
                image_subtracted_bias, traces, remove_background=False)
            
            if config['calib_type'] == 'Static':
                final_wave = static_wave
            elif config['calib_type'] == 'Interpolate':
                final_wave = veloce_wavecalib.interpolate_wave(extracted_science_orders, hdr)
            elif config['calib_type'] == 'SimThXe':
                final_wave = veloce_wavecalib.calibrate_simTh(extracted_science_orders, hdr)
            elif config['calib_type'] == 'SimLC':
                final_wave = veloce_wavecalib.calibrate_simLC(extracted_science_orders, hdr)
            
            final_flux = extracted_science_orders

            if config['plot_diagnostic']:
                        veloce_diagnostic.plot_order_cross_section(
                            image_subtracted_bias, traces, 10, filename,
                            veloce_paths, plot_type='median')
                        if config['flat_field']:
                            veloce_diagnostic.plot_extracted_2D_order(
                                extracted_order_imgs, order=10, traces=traces, filename=filename,
                                veloce_paths=veloce_paths, flatfielded=True, flatfield=flat)
                        else:
                            veloce_diagnostic.plot_extracted_2D_order(
                                extracted_order_imgs, order=10, traces=traces, filename=filename,
                                veloce_paths=veloce_paths)

            # save extracted spectrum as fits file
            fits_filename = os.path.join(veloce_paths.output_dir, f"veloce_{arm}_{filename}")
            veloce_reduction_tools.save_extracted_spectrum_fits(
                filename=fits_filename, wave=final_wave, flux=final_flux, hdr=hdr)

# def extract_run_with_blaze(obs_list_filename, run, arm, amp_mode, sim_calib=False, remove_background=True, blaze_path=None, veloce_paths=None, output_path=None):
#     """
#     Extracts spectral data from Veloce observations with blaze correction applied.

#     This function processes observation data for a specific run and spectrograph arm from Veloce.
#     It loads wave calibration data, trace data, and modifies summing ranges based on the specified arm.
#     The function also applies blaze correction using precomputed blaze functions. The function is designed
#     to work with 'green' and 'red' arms, with a note to add support for the 'blue' arm in the future.

#     Parameters:
#     - obs_list_filename (str): The filename of the observation list to be processed.
#     - run (str): The specific run identifier for which the data is to be extracted.
#     - arm (str): The spectrograph arm to be processed. Currently supports 'green' and 'red'.
#     - blaze_path (str, optional): The path to the blaze function file. If None, a default path is used.
#     - output_path (str, optional): The path to save the extracted data. If None, a default path is used.

#     Raises:
#     - ValueError: If an unsupported arm is specified.

#     Notes:
#     - The function assumes a standard directory structure for storing Veloce data and extracted data.
#     - Wave calibration is based on pre-fitted Thorium-Argon (ThAr) lamp observations.
#     - Trace data is loaded from a predefined location and may need adjustments for summing ranges,
#       especially if the spectrograph setup has been altered.
#     - Blaze correction is applied using precomputed blaze functions loaded from a specified or default path.

#     Returns:
#     None. The function is designed to perform data extraction and processing, with outputs
#     saved to files (used in further processing).
#     """
#     if veloce_paths is None:
#         veloce_paths = veloce_path.VelocePaths(run)
#     if output_path is None:
#         output_path = veloce_paths.extracted_dir
#         if not os.path.exists(output_path):
#             os.makedirs(output_path)

#     if arm == 'blue':
#         ccd = 'ccd_1'
#     elif arm == 'green':
#         ccd = 'ccd_2'
#     elif arm == 'red':
#         ccd = 'ccd_3'
#     else:
#         raise ValueError('Unsupported arm')
    
#     # load traces
#     if sim_calib:
#         # trace_data = np.load(os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_4amp_trace.npz'))
#         filename = os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_4amp_sim_calib_trace.pkl')
#         traces = veloce_reduction_tools.Traces.load_traces(filename)
#     else:
#         # trace_data = np.load(os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_4amp_no_sim_calib_trace.npz'))
#         filename = os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_4amp_no_sim_calib_trace.pkl')
#         traces = veloce_reduction_tools.Traces.load_traces(filename)
#     # traces, summing_ranges, wave_calib_slice = \
#     #     trace_data['traces'], trace_data['summing_ranges'], trace_data['wave_calib_slice']
#     # load wave calibration based on ThAr
#     ORDER, COEFFS, MATCH_LAM, MATCH_PIX, MATCH_LRES, GUESS_LAM, Y0 = \
#         veloce_reduction_tools.load_prefitted_wave(arm=arm, wave_calib_slice=traces.wave_calib_slice,
#                                                    wave_path=veloce_paths.wave_dir)

#     ### load blazes
#     if blaze_path in None:
#         blaze_path = os.path.join(veloce_paths.blaze_dir, f'veloce_blaze_{arm}_pix.npz')
#     blazes = np.load(blaze_path)
#     blazes = blazes['blazes']

#     with open(os.path.join(veloce_paths.obs_list_dir, obs_list_filename), 'rb') as f:
#         obs_list = pickle.load(f)
    
#     # number of pixels to discard from both sides
#     # (so only well wave calibrated high snr part in the middle is left)
#     cutoff = 1250
#     # pix_max = 4112
#     for date in obs_list.keys(): 
#         for obs in obs_list[date]:
#             target, filename = obs
#             print(target, filename)
#             spectrum_filename =  os.path.join(veloce_paths.input_dir, date, ccd, filename)
#             with fits.open(spectrum_filename) as hdul:
#                 image_data = hdul[0].data
#                 hdr = hdul[0].header
#                 # times.append(hdr['MJD-OBS'])
#                 image_subtracted_bias = veloce_reduction_tools.remove_overscan_bias(
#                     image_data, hdr, amplifier_mode=amp_mode, overscan_range=32)
#                 if remove_background:
#                     # this models scattered light and subtracts it
#                     background = veloce_reduction_tools.fit_background(image_subtracted_bias, traces)
#                     head = f'scattered light corrected\n---\nBackground statistics:\n---'
#                     median_str = f'median = {np.median(background)}'
#                     max_str = f'max = {np.max(background)}'
#                     std_str = f'stdev = {np.std(background)}'
#                     print('\n'.join([head, median_str, max_str, std_str]))
#                     image_subtracted_bias -= background
#                     image_subtracted_bias[image_subtracted_bias < 0] = 0
#                 extracted_science_orders, extracted_order_imgs = veloce_reduction_tools.extract_orders_with_trace(
#                     image_subtracted_bias, traces, remove_background=False)
#                 waves = veloce_reduction_tools.calibrate_orders_to_wave(
#                     extracted_science_orders, Y0, COEFFS, traces=traces)
#                 fluxes = extracted_science_orders
#                 # if arm == 'green':
#                 #     waves = np.array(veloce_reduction_tools.calibrate_orders_to_wave(
#                 #         extracted_science_orders, Y0[0], COEFFS))
#                 #     fluxes = np.array(extracted_science_orders[1:])
#                 #     # blazes = blazes[1:]
#                 # else:
#                 #     waves = np.array(veloce_reduction_tools.calibrate_orders_to_wave(
#                 #         extracted_science_orders, Y0[0], COEFFS))
#                 #     fluxes = np.array(extracted_science_orders)

#                 final_wave = []
#                 final_flux = []
#                 for wave, extracted_science_order, blaze in zip(waves, fluxes, blazes):
#                     y = np.array(blaze, dtype=np.float64)
#                     ysm = median_filter(y,50)
#                     ysm /= max(ysm)
#                     flux = extracted_science_order.copy()
#                     flux /= ysm
#                     flux /= np.median(flux)
#                     final_wave.append(wave[cutoff:-cutoff])
#                     final_flux.append(flux[cutoff:-cutoff])
#                 final_wave = np.array(final_wave)
#                 final_flux = np.array(final_flux)

#                 # save extracted spectrum as fits file
#                 fits_filename = os.path.join(output_path, f"{target}_veloce_{arm}_{filename}")
#                 veloce_reduction_tools.save_extracted_spectrum_fits(
#                     filename=fits_filename, output_path=output_path, wave=final_wave, flux=final_flux, hdr=hdr)
#                 # np.savez(
#                 #     os.path.join(output_path, f"{target}_veloce_{arm}_{filename.split('.')[0]}"),
#                 #     wave=final_wave, flux=final_flux, mjd=float(hdr['MJD-OBS']))

# def extract_blaze(file_name, arm, amp_mode, remove_background=True, blaze_path=None, master_path=None, veloce_paths=None):
#     """
#     Extracts blaze function from a master flat field file for a specific spectrograph arm.

#     This function processes a master flat field file to extract the blaze function for a specific
#     spectrograph arm. It loads trace data, adjusts summing ranges, and extracts the blaze function
#     from the flat field image. The extracted blaze function is saved to a specified or default path.

#     Parameters:
#     - file_name (str): The filename of the master flat field file to be processed.
#     - arm (str): The spectrograph arm to be processed. Currently supports 'green' and 'red'.
#     - blaze_path (str, optional): The path to save the extracted blaze function. If None, a default path is used.

#     Raises:
#     - ValueError: If an unsupported arm is specified.

#     Notes:
#     - The function assumes a standard directory structure for storing Veloce data and extracted data.
#     - Trace data is loaded from a predefined location and may need adjustments for summing ranges,
#       especially if the spectrograph setup has been altered.
#     - The blaze function is extracted by summing the flat field image over the adjusted summing ranges
#       for each trace.

#     Returns:
#     None. The function is designed to perform data extraction and processing, with outputs
#     saved to files (used in further processing).
#     """
#     if veloce_paths is None:
#         veloce_paths = veloce_path.VelocePaths()
#     if blaze_path is None:
#         blaze_path = veloce_paths.blaze_dir
#     if master_path is None:
#         master_path = veloce_paths.master_dir
    
#     # load traces
#     trace_data = np.load(os.path.join(veloce_paths.trace_dir, f'veloce_{arm}_4amp_trace.npz'))
#     traces, summing_ranges = trace_data['traces'], trace_data['summing_ranges']

#     with fits.open(os.path.join(master_path, file_name)) as hdul:
#         image_data = hdul[0].data
#         hdr = hdul[0].header
#         image_subtracted_bias = veloce_reduction_tools.remove_overscan_bias(image_data, hdr, amplifier_mode=amp_mode, overscan_range=32)
#         if remove_background:
#                     # this models scattered light and subtracts it
#                     background = veloce_reduction_tools.fit_background(image_subtracted_bias, traces)
#                     head = f'scattered light corrected\n---\nBackground statistics:\n---'
#                     median_str = f'median = {np.median(background)}'
#                     max_str = f'max = {np.max(background)}'
#                     std_str = f'stdev = {np.std(background)}'
#                     print('\n'.join([head, median_str, max_str, std_str]))
#               {target}_      image_subtracted_bias -= background
#                     image_subtracted_bias[image_subtracted_bias < 0] = 0
#         extracted_orders, extracted_order_imgs = veloce_reduction_tools.extract_orders_with_trace(image_subtracted_bias, traces, summing_ranges, remove_background=False)

#     np.savez(os.path.join(blaze_path, f"blaze_{file_name.split('.')[0]}"), blazes=np.array(extracted_orders))

if __name__ == '__main__':
    pass