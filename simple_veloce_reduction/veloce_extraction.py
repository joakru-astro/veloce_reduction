from astropy.io import fits
import os
import numpy as np

from . import veloce_reduction_tools
from . import veloce_wavecalib
from . import veloce_diagnostic

data_dirs = {'red': 'ccd_3', 'green': 'ccd_2', 'blue': 'ccd_1'}
arm_nums = {'red': 3, 'green': 2, 'blue': 1}

# class ExtractedSpectrum():
#     def __init__(self, wave, flux, hdr):
#         self.wave = wave
#         self.flux = flux
#         self.header = hdr

#     def save(self, filename):
#         np.savez(filename, wave=self.wave, flux=self.flux, hdr=self.header)

def load_trace_data(arm, trace_path, sim_calib=False, filename=None):
    if filename == 'Default':
        if sim_calib:
            filename = os.path.join(trace_path, f'veloce_{arm}_4amp_sim_calib_trace.pkl')
        else:
            filename = os.path.join(trace_path.trace_dir, f'veloce_{arm}_4amp_no_sim_calib_trace.pkl')
    else:
        if arm.lower() not in filename.lower():
            # only checks the filename at this stage
            raise ValueError(f"Trace data filename '{filename}' does not match selected arm '{arm}'.")
        if not os.path.abspath(filename):
            # if not absolute path, assume it is relative to trace_path
            # filename = os.path.join(trace_path, filename)
            filename = os.path.join(trace_path, filename)
    traces = veloce_reduction_tools.Traces.load_traces(filename)
    return traces

def remove_scattered_light(frame, hdr, traces, diagnostic=False):
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

def get_flat(veloce_paths, arm, amplifier_mode, date, obs_list):
    norm_flat_filename = os.path.join(veloce_paths.master_dir, f'master_flat_{arm}_{date}_norm.fits')
    if os.path.exists(norm_flat_filename):
        with fits.open(norm_flat_filename) as hdul:
            flat = hdul[0].data
            hdr = hdul[0].header
    else:
        master_flat_filename = os.path.join(veloce_paths.master_dir, f'master_flat_{arm}_{date}.fits')
        if os.path.exists(master_flat_filename):
            with fits.open(master_flat_filename) as hdul:
                master_flat = hdul[0].data
                hdr = hdul[0].header
            ### TODO: swicht all flats to have removed overscan?
            master_flat = veloce_reduction_tools.remove_overscan_bias(
                master_flat, hdr, overscan_range=32, amplifier_mode=amplifier_mode)
        else:
            master_flat, hdr = veloce_reduction_tools.get_master_mmap(
                obs_list, f"flat_{arm}", veloce_paths.input_dir,
                date, arm)
            veloce_reduction_tools.save_image_fits(master_flat_filename, master_flat, hdr)
            master_flat = veloce_reduction_tools.remove_overscan_bias(
                master_flat, hdr, overscan_range=32, amplifier_mode=amplifier_mode)
        flat, hdr = veloce_reduction_tools.get_normalised_master_flat(master_flat, hdr)
        veloce_reduction_tools.save_image_fits(norm_flat_filename, flat, hdr)

    return flat

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
        print(arm)
        ccd = data_dirs[arm]
        ### load traces
        traces = load_trace_data(arm, veloce_paths.trace_dir, sim_calib=config['sim_calib'], filename=config['trace_file'])

        ### load wave calibration based on ThAr
        ORDER, COEFFS, MATCH_LAM, MATCH_PIX, MATCH_LRES, GUESS_LAM, Y0 = \
            veloce_reduction_tools.load_prefitted_wave(arm=arm, wave_calib_slice=traces.wave_calib_slice,
                                                    wave_path=veloce_paths.wave_dir)
        if config['calib_type'] == 'Static':
            static_wave = veloce_reduction_tools.calibrate_orders_to_wave(None, Y0, COEFFS, traces=traces)
        elif config['calib_type'] == 'Interpolate':
            wave_interp_base = veloce_wavecalib.load_wave_calibration_for_interpolation(target_list, obs_list, veloce_paths)
        else:
            pass #load LC reference

        for date in target_list.keys(): 
            if config['flat_field']:
                flat = get_flat(config, veloce_paths, arm, config['amplifier_mode'], date, obs_list)
                
            for obs in target_list[date]:
                target, filename = obs
                if int(filename[5]) == arm_nums[arm]:
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
                            vacuum_wave = static_wave
                            final_flux = extracted_science_orders
                        elif config['calib_type'] == 'Interpolate':
                            vacuum_wave, final_flux = veloce_wavecalib.interpolate_wave(
                                extracted_science_orders, hdr)
                        elif config['calib_type'] == 'SimThXe':
                            vacuum_wave = veloce_wavecalib.calibrate_simTh(
                                extracted_science_orders, hdr)
                            final_flux = extracted_science_orders
                        elif config['calib_type'] == 'SimLC':
                            vacuum_wave, final_flux = veloce_wavecalib.calibrate_simLC(
                                extracted_science_orders, veloce_paths, image_subtracted_bias,
                                hdr, arm, plot=config['plot_diagnostic'])
                        
                        final_wave = [veloce_reduction_tools.vacuum_to_air(wave) for wave in vacuum_wave]
                        
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
        print(arm)
        ccd = data_dirs[arm]
        ### load traces
        traces = load_trace_data(arm, veloce_paths.trace_dir, sim_calib=config['sim_calib'], filename=config['trace_file'])

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

        date = config['date']
        if config['flat_field']:
            flat = get_flat(config, veloce_paths, arm, config['amplifier_mode'], date, obs_list)

        for target, filename in target_list[date]:
        # for target, filename in target_list:
            if int(filename[5]) == arm_nums[arm]:
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
                        vacuum_wave = static_wave
                        final_flux = extracted_science_orders
                    elif config['calib_type'] == 'Interpolate':
                        vacuum_wave = veloce_wavecalib.interpolate_wave(extracted_science_orders, hdr)                    
                        final_flux = extracted_science_orders
                    elif config['calib_type'] == 'SimThXe':
                        vacuum_wave = veloce_wavecalib.calibrate_simTh(extracted_science_orders, hdr)                    
                        final_flux = extracted_science_orders
                    elif config['calib_type'] == 'SimLC':
                            vacuum_wave, final_flux = veloce_wavecalib.calibrate_simLC(
                                extracted_science_orders, veloce_paths, image_subtracted_bias,
                                hdr, arm, plot=config['plot_diagnostic'])
                        
                    final_wave = [veloce_reduction_tools.vacuum_to_air(wave) for wave in vacuum_wave]

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

def extract_single_file(filename, config, veloce_paths, obs_list):
    # pick which arm to reduce
    if config['arm'] in data_dirs.keys():
        arms = [config['arm']]
    elif config['arm'] == 'all':
        raise ValueError('Cannot use "all" for single file extraction')
        # arms = data_dirs.keys()
        # can manipulate filename to match arm and make it work but then it's extract exposure and not file
    else:
        raise ValueError('Unsupported arm')
    
    for arm in arms:
        print(arm)
        ccd = data_dirs[arm]
        ### load traces
        traces = load_trace_data(arm, veloce_paths.trace_dir, sim_calib=config['sim_calib'], filename=config['trace_file'])
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

        if int(filename[5]) == arm_nums[arm]:
            print(filename)

            spectrum_filename =  os.path.join(veloce_paths.input_dir, config['date'], ccd, filename)

            with fits.open(spectrum_filename) as hdul:
                image_data = hdul[0].data
                hdr = hdul[0].header
                # times.append(hdr['MJD-OBS'])
                image_subtracted_bias = veloce_reduction_tools.remove_overscan_bias(
                    image_data, hdr, overscan_range=32, amplifier_mode=config['amplifier_mode'])
                
                if config['flat_field']:
                    flat = get_flat(config, veloce_paths, arm, config['amplifier_mode'], config['date'], obs_list)
                    image_subtracted_bias, hdr = veloce_reduction_tools.flat_field_correction(image_subtracted_bias, flat, hdr)
                
                if config['scattered_light']:
                    image_subtracted_bias, hdr = remove_scattered_light(image_subtracted_bias, hdr, traces)
                
                extracted_science_orders, extracted_order_imgs = veloce_reduction_tools.extract_orders_with_trace(
                    image_subtracted_bias, traces, remove_background=False)
      
                if config['calib_type'] == 'Static':
                    vacuum_wave = static_wave                    
                    final_flux = extracted_science_orders
                elif config['calib_type'] == 'Interpolate':
                    vacuum_wave = veloce_wavecalib.interpolate_wave(extracted_science_orders, hdr)                    
                    final_flux = extracted_science_orders
                elif config['calib_type'] == 'SimThXe':
                    vacuum_wave = veloce_wavecalib.calibrate_simTh(extracted_science_orders, hdr)                    
                    final_flux = extracted_science_orders
                elif config['calib_type'] == 'SimLC':
                    vacuum_wave, final_flux = veloce_wavecalib.calibrate_simLC(
                                extracted_science_orders, veloce_paths, image_subtracted_bias,
                                hdr, arm, plot=config['plot_diagnostic'])
                final_wave = [veloce_reduction_tools.vacuum_to_air(wave) for wave in vacuum_wave]

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
                
if __name__ == '__main__':
    pass
