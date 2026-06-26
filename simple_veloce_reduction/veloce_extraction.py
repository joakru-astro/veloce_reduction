from astropy.io import fits
import os
import numpy as np

from . import veloce_reduction_tools
from . import veloce_wavecalib
from . import veloce_diagnostic
from . import veloce_normalise

data_dirs = {'red': 'ccd_3', 'green': 'ccd_2', 'blue': 'ccd_1'}
arm_nums = {'red': 3, 'green': 2, 'blue': 1}

def extract_run(target_list, config, veloce_paths, obs_list):
    """
    Extracts spectral data from Veloce observations for a specific run without applying blaze correction.
    
    This function processes observation data for a specific run and spectrograph arm from Veloce.
    It loads wave calibration data, trace data, and modifies summing ranges based on the
    specified arm. The function is designed to work with 'green', 'red', and 'blue' arms.

    Parameters:
    - target_list (dict): A dictionary containing observation targets grouped by date.
    - config (dict): Configuration settings for the extraction process, including arm, amplifier mode, calibration type, etc.
    - veloce_paths (VelocePaths): An object containing paths to Veloce data directories.
    - obs_list (list): A list of observation metadata for the run.

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
    if config['arm'] == 'all':
        if 'LC' in config['calib_type']:
            arms = ['red', 'green'] # LC calibration not supported for blue arm
        else:
            arms = list(data_dirs.keys())
    elif config['arm'] in data_dirs.keys():
        arms = [config['arm']]
    else:
        raise ValueError('Unsupported arm')
    
    for arm in arms:
        print(arm)
        ccd = data_dirs[arm]
        ### load traces
        traces = veloce_reduction_tools.load_trace_data(arm, veloce_paths.trace_dir, filename=config['trace_file'])
        if config['validate_trace']:
            traces = veloce_reduction_tools.get_trace_shift(traces, veloce_paths, arm, config['amplifier_mode'], obs_list)
        
        if config['calib_type'] == 'Static':
            ### load static wave calibration based on ThAr
            # ORDER, COEFFS, MATCH_LAM, MATCH_PIX, MATCH_LRES, GUESS_LAM, Y0 = \
            #     veloce_reduction_tools.load_prefitted_wave(
            #         arm=arm, wave_calib_slice=traces.wave_calib_slice, wave_path=veloce_paths.wave_dir)
            # static_wave = veloce_reduction_tools.calibrate_orders_to_wave(None, Y0, COEFFS, traces=traces)
            static_wave, ref_arcTh, _ = veloce_reduction_tools.load_extracted_spectrum_fits(
                os.path.join(veloce_paths.wave_dir, f"arcTh_wave_{arm}_230828.fits"))
            veloce_wavecalib.add_predetermined_resolution(arm, date, veloce_paths)
        elif config['calib_type'] == 'SimLC':
            veloce_wavecalib.add_predetermined_resolution(arm, date, veloce_paths)
        # elif config['calib_type'] == 'Interpolate':
        #     wave_interp_base = veloce_wavecalib.load_wave_calibration_for_interpolation(target_list, obs_list, veloce_paths)

        for date in target_list.keys(): 
            if config['flat_field']:
                flat, norm_flat = veloce_reduction_tools.get_flat(veloce_paths, arm, config['amplifier_mode'], date, obs_list)
            # traces.adjust_traces_with_ccf(flat, arm)

            if config['calib_type'] == 'arcTh':
                arcTh_wave = veloce_wavecalib.calibrate_absolute_Th(traces, veloce_paths, obs_list,
                                                   date, arm, config['amplifier_mode'],
                                                   estimate_resolution=config['estimate_resolution'],
                                                   plot=config['plot_diagnostic'], plot_filename=f'arcTh_wavecalib_{arm}_{date}',
                                                   th_linelist_filename='Default')
            elif config['calib_type'] == 'buildLC':
                if arm == 'blue':
                    print('[Warning] LC calibration not supported for blue arm')
                    continue
                vacuum_wave, calibrated_pixels_per_order = veloce_wavecalib.build_LC_wavelength_solution(
                    traces, veloce_paths, date,
                    arm, config['amplifier_mode'], obs_list,
                    estimate_resolution=config['estimate_resolution'], plot=config['plot_diagnostic'], filename=None)
                          
            for obs in target_list[date]:
                target, filename = obs
                if int(filename[5]) == arm_nums[arm]:
                    print(target, filename)
                    spectrum_filename =  os.path.join(veloce_paths.input_dir, date, ccd, filename)
                    with fits.open(spectrum_filename) as hdul:
                        image_data = hdul[0].data
                        hdr = hdul[0].header

                        image_subtracted_bias  = veloce_reduction_tools.remove_overscan_bias(
                            image_data, hdr, arm, config['amplifier_mode'], overscan_range=32)
                        
                        if config['flat_field']:
                            image_subtracted_bias, hdr = veloce_reduction_tools.flat_field_correction(image_subtracted_bias, norm_flat, hdr)
                        
                        if config['scattered_light']:
                            image_subtracted_bias, hdr, background = veloce_reduction_tools.remove_scattered_light(image_subtracted_bias, hdr, traces)
                        
                        extracted_science_orders, extracted_uncertainties_orders, extracted_order_imgs = veloce_reduction_tools.extract_orders_with_trace(
                            image_subtracted_bias, traces, remove_background=False)
                        
                        if config['calib_type'] == 'Static':
                            final_wave = static_wave
                            final_flux = extracted_science_orders
                            final_uncertainty = extracted_uncertainties_orders
                        # elif config['calib_type'] == 'Interpolate':
                        #     vacuum_wave, final_flux = veloce_wavecalib.interpolate_wave(
                        #         extracted_science_orders, hdr)
                        # elif config['calib_type'] == 'SimThXe':
                        #     vacuum_wave = veloce_wavecalib.calibrate_simTh(
                        #         extracted_science_orders, hdr)
                        #     final_flux = extracted_science_orders
                        elif config['calib_type'] == 'SimLC':
                            vacuum_wave, final_flux, final_uncertainty = veloce_wavecalib.calibrate_simLC(
                                extracted_science_orders, veloce_paths, image_subtracted_bias,
                                hdr, arm, plot=config['plot_diagnostic'], filename=filename, flux_error=extracted_uncertainties_orders)
                            final_wave = [veloce_reduction_tools.vacuum_to_air(wave) for wave in vacuum_wave]
                        elif config['calib_type'] == 'buildLC':
                            final_wave = [veloce_reduction_tools.vacuum_to_air(wave) for wave in vacuum_wave]
                            final_flux = [flux[calibrated_pixels] for flux, calibrated_pixels in zip(extracted_science_orders, calibrated_pixels_per_order) if np.any(calibrated_pixels)]
                            final_uncertainty = [err[calibrated_pixels] for err, calibrated_pixels in zip(extracted_uncertainties_orders, calibrated_pixels_per_order) if np.any(calibrated_pixels)]
                        elif config['calib_type'] == 'arcTh':
                            final_wave, final_flux, final_uncertainty = arcTh_wave, extracted_science_orders, extracted_uncertainties_orders
                        else:
                            # TODO: change to best available(?), same for if missing files/good data
                            raise ValueError('Unsupported calib_type')
                        
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
                            if config['scattered_light']:
                                veloce_diagnostic.plotted_scattered_light(image_subtracted_bias+background, background, image_subtracted_bias, traces, filename=filename, veloce_paths=veloce_paths)

                        # save extracted spectrum as fits file
                        fits_filename = os.path.join(veloce_paths.extracted_spectra_dir, f"{target}_veloce_{arm}_{filename}")
                        veloce_reduction_tools.save_extracted_spectrum_fits(
                            filename=fits_filename, wave=final_wave, flux=final_flux, hdr=hdr, err=final_uncertainty)
    if config['merge_orders']:
        veloce_normalise.merge_veloce_orders(veloce_paths, target_list, arms, config, obs_list, single_file=False, skip_edge_orders=True)

def extract_night(target_list, config, veloce_paths, obs_list):
    """
    Extracts spectral data from Veloce observations for a specific night.

    This function processes observation data for a specific night and spectrograph arm from Veloce.
    It loads wave calibration data, trace data, and modifies summing ranges based on the specified arm.
    The function is designed to work with 'green' and 'red' arms, with a note to add support for the 'blue' arm in the future.
                                                     
    Parameters:
    - target_list (dict): A dictionary containing observation targets grouped by date.
    - config (dict): Configuration settings for the extraction process, including arm, amplifier mode, calibration type, etc.
    - veloce_paths (VelocePaths): An object containing paths to Veloce data directories.
    - obs_list (list): A list of observation metadata for the night.

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
    if config['arm'] == 'all':
        if 'LC' in config['calib_type']:
            arms = ['red', 'green'] # LC calibration not supported for blue arm
        else:
            arms = list(data_dirs.keys())
    elif config['arm'] in data_dirs.keys():
        arms = [config['arm']]
    else:
        raise ValueError('Unsupported arm')
    
    for arm in arms:
        print(f"Processing arm: {arm}")
        date = config['date']
        ccd = data_dirs[arm]
        ### load traces
        traces = veloce_reduction_tools.load_trace_data(arm, veloce_paths.trace_dir, filename=config['trace_file'])
        if config['validate_trace']:
            traces = veloce_reduction_tools.get_trace_shift(traces, veloce_paths, arm, config['amplifier_mode'], obs_list)
        
        if config['flat_field']:
            flat, norm_flat = veloce_reduction_tools.get_flat(veloce_paths, arm, config['amplifier_mode'], date, obs_list)
        # traces.adjust_traces_with_ccf(flat, arm)

        if config['calib_type'] == 'Static':
            # load wave calibration based on ThAr
            # ORDER, COEFFS, MATCH_LAM, MATCH_PIX, MATCH_LRES, GUESS_LAM, Y0 = \
            #     veloce_reduction_tools.load_prefitted_wave(
            #         arm=arm, wave_calib_slice=traces.wave_calib_slice, wave_path=veloce_paths.wave_dir)
            # static_wave = veloce_reduction_tools.calibrate_orders_to_wave(None, Y0, COEFFS, traces=traces)
            static_wave, ref_arcTh, _ = veloce_reduction_tools.load_extracted_spectrum_fits(
                os.path.join(veloce_paths.wave_dir, f"arcTh_wave_{arm}_230828.fits"))
            veloce_wavecalib.add_predetermined_resolution(arm, date, veloce_paths)
        elif config['calib_type'] == 'arcTh':
            arcTh_wave = veloce_wavecalib.calibrate_absolute_Th(traces, veloce_paths, obs_list,
                                               date, arm, config['amplifier_mode'],
                                               plot=config['plot_diagnostic'], plot_filename=f'arcTh_wavecalib_{arm}_{date}',
                                               th_linelist_filename='Default')
        elif config['calib_type'] == 'buildLC':
                if arm == 'blue':
                    print('[Warning] LC calibration not supported for blue arm')
                    continue
                vacuum_wave, calibrated_pixels_per_order = veloce_wavecalib.build_LC_wavelength_solution(
                    traces, veloce_paths, date,
                    arm, config['amplifier_mode'], obs_list,
                    estimate_resolution=config['estimate_resolution'], plot=config['plot_diagnostic'], filename=None)
        elif config['calib_type'] == 'SimLC':
            veloce_wavecalib.add_predetermined_resolution(arm, date, veloce_paths)
        # elif config['calib_type'] == 'Interpolate':
        #     wave_interp_base = veloce_wavecalib.load_wave_calibration_for_interpolation() 

        for target, filename in target_list[date]:
        # for target, filename in target_list:
            if int(filename[5]) == arm_nums[arm]:
                print(target, filename)
                spectrum_filename =  os.path.join(veloce_paths.input_dir, date, ccd, filename)
                with fits.open(spectrum_filename) as hdul:
                    image_data = hdul[0].data
                    hdr = hdul[0].header
                    # times.append(hdr['MJD-OBS'])
                    image_subtracted_bias = veloce_reduction_tools.remove_overscan_bias(
                            image_data, hdr, arm, config['amplifier_mode'], overscan_range=32)
                    
                    if config['flat_field']:
                        image_subtracted_bias, hdr = veloce_reduction_tools.flat_field_correction(image_subtracted_bias, norm_flat, hdr)

                    if config['scattered_light']:
                        image_subtracted_bias, hdr = veloce_reduction_tools.remove_scattered_light(image_subtracted_bias, hdr, traces)

                    extracted_science_orders,  extracted_uncertainties_orders, extracted_order_imgs = veloce_reduction_tools.extract_orders_with_trace(
                        image_subtracted_bias, traces, remove_background=False)
                    
                    if config['calib_type'] == 'Static':
                        final_wave = static_wave
                        final_flux = extracted_science_orders
                        final_uncertainty = extracted_uncertainties_orders
                    # elif config['calib_type'] == 'Interpolate':
                    #     vacuum_wave = veloce_wavecalib.interpolate_wave(extracted_science_orders, hdr)                    
                    #     final_flux = extracted_science_orders
                    # elif config['calib_type'] == 'SimThXe':
                    #     vacuum_wave = veloce_wavecalib.calibrate_simTh(extracted_science_orders, hdr)                    
                    #     final_flux = extracted_science_orders
                    elif config['calib_type'] == 'SimLC':
                        vacuum_wave, final_flux, final_uncertainty = veloce_wavecalib.calibrate_simLC(
                                extracted_science_orders, veloce_paths, image_subtracted_bias,
                                hdr, arm, flux_error=extracted_uncertainties_orders, plot=config['plot_diagnostic'], filename=filename)
                        final_wave = [veloce_reduction_tools.vacuum_to_air(wave) for wave in vacuum_wave]
                    elif config['calib_type'] == 'buildLC':
                        final_wave = [veloce_reduction_tools.vacuum_to_air(wave) for wave in vacuum_wave]
                        final_flux = [flux[calibrated_pixels] for flux, calibrated_pixels in zip(extracted_science_orders, calibrated_pixels_per_order) if np.any(calibrated_pixels)]
                        final_uncertainty = [flux_err[calibrated_pixels] for flux_err, calibrated_pixels in zip(extracted_uncertainties_orders, calibrated_pixels_per_order) if np.any(calibrated_pixels)]
                    elif config['calib_type'] == 'arcTh':
                        final_wave, final_flux, final_uncertainty = arcTh_wave, extracted_science_orders, extracted_uncertainties_orders
                    else:
                        raise ValueError('Unsupported calib_type')
                    
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
                    fits_filename = os.path.join(veloce_paths.extracted_spectra_dir, f"{target}_veloce_{arm}_{filename}")
                    veloce_reduction_tools.save_extracted_spectrum_fits(
                        filename=fits_filename, wave=final_wave, flux=final_flux, hdr=hdr, err=final_uncertainty)
    if config['merge_orders']:
        veloce_normalise.merge_veloce_orders(veloce_paths, target_list, arms, config, obs_list, single_file=False, skip_edge_orders=True)

def extract_single_file(filename, config, veloce_paths, obs_list):
    """
    Extracts the spectrum from a single file for a specified arm of the Veloce spectrograph.
    This function processes a single file by loading trace data, applying calibrations, 
    removing overscan bias, performing flat-field correction, removing scattered light, 
    and extracting spectral orders. The extracted spectrum is saved as a FITS file.
    
    Parameters:
        - filename (str): Name of the file to be processed.
        - config (dict): Configuration settings for the extraction process, including arm, amplifier mode, calibration type, etc.
        - veloce_paths (VelocePaths): An object containing paths to Veloce data directories.
        - obs_list (list): List of observation metadata.
    
    Raises:
        - ValueError: If an unsupported arm is specified or if 'all' is used for single file extraction.
    
    Returns:
        - None: The extracted spectrum is saved as a FITS file in the specified output directory.
    """
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
        print(f"Processing {arm} arm")
        ccd = data_dirs[arm]
        ### load traces
        traces = veloce_reduction_tools.load_trace_data(arm, veloce_paths.trace_dir, filename=config['trace_file'])
        if config['validate_trace']:
            traces = veloce_reduction_tools.get_trace_shift(traces, veloce_paths, arm, config['amplifier_mode'], obs_list)
        
        if config['calib_type'] == 'Static':
            # load static wave calibration based on ThAr
            # ORDER, COEFFS, MATCH_LAM, MATCH_PIX, MATCH_LRES, GUESS_LAM, Y0 = veloce_reduction_tools.load_prefitted_wave(
            #     arm=arm, wave_calib_slice=traces.wave_calib_slice, wave_path=veloce_paths.wave_dir)
            # static_wave = veloce_reduction_tools.calibrate_orders_to_wave(None, Y0, COEFFS, traces=traces)
            static_wave, ref_arcTh, _ = veloce_reduction_tools.load_extracted_spectrum_fits(
                os.path.join(veloce_paths.wave_dir, f"arcTh_wave_{arm}_230828.fits"))
            veloce_wavecalib.add_predetermined_resolution(arm, config['date'], veloce_paths)
        # elif config['calib_type'] == 'Interpolate':
        #     wave_interp_base = veloce_wavecalib.load_wave_calibration_for_interpolation()
        elif config['calib_type'] == 'arcTh':
            arcTh_wave = veloce_wavecalib.calibrate_absolute_Th(traces, veloce_paths, obs_list,
                                               config['date'], arm, config['amplifier_mode'],
                                               plot=config['plot_diagnostic'], plot_filename=f'arcTh_wavecalib_{arm}_{config["date"]}',
                                               th_linelist_filename='Default')
        elif config['calib_type'] == 'buildLC':
                if arm == 'blue':
                    raise ValueError('buildLC calibration not supported for blue arm')
                vacuum_wave, calibrated_pixels_per_order = veloce_wavecalib.build_LC_wavelength_solution(
                    traces, veloce_paths, config['date'],
                    arm, config['amplifier_mode'], obs_list,
                    estimate_resolution=config['estimate_resolution'], plot=config['plot_diagnostic'], filename=None)   
        elif config['calib_type'] == 'SimLC':
            veloce_wavecalib.add_predetermined_resolution(arm, config['date'], veloce_paths)

        if int(filename[5]) == arm_nums[arm]:
            print(filename)

            spectrum_filename =  os.path.join(veloce_paths.input_dir, config['date'], ccd, filename)

            with fits.open(spectrum_filename) as hdul:
                image_data = hdul[0].data
                hdr = hdul[0].header
                # times.append(hdr['MJD-OBS'])
                image_subtracted_bias = veloce_reduction_tools.remove_overscan_bias(
                            image_data, hdr, arm, config['amplifier_mode'], overscan_range=32)
                
                if config['flat_field']:
                    flat, norm_flat = veloce_reduction_tools.get_flat(veloce_paths, arm, config['amplifier_mode'], config['date'], obs_list)
                    image_subtracted_bias, hdr = veloce_reduction_tools.flat_field_correction(image_subtracted_bias, norm_flat, hdr)
                
                if config['scattered_light']:
                    image_subtracted_bias, hdr = veloce_reduction_tools.remove_scattered_light(image_subtracted_bias, hdr, traces)
                
                extracted_science_orders, extracted_uncertainties_orders, extracted_order_imgs = veloce_reduction_tools.extract_orders_with_trace(
                    image_subtracted_bias, traces, remove_background=False)
      
                if config['calib_type'] == 'Static':
                    final_wave = static_wave                    
                    final_flux = extracted_science_orders
                    final_uncertainty = extracted_uncertainties_orders
                # elif config['calib_type'] == 'Interpolate':
                #     vacuum_wave = veloce_wavecalib.interpolate_wave(extracted_science_orders, hdr)                    
                #     final_flux = extracted_science_orders
                # elif config['calib_type'] == 'SimThXe':
                #     vacuum_wave = veloce_wavecalib.calibrate_simTh(extracted_science_orders, hdr)                    
                #     final_flux = extracted_science_orders
                elif config['calib_type'] == 'SimLC':
                    vacuum_wave, final_flux, final_uncertainty = veloce_wavecalib.calibrate_simLC(
                                extracted_science_orders, veloce_paths, image_subtracted_bias,
                                hdr, arm, plot=config['plot_diagnostic'], filename=filename, flux_error=extracted_uncertainties_orders)
                    final_wave = [veloce_reduction_tools.vacuum_to_air(wave) for wave in vacuum_wave]
                elif config['calib_type'] == 'buildLC':
                    final_wave = [veloce_reduction_tools.vacuum_to_air(wave) for wave in vacuum_wave]
                    final_flux = [flux[calibrated_pixels] for flux, calibrated_pixels in zip(extracted_science_orders, calibrated_pixels_per_order) if np.any(calibrated_pixels)]
                    final_uncertainty = [err[calibrated_pixels] for err, calibrated_pixels in zip(extracted_uncertainties_orders, calibrated_pixels_per_order) if np.any(calibrated_pixels)]
                elif config['calib_type'] == 'arcTh':
                    final_wave, final_flux, final_uncertainty = arcTh_wave, extracted_science_orders, extracted_uncertainties_orders
                else:
                    raise ValueError('Unsupported calib_type')

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
                fits_filename = os.path.join(veloce_paths.extracted_spectra_dir, f"veloce_{arm}_{filename}")
                veloce_reduction_tools.save_extracted_spectrum_fits(
                    filename=fits_filename, wave=final_wave, flux=final_flux, hdr=hdr, err=final_uncertainty)
    if config['merge_orders']:
        veloce_normalise.merge_veloce_orders(veloce_paths, hdr['OBJECT'], arms, config, obs_list, single_file=fits_filename, skip_edge_orders=True)


if __name__ == '__main__':
    pass
