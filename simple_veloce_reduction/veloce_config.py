import os
import yaml

from astropy.io import fits
import os, re

from datetime import datetime

data_dirs = {'red': 'ccd_3', 'green': 'ccd_2', 'blue': 'ccd_1'}

class VelocePaths:
    def __init__(self, input_dir=None, output_dir=None):
        # self.reduction_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.reduction_parent_dir = os.path.dirname(os.path.abspath(__file__))
        
        if input_dir is not None:
            input_dir = os.path.abspath(input_dir)
            if not os.path.exists(input_dir):
                raise FileNotFoundError(f"Input path does not exist: {input_dir}")
            else:
                self.input_dir = input_dir

        if output_dir is not None:
            output_dir = os.path.abspath(output_dir)
            if not os.path.exists(output_dir):
                # raise FileNotFoundError(f'Output path does not exist: {output_dir}')
                os.makedirs(output_dir)
                self.output_dir = output_dir
            else:
                self.output_dir = output_dir
        
        self.wave_dir = os.path.join(self.reduction_parent_dir, 'Wave')
        self.trace_dir = os.path.join(self.reduction_parent_dir, 'Trace')
        
        if output_dir is not None:
            self.intermediate_dir = os.path.join(self.output_dir, 'Intermediate_results')
            if not os.path.exists(self.intermediate_dir):
                os.makedirs(self.intermediate_dir)
            self.master_dir = os.path.join(self.intermediate_dir, 'Master')
            if not os.path.exists(self.master_dir):
                os.makedirs(self.master_dir)
            self.plot_dir = os.path.join(self.intermediate_dir, 'Plots')
            if not os.path.exists(self.plot_dir):
                os.makedirs(self.plot_dir)
            self.trace_shift_dir = os.path.join(self.intermediate_dir, 'Trace_shifts')
            if not os.path.exists(self.trace_shift_dir):
                os.makedirs(self.trace_shift_dir)
            self.wavelength_calibration_dir = os.path.join(self.intermediate_dir, 'Wavelength_calibration')
            if not os.path.exists(self.wavelength_calibration_dir):
                os.makedirs(self.wavelength_calibration_dir)

    @classmethod
    def from_config(cls, config):
        
        paths = cls(config['input_dir'], config['output_dir'])

        if config['wave_dir'] != 'Default':
            paths.wave_dir = os.path.abspath(config['wave_dir'])
        if config['trace_dir'] != 'Default':
            paths.trace_dir = os.path.abspath(config['trace_dir'])

        nondefault_dirs = True
        if config['master_dir'] != 'Default':
            os.rmdir(os.path.join(paths.intermediate_dir, 'Master'))
            paths.master_dir = os.path.abspath(config['master_dir'])
            if not os.path.exists(paths.master_dir):
                os.makedirs(paths.master_dir)
        else:
            nondefault_dirs = False
        if config['wavelength_calibration_dir'] != 'Default':
            os.rmdir(os.path.join(paths.intermediate_dir, 'Wavelength_calibration'))
            paths.wavelength_calibration_dir = os.path.abspath(config['wavelength_calibration_dir'])
            if not os.path.exists(paths.wavelength_calibration_dir):
                os.makedirs(paths.wavelength_calibration_dir)
        else:
            nondefault_dirs = False
        if config['trace_shift_dir'] != 'Default':
            os.rmdir(os.path.join(paths.intermediate_dir, 'Trace_shifts'))
            paths.trace_shift_dir = os.path.abspath(config['trace_shift_dir'])
            if not os.path.exists(paths.trace_shift_dir):
                os.makedirs(paths.trace_shift_dir)
        else:
            nondefault_dirs = False
        if config['plot_dir'] != 'Default':
            os.rmdir(os.path.join(paths.intermediate_dir, 'Plots'))
            paths.plot_dir = os.path.abspath(config['plot_dir'])
            if not os.path.exists(paths.plot_dir):
                os.makedirs(paths.plot_dir)
        else:
            nondefault_dirs = False
        if nondefault_dirs:
            # os.rmdir(paths.intermediate_dir)
            paths.intermediate_dir = None

        return paths
    
    def __repr__(self):
        return f'VelocePaths({self.input_dir}, {self.output_dir})'
    
    def __str__(self):
        message = f'VelocePaths instance:\n'
        message += f'Raw data directory: {self.input_dir}\n'
        message += f'Extracted data directory: {self.output_dir}\n'
        message += f'Wave directory: {self.wave_dir}\n'
        message += f'Trace directory: {self.trace_dir}\n'
        return message
    # remove methods below if not needed
    # def __eq__(self, other):
    #     equal = True
    #     for key, value in self.__dict__.items():
    #         if value != other.__dict__[key]:
    #             equal = False
    #             break
    #     return equal
    
    # def __ne__(self, other):
    #     return not self.__eq__(other)
    
    # def __hash__(self):
    #     return hash(tuple(self.__dict__.values()))
    
    # def __getitem__(self, key):
    #     return self.__dict__[key]
    
    # def __setitem__(self, key, value):
    #     self.__dict__[key] = value

    
def load_config(config_file):
    if not os.path.isabs(config_file):
        config_file = os.path.join(os.getcwd(), config_file)

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    if validate_config(config):
        veloce_paths = VelocePaths.from_config(config)
        science_targets = load_target_list(config['science_targets'])
        config['science_targets'] = science_targets
        config = {key: value for key, value in config.items() if 'dir' not in key}
        return config, veloce_paths
    else:
        raise ValueError('Invalid configuration file')

def validate_config(config):
    # validate options set in config file
    if config['reduce'] not in ['run', 'night', 'file']:
        raise ValueError(f'reduce must be "run", "night" or "file", not {config["reduce"]}')
    if config['calib_type'] not in ['SimLC', 'SimThXe', 'Interpolate', 'Static', 'arcTh']:
        raise ValueError(f'calib_type must be "SimLC", "SimThXe", "Interpolate", "arcTh or "Static", not {config["calib_type"]}')
    if config['arm'] not in ['all', 'red', 'green', 'blue']:
        raise ValueError(f'arm must be "all", "red", "green" or "blue", not {config["arm"]}')
    if config['amplifier_mode'] not in [2, 4]:
        raise ValueError(f'amplifier_mode must be 2 or 4, not {config["amplifier_mode"]}')
    if not isinstance(config['plot_diagnostic'], bool):
        raise ValueError(f'plot_diagnostic must be True or False, not {config["plot_diagnostic"]}')
    # validate targets list
    if config['science_targets'] != 'Default' and not (os.path.exists(os.path.join(os.getcwd(), config['science_targets'])) or os.path.exists(config['science_targets'])):
        raise FileNotFoundError(f'{config["science_targets"]} does not exist.')
    else:
        pass # maybe verify if targets are present for the run/night/file
    # validate input paths
    if not os.path.exists(os.path.abspath(config['input_dir'])):
        raise FileNotFoundError(f'{os.path.abspath(config["input_dir"])} does not exist.')
    if config['reduce'] == 'night' and not os.path.exists(os.path.abspath(os.path.join(config['input_dir'], config['date']))):
        raise FileNotFoundError(f'{config["date"]} does not exist in {os.path.abspath(config["input_dir"])}')
    # if not os.path.exists(os.path.abspath(config['output_dir'])):
    #     raise FileNotFoundError(f'{os.path.abspath(config["output_dir"])} does not exist.')
    # validate internal paths
    if config['wave_dir'] != 'Default' and not os.path.exists(os.path.abspath(config['wave_dir'])):
        raise FileNotFoundError(f'{os.path.abspath(config["wave_dir"])} does not exist.')
    if config['trace_dir'] != 'Default' and not os.path.exists(os.path.abspath(config['trace_dir'])):
        raise FileNotFoundError(f'{os.path.abspath(config["trace_dir"])} does not exist.')
    
    return True

def load_target_list(target_file):
    if target_file == 'Default':
        return [None]
    elif not os.path.isabs(target_file):
        target_file = os.path.join(os.getcwd(), target_file)
    with open(target_file, 'r') as f:
        targets = f.read().splitlines()
    return targets

def make_config(input_dir, output_dir,
                wave_dir='Default', trace_dir='Default',
                master_dir='Default', wavelength_calibration_dir='Default',
                trace_shift_dir='Default', plot_dir='Default',
                trace_file='Default',
                reduce='run', date=None, filename=None,
                calib_type='arcTh', science_targets='Default',
                arm='all', amplifier_mode=4,
                use_log=False, plot_diagnostic=False,
                validate_trace=True, sim_calib=True,
                scattered_light=False, flat_field=False):

    config = {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'reduce': reduce,
        'date': date,
        'filename': filename,
        'calib_type': calib_type,
        'science_targets': science_targets,
        'arm': arm,
        'amplifier_mode': amplifier_mode,
        'use_log': use_log,
        'plot_diagnostic': plot_diagnostic,
        'validate_trace': validate_trace,
        'sim_calib': sim_calib,
        'scattered_light': scattered_light,
        'flat_field': flat_field,
        'wave_dir': wave_dir,
        'trace_dir': trace_dir,
        'master_dir': master_dir,
        'wavelength_calibration_dir': wavelength_calibration_dir,
        'trace_shift_dir': trace_shift_dir,
        'plot_dir': plot_dir,
        'trace_file': trace_file
    }
    if validate_config(config):
        return config
    else:
        raise ValueError('Invalid configuration')

def save_config(config, veloce_paths, config_filename='config.yaml'):
    if not os.path.exists(veloce_paths.output_dir):
        os.makedirs(veloce_paths.output_dir)
    config_file = os.path.join(veloce_paths.output_dir, config_filename)
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    print(f'Configuration saved to {config_file}')

    return config_file

def format_date(date_str):
    # Parse the date string using datetime.strptime
    date_obj = datetime.strptime(date_str, '%y%m%d')
    
    # Format the date object to the desired format
    formatted_date = date_obj.strftime('%d%b').lower()
    
    return formatted_date

def load_run_logs(science_targets, veloce_paths, config):
    # Define the regular expression pattern for YYMMDD format
    date_pattern = re.compile(r'^\d{6}$')

    # Use list comprehension to filter and sort directories
    dates = sorted(
        [item for item in os.listdir(veloce_paths.input_dir)
         if os.path.isdir(os.path.join(veloce_paths.input_dir, item)) and date_pattern.match(item)])
    
    days = [format_date(date) for date in dates]

    obs_list = {'flat_red': {}, 'flat_green': {}, 'flat_blue': {}, 'flat_blue_long': {},
                'ARC-ThAr_red': {}, 'ARC-ThAr_green': {}, 'ARC-ThAr_blue': {},
                'SimThLong': {}, 'SimTh': {}, 'SimLC': {},
                'dark': {}, 'bias': {}, 'science': {}}
    
    for day, date in zip(days, dates):
        
        if config['use_log']:
            log_path = os.path.join(veloce_paths.input_dir, date)
            log_name = [name for name in os.listdir(log_path) if name.split('.')[-1] == 'log'][0]
            log_path = os.path.join(log_path, log_name)
            temp_obs_list, _ = load_log_info(log_path, science_targets, config['arm'], day)
        else:
            temp_obs_list, _ = scan_directory(veloce_paths, date, config['arm'], science_targets=science_targets)
        ### TODO: compare science targets with input list and save ones that were found
        for key in temp_obs_list:
            obs_list[key][date] = temp_obs_list[key]

    return obs_list

def load_night_logs(science_targets, veloce_paths, config):
    day = format_date(config['date'])

    obs_list = {'flat_red': {}, 'flat_green': {}, 'flat_blue': {}, 'flat_blue_long': {},
                'ARC-ThAr_red': {}, 'ARC-ThAr_green': {}, 'ARC-ThAr_blue': {},
                'SimThLong': {}, 'SimTh': {}, 'SimLC': {},
                'dark': {}, 'bias': {}, 'science': {}}

    if config['use_log']:
        log_path = os.path.join(veloce_paths.input_dir, config['date'])
        log_name = [name for name in os.listdir(log_path) if name.split('.')[-1] == 'log'][0]
        log_path = os.path.join(log_path, log_name)
        temp_obs_list, _ = load_log_info(log_path, science_targets, config['arm'], day)
    else:
        temp_obs_list, _ = scan_directory(veloce_paths, config['date'], config['arm'], science_targets=science_targets)

    for key in temp_obs_list:
        obs_list[key][config['date']] = temp_obs_list[key]

    return obs_list

def load_log_info(log_path, science_targets, selected_arm, day):
    """
    Loads observation log information and categorizes files based on their type and selected arm.

    This function reads an observation log file and categorizes the entries into different types such as
    'flat', 'dark', 'bias', 'science', and 'wave_calib'. It filters the entries based on the selected
    spectrograph arm and the specified calibration type.

    Parameters:
    - log_path (str): The path to the observation log file.
    - science_targets (list of str): A list of science target names to filter the science observations.
    - selected_arm (str): The spectrograph arm to be processed. Valid values are 'red', 'green', and 'blue'.
    - day (str): The day identifier to construct the file names.
    - calib_type (str): The calibration type to filter the wave calibration observations.

    Returns:
    - dict: A dictionary with keys representing different observation types ('flat_red', 'flat_green', 'flat_blue',
      'dark', 'bias', 'science', 'wave_calib') and values being lists of file names or tuples of target names and file names.

    Note:
    - The function assumes a specific format for the observation log file.
    - The function categorizes flat fields based on their exposure times (0.1s for 'flat_red', 1.0s for 'flat_green',
      and 10.0s for 'flat_blue').
    """
    obs_list = {'flat_red': [], 'flat_green': [], 'flat_blue': [], 'flat_blue_long': [],
                'ARC-ThAr_red': [], 'ARC-ThAr_green': [], 'ARC-ThAr_blue': [],
                'SimThLong': [], 'SimTh': [], 'SimLC': [],
                'dark': [], 'bias': [], 'science': []}
    arms = {'red': 3, 'green': 2, 'blue': 1, 'all': None}
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines[10:]:
            if line[0:4].isdigit():
                run, arm, target, exp_time = [line.split()[i] for i in [0, 1, 2, 5]]
                file_name = f'{day}{arm}{run}.fits'
                if int(arm) == arms[selected_arm] or selected_arm == 'all':
                    if target.strip() == 'BiasFrame':
                        obs_list['bias'].append(file_name)
                    elif target.strip() == 'FlatField-Quartz':
                        if float(exp_time) == 0.1 and int(arm) == 3:
                            obs_list['flat_red'].append(file_name)
                        elif float(exp_time) == 1.0 and int(arm) == 2:
                            obs_list['flat_green'].append(file_name)
                        elif float(exp_time) == 10.0 and int(arm) == 1:
                            obs_list['flat_blue'].append(file_name)
                        elif float(exp_time) == 60.0 and int(arm) == 1:
                            obs_list['flat_blue_long'].append(file_name)
                        else:
                            pass
                            # print(f"[Warning]: Non standard flat exp time = {exp_time} for {file_name}")
                    elif target.strip() == 'ARC-ThAr':
                        if float(exp_time) == 15 and int(arm) == 3:
                            obs_list['ARC-ThAr_red'].append(file_name)
                        elif float(exp_time) == 60 and int(arm) == 2:
                            obs_list['ARC-ThAr_green'].append(file_name)
                        elif float(exp_time) == 180 and int(arm) == 1:
                            obs_list['ARC-ThAr_blue'].append(file_name)
                    elif target.strip() == 'SimThLong':
                        obs_list['SimThLong'].append(file_name)
                    elif target.strip() == 'SimTh':
                        obs_list['SimTh'].append(file_name)
                    elif target.strip() == 'SimLC':
                        obs_list['SimLC'].append(file_name)
                    elif target.strip() == 'DarkFrame':
                        obs_list['dark'].append(file_name)
                    elif target.strip() in science_targets or science_targets is None:
                        obs_list['science'].append([target.strip(), file_name])
                    else:
                        pass
                    
    return obs_list, science_targets

def scan_directory(veloce_paths, date, selected_arm, science_targets=[None]):
    """
    Scan input directory and create an obs_list based on FITS headers.

    Parameters:
    - veloce_paths (object): VelocePaths object containing directory paths.
    - date (str): Date string in 'YYYYMMDD' format pointing to date to be scanned.
    - selected_arm (str): The spectrograph arm to be processed. Valid values are 'red', 'green', 'blue', and 'all'.
    - science_targets (list or None): List of science target names to filter by. If None, include all science targets.
    """
    ### TODO: validate data quality
    obs_list = {'flat_red': [], 'flat_green': [], 'flat_blue': [], 'flat_blue_long': [],
                'ARC-ThAr_red': [], 'ARC-ThAr_green': [], 'ARC-ThAr_blue': [],
                'SimThLong': [], 'SimTh': [], 'SimLC': [],
                'dark': [], 'bias': [], 'science': []}
    data_dirs = {'red': 'ccd_3', 'green': 'ccd_2', 'blue': 'ccd_1'}
    # pick which arm to reduce
    if selected_arm in data_dirs.keys():
        arms = [selected_arm]
    elif selected_arm == 'all':
        arms = data_dirs.keys()
    else:
        raise ValueError('Unsupported arm')
    for arm in arms:
        obs_dir = os.path.join(veloce_paths.input_dir, date, data_dirs[arm])
        if not os.path.exists(obs_dir):
            print(f"Directory {obs_dir} does not exist. Skipping.")
            continue
        fits_files = sorted([f for f in os.listdir(obs_dir) if f.endswith('.fits')])
        for file_name in fits_files:
            fits_path = os.path.join(obs_dir, file_name)

            try:
                with fits.open(fits_path) as hdul:
                    header = hdul[0].header
                    exp_time = header.get('EXPTIME', None)
                    target = header.get('OBJECT', '').strip().replace(' ', '')

                    if target == 'BiasFrame':
                        obs_list['bias'].append(file_name)
                    elif target == 'FlatField-Quartz':
                        if float(exp_time) == 0.1 and arm == 'red':
                            obs_list['flat_red'].append(file_name)
                        elif float(exp_time) == 1.0 and arm == 'green':
                            obs_list['flat_green'].append(file_name)
                        elif float(exp_time) == 10.0 and arm == 'blue':
                            obs_list['flat_blue'].append(file_name)
                        elif float(exp_time) == 60.0 and arm == 'blue':
                            obs_list['flat_blue_long'].append(file_name)
                        else:
                            pass
                            # print(f"[Warning]: Non standard flat exp time = {exp_time} for {file_name}")
                    elif target == 'ARC-ThAr':
                        if float(exp_time) == 15 and arm == 'red':
                            obs_list['ARC-ThAr_red'].append(file_name)
                        elif float(exp_time) == 60 and arm == 'green':
                            obs_list['ARC-ThAr_green'].append(file_name)
                        elif float(exp_time) == 180 and arm == 'blue':
                            obs_list['ARC-ThAr_blue'].append(file_name)
                    elif target == 'SimThLong':
                        obs_list['SimThLong'].append(file_name)
                    elif target.strip() == 'SimTh':
                        obs_list['SimTh'].append(file_name)
                    elif target == 'SimLC':
                        obs_list['SimLC'].append(file_name)
                    elif target == 'DarkFrame':
                        obs_list['dark'].append(file_name)
                    elif target == 'Acquire':
                        pass
                    elif target in science_targets or science_targets[0] is None:
                        obs_list['science'].append([target, file_name])
                    else:
                        pass
            except Exception as e:
                print(f"Error reading {fits_path}: {e}")
    if science_targets[0] is None:
        science_targets = list(set([target[0] for target in obs_list['science']]))

    return obs_list, science_targets

def get_obs_list(summary, target=None):
    """
    Drops empty lists from the summary dictionary and optionally filters the observations based on the target name.

    Parameters:
    - summary (dict): A dictionary with observation dates as keys and lists of observations as values.
    - target (str, optional): The target name to filter the observations. If provided, the list will be filtered
      based on the target.

    Returns:
    - dict: A dictionary with non-empty lists of observations.
    """
    if target is not None:
        summary_final = {k:[obs for obs in v if obs[0]==target] for k,v in summary.items() if v}
    else:
        summary_final = {k:v for k,v in summary.items() if v}

    if not summary_final:
        raise ValueError("No observations found for the specified target.")
    
    return summary_final