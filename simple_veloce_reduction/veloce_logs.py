import astropy
from astropy.io import fits
import os, re
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from . import veloce_path

from datetime import datetime

def format_date(date_str):
    # Parse the date string using datetime.strptime
    date_obj = datetime.strptime(date_str, '%y%m%d')
    
    # Format the date object to the desired format
    formatted_date = date_obj.strftime('%d%b').lower()
    
    return formatted_date

def load_run_logs(run, science_targets, arm, calib_type=None, veloce_paths=None):
    if veloce_paths is None:
        veloce_paths = veloce_path.VelocePaths(run)
        veloce_paths.__post_init__()
    veloce_paths.raw_dir

    # Define the regular expression pattern for YYMMDD format
    date_pattern = re.compile(r'^\d{6}$')

    # Use list comprehension to filter and sort directories
    dates = sorted(
        [item for item in os.listdir(veloce_paths.raw_dir)
         if os.path.isdir(os.path.join(veloce_paths.raw_dir, item)) and date_pattern.match(item)])
    
    days = [format_date(date) for date in dates]

    obs_list = {'science': {},
            'flat_red': {},
            'flat_green': {},
            'flat_blue': {},
            'flat_blue_long': {},
            'dark': {},
            'bias': {},
            'wave_calib': {}}
    
    for day, date in zip(days, dates):
        log_path = os.path.join(veloce_paths.raw_parent_dir, run, date)
        log_name = [name for name in os.listdir(log_path) if name.split('.')[-1] == 'log'][0]
        log_path = os.path.join(log_path, log_name)
        temp_obs_list = load_log_info(log_path, science_targets, arm, day, calib_type)
        for key in temp_obs_list:
            obs_list[key][date] = temp_obs_list[key]

    return obs_list

def load_log_info(log_path, science_targets, selected_arm, day, calib_type):
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
    # obs_list = {'flat': [], 'dark': [], 'bias': [], 'science': [], 'wave_calib': []}
    obs_list = {'flat_red': [], 'flat_green': [], 'flat_blue': [],
                'flat_blue_long': [],  'dark': [], 'bias': [],
                'science': [], 'wave_calib': []}
    arms = {'red': 3, 'green': 2, 'blue': 1}
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines[10:]:
            if line[0:4].isdigit():
                run, arm, target, exp_time = [line.split()[i] for i in [0, 1, 2, 5]]
                file_name = f'{day}{arm}{run}.fits'
                if int(arm) == arms[selected_arm]:
                    if target.strip() == 'BiasFrame':
                        obs_list['bias'].append(file_name)
                    elif target.strip() == 'FlatField-Quartz':
                        if float(exp_time) == 0.1:
                            obs_list['flat_red'].append(file_name)
                        elif float(exp_time) == 1.0:
                            obs_list['flat_green'].append(file_name)
                        elif float(exp_time) == 10.0:
                            obs_list['flat_blue'].append(file_name)
                        elif float(exp_time) == 60.0:
                            obs_list['flat_blue_long'].append(file_name)
                        else:
                            pass
                            # print(f"[Warning]: Non standard flat exp time = {exp_time} for {file_name}")
                    elif target.strip() == calib_type:
                        obs_list['wave_calib'].append(file_name)
                    elif target.strip() == 'DarkFrame':
                        obs_list['dark'].append(file_name)
                    elif target.strip() in science_targets or science_targets is None:
                        obs_list['science'].append([target.strip(), file_name])
                    else:
                        pass
                    
    return obs_list

def save_science_target_list(summary, run=None, target=None, list_name=None, obs_list_path=None):
    """
    Saves a filtered list of science targets to a pickle file.

    This function filters a summary dictionary of observations based on a specified run or target and saves
    the filtered list to a pickle file. The file is saved in a predefined directory.

    Parameters:
    - summary (dict): A dictionary with observation types as keys and lists of observations as values.
    - run (str, optional): The run identifier to filter the observations. If provided, the list will be filtered
      based on the run.
    - target (str, optional): The target name to filter the observations. If provided, the list will be filtered
      based on the target.
    - list_name (str, optional): The name to use for the saved list file. If None, the run or target name will be used.

    Returns:
    - str: The filename of the saved pickle file.

    Raises:
    - ValueError: If neither `run` nor `target` is provided.

    Note:
    - The function assumes a standard directory structure for storing observation lists.
    - The function uses the `pickle` module to save the filtered list to a file.
    """
    if obs_list_path is None:
        veloce_paths = veloce_path.VelocePaths(run)
        obs_list_path = veloce_paths.obs_list_dir
    # obs_list_path = "/home/usqobserver2/Joachim_veloce/Veloce_reduction/Obs_lists"
    
    if target is not None:
        if list_name is None: list_name = target
        summary_final = {k:[obs for obs in v if obs[0]==target] for k,v in summary.items() if v}
    elif run is not None:
        if list_name is None: list_name = run
        summary_final = {k:v for k,v in summary.items() if v}
    else:
        raise ValueError("No run or target provided.")

    with open(os.path.join(obs_list_path, f'obs_list_{list_name}.pkl'), 'wb') as f:
        pickle.dump(summary_final, f)
    
    return f'obs_list_{list_name}.pkl'

# def load_multi_logs(science_targets, data_path, run, days, dates, logs_names):