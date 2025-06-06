#!/usr/bin/env python3
import yaml
import os
from astropy.io import fits
from simple_veloce_reduction import veloce_extraction, veloce_logs, veloce_config

def main():
    # Load configuration
    # config_filename = '/home/usqobserver2/Joachim_veloce/veloce_reduction_dev/veloce_reduction/config.yaml'
    # config, veloce_paths = veloce_config.load_config(config_filename)

    config, veloce_paths = veloce_config.load_config('config.yaml')

    if config['reduce'] == 'run':
        # Run the reduction
        obs_list = veloce_logs.load_run_logs(
            config['science_targets'], config['arm'], veloce_paths)
        target_list = veloce_logs.get_obs_list(obs_list['science'])
        veloce_extraction.extract_run(target_list, config, veloce_paths, obs_list)
    elif config['reduce'] == 'night':
        obs_list = veloce_logs.load_night_logs(
            config['date'], config['science_targets'], config['arm'], veloce_paths)
        target_list = veloce_logs.get_obs_list(obs_list['science'])
        if obs_list['science']:
            veloce_extraction.extract_night(target_list, config, veloce_paths, obs_list)
    elif config['reduce'] == 'file':
        obs_list = veloce_logs.load_night_logs(
            config['date'], config['science_targets'], config['arm'], veloce_paths)
        veloce_extraction.extract_single_file(config['filename'], config, veloce_paths, obs_list)


if __name__ == "__main__":
    main()