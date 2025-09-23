#!/usr/bin/env python3
import yaml
import os
import sys
import argparse
from astropy.io import fits
from simple_veloce_reduction import veloce_extraction, veloce_config

def main():
    parser = argparse.ArgumentParser(
        description="Reduce Veloce data from raw FITS files (in default Veloce dir structure) to extracted 1D spectra",
        epilog="Example: python veloce_reduction.py /absolute/path/to/config.yaml"
    )
    parser.add_argument(
        "config_filename",
        nargs="?",
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml)"
    )
    args = parser.parse_args()

    config, veloce_paths = veloce_config.load_config(args.config_filename)
    _config_filename = veloce_config.save_config(config, veloce_paths, args.config_filename)

    if config['reduce'] == 'run':
        obs_list = veloce_config.load_run_logs(
            config['science_targets'], veloce_paths, config)
        target_list = veloce_config.get_obs_list(obs_list['science'])
        veloce_extraction.extract_run(target_list, config, veloce_paths, obs_list)
    elif config['reduce'] == 'night':
        obs_list = veloce_config.load_night_logs(
            config['science_targets'], veloce_paths, config)
        target_list = veloce_config.get_obs_list(obs_list['science'])
        if obs_list['science']:
            veloce_extraction.extract_night(target_list, config, veloce_paths, obs_list)
    elif config['reduce'] == 'file':
        obs_list = veloce_config.load_night_logs(
            config['science_targets'], veloce_paths, config)
        veloce_extraction.extract_single_file(config['filename'], config, veloce_paths, obs_list)

if __name__ == "__main__":
    main()