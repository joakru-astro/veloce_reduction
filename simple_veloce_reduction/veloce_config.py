### Put here the directory structure for the Veloce reduction pipeline
import os
import yaml

data_dirs = {'red': 'ccd_3', 'green': 'ccd_2', 'blue': 'ccd_1'}

class VelocePaths:
    def __init__(self, input_dir=None, output_dir=None):
        self.reduction_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
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
            os.rmdir(paths.intermediate_dir)
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
    if not (os.path.exists(os.path.join(os.getcwd(), config['science_targets'])) or os.path.exists(config['science_targets'])):
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
    if not os.path.isabs(target_file):
        target_file = os.path.join(os.getcwd(), target_file)
    with open(target_file, 'r') as f:
        targets = f.read().splitlines()
    return targets