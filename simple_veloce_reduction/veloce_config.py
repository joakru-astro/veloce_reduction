### Put here the directory structure for the Veloce reduction pipeline
import os
import yaml

data_dirs = {'red': 'ccd_3', 'green': 'ccd_2', 'blue': 'ccd_1'}

class VelocePaths:
    # what if I drop using 'run' and just use input/output dir?
    # def __init__(self, input_path, output_path, run=None):
    def __init__(self, input_dir, output_dir, run=None):
        self.reduction_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # self.raw_parent_dir = os.path.join(self.reduction_parent_dir, 'Data', 'Raw')
        input_dir = os.path.abspath(input_dir)
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input path does not exist: {input_dir}")
        else:
            self.input_dir = input_dir
            if run is not None:
                if input_dir.split('/')[-1] != run:
                    raise FileNotFoundError(f"A run {run} was selected but doesn't match input directory name: {os.path.join(os.getcwd(), input_dir)}")
            else:
                run = input_dir.split('/')[-1]
                print(f"Warning: No run selected, using input directory name {run}.")
        # if run in os.listdir(os.path.join(os.getcwd(), input_dir)):
        #     self.raw_parent_dir = os.path.join(os.getcwd(), input_dir)
        # elif input_dir.split('/')[-1] == run:
        #     self.raw_parent_dir = os.path.dirname(os.path.join(os.getcwd(), input_dir))
        # elif run is not None:
        #     raise FileNotFoundError(f'A run {run} was selected but does not exist in input path: {os.path.join(os.getcwd(), input_dir)}')
        # elif os.listdir(os.path.join(os.getcwd(), input_dir)):
        #     self.raw_parent_dir = input_dir
            
        # self.extracted_parent_dir = os.path.join(self.reduction_parent_dir, 'Data', 'Extracted')
        output_dir = os.path.abspath(output_dir)
        if not os.path.exists(output_dir):
            raise FileNotFoundError(f'Output path does not exist: {output_dir}')
        else:
            if output_dir.split('/')[-1] == run:
                self.output_dir = output_dir
            else:
                print(f"Warning: Output directory name: {output_dir},\n does not match run name: {run}")
                print("Creating output subdirectory with run name.")
                self.output_dir = os.path.join(output_dir, run)
        
        self.wave_dir = os.path.join(self.reduction_parent_dir, 'Wave')
        self.trace_dir = os.path.join(self.reduction_parent_dir, 'Trace')
        # self.blaze_dir = os.path.join(self.reduction_parent_dir, 'Blaze')
        
        self.intermediate_dir = os.path.join(self.output_dir, 'Intermediate_results')
        if not os.path.exists(self.intermediate_dir):
            os.makedirs(self.intermediate_dir)
        self.master_dir = os.path.join(self.intermediate_dir, 'Master')
        if not os.path.exists(self.master_dir):
            os.makedirs(self.master_dir)
        # self.obs_list_dir = os.path.join(self.intermediate_dir, 'Obs_lists')
        self.plot_dir = os.path.join(self.intermediate_dir, 'Plots')
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        self.log_dir = os.path.join(self.intermediate_dir, 'Reduction_logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.wavelength_calibration_dir = os.path.join(self.intermediate_dir, 'Wavelength_calibration')
        if not os.path.exists(self.wavelength_calibration_dir):
            os.makedirs(self.wavelength_calibration_dir)
        self.run = run
    #     if self.run is not None:
    #         self.__post_init__()

    # def __post_init__(self):
    #     # if self.run is not None:
    #     self.raw_dir = os.path.join(self.raw_parent_dir, f'{self.run}/')
    #     self.extracted_dir = os.path.join(self.extracted_parent_dir, f'{self.run}/')

    # def update_run(self, run):
    #     self.run = run
    #     self.__post_init__()

    @classmethod
    def from_config(cls, config):
        
        # paths.reduction_parent_dir = config['reduction_parent_dir']
        # input_dir = os.path.abspath(config['input_dir'])
        # output_dir = os.path.abspath(config['output_dir'])
        # if config['run'] is not None:
        #     run = config['run']
        paths = cls(config['input_dir'], config['output_dir'], config['run'])
        # if not os.path.isabs(config['raw_parent_dir']):
        #     paths.raw_parent_dir = os.path.join(os.getcwd(), config['raw_parent_dir'])
        # else:
        #     paths.raw_parent_dir = config['raw_parent_dir']
        # if not os.path.isabs(config['extracted_parent_dir']):
        #     paths.extracted_parent_dir = os.path.join(os.getcwd(), config['extracted_parent_dir'])
        # else:
        #     paths.extracted_parent_dir = config['extracted_parent_dir']
        # internal paths
        if config['wave_dir'] != 'Default':
            paths.wave_dir = os.path.abspath(config['wave_dir'])
        if config['trace_dir'] != 'Default':
            paths.trace_dir = os.path.abspath(config['trace_dir'])
        # intermediate paths
        if config['blaze_dir'] != 'Default':
            paths.blaze_dir = os.path.abspath(config['blaze_dir'])
        if config['master_dir'] != 'Default':
            paths.master_dir = os.path.abspath(config['master_dir'])
        if config['wavelength_calibration_dir'] != 'Default':
            paths.wavelength_calibration_dir = os.path.abspath(config['wavelength_calibration_dir'])
        # if config['obs_list_dir'] != 'Default':
        #     paths.obs_list_dir = config['obs_list_dir']
        if config['plot_dir'] != 'Default':
            paths.plot_dir = config['plot_dir']
        if config['log_dir'] != 'Default':
            paths.log_dir = config['log_dir']
        # paths.update_run(config['run'])
        return paths
    
    def __repr__(self):
        return f'VelocePaths({self.run})'
    
    def __str__(self):
        message = f'VelocePaths instance for run: {self.run}\n'
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
    if config['calib_type'] not in ['SimLC', 'SimThXe', 'Interpolate', 'Static']:
        raise ValueError(f'calib_type must be "SimLC", "SimThXe", "Interpolate" or "Static", not {config["calib_type"]}')
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
    # elif not os.path.exists(os.path.join(config['raw_parent_dir'], config['run'])):
    #     raise FileNotFoundError(f'{config["run"]} does not exist in {config["raw_parent_dir"]}')
    if config['reduce'] == 'night' and not os.path.exists(os.path.abspath(os.path.join(config['input_dir'], config['date']))):
        raise FileNotFoundError(f'{config["date"]} does not exist in {os.path.abspath(config["input_dir"])}')
    # if config['reduce'] == 'file' and not os.path.exists(os.path.join(config['raw_parent_dir'], config['run'], config['date'], config['filename'])):
    #     raise FileNotFoundError(f'{config["filename"]} does not exist in {config["raw_parent_dir"]}/{config["run"]}/{config["date"]}')
    # validate output paths
    if not os.path.exists(os.path.abspath(config['output_dir'])):
        raise FileNotFoundError(f'{os.path.abspath(config["output_dir"])} does not exist.')
    # validate internal paths
    if config['wave_dir'] != 'Default' and not os.path.exists(os.path.abspath(config['wave_dir'])):
        raise FileNotFoundError(f'{os.path.abspath(config["wave_dir"])} does not exist.')
    if config['trace_dir'] != 'Default' and not os.path.exists(os.path.abspath(config['trace_dir'])):
        raise FileNotFoundError(f'{os.path.abspath(config["trace_dir"])} does not exist.')
    
    return True
# blaze_dir: Default
# master_dir: Default
# obs_list_dir: Default
# plot_dir: Default
# log_dir: Default

def load_target_list(target_file):
    if not os.path.isabs(target_file):
        target_file = os.path.join(os.getcwd(), target_file)
    with open(target_file, 'r') as f:
        targets = f.read().splitlines()
    return targets