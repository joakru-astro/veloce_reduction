### Put here the directory structure for the Veloce reduction pipeline
import os

class VelocePaths:
    def __init__(self, run=None):
        self.raw_parent_dir = '/home/usqobserver2/VeloceData/'
        self.extracted_parent_dir = '/home/usqobserver2/Joachim_veloce/Extracted_data/'
        self.reduction_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))#'~/Joachim_veloce/Veloce_reduction/'
        # print(self.reduction_parent_dir)
        self.wave_dir = os.path.join(self.reduction_parent_dir, 'Wave')
        self.trace_dir = os.path.join(self.reduction_parent_dir, 'Trace')
        self.blaze_dir = os.path.join(self.reduction_parent_dir, 'Blaze')
        self.master_dir = os.path.join(self.reduction_parent_dir, 'Master_data')
        self.obs_list_dir = os.path.join(self.reduction_parent_dir, 'Obs_lists')
        self.run = run

    def __post_init__(self):
        if self.run is not None:
            self.raw_dir = os.path.join(self.raw_parent_dir, f'{self.run}/')
            self.extracted_dir = os.path.join(self.extracted_parent_dir, f'{self.run}/')

    @classmethod
    def update_run(cls, run):
        cls.run = run
        cls.__post_init__()