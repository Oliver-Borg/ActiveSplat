import os
import threading
import time
from enum import Enum
from typing import List, Tuple, Union

import torch
import numpy as np

PROJECT_NAME = 'ActiveSplat'

OPENCV_TO_OPENGL = np.array(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ]
)

CURRENT_FRUSTUM = {
    'color': [0.961, 0.475, 0.000],
    'scale': 0.2,
    'material': 'unlit_line_mat',
}
CURRENT_AGENT = {
    'color': [0.961, 0.475, 0.000],
    'material': 'lit_mat_transparency',
}
CURRENT_HORIZON = {
    'color': [0.0, 1.0, 0.0],
}

# NOTE: Functions for TIMING
def start_timing(use_cuda:bool=True) -> Tuple[Union[torch.cuda.Event, float], Union[torch.cuda.Event, None]]:
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start = time.perf_counter()
        end = None
    return start, end

def end_timing(start:Union[torch.cuda.Event, float], end:Union[torch.cuda.Event, None], use_cuda:bool=True) -> float:
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
    else:
        end = time.perf_counter()
        elapsed_time = end - start
        # Convert to milliseconds to have the same units
        # as torch.cuda.Event.elapsed_time
        elapsed_time = elapsed_time * 1000
    return elapsed_time

class CsvRecorder:
    '''
    Appends rows to <results_dir>/<file_name>, prefixed with the wall time. Recording is a no-op until open() is
    called, and the file is only created on the first row, so runs that never record don't get an empty file.
    '''
    
    def __init__(self, file_name:str, columns:List[str]) -> None:
        self.__file_name = file_name
        self.__columns = columns
        self.__results_dir = None
        self.__file = None
        self.__lock = threading.Lock()
        
    def open(self, results_dir:str) -> None:
        with self.__lock:
            if self.__file is not None:
                self.__file.close()
                self.__file = None
            self.__results_dir = results_dir
            
    def write_row(self, values:List) -> None:
        with self.__lock:
            if self.__results_dir is None:
                return
            if self.__file is None:
                path = os.path.join(self.__results_dir, self.__file_name)
                write_header = not os.path.isfile(path)
                self.__file = open(path, 'a', buffering=1)
                if write_header:
                    self.__file.write(','.join(['wall_time'] + self.__columns) + '\n')
            self.__file.write(','.join([f'{time.time():.6f}'] + [str(value) for value in values]) + '\n')

class TimingRecorder(CsvRecorder):
    '''Per-step timings: <results_dir>/timing.csv as (wall_time, stage, step, ms).'''
    
    def __init__(self, file_name:str='timing.csv') -> None:
        super().__init__(file_name, ['stage', 'step', 'ms'])
        
    def record(self, stage:str, step:int, elapsed_ms:float) -> None:
        self.write_row([stage, step, f'{elapsed_ms:.3f}'])

timing_recorder = TimingRecorder()
# NOTE: VGGT-Omega's predicted pose of each window frame relative to the current frame, against the true poses
pose_error_recorder = CsvRecorder('pose_errors.csv', ['step', 'frame', 'window_size', 'rot_err_deg', 'trans_err_m',
                                                      'trans_dir_err_deg', 'gt_baseline_m', 'scale'])

class GlobalState(Enum):
    REPLAY = 'REPLAY'
    AUTO_PLANNING = 'AUTO_PLANNING'
    MANUAL_PLANNING = 'MANUAL_PLANNING'
    MANUAL_CONTROL = 'MANUAL_CONTROL'
    PAUSE = 'PAUSE'
    QUIT = 'QUIT'