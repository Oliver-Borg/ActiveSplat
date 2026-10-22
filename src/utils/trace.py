# Lightweight span tracer. Every process of a run (mapper node, planner node) appends spans to <results_dir>/trace_<process>.jsonl as
# Chrome Trace Event Format "X" (complete) events with the real pid and tid. Spans on one thread nest by time containment; spans on
# different threads or processes overlap in time when they run in parallel. scripts/tools/timeline.py draws the files (and merges them into
# one trace.json that opens in Perfetto, https://ui.perfetto.dev, or chrome://tracing).
# Tracing is a no-op until open() is called. One event is one short line write, so it is meant for stages (>= 1 ms), not for inner loops.
import functools
import json
import os
import threading
import time
from typing import Callable, Tuple

class Tracer:

    def __init__(self) -> None:
        self.__lock = threading.Lock()
        self.__file = None
        self.__process_name = None
        self.__results_dir = None
        self.__named_threads = set()

    def open(self, results_dir:str, process_name:str) -> None:
        with self.__lock:
            if self.__file is not None:
                self.__file.close()
                self.__file = None
            self.__results_dir = results_dir
            self.__process_name = process_name
            self.__named_threads = set()

    @property
    def enabled(self) -> bool:
        return self.__results_dir is not None

    @property
    def results_dir(self):
        return self.__results_dir

    def __synchronize(self) -> None:
        # NOTE: CUDA kernels run asynchronously, so a span around GPU work only measures the launches unless the GPU is drained first
        if self.__results_dir is not None:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def begin(self, sync:bool=False) -> Tuple[float, float]:
        '''Start token: (wall time, monotonic time). sync: wait for the GPU first (only while tracing, it perturbs the timing slightly).'''
        if sync:
            self.__synchronize()
        return time.time(), time.perf_counter()

    def finish(self, name:str, token:Tuple[float, float], sync:bool=False, **args) -> float:
        '''Writes the span that started at token and ends now. Returns its duration in ms.'''
        if sync:
            self.__synchronize()
        elapsed = time.perf_counter() - token[1]
        self.__write(name, token[0], elapsed, args)
        return elapsed * 1000

    def complete(self, name:str, elapsed_ms:float, **args) -> None:
        '''Writes a span of the given duration that ends now (for durations measured elsewhere).'''
        self.__write(name, time.time() - elapsed_ms / 1000, elapsed_ms / 1000, args)

    def span(self, name:str, sync:bool=False, **args):
        tracer = self
        class Span:
            def __enter__(self):
                self.token = tracer.begin(sync)
            def __exit__(self, *exc):
                tracer.finish(name, self.token, sync, **args)
        return Span()

    def traced(self, name:str) -> Callable:
        '''Decorator: one span per call.'''
        def decorator(function):
            @functools.wraps(function)
            def wrapper(*args, **kwargs):
                token = self.begin()
                try:
                    return function(*args, **kwargs)
                finally:
                    self.finish(name, token)
            return wrapper
        return decorator

    def __write(self, name:str, start_wall:float, duration_s:float, args:dict) -> None:
        if self.__results_dir is None:
            return
        thread = threading.current_thread()
        events = []
        with self.__lock:
            if self.__results_dir is None:
                return
            if self.__file is None:
                self.__file = open(os.path.join(self.__results_dir, f"trace_{self.__process_name.replace(' ', '_')}.jsonl"), 'a', buffering=1)
                events.append({'ph':'M', 'name':'process_name', 'pid':os.getpid(), 'args':{'name':self.__process_name}})
            if thread.ident not in self.__named_threads:
                self.__named_threads.add(thread.ident)
                events.append({'ph':'M', 'name':'thread_name', 'pid':os.getpid(), 'tid':thread.ident, 'args':{'name':thread.name}})
            events.append({'ph':'X', 'name':name, 'pid':os.getpid(), 'tid':thread.ident, 'ts':round(start_wall * 1e6), 'dur':round(duration_s * 1e6), 'args':args})
            self.__file.write(''.join(json.dumps(event) + '\n' for event in events))

tracer = Tracer()
