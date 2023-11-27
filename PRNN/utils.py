from pathlib import Path
import matplotlib
import platform
import time

install_path = Path(__file__)
top = install_path.parents[1].absolute()

paths = {
    "raw": top.joinpath("data/raw"),
    "processed": top.joinpath("data/processed"),
    "models": top.joinpath("models"),
    "notebooks": top.joinpath("notebooks"),
    "scripts": top.joinpath("scripts")
}

def get_system_and_backend():
    if platform.system()=='Linux':
        matplotlib.use("TkAgg")

def time_func(func, units='sec'):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        if units=='sec':
            print(f"Time elapsed: {end-start:.2f} sec")
        elif units=='ms':
            print(f"Time elapsed: {(end-start)*1e3:.2f} ms")
    return wrapper