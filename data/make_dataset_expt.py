import numpy as np
import h5py

acquire_t = 0.025
inputs_data = np.load(f"../data/expt/20230829/raw_frames_{acquire_t}ms.npy").astype(np.float32)
truths_data = np.load(f"../data/expt/20230829/theory_phase_{acquire_t}ms.npy").astype(np.float32)

""" Save the data to .h5 file """
basepath = "raw/"
filepath = f"flowers_expt_n{inputs_data.shape[0]}_npix{inputs_data.shape[-1]}_{acquire_t}ms.h5"

with h5py.File(basepath+filepath, "a") as h5_data:
    h5_data["truths"] = truths_data
    h5_data["inputs"] = inputs_data
    h5_data["E1"] = np.array([1])
    h5_data["E2"] = np.array([1])
    h5_data["vis"] = np.array([1], dtype=np.float32)