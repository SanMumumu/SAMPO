import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.nn.functional import interpolate
from cotracker.utils.visualizer import Visualizer

def apply_cotracker_on_first_two_frames(pixel_values, model, device, grid_size=12, grid_query_frame=0, time_window=8):
    # TODO: This function will be released after the paper is accepted.
    return None

def _process_step(window_frames, is_first_step, grid_size, grid_query_frame, model, device):
    window_frames_cpu = [frame.cpu() for frame in window_frames[-model.step * 2 :]]
    video_chunk = np.stack(window_frames_cpu)[None]
    return model(
        video_chunk,
        is_first_step=is_first_step,
        grid_size=grid_size,
        grid_query_frame=grid_query_frame,
    )