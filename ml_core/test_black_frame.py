import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ml_core.video_processor import is_frame_valid

def test_black_frame():
    # Create black frame (all zeros)
    black_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert not is_frame_valid(black_frame), "Black frame should be invalid"
    print("Black frame detected correctly.")

def test_dark_frame():
    # Create very dark frame (values < 10)
    dark_frame = np.full((100, 100, 3), 5, dtype=np.uint8)
    assert not is_frame_valid(dark_frame), "Dark frame should be invalid"
    print("Dark frame detected correctly.")

def test_normal_frame():
    # Create normal frame
    normal_frame = np.full((100, 100, 3), 100, dtype=np.uint8)
    assert is_frame_valid(normal_frame), "Normal frame should be valid"
    print("Normal frame detected correctly.")

if __name__ == "__main__":
    test_black_frame()
    test_dark_frame()
    test_normal_frame()
