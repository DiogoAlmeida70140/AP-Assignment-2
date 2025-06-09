import numpy as np
import cv2
def preprocess(state):
    """
    Preprocesses the environment image state.
    Converts it to grayscale, normalizes, and adds channel dimension.
    """
    # state_raw is (H, W, 3) float32 in range [0, 1]
    # For cv2.cvtColor, we need uint8 or make sure the type is float32 and cv2 accepts it
    # To be safe, let's cast to uint8 for cv2.cvtColor, and then normalize
    state_uint8 = (state * 255).astype(np.uint8)
    state_gray = cv2.cvtColor(state_uint8, cv2.COLOR_RGB2GRAY) 
    state_gray = state_gray / 255.0 # Normalize back to [0, 1]
    return np.expand_dims(state_gray, axis=0) # (1, H, W)