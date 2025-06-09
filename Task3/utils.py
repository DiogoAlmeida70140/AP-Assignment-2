import numpy as np
import cv2

def preprocess(state):
    """
    Pré-processa o estado da imagem para a rede neural.
    Converte para grayscale e garante que esteja no intervalo [0, 1].
    Assume que o input já está no intervalo [0, 1] (float32).
    """
    # state_raw é (H, W, 3) float32 no intervalo [0, 1]
    # Para cv2.cvtColor, precisamos de uint8 ou de ter certeza que o tipo é float32 e o cv2 aceita
    # Por segurança, vamos converter para uint8 para cv2.cvtColor, e depois normalizar
    state_uint8 = (state * 255).astype(np.uint8)
    state_gray = cv2.cvtColor(state_uint8, cv2.COLOR_RGB2GRAY) # Correção aqui
    state_gray = state_gray / 255.0 # Normaliza de volta para [0, 1]
    return np.expand_dims(state_gray, axis=0) # (1, H, W)