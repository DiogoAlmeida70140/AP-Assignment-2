# --- Main ---------------------------------------------------------------
import os
from train import *
from model import CNN_QNet
from snake_game import SnakeGame
import torch

if __name__ == '__main__':
    ####### ONLY RUN ONE TIME ##########
    #train()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = SnakeGame(30, 30, border=1)

    # 1) Avalia a heurística sozinha como baseline
    evaluate_heuristic_baseline(env, num_episodes=100)

    # 2) Carrega o modelo treinado
    model = CNN_QNet().to(device)
    model.load_state_dict(torch.load('./Task1/model/model.pth'))
    model.eval()

    # 3) Avalia e mostra as top 3 partidas entre 5000
    evaluate_and_show_best(model, env, num_eval_episodes=500, top_k=3, scale=10, slow_fps=5)
