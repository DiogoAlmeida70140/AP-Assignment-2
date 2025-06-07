# --- Main ---------------------------------------------------------------
import os
from train import *
from model import CNN_QNet
from snake_game import SnakeGame
import torch



def get_model(env, force_train=False):
    """
    Returns the trained model, optionally retraining it.
    """

    if force_train:
        model = train(env)
        model.save()
        print("Modelo salvo em:", model.save_path)
    else:
        model = CNN_QNet(input_shape=(1, env.height + 2 * env.border, env.width + 2 * env.border), num_actions=3)
        model.load_state_dict(torch.load('./Task1/model/model.pth'))
        model.to(device)
    return model


if __name__ == '__main__':
    # Check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")
        
    board_size = (14, 14)
    border =1
    env = SnakeGame(*board_size, border=border)

    model = get_model(env, force_train=False)

    # evaluate_heuristic_baseline(env, num_episodes=100)

        
    # model.eval()
    # # # Mostra o modelo com a heuristica
    # # # play_with_heuristic(env, scale=10, fps=30, num_episodes=5)
    # # # 3) Avalia e mostra as top 3 partidas entre 5000
    #evaluate_and_show_best(model, env, num_eval_episodes=10, top_k=3, scale=10, slow_fps=5)
    play(model, env, num_eval_episodes=1000, top_k=5)
    
    
    
    
        
    
    
    
    
# import torch
# from snake_game import SnakeGame
# from train import train, play_with_policy, play_with_model
# from Task1.policies import random_policy, heuristic_policy
# from model import CNN_QNet

# if __name__ == '__main__':
#     # Check CUDA
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     if torch.cuda.is_available():
#         print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
#     else:
#         print("Running on CPU")

#     # Initialize environment and model
#     env = SnakeGame(30, 30, border=1)
#     model = CNN_QNet().to(device)

#     # Train
#     print("\nStarting training...")
#     model = train()

#     # Visualize random policy
#     print("\nRunning random policy visualization...")
#     play_with_policy(env, random_policy, policy_name="Random", scale=10, fps=10, num_episodes=5)

#     # Visualize heuristic policy
#     print("\nRunning heuristic policy visualization...")
#     play_with_policy(env, heuristic_policy, policy_name="Heuristic", scale=10, fps=10, num_episodes=5)

#     # Visualize trained model
#     print("\nRunning trained model visualization...")
#     play_with_model(env, model, scale=10, fps=10, num_episodes=5, device=device)
