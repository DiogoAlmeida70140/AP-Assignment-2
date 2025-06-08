from collections import deque
import numpy as np
import torch
import random
import time
import csv
from model import CNN_QNet, QTrainer, unique_color_num
import matplotlib.pyplot as plt
import cv2
import pygame
from policies import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(state):
    """
    Converts RGB state image to one-hot encoded tensor (5, H, W) for CNN.
    Assumes only 5 unique RGB values (for head, body, apple, wall, background).
    """
    H, W, _ = state.shape
    state_reshaped = state.reshape(-1, 3)
    unique_colors = np.unique(state_reshaped, axis=0)
    unique_colors = np.array(sorted([tuple(c) for c in unique_colors]))  # Consistent ordering

    color_to_idx = {tuple(color): idx for idx, color in enumerate(unique_colors)}
    idx_map = np.array([color_to_idx[tuple(pixel)] for pixel in state_reshaped])
    idx_img = idx_map.reshape(H, W)

    # One-hot encode (H, W) -> (5, H, W)
    one_hot = np.eye(unique_color_num)[idx_img]  # (H, W, 5)
    one_hot = one_hot.transpose(2, 0, 1)  # (5, H, W)

    return one_hot.astype(np.float32)

def get_action(model, state, epsilon, step_count=0, print_qvalues=False):
    if random.random() < epsilon:
        return random.choice([-1, 0, 1]), step_count + 1
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor)
    if print_qvalues and step_count % 100 == 0:  # Imprime a cada 100 passos
        print(f"Step {step_count}, Q-values: {q_values.tolist()}")
    return torch.argmax(q_values).item() - 1, step_count + 1

# --- Training -----------------------------------------------------------
def train(env):
    EPISODES = 2000
    EPS_START = 1.0
    EPS_DECAY = 0.999
    EPS_MIN = 0.05
    GAMMA = 0.5
    LR = 1e-5

    heuristic_prob = 0.5  # Probabilidade de usar heurística nos primeiros episódios
    heuristic_epoch_limit = 500  # Limite de episódios para usar heurística

    model = CNN_QNet(input_shape=(unique_color_num, env.height + 2 * env.border, env.width + 2 * env.border))
    trainer = QTrainer(model, lr=LR, gamma=GAMMA)

    metrics = []
    losses = []
    epsilon = EPS_START
    start_time = time.time()
    step_count = 0

    for ep in range(1, EPISODES + 1):
        state, _, done, _ = env.reset()
        state = preprocess(state)
        total_reward = 0
        episode_losses = []

        used_heuristic = ep <= heuristic_epoch_limit and random.random() < heuristic_prob

        i = int(ep * heuristic_prob)
        reset_path()
        while not done:
            if used_heuristic and i <= heuristic_epoch_limit:
                action = heuristic_policy(env, pathfind="astar")
                update_model = False
            else:
                action, step_count = get_action(trainer.model, state, epsilon, step_count, print_qvalues=True)
                update_model = True
            i += 1

            next_state, reward, done, _ = env.step(action)
            next_state = preprocess(next_state)
            if update_model:
                # Train step: note que action+1 faz mapear {–1,0,1} → {0,1,2}
                loss = trainer.train_step(state, action + 1, reward, next_state, done)
                episode_losses.append(loss)

            state = next_state
            total_reward += reward

        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)
        metrics.append((ep, total_reward, epsilon))

        # Cálculo da loss média do episódio
        if episode_losses:
            avg_loss = sum(episode_losses) / len(episode_losses)
            losses.append(avg_loss)
        else:
            losses.append(0)

        if ep % 10 == 0:
            elapsed = time.time() - start_time
            avg10 = np.mean([m[1] for m in metrics[-10:]])
            avg_loss_10 = np.mean(losses[-10:])
            print(f"Ep {ep}/{EPISODES} | Score: {total_reward:.2f} | "
                  f"Avg10: {avg10:.2f} | Avg Loss: {avg_loss_10:.4f} | "
                  f"Eps: {epsilon:.3f} | Elap: {elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"Treino concluído em {total_time:.1f}s, score médio: "
          f"{np.mean([m[1] for m in metrics]):.2f}")
    

    save_metrics(metrics, losses)
    show_metrics(metrics, losses)

    return trainer.model

def save_metrics(metrics, losses):
    # Salvando métricas + losses em CSV
    with open('./Task1/training_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        # Adiciona coluna usada_heuristic
        writer.writerow(['episode', 'score', 'epsilon', 'avg_loss', 'used_heuristic'])
        for i in range(len(metrics)):
            # metrics[i] = (ep, total_reward, epsilon)
            # losses[i] = avg_loss
            # Precisamos saber se, naquele episódio, usamos heurística. 
            # Uma forma simples: se o último episódio usou heurística ao menos uma vez → True. 
            # Para simplificar: considere que “usou heurística” se ep<=200.
            used = (metrics[i][0] <= 200)
            writer.writerow([metrics[i][0], metrics[i][1], metrics[i][2], losses[i], used])
    return

def show_metrics(metrics, losses):
    # Plot charts (score, epsilon, loss)
    epis = [m[0] for m in metrics]
    scores = [m[1] for m in metrics]
    epsilons = [m[2] for m in metrics]

    plt.figure(figsize=(8,4))
    plt.plot(epis, scores, label='Score por episódio')
    plt.xlabel('Episódio')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./Task1/images/treino_score.png')
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(epis, epsilons, label='Epsilon por episódio')
    plt.xlabel('Episódio')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./Task1/images/treino_epsilon.png')
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(epis, losses, label='Loss média por episódio')
    plt.xlabel('Episódio')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./Task1/images/treino_loss.png')
    plt.close()
    return


def play(model, env, num_eval_episodes=500, top_k=3, scale=10, slow_fps=5):
    """
    Carrega um modelo a partir de model_path, avalia em múltiplos episódios e mostra os melhores jogos em Pygame.
    
    Args:
        model_path (str): Caminho para o arquivo do modelo (ex.: './Task1/model/model.pth').
        width (int): Largura do tabuleiro do jogo.
        height (int): Altura do tabuleiro do jogo.
        food_amount (int): Quantidade de comida no tabuleiro.
        border (int): Tamanho da borda ao redor do tabuleiro.
        grass_growth (float): Taxa de crescimento da grama.
        max_grass (float): Nível máximo de grama.
        num_eval_episodes (int): Número de episódios para avaliação.
        top_k (int): Número de melhores episódios a serem exibidos.
        scale (int): Fator de escala para a janela do Pygame.
        slow_fps (int): FPS para a reprodução lenta.
    """
    
    # Avaliar o modelo em múltiplos episódios
    results = []  # Lista de (pontuação final, [frame0, frame1, ...])
    for ep in range(1, num_eval_episodes + 1):
        raw_frames = []
        raw_state, _, done, _ = env.reset()
        
        if ep == 1:
            # Imprimir os Q-values do estado inicial
            state_tensor = preprocess(raw_state)
            with torch.no_grad():
                qs = model(torch.tensor(state_tensor, dtype=torch.float32).unsqueeze(0).to(device))
            print("Q-values iniciais para estado inicial:", qs.cpu().numpy().tolist())
        
        raw_frames.append(raw_state.copy())
        state = preprocess(raw_state)
        total_reward = 0
        step_count = 0
        
        while not done:
            action, step_count = get_action(model, state, epsilon=0, step_count=step_count)
            raw_next, reward, done, _ = env.step(action)
            raw_frames.append(raw_next.copy())
            state = preprocess(raw_next)
            total_reward += reward
        
        results.append((total_reward, raw_frames))
        print(f"[Eval] Ep {ep}/{num_eval_episodes} | Score: {total_reward:.2f} | Steps: {step_count}")
    
    # Selecionar os top_k melhores episódios
    results.sort(key=lambda x: len(x[1]), reverse=True)
    top_results = results[:top_k]
    print(f"\nTop {top_k} resultados (score, n_passos):")
    for idx, (sc, frames) in enumerate(top_results, start=1):
        print(f"  #{idx}: Score={sc:.2f}, Passos={len(frames)-1}")
    
    # Mostrar os melhores jogos em Pygame
    pygame.init()
    window_size = ((env.width + 2 * env.border) * scale, (env.height + 2 * env.border) * scale)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Top Jogadas - Snake DQN")
    clock = pygame.time.Clock()
    
    for rank, (score_final, raw_frames) in enumerate(top_results, start=1):
        print(f"\nReproduzindo partida #{rank} com Score={score_final:.2f} em {slow_fps} FPS...")
        for raw in raw_frames:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            disp = (raw * 255).astype(np.uint8)
            surf = pygame.surfarray.make_surface(disp)
            surf = pygame.transform.scale(surf, window_size)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(slow_fps)
        
        # Pausa de 1 segundo entre partidas
        time.sleep(1)
    
    print("Fim da reprodução das melhores partidas.")
    pygame.quit()

