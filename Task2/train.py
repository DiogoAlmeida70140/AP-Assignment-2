from collections import deque
import os
import numpy as np
import torch
import random
import time
import csv
from model import CNN_QNet, QTrainer
import matplotlib.pyplot as plt
import cv2
import pygame
from policies import *
from utils import preprocess
# Importa o ReplayBuffer
from replay_buffer import ReplayBuffer 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Treinamento --------------------------------------------------------
def train(env, num_episodes=20000, max_steps_per_episode=1000,
          epsilon_start=1.0, epsilon_end=0.005, epsilon_decay=0.9995,
          batch_size=64, learning_rate=0.0001, gamma=0.99,
          buffer_capacity=10000, num_warmup_steps=5000,
          target_update_freq=100, max_train_time = 2*3600):

    # Modelo principal (online network)
    model = CNN_QNet(input_shape=(1, env.height + 2 * env.border, env.width + 2 * env.border)).to(device)
    # Target Network - CÓPIA IDÊNTICA DA ONLINE NETWORK
    target_model = CNN_QNet(input_shape=(1, env.height + 2 * env.border, env.width + 2 * env.border)).to(device)
    target_model.load_state_dict(model.state_dict()) # Inicializa a target com os pesos da online
    target_model.eval() # Coloca a target network em modo de avaliação (sem updates de gradiente)

    trainer = QTrainer(model, lr=learning_rate, gamma=gamma)
    
    # Replay Buffer
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    
    # Popula o buffer inicialmente (warm-start)
    replay_buffer.populate(env, heuristic_policy, num_warmup_steps)

    epsilon = epsilon_start
    total_scores = []
    total_steps = []
    plot_scores = []
    plot_mean_scores = []
    record = 0
    
    # Variável para contar os passos globais e atualizar a target network
    global_step_counter = 0 
    
    print(f"A treinar no dispositivo: {device}...")
    start_time = time.time()

    for episode in range(num_episodes):     
        state_raw, _, _, _ = env.reset()
        preprocessed_state = preprocess(state_raw)
        score = 0
        steps_in_episode = 0
        done = False

        while not done and steps_in_episode < max_steps_per_episode:
            # Seleciona a ação (epsilon-greedy)
            if random.random() < epsilon:
                action = random.choice([-1, 0, 1])
            else:
                state_tensor = torch.tensor(preprocessed_state, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    prediction = model(state_tensor)
                action = [-1, 0, 1][torch.argmax(prediction).item()]

            # Executa a ação
            next_state_raw, reward, done, info = env.step(action)
            preprocessed_next_state = preprocess(next_state_raw)

            # Armazena a transição no replay buffer
            action_idx = action + 1
            replay_buffer.add(preprocessed_state, action_idx, reward, preprocessed_next_state, done)

            # Atualiza o estado
            preprocessed_state = preprocessed_next_state
            score += reward
            steps_in_episode += 1
            global_step_counter += 1

            # Treina o modelo se houver experiências suficientes no buffer e a cada 4 passos (para eficiência)
            if len(replay_buffer) > batch_size and global_step_counter % 4 == 0:
                # Amostra um batch do buffer
                states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = replay_buffer.sample(batch_size)
                
                # Converte para tensores PyTorch
                states_batch = torch.tensor(states_batch, dtype=torch.float32).to(device)
                actions_batch = torch.tensor(actions_batch, dtype=torch.int64).to(device)
                rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32).to(device)
                next_states_batch = torch.tensor(next_states_batch, dtype=torch.float32).to(device)
                dones_batch = torch.tensor(dones_batch, dtype=torch.bool).to(device)

                # Calcular Q-values para o estado atual (online network)
                q_values_online = model(states_batch)
                q_current_action = q_values_online.gather(1, actions_batch.unsqueeze(1)).squeeze(1)

                # Calcular Q-values para o próximo estado (target network)
                with torch.no_grad():
                    q_next_state = target_model(next_states_batch).max(1)[0]

                # Calcular os alvos Q-values
                q_targets = rewards_batch + trainer.gamma * q_next_state * (~dones_batch)

                # Chamar o train_step com os Q_current_action e q_targets
                loss = trainer.train_step(q_current_action, q_targets)
            
            # Atualiza a Target Network a cada 'target_update_freq' passos globais
            if global_step_counter % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())
            
        # Reduz epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        total_scores.append(score)
        total_steps.append(steps_in_episode)
        mean_score = np.mean(total_scores[-100:])

        if score > record:
            record = score
            model.save("best_model.pth")

        elapsed_time = time.time() - start_time
        if (episode % 10 == 0):
            print(f'Episódio {episode+1}/{num_episodes} | Score: {score:.2f} | Recorde: {record:.2f} | Epsilon: {epsilon:.2f} | Média Score (100): {mean_score:.2f} | Passos no episódio: {steps_in_episode} | Tempo Decorrido: {elapsed_time:.1f}s')
        
        plot_scores.append(score)
        plot_mean_scores.append(mean_score)

        if elapsed_time > max_train_time:
            print("Tempo de treino limite excedido")
            break

    total_training_time = time.time() - start_time
    print(f"\nTreino concluído em {total_training_time:.1f}s.")
    print(f"Pontuação média total: {np.mean(total_scores):.2f}")

    # Plotar resultados
    os.makedirs('./Task1/images', exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(plot_scores, label='Score por Episódio')
    plt.plot(plot_mean_scores, label='Média de Scores (últimos 100 episódios)')
    plt.title('Treinamento da DQN com Experience Replay e Target Network')
    plt.xlabel('Episódio')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('./Task1/images/treino_score_dqn_enhanced.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(plot_scores)), [epsilon_start * (epsilon_decay ** i) for i in range(len(plot_scores))], label='Epsilon')
    plt.title('Decaimento do Epsilon')
    plt.xlabel('Episódio')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.grid(True)
    plt.savefig('./Task1/images/treino_epsilon_dqn_enhanced.png')
    plt.close()


    return model


def evaluate_and_show(env, model, num_eval_episodes=100, idle_tolerance=100, top_k=10, fps=10, scale=10):
    # Avaliar o modelo em múltiplos episódios
    results = []  # Lista de (pontuação final, [frame0, frame1, ...])
    for ep in range(1, num_eval_episodes + 1):
        state_raw, _, _, _ = env.reset()
        preprocessed_state = preprocess(state_raw)
        raw_frames = [state_raw.copy()]
        done = False
        total_reward = 0
        steps = 0
        idle_steps = 0  # Contador de passos sem progresso
        while not done and idle_steps < idle_tolerance:
            state_tensor = torch.tensor(preprocessed_state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = model(state_tensor)
            action = [-1, 0, 1][torch.argmax(prediction).item()]

            next_state_raw, reward, done, info = env.step(action)
            raw_frames.append(next_state_raw.copy())
            preprocessed_next_state = preprocess(next_state_raw)
            total_reward += reward
            preprocessed_state = preprocessed_next_state
            idle_steps += 1 if reward == 0 else 0
            steps += 1

        results.append((total_reward, raw_frames))
        print(f"[Eval] Ep {ep}/{num_eval_episodes} | Score: {total_reward:.2f} | Steps: {steps}")

    # Selecionar os top_k melhores episódios
    results.sort(key=lambda x: x[0], reverse=True)
    top_results = results[:top_k]
    print(f"\nTop {top_k} resultados (score, n_passos):")
    for idx, (sc, frames) in enumerate(top_results, start=1):
        print(f"  #{idx}: Score={sc:.2f}, Passos={len(frames) - 1}")

    # Mostrar os melhores jogos em Pygame
    pygame.init()
    window_size = ((env.width + 2 * env.border) * scale, (env.height + 2 * env.border) * scale)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Top Jogadas - Snake DQN")
    clock = pygame.time.Clock()

    for rank, (score_final, raw_frames) in enumerate(top_results, start=1):
        print(f"\nReproduzindo partida #{rank} com Score={score_final:.2f} em {fps} FPS...")
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
            clock.tick(fps)

        # Pausa de 1 segundo entre partidas
        time.sleep(1)

    print("Fim da reprodução das melhores partidas.")
    pygame.quit()
    return top_results
