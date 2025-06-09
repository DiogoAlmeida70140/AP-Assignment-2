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
from replay_buffer import ReplayBuffer 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def boltzmann_policy(model, state, temperature):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor).cpu().numpy()[0]
    q_values = q_values / temperature
    exp_q = np.exp(q_values - np.max(q_values))  # Estabilidade numérica
    probs = exp_q / np.sum(exp_q)
    action_idx = np.random.choice(len(probs), p=probs)
    return [-1, 0, 1][action_idx]

def train(env, num_episodes=50000, max_steps_per_episode=1000,
          exploration_strategy='epsilon_greedy', 
          epsilon_start=1.0, epsilon_end=0.005, epsilon_decay=0.9998,
          temperature_start=1.0, temperature_end=0.01, temperature_decay=0.999,
          batch_size=64, learning_rate=0.0001, gamma=0.99,
          buffer_capacity=10000, num_warmup_steps=5000,
          target_update_freq=100):

    model = CNN_QNet(input_shape=(1, env.height + 2 * env.border, env.width + 2 * env.border)).to(device)
    target_model = CNN_QNet(input_shape=(1, env.height + 2 * env.border, env.width + 2 * env.border)).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    trainer = QTrainer(model, lr=learning_rate, gamma=gamma)
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    replay_buffer.populate(env, heuristic_policy, num_warmup_steps)

    epsilon = epsilon_start
    temperature = temperature_start
    total_scores = []
    total_steps = []
    plot_scores = []
    plot_mean_scores = []
    exploration_metrics = []  # Para rastrear ε ou temperatura
    record = 0
    global_step_counter = 0

    print(f"A treinar com {exploration_strategy} no dispositivo: {device}...")
    start_time = time.time()

    for episode in range(num_episodes):
        state_raw, _, _, _ = env.reset()
        preprocessed_state = preprocess(state_raw)
        score = 0
        steps_in_episode = 0
        done = False

        while not done and steps_in_episode < max_steps_per_episode:
            if exploration_strategy == 'epsilon_greedy':
                if random.random() < epsilon:
                    action = random.choice([-1, 0, 1])
                else:
                    state_tensor = torch.tensor(preprocessed_state, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        prediction = model(state_tensor)
                    action = [-1, 0, 1][torch.argmax(prediction).item()]
            elif exploration_strategy == 'boltzmann':
                action = boltzmann_policy(model, preprocessed_state, temperature)

            next_state_raw, reward, done, info = env.step(action)
            preprocessed_next_state = preprocess(next_state_raw)
            action_idx = action + 1
            replay_buffer.add(preprocessed_state, action_idx, reward, preprocessed_next_state, done)

            preprocessed_state = preprocessed_next_state
            score += reward
            steps_in_episode += 1
            global_step_counter += 1

            if len(replay_buffer) > batch_size and global_step_counter % 4 == 0:
                states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = replay_buffer.sample(batch_size)
                states_batch = torch.tensor(states_batch, dtype=torch.float32).to(device)
                actions_batch = torch.tensor(actions_batch, dtype=torch.int64).to(device)
                rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32).to(device)
                next_states_batch = torch.tensor(next_states_batch, dtype=torch.float32).to(device)
                dones_batch = torch.tensor(dones_batch, dtype=torch.bool).to(device)

                q_values_online = model(states_batch)
                q_current_action = q_values_online.gather(1, actions_batch.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    q_next_state = target_model(next_states_batch).max(1)[0]
                q_targets = rewards_batch + trainer.gamma * q_next_state * (~dones_batch)

                loss = trainer.train_step(q_current_action, q_targets)

            if global_step_counter % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())

        if exploration_strategy == 'epsilon_greedy':
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            exploration_metrics.append(epsilon)
        elif exploration_strategy == 'boltzmann':
            temperature = max(temperature_end, temperature * temperature_decay)
            exploration_metrics.append(temperature)

        total_scores.append(score)
        total_steps.append(steps_in_episode)
        mean_score = np.mean(total_scores[-100:])

        if score > record:
            record = score
            model.save(f"best_model_{exploration_strategy}.pth")

        elapsed_time = time.time() - start_time
        if episode % 10 == 0:
            metric = epsilon if exploration_strategy == 'epsilon_greedy' else temperature
            metric_name = 'Epsilon' if exploration_strategy == 'epsilon_greedy' else 'Temperature'
            print(f'Episódio {episode+1}/{num_episodes} | Score: {score:.2f} | Recorde: {record:.2f} | {metric_name}: {metric:.2f} | Média Score (100): {mean_score:.2f} | Passos: {steps_in_episode} | Tempo: {elapsed_time:.1f}s')

        plot_scores.append(score)
        plot_mean_scores.append(mean_score)

    total_training_time = time.time() - start_time
    print(f"\nTreino concluído em {total_training_time:.1f}s.")
    print(f"Pontuação média total: {np.mean(total_scores):.2f}")

    os.makedirs('./Task3/images', exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(plot_scores, label='Score por Episódio')
    plt.plot(plot_mean_scores, label='Média de Scores (últimos 100 episódios)')
    plt.title(f'Treinamento DQN com {exploration_strategy}')
    plt.xlabel('Episódio')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./Task3/images/treino_score_{exploration_strategy}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(exploration_metrics, label='Epsilon' if exploration_strategy == 'epsilon_greedy' else 'Temperature')
    plt.title(f'Decaimento de {"Epsilon" if exploration_strategy == "epsilon_greedy" else "Temperature"}')
    plt.xlabel('Episódio')
    plt.ylabel('Epsilon' if exploration_strategy == 'epsilon_greedy' else 'Temperature')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./Task3/images/treino_exploration_{exploration_strategy}.png')
    plt.close()

    return model

def play(model, env, num_eval_episodes=10, top_k=10, scale=10):
    pygame.init()
    board_h, board_w, _ = env.board_state().shape 
    screen_width = board_w * scale
    screen_height = board_h * scale
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Jogo da Cobra - Modelo Treinado")
    clock = pygame.time.Clock()
    fps = 10

    model.eval()

    eval_scores = []
    
    print("\nA iniciar a avaliação com o modelo treinado...")
    for ep in range(num_eval_episodes):
        state_raw, _, _, _ = env.reset()
        preprocessed_state = preprocess(state_raw)
        done = False
        total_reward = 0
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return

            state_tensor = torch.tensor(preprocessed_state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = model(state_tensor)
            action = [-1, 0, 1][torch.argmax(prediction).item()]

            next_state_raw, reward, done, info = env.step(action)
            preprocessed_next_state = preprocess(next_state_raw)
            total_reward += reward

            disp = (next_state_raw * 255).astype(np.uint8)
            if disp.shape[2] == 1: 
                disp = np.stack([disp.squeeze(), disp.squeeze(), disp.squeeze()], axis=-1)
            
            surf = pygame.surfarray.make_surface(np.transpose(disp, (1, 0, 2))) 
            surf = pygame.transform.scale(surf, (screen_width, screen_height))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(fps)

            preprocessed_state = preprocessed_next_state
        
        eval_scores.append(total_reward)
        print(f"Episódio de Avaliação {ep+1}/{num_eval_episodes} | Pontuação: {total_reward:.2f}")

    print(f"\nPontuação média de avaliação em {num_eval_episodes} episódios: {np.mean(eval_scores):.2f}")
    pygame.quit()

def play_with_policy(env, policy, policy_name="Política", num_episodes=10, fps=10, scale=10):
    pygame.init()
    board_h, board_w, _ = env.board_state().shape
    screen_width = board_w * scale
    screen_height = board_h * scale
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption(f"Jogo da Cobra - {policy_name}")
    clock = pygame.time.Clock()

    scores = []
    print(f"\nA executar a política: {policy_name}...")
    for ep in range(num_episodes):
        state_raw, _, _, _ = env.reset()
        done = False
        total = 0
        step_count = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return

            action = policy(env)
            next_state_raw, reward, done, info = env.step(action)
            total += reward
            step_count += 1

            disp = (next_state_raw * 255).astype(np.uint8)
            if disp.shape[2] == 1:
                disp = np.stack([disp.squeeze(), disp.squeeze(), disp.squeeze()], axis=-1)
            
            surf = pygame.surfarray.make_surface(np.transpose(disp, (1, 0, 2)))
            surf = pygame.transform.scale(surf, (screen_width, screen_height))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(fps)
        
        scores.append(total)
        print(f"{policy_name} Ep {ep+1}/{num_episodes} | Pontuação: {total:.2f}, Passos: {step_count}")

    print(f"Pontuação média para {policy_name} em {num_episodes} episódios: {np.mean(scores):.2f}")
    pygame.quit()