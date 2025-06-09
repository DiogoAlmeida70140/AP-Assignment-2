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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def get_action(model, state, epsilon, step_count=0, print_qvalues=False):
    """
    Selects an action using the epsilon-greedy policy.
    Optionally prints Q-values at regular intervals.
    """
    if random.random() < epsilon:
        return random.choice([-1, 0, 1]), step_count + 1
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor)
    if print_qvalues and step_count % 100 == 0: 
        print(f"Step {step_count}, Q-values: {q_values.tolist()}")
    return torch.argmax(q_values).item() - 1, step_count + 1

# --- Training -----------------------------------------------------------
def train(env, file_name, num_episodes=30000, max_steps_per_episode=1000,
          epsilon_start=1.0, epsilon_end=0.005, epsilon_decay=0.9995,
          learning_rate=0.00015, gamma=0.99, max_train_time=60):
    """
    Trains a neural network using Deep Q-Learning (DQN) with an epsilon-greedy policy.
    Saves training metrics and the best-performing model based on score.
    """
    # Main model
    model = CNN_QNet(input_shape=(1, env.height + 2 * env.border, env.width + 2 * env.border)).to(device)

    trainer = QTrainer(model, lr=learning_rate, gamma=gamma)

    epsilon = epsilon_start
    total_scores = []
    total_steps = []
    plot_scores = []
    plot_mean_scores = []
    record = 0
    metrics = []
    losses = []

    # Variable to count global steps and update the target network
    global_step_counter = 0

    print(f"Training on the device: {device}...")
    start_time = time.time()

    for episode in range(num_episodes):
        state_raw, _, _, _ = env.reset()
        preprocessed_state = preprocess(state_raw)
        score = 0
        steps_in_episode = 0
        done = False
        episode_losses = []

        while not done and steps_in_episode < max_steps_per_episode:
            # Select the action (epsilon-greedy)
            if random.random() < epsilon:
                action = random.choice([-1, 0, 1])
            else:
                state_tensor = torch.tensor(preprocessed_state, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    prediction = model(state_tensor)
                action = [-1, 0, 1][torch.argmax(prediction).item()]

            # Execute the action
            next_state_raw, reward, done, info = env.step(action)
            preprocessed_next_state = preprocess(next_state_raw)

            # Call train_step with Q_current_action and q_targets
            loss = trainer.train_step(preprocessed_state, action + 1, reward, preprocessed_next_state, done)
            episode_losses.append(loss)

            # Update status
            preprocessed_state = preprocessed_next_state
            score += reward
            steps_in_episode += 1
            global_step_counter += 1

        # Reduz epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        total_scores.append(score)
        total_steps.append(steps_in_episode)
        mean_score = np.mean(total_scores[-100:])

        if score > record:
            record = score
            model.save("best_model.pth")

        metrics.append((episode, score, epsilon))
        if episode_losses:
            avg_loss = sum(episode_losses) / len(episode_losses)
            losses.append(avg_loss)
        else:
            losses.append(0)

        elapsed_time = time.time() - start_time
        if (episode % 10 == 0):
            print(
                f'Episódio {episode + 1}/{num_episodes} | Score: {score:.2f} | Recorde: {record:.2f} | Epsilon: {epsilon:.2f} | Média Score (100): {mean_score:.2f} | Passos no episódio: {steps_in_episode} | Tempo Decorrido: {elapsed_time:.1f}s')

        plot_scores.append(score)
        plot_mean_scores.append(mean_score)

        if elapsed_time > max_train_time:
            print("Tempo de treino limite excedido")
            break

    total_training_time = time.time() - start_time
    print(f"\nTreino concluído em {total_training_time:.1f}s.")
    print(f"Pontuação média total: {np.mean(total_scores):.2f}")

    save_metrics(metrics, losses, file_name)
    show_metrics(metrics, losses, file_name)

    return model

def save_metrics(metrics, losses, file_name):
    """
    Saves training metrics (episode, score, epsilon, loss) to a CSV file.
    Flags whether heuristics were used during training.
    """
    # Saving metrics + losses in CSV
    with open(f'./Task1/{file_name}_train_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        # Add column used heuristic
        writer.writerow(['episode', 'score', 'epsilon', 'avg_loss', 'used_heuristic'])
        for i in range(len(metrics)):
            # metrics[i] = (ep, total_reward, epsilon)
            # loss[i] = avg_loss
            # We need to know if, in that episode, we used heuristics.
            # A simple way: if the last episode used heuristics at least once → True.
            # To simplify: consider that you “used heuristics” if ep<=200.
            used = (metrics[i][0] <= 200)
            writer.writerow([metrics[i][0], metrics[i][1], metrics[i][2], losses[i], used])
    return

def show_metrics(metrics, losses, file_name):
    """
    Generates and saves line plots of score, epsilon, and average loss over training episodes.
    """
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
    plt.savefig(f'./Task1/images/{file_name}_train_score.png')
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(epis, epsilons, label='Epsilon por episódio')
    plt.xlabel('Episódio')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./Task1/images/{file_name}_train_epsilon.png')
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(epis, losses, label='Loss média por episódio')
    plt.xlabel('Episódio')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./Task1/images/{file_name}_train_loss.png')
    plt.close()
    return

def play(model, env, num_eval_episodes=500, top_k=3, scale=10, slow_fps=5):
    """
    Evaluates the trained model over multiple episodes and visualizes the top-performing ones using Pygame.
    Displays top_k episodes with highest step counts.
    """

    # Evaluate the model across multiple episodes
    results = []  # List of (final score, [frame0, frame1, ...])
    for ep in range(1, num_eval_episodes + 1):
        raw_frames = []
        raw_state, _, done, _ = env.reset()

        if ep == 1:
            # Print the Q values ​​of the initial state
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

    # Select top_k best episodes
    results.sort(key=lambda x: len(x[1]), reverse=True)
    top_results = results[:top_k]
    print(f"\nTop {top_k} resultados (score, n_passos):")
    for idx, (sc, frames) in enumerate(top_results, start=1):
        print(f"  #{idx}: Score={sc:.2f}, Passos={len(frames)-1}")

    # Show the best Pygame games
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

        # 1 second pause between matches
        time.sleep(1)

    print("Fim da reprodução das melhores partidas.")
    pygame.quit()

def evaluate_and_show(env, model, num_eval_episodes=100, idle_tolerance=100, top_k=10, fps=10, scale=10):
    """
    Evaluate the model and display top episodes with Pygame.
    """
    # Evaluate the model across multiple episodes
    results = []  # List of (final score, [frame0, frame1, ...])
    for ep in range(1, num_eval_episodes + 1):
        state_raw, _, _, _ = env.reset()
        preprocessed_state = preprocess(state_raw)
        raw_frames = [state_raw.copy()]
        done = False
        total_reward = 0
        apple_count = 0
        steps = 0
        idle_steps = 0  # Step counter with no progress
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
            if reward < 0.2:
                idle_steps += 1
            else:
                idle_steps = 0
                apple_count += 1
            steps += 1

        results.append((total_reward, raw_frames, apple_count))
        print(f"[Eval] Ep {ep}/{num_eval_episodes} | Score: {total_reward:.2f} | Steps: {steps} | Apples: {apple_count}")

    # Select top_k best episodes
    results.sort(key=lambda x: x[0], reverse=True)
    top_results = results[:top_k]
    print(f"\nTop {top_k} resultados (score, n_passos):")
    for idx, (sc, frames, apples) in enumerate(top_results, start=1):
        print(f"  #{idx}: Score={sc:.2f}, Passos={len(frames) - 1}, apples={apples}")

    print("Top average score:", np.mean([score for score, _, _ in top_results]))
    print("Top average apples:", np.mean([apples for _, _, apples in top_results]))

    # Show the best Pygame games
    pygame.init()
    window_size = ((env.width + 2 * env.border) * scale, (env.height + 2 * env.border) * scale)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Top Jogadas - Snake DQN")
    clock = pygame.time.Clock()

    for rank, (score_final, raw_frames, apples) in enumerate(top_results, start=1):
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

        time.sleep(1)

    print("Fim da reprodução das melhores partidas.")
    pygame.quit()
    return top_results
