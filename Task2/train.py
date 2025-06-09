import numpy as np
import torch
import random
import time
import csv
from model import CNN_QNet, QTrainer
import matplotlib.pyplot as plt
import pygame
from policies import *
from replay_buffer import ReplayBuffer 
from utils import preprocess
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(env, file_name, num_episodes=30000, max_steps_per_episode=1000,
          epsilon_start=1.0, epsilon_end=0.005, epsilon_decay=0.9995,
          batch_size=64, learning_rate=0.0001, gamma=0.99,
          buffer_capacity=10000, num_warmup_steps=5000,
          target_update_freq=100, max_train_time = 2*3600,
          ):
    """
    Train the DQN agent using CNN and replay buffer.
    """
    # Main model 
    model = CNN_QNet(input_shape=(1, env.height + 2 * env.border, env.width + 2 * env.border)).to(device)
    # Target Network - IDENTICAL COPY 
    target_model = CNN_QNet(input_shape=(1, env.height + 2 * env.border, env.width + 2 * env.border)).to(device)
    target_model.load_state_dict(model.state_dict()) # Initialize the target with the online weights
    target_model.eval() # Put the target network into evaluation mode (no gradient updates)

    trainer = QTrainer(model, lr=learning_rate, gamma=gamma)
    
    # Replay Buffer
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    
    # Populate the buffer initially (warm-start)
    replay_buffer.populate(env, heuristic_policy, num_warmup_steps)

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
    
    print(f"A treinar no dispositivo: {device}...")
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

            next_state_raw, reward, done, info = env.step(action)
            preprocessed_next_state = preprocess(next_state_raw)

            # Store the transition in the replay buffer
            action_idx = action + 1
            replay_buffer.add(preprocessed_state, action_idx, reward, preprocessed_next_state, done)

            # Update status
            preprocessed_state = preprocessed_next_state
            score += reward
            steps_in_episode += 1
            global_step_counter += 1

            # Train the model if there are enough experiences in the buffer and every 4 steps (for efficiency)
            if len(replay_buffer) > batch_size and global_step_counter % 4 == 0:
                # Sample a batch from the buffer
                states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = replay_buffer.sample(batch_size)
                
                # Convert to PyTorch tensors
                states_batch = torch.tensor(states_batch, dtype=torch.float32).to(device)
                actions_batch = torch.tensor(actions_batch, dtype=torch.int64).to(device)
                rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32).to(device)
                next_states_batch = torch.tensor(next_states_batch, dtype=torch.float32).to(device)
                dones_batch = torch.tensor(dones_batch, dtype=torch.bool).to(device)

                # Calculate Q-values ​​for the current state (online network)
                q_values_online = model(states_batch)
                q_current_action = q_values_online.gather(1, actions_batch.unsqueeze(1)).squeeze(1)

                # Calculate Q-values ​​for the next state (target network)
                with torch.no_grad():
                    q_next_state = target_model(next_states_batch).max(1)[0]

                # Calculate target Q-values
                q_targets = rewards_batch + trainer.gamma * q_next_state * (~dones_batch)

                # Call train_step with Q_current_action and q_targets
                loss = trainer.train_step(q_current_action, q_targets)
                episode_losses.append(loss)
            
            # Updates Target Network every 'target_update_freq' global steps
            if global_step_counter % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())
            
        # Reduce epsilon
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
            print(f'Episódio {episode+1}/{num_episodes} | Score: {score:.2f} | Recorde: {record:.2f} | Epsilon: {epsilon:.2f} | Média Score (100): {mean_score:.2f} | Passos no episódio: {steps_in_episode} | Tempo Decorrido: {elapsed_time:.1f}s')
        
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
    with open(f'./Task2/{file_name}_train_metrics.csv', 'w', newline='') as f:
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
    plt.savefig(f'./Task2/images/{file_name}_train_score.png')
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(epis, epsilons, label='Epsilon por episódio')
    plt.xlabel('Episódio')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./Task2/images/{file_name}_train_epsilon.png')
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(epis, losses, label='Loss média por episódio')
    plt.xlabel('Episódio')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./Task2/images/{file_name}_train_loss.png')
    plt.close()
    return

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
