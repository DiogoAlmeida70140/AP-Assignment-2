import numpy as np
import torch
import random
import time
import csv
from snake_game import SnakeGame
from model import CNN_QNet, QTrainer

# --- Helpers ------------------------------------------------------------
def preprocess(state):
    # RGB state to PyTorch tensor (C, H, W)
    state = state / 255.0
    return np.transpose(state, (2, 0, 1))


def get_action(model, state, epsilon):
    if random.random() < epsilon:
        return random.choice([-1, 0, 1])
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor)
    return torch.argmax(q_values).item() - 1


def heuristic_policy(env):
    # Usa get_state() para coordenadas e calcula ação heurística
    score, apples, head, tail, direction = env.get_state()
    if not apples:
        return 0
    apple = apples[0]
    hy, hx = head
    ay, ax = apple
    # determina direção desejada
    if ay < hy:
        desired = 0
    elif ay > hy:
        desired = 2
    elif ax > hx:
        desired = 1
    else:
        desired = 3
    # converte diferença em ação
    diff = (desired - direction) % 4
    if diff == 0:
        action = 0
    elif diff == 1:
        action = 1
    elif diff == 3:
        action = -1
    else:
        action = 0
    # evita colisão
    ny, nx = hy, hx
    if direction == 0:
        ny -= 1
    elif direction == 1:
        nx += 1
    elif direction == 2:
        ny += 1
    else:
        nx -= 1
    if ny < 0 or ny >= env.height or nx < 0 or nx >= env.width or (ny, nx) in tail:
        for alt in [-1, 0, 1]:
            nd = (direction + alt) % 4
            ty, tx = hy, hx
            if nd == 0:
                ty -= 1
            elif nd == 1:
                tx += 1
            elif nd == 2:
                ty += 1
            else:
                tx -= 1
            if 0 <= ty < env.height and 0 <= tx < env.width and (ty, tx) not in tail:
                return alt
        return 0
    return action

# --- Training -----------------------------------------------------------
def train():
    EPISODES = 2000
    EPS_START = 1.0
    EPS_DECAY = 0.999
    EPS_MIN = 0.05
    GAMMA = 0.99
    LR = 1e-3

    env = SnakeGame(30, 30, border=1)
    model = CNN_QNet()
    trainer = QTrainer(model, lr=LR, gamma=GAMMA)

    metrics = []
    epsilon = EPS_START
    start_time = time.time()

    for ep in range(1, EPISODES + 1):
        state, _, done, _ = env.reset()
        state = preprocess(state)
        total_reward = 0

        while not done:
            if ep <= 200 and random.random() < 0.5:
                action = heuristic_policy(env)
            else:
                action = get_action(model, state, epsilon)

            next_state, reward, done, _ = env.step(action)
            next_state = preprocess(next_state)
            trainer.train_step(state, action + 1, reward, next_state, done)
            state = next_state
            total_reward += reward

        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)
        metrics.append((ep, total_reward, epsilon))

        if ep % 10 == 0:
            elapsed = time.time() - start_time
            avg10 = np.mean([m[1] for m in metrics[-10:]])
            print(f"Ep {ep}/{EPISODES} | Score: {total_reward:.2f} | Avg10: {avg10:.2f} | Eps: {epsilon:.3f} | Elap: {elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"Treino concluído em {total_time:.1f}s, score médio: {np.mean([m[1] for m in metrics]):.2f}")
    model.save()

    with open('training_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'score', 'epsilon'])
        writer.writerows(metrics)

# --- Play & Visualization ----------------------------------------------
def play(model, env, scale=10, fps=30, num_episodes=1000):
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((32*scale,32*scale))
    pygame.display.set_caption("Snake DQN Play")
    clock = pygame.time.Clock()
    scores = []
    start = time.time()

    for ep in range(1, num_episodes+1):
        raw_state, _, done, _ = env.reset()
        state = preprocess(raw_state)
        total = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = get_action(model, state, epsilon=0)
            raw_next, reward, done, _ = env.step(action)
            state = preprocess(raw_next)
            total += reward

            # render using raw_next
            disp = (raw_next * 255).astype(np.uint8)
            surf = pygame.surfarray.make_surface(disp)
            surf = pygame.transform.scale(surf,(32*scale,32*scale))
            screen.blit(surf,(0,0))
            pygame.display.flip()
            clock.tick(fps)

        scores.append(total)
        print(f"Play Ep {ep} | Score: {total:.2f}")

    print(f"Score médio: {np.mean(scores):.2f}, Tempo: {time.time()-start:.1f}s")


# --- Main ---------------------------------------------------------------
if __name__ == '__main__':
    #train()
    env = SnakeGame(30,30,border=1)
    model = CNN_QNet()
    model.load_state_dict(torch.load('./model/model.pth'))
    play(model, env)
