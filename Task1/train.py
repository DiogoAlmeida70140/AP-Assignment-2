from collections import deque
import numpy as np
import torch
import random
import time
import csv
from snake_game import SnakeGame
from model import CNN_QNet, QTrainer
import matplotlib.pyplot as plt
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Helpers ------------------------------------------------------------
def preprocess(state):
    # Converte para grayscale
    state_gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state_gray = state_gray / 255.0  # Normaliza
    return np.expand_dims(state_gray, axis=0)  # (1, H, W)


def get_action(model, state, epsilon, step_count=0):
    if random.random() < epsilon:
        return random.choice([-1, 0, 1]), step_count + 1
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor)
    if step_count % 100 == 0:  # Imprime a cada 100 passos
        print(f"Step {step_count}, Q-values: {q_values.tolist()}")
    return torch.argmax(q_values).item() - 1, step_count + 1

def bfs_path(head, apple, snake_body, width, height, border):
    """
    BFS para encontrar o caminho mais curto de `head` até `apple`, evitando snake_body e border.
    - head: tuplo (y_head, x_head)
    - apple: tuplo (y_apple, x_apple)
    - snake_body: lista de (y,x) do corpo (inclui a cabeça em snake_body[0], mas vamos tratar a cabeça como livre logo no início).
    - width, height: dimensões da região livre (excluindo border)
    - border: número de camadas de border cinzenta em torno do mapa.
    Retorna: lista de tuplos (y,x) incluindo head e apple. Ex: [(y_head,x_head), (y1,x1), …, (y_apple,x_apple)].
             Se não existir caminho, retorna None.
    """

    # 1) Constrói uma grelha 2D booleana “livre ou não” (com border ignorado/incluído)
    # A grelha interna começa (border, border) a (border+height-1, border+width-1)
    rows = height + 2 * border
    cols = width + 2 * border

    # Cria matriz “ocupada = True” sempre que for parte do corpo ou border.
    occupied = [[False]*cols for _ in range(rows)]
    # Marca border como ocupada
    for i in range(rows):
        for j in range(cols):
            if i < border or i >= border+height or j < border or j >= border+width:
                occupied[i][j] = True

    # Marca cada segmento da cobra como ocupado
    # (Inclui a cabeça; mas vamos supor que podemos “mover a cabeça” para si mesma no primeiro passo)
    for (cy, cx) in snake_body:
        occupied[cy + border][cx + border] = True

    # Calcular coordenadas “globais” (com border deslocado)
    hy, hx = head
    ay, ax = apple
    start = (hy + border, hx + border)
    goal = (ay + border, ax + border)

    # 2) BFS normal em grid 4-conectado
    queue = deque([start])
    visited = {start: None}  # dicionário de “pai” para reconstruir o caminho

    while queue:
        curr = queue.popleft()
        if curr == goal:
            break

        y, x = curr
        # Quatro vizinhos ortogonais
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                if not occupied[ny][nx] and (ny,nx) not in visited:
                    visited[(ny,nx)] = curr
                    queue.append((ny,nx))

    # Se não “visitou” goal, não há caminho
    if goal not in visited:
        return None

    # Reconstrói o caminho de trás para a frente
    path = []
    node = goal
    while node is not None:
        # Converte de volta para coordenadas sem border (subtraindo border)
        py, px = node
        path.append((py - border, px - border))
        node = visited[node]

    path.reverse()
    return path  # ex: [(y_head,x_head), (y1,x1), …, (y_apple,x_apple)]

def heuristic_policy(env):
    score, apples, head, tail, direction = env.get_state()
    if not apples:
        return 0
    apple = apples[0]
    hy, hx = head
    ay, ax = apple

    # 1) Usa BFS para encontrar o caminho mais curto até à maçã
    path = bfs_path(head, apple, [head] + tail, env.width, env.height, env.border)
    if path is not None and len(path) >= 2:
        # O segundo nó de path é a próxima célula para onde nos devemos mover
        next_cell = path[1]
        ny, nx = next_cell
        # Calcula “direção desejada” com base na diferença entre head e next_cell
        # Se head=(hy,hx) e next_cell=(ny,nx), então:
        if ny < hy:    desired = 0  # N
        elif ny > hy:  desired = 2  # S
        elif nx > hx:  desired = 1  # E
        else:          desired = 3  # W

        # Converte desired em action = {-1,0,1} baseando-se em “direction”
        diff = (desired - direction) % 4
        if diff == 0:      action = 0
        elif diff == 1:    action = 1
        elif diff == 3:    action = -1
        else:              action = 0

        return action

    # 2) Se não existir caminho via BFS (path is None), volta ao “look-ahead” de 1 passo:
    #    tenta a ação que evita colisões no próximo movimento
    hy, hx = head
    # Tenta ação “direção desejada” original antes usada (apontar à fruta)
    # Reutiliza a lógica anterior para “desired” e “diff”
    if ay < hy:    desired = 0
    elif ay > hy:  desired = 2
    elif ax > hx:  desired = 1
    else:          desired = 3
    diff = (desired - direction) % 4
    if diff == 0:      action = 0
    elif diff == 1:    action = 1
    elif diff == 3:    action = -1
    else:              action = 0

    # Simula “ver qual a célula seguinte” se usares action
    nd = (direction + action) % 4
    ty, tx = hy, hx
    if nd == 0:   ty -= 1
    elif nd == 1: tx += 1
    elif nd == 2: ty += 1
    else:         tx -= 1

    # Se colidir de imediato, procura ação alternativa menos pior
    if ty < 0 or ty >= env.height or tx < 0 or tx >= env.width or (ty, tx) in tail:
        for alt in [-1, 0, 1]:
            nd2 = (direction + alt) % 4
            ty2, tx2 = hy, hx
            if nd2 == 0:   ty2 -= 1
            elif nd2 == 1: tx2 += 1
            elif nd2 == 2: ty2 += 1
            else:          tx2 -= 1
            if 0 <= ty2 < env.height and 0 <= tx2 < env.width and (ty2, tx2) not in tail:
                return alt
        return 0

    return action

# --- Training -----------------------------------------------------------
def train():
    EPISODES = 2000
    EPS_START = 1.0
    EPS_DECAY = 0.995
    EPS_MIN = 0.1
    GAMMA = 0.99
    LR = 1e-3

    env = SnakeGame(30, 30, border=1)
    model = CNN_QNet().to(device)
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

        while not done:
            # Aqui: mix de heurística nos primeiros 200 episódios (50% das vezes)
            if ep <= 200 and random.random() < 0.5:
                action = heuristic_policy(env)
                used_heuristic = True
            else:
                action, step_count = get_action(model, state, epsilon, step_count)
                used_heuristic = False

            next_state, reward, done, _ = env.step(action)
            next_state = preprocess(next_state)
            # Train step: note que action+1 faz mapear {–1,0,1} → {0,1,2}
            loss = trainer.train_step(state, action + 1, reward, next_state, done)
            episode_losses.append(loss)

            state = next_state
            total_reward += reward

        # Nova lógica de ε:
        if ep <= 200:
            # Nos primeiros 200 episódios, decai “normalmente” (com EPS_DECAY)
            epsilon = max(EPS_MIN, epsilon * EPS_DECAY)
        else:
            # A partir do episódio 201, fixa ε bem baixo para quase nenhum “explore”
            epsilon = 0.0
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
    model.save()

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
    
    print("Treinando em:", next(model.parameters()).device)


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

# --- Avaliação e Reprodução das Melhores Partidas ------------------------
def evaluate_and_show_best(model, env, num_eval_episodes=500, top_k=3, scale=10, slow_fps=5):
    """
    Avalia o agente em `num_eval_episodes` partidas (epsilon=0), 
    mantém na memória todas as frames + a pontuação final de cada partida,
    seleciona as `top_k` partidas de maior pontuação e
    reproduz (devagar) essas partidas usando Pygame a slow_fps.
    """
    import pygame
    import time
    import numpy as np
    from train import preprocess, get_action

    # 1. Executar N episódios de avaliação, guardando trajes
    results = []  # list of (score_final, [raw_frame0, raw_frame1, ...])

    for ep in range(1, num_eval_episodes + 1):
        raw_frames = []
        raw_state, _, done, _ = env.reset()
        
        if ep == 1:
            # imprime os Q‐values do estado inicial (fora do laço de coleta de frames)
            state_tensor = preprocess(raw_state)
            with torch.no_grad():
                qs = model(torch.tensor(state_tensor, dtype=torch.float32).unsqueeze(0).to(device))
            print("Q‐values iniciais para estado inicial:", qs.cpu().numpy().tolist())
        
        raw_frames.append(raw_state.copy())
        state = preprocess(raw_state)
        total = 0

        while not done:
            action = get_action(model, state, epsilon=0)[0]
            raw_next, reward, done, _ = env.step(action)
            raw_frames.append(raw_next.copy())
            state = preprocess(raw_next)
            total += reward

        results.append((total, raw_frames))
        print(f"[Eval] Ep {ep}/{num_eval_episodes} | Score: {total:.2f}")

    # 2. Ordenar por pontuação decrescente e selecionar top_k
    results.sort(key=lambda x: x[0], reverse=True)
    top_results = results[:top_k]
    print(f"\nTop {top_k} resultados (score, n_passos):")
    for idx, (sc, frames) in enumerate(top_results, start=1):
        print(f"  #{idx}: Score={sc:.2f}, Passos={len(frames)-1}")

    # 3. Para cada partida entre as top_k, reproduzir em Pygame a slow_fps
    pygame.init()
    window_size = (32*scale, 32*scale)
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
            surf = pygame.transform.scale(surf, (32*scale, 32*scale))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(slow_fps)

        # Após terminar os frames de uma partida, pausar 1s antes de ir para a próxima
        time.sleep(1)

    print("Fim da reprodução das melhores partidas.")
    pygame.quit()
    
    
def evaluate_heuristic_baseline(env, num_episodes=100):
    scores = []
    for ep in range(num_episodes):
        _, _, done, _ = env.reset()
        total = 0
        while not done:
            a = heuristic_policy(env)
            _, r, done, _ = env.step(a)
            total += r
        scores.append(total)
    scores = np.array(scores)
    print(f"Heuristic baseline em {num_episodes} episódios:")
    print(f"  Média: {scores.mean():.2f}")
    print(f"  Desvio padrão: {scores.std():.2f}")
    print(f"  Máximo: {scores.max():.2f}")
    print(f"  Mínimo: {scores.min():.2f}")
    return scores
