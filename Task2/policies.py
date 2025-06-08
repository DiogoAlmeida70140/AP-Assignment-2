import heapq
import random
from collections import deque
import pygame
import numpy as np
import time

def random_policy():
    """Selects a random action from {-1, 0, 1}."""
    return random.choice([-1, 0, 1])


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

########################## A* Pathfinding with A* Search Algorithm ##########################
def astar_path(head, apple, snake_body, width, height, border):
    rows = height + 2 * border
    cols = width + 2 * border
    occupied = [[False] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            if i < border or i >= border + height or j < border or j >= border + width:
                occupied[i][j] = True

    for cy, cx in snake_body:
        occupied[cy + border][cx + border] = True

    hy, hx = head
    ay, ax = apple
    start = (hy + border, hx + border)
    goal = (ay + border, ax + border)

    open_set = []
    heapq.heappush(open_set, (0 + manhattan(start, goal), 0, start, [head]))  # path inicia com [head]

    visited = set()

    while open_set:
        _, cost, current, path = heapq.heappop(open_set)

        if current == goal:
            return path  # já está na ordem correta

        if current in visited:
            continue
        visited.add(current)

        y, x = current
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            next_node = (ny, nx)
            if 0 <= ny < rows and 0 <= nx < cols and not occupied[ny][nx]:
                if next_node not in visited:
                    game_yx = (ny - border, nx - border)
                    heapq.heappush(open_set, (
                        cost + 1 + manhattan(next_node, goal),
                        cost + 1,
                        next_node,
                        path + [game_yx]
                    ))

    return None
########################## Breadth-First Search (BFS) Pathfinding ##########################
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

path = []

def reset_path():
    global path
    path=[]


def heuristic_policy(env, pathfind="bfs"):
    """
    Heuristic policy to move the snake toward the apple using BFS or a one-step look-ahead.
    Args:
        env: SnakeGame environment instance
    Returns:
        Action (-1, 0, 1) for left, straight, or right
    """
    global path
    score, apples, head, tail, direction = env.get_state()
    if not apples:
        return 0
    apple = apples[0]
    hy, hx = head
    ay, ax = apple

    # 1) Usa BFS para encontrar o caminho mais curto até à maçã
    pathfinder = bfs_path if pathfind == "bfs" else astar_path
    if path is None or len(path) == 0:
        path = pathfinder(head, apple, [head] + tail, env.width, env.height, env.border)
        if path is not None:
            path.pop(0)  # Remove o head do início do caminho

    if path is not None and len(path) >= 1:
        # O segundo nó de path é a próxima célula para onde nos devemos mover
        next_cell = path.pop(0)
        ny, nx = next_cell
        # Calcula “direção desejada” com base na diferença entre head e next_cell
        # Se head=(hy,hx) e next_cell=(ny,nx), então:
        if ny < hy:
            desired = 0  # N
        elif ny > hy:
            desired = 2  # S
        elif nx > hx:
            desired = 1  # E
        else:
            desired = 3  # W

        # Converte desired em action = {-1,0,1} baseando-se em “direction”
        action = ((desired - direction + 2) & 3) - 2
        return action

    # 2) Se não existir caminho via BFS (path is None), volta ao “look-ahead” de 1 passo:
    #    tenta a ação que evita colisões no próximo movimento
    hy, hx = head
    # Tenta ação “direção desejada” original antes usada (apontar à fruta)
    # Reutiliza a lógica anterior para “desired” e “diff”
    if ay < hy:
        desired = 0
    elif ay > hy:
        desired = 2
    elif ax > hx:
        desired = 1
    else:
        desired = 3

    action = ((desired - direction + 2) & 3) - 2

    # Simula “ver qual a célula seguinte” se usares action
    nd = (direction + action) % 4
    ty, tx = hy, hx
    if nd == 0:
        ty -= 1
    elif nd == 1:
        tx += 1
    elif nd == 2:
        ty += 1
    else:
        tx -= 1

    # Se colidir de imediato, procura ação alternativa menos pior
    if ty < 0 or ty >= env.height or tx < 0 or tx >= env.width or (ty, tx) in tail:
        for alt in [-1, 0, 1]:
            nd2 = (direction + alt) % 4
            ty2, tx2 = hy, hx
            if nd2 == 0:
                ty2 -= 1
            elif nd2 == 1:
                tx2 += 1
            elif nd2 == 2:
                ty2 += 1
            else:
                tx2 -= 1
            if 0 <= ty2 < env.height and 0 <= tx2 < env.width and (ty2, tx2) not in tail:
                return alt
        return 0

    return action

def play_with_heuristic(env, scale=10, fps=5, num_episodes=10):
    pygame.init()
    window_size = ((env.width + 2 * env.border) * scale, (env.height + 2 * env.border) * scale)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Snake Heuristic Play")
    clock = pygame.time.Clock()
    scores = []
    start_time = time.time()
    paused = False

    for ep in range(1, num_episodes + 1):
        raw_state, _, done, info = env.reset()
        total = 0
        step_count = 0
        reset_path()
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    paused = not paused

            if paused:
                clock.tick(fps)
                continue

            # Get heuristic action
            action = heuristic_policy(env, pathfind="astar")
            print(f"Ep {ep}, Step {step_count}, Action: {action}, Head: {env.snake[0]}, Apple: {env.apples[0] if env.apples else None}")
            raw_next, reward, done, info = env.step(action)
            total += reward
            step_count += 1

            # Render current frame
            disp = (raw_next * 255).astype(np.uint8)
            surf = pygame.surfarray.make_surface(disp)
            surf = pygame.transform.scale(surf, window_size)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(fps)

        scores.append(total)
        print(f"Heuristic Ep {ep}/{num_episodes} | Score: {total:.2f}, Steps: {step_count}")

        # Pause briefly to show final state
        pygame.time.wait(500)
        for _ in range(int(fps * 0.5)):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    paused = not paused
            if paused:
                clock.tick(fps)

    print(f"Score médio: {np.mean(scores):.2f}, Desvio padrão: {np.std(scores):.2f}, Tempo: {time.time() - start_time:.1f}s")
    pygame.quit()
