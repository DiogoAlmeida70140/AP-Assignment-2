import heapq
import random
from collections import deque
import pygame
import numpy as np
import time

def random_policy():
    """
    Selects a random action from {-1, 0, 1}.
    """
    return random.choice([-1, 0, 1])


def manhattan(a, b):
    """
    Returns the Manhattan distance between two points a and b.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

########################## A* Pathfinding with A* Search Algorithm ##########################
def astar_path(head, apple, snake_body, width, height, border):
    """
    Finds a path from head to apple using the A* search algorithm.
    Takes into account snake body and borders as obstacles.
    Returns a list of (y, x) positions from head to apple, or None if no path exists.
    """
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
    heapq.heappush(open_set, (0 + manhattan(start, goal), 0, start, [head]))

    visited = set()

    while open_set:
        _, cost, current, path = heapq.heappop(open_set)

        if current == goal:
            return path  

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
    Finds the shortest path from head to apple using Breadth-First Search (BFS).
    Avoids the snake body and borders.
    Returns a list of (y, x) positions from head to apple, or None if no path exists.
    """

    # 1) Construct a boolean “free or not” 2D grid (with border ignored/included)
    # The inner grid starts (border, border) at (border+height-1, border+width-1)
    rows = height + 2 * border
    cols = width + 2 * border

    # Creates “occupied = True” array whenever it is part of body or border.
    occupied = [[False]*cols for _ in range(rows)]
    # Mark border as occupied
    for i in range(rows):
        for j in range(cols):
            if i < border or i >= border+height or j < border or j >= border+width:
                occupied[i][j] = True

    # Mark each segment of the snake as occupied
    # (Includes the head; but let's assume we can "move the head" to itself in the first step)
    for (cy, cx) in snake_body:
        occupied[cy + border][cx + border] = True

    # Calculate “global” coordinates (with offset edge)
    hy, hx = head
    ay, ax = apple
    start = (hy + border, hx + border)
    goal = (ay + border, ax + border)

    #2) Normal BFS on 4-connected grid
    queue = deque([start])
    visited = {start: None} 

    while queue:
        curr = queue.popleft()
        if curr == goal:
            break

        y, x = curr
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                if not occupied[ny][nx] and (ny,nx) not in visited:
                    visited[(ny,nx)] = curr
                    queue.append((ny,nx))

    # If you didn't "visit" the goal, there's no way
    if goal not in visited:
        return None

    # Reconstructs the path backwards
    path = []
    node = goal
    while node is not None:
        # Convert back to borderless coordinates (subtracting border)
        py, px = node
        path.append((py - border, px - border))
        node = visited[node]

    path.reverse()
    return path  

path = []

def reset_path():
    """
    Resets the global path list used in heuristic policy.
    """
    global path
    path=[]

def heuristic_policy(env, pathfind="bfs"):
    """
    Heuristic policy to control the snake using either BFS or A*.
    Returns an action (-1, 0, 1) based on the next desired direction.
    If no path is found, performs a safe one-step look-ahead to avoid collisions.
    """
    global path
    score, apples, head, tail, direction = env.get_state()
    if not apples:
        return 0

    apple = min(apples, key=lambda a: manhattan(head, a))
    hy, hx = head
    ay, ax = apple

    pathfinder = bfs_path if pathfind == "bfs" else astar_path
    if path is None or len(path) == 0:
        path = pathfinder(head, apple, [head] + tail, env.width, env.height, env.border)
        if path is not None:
            path.pop(0)  # Remove the head from the beginning of the path

    if path is not None and len(path) >= 1:
        # The second path node is the next cell we should move to
        next_cell = path.pop(0)
        ny, nx = next_cell
        # Calculate “desired direction” based on the difference between head and next_cell
        # If head=(hy,hx) and next_cell=(ny,nx), then:
        if ny < hy:
            desired = 0  # N
        elif ny > hy:
            desired = 2  # S
        elif nx > hx:
            desired = 1  # E
        else:
            desired = 3  # W

        # Convert desired to action = {-1,0,1} based on “direction”
        action = ((desired - direction + 2) & 3) - 2
        return action

    
    hy, hx = head
    # Replace original “desired direction” action previously used (point to fruit) 
    # Reuse previous logic for “desired” and “diff”
    if ay < hy:
        desired = 0
    elif ay > hy:
        desired = 2
    elif ax > hx:
        desired = 1
    else:
        desired = 3

    action = ((desired - direction + 2) & 3) - 2

    # Simulates “see which cell next” if using action
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

    # If it collides immediately, seeks least worst alternative action
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
    """
    Plays the Snake game using a heuristic policy (A* or BFS) with Pygame rendering.
    Handles display, pause, and quit events.
    Outputs score per episode and statistics at the end.
    """
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
