import heapq
import random
from collections import deque

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
    BFS to find the shortest path from `head` to `apple`, avoiding snake_body and border.
    Args:
        head: Tuple (y_head, x_head)
        apple: Tuple (y_apple, x_apple)
        snake_body: List of (y, x) for body (includes head at snake_body[0])
        width, height: Dimensions of the game board (excluding border)
        border: Number of border layers
    Returns:
        List of (y, x) tuples from head to apple, or None if no path exists
    """
    rows = height + 2 * border
    cols = width + 2 * border
    occupied = [[False] * cols for _ in range(rows)]
    
    # Mark border as occupied
    for i in range(rows):
        for j in range(cols):
            if i < border or i >= border + height or j < border or j >= border + width:
                occupied[i][j] = True
    
    # Mark snake body as occupied)
    for cy, cx in snake_body:
        occupied[cy + border][cx + border] = True
    
    # Convert to border-adjusted coordinates
    hy, hx = head
    ay, ax = apple
    start = (hy + border, hx + border)
    goal = (ay + border, ax + border)
    
    # BFS
    queue = deque([start])
    visited = {start: None}
    while queue:
        curr = queue.popleft()
        if curr == goal:
            break
        y, x = curr
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                if not occupied[ny][nx] and (ny, nx) not in visited:
                    visited[(ny, nx)] = curr
                    queue.append((ny, nx))
    
    # Reconstruct path
    if goal not in visited:
        return None
    path = []
    node = goal
    while node is not None:
        py, px = node
        path.append((py - border, px - border))
        node = visited[node]
    path.reverse()
    return path

def heuristic_policy(env):
    """
    Heuristic policy to move the snake toward the apple using BFS or a one-step look-ahead.
    Args:
        env: SnakeGame environment instance
    Returns:
        Action (-1, 0, 1) for left, straight, or right
    """
    score, apples, head, tail, direction = env.get_state()
    if not apples:
        return 0
    apple = apples[0]
    hy, hx = head
    ay, ax = apple

    # BFS path
    # path = bfs_path(head, apple, [head] + tail, env.width, env.height, env.border)
    path = bfs_path(head, apple, [head] + tail, env.width, env.height, env.border)
    if path is not None and len(path) >= 2:
        # O segundo nó de path é a próxima célula para onde nos devemos mover
        next_cell = path[1]
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
        diff = (desired - direction) % 4
        if diff == 0:
            action = 0
        elif diff == 1:
            action = 1
        elif diff == 3:
            action = -1
        else:
            action = 0

        return action

    # Fallback: one-step look-ahead
    if ay < hy:
        desired = 0
    elif ay > hy:
        desired = 2
    elif ax > hx:
        desired = 1
    else:
        desired = 3
    action = (desired - direction + 2) & 3 - 2

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