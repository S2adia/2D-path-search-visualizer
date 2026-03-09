import numpy as np
import networkx as nx
import random

# Constants
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)] #North, South, West, East
FREE = 0
OBSTACLE = 1

#grid generation
def generate_grid(n: int, obstacle_pct: float, seed: int = None) -> np.ndarray:
    """create an NxN grid with random obstacles.
    args:
        n: grid dimension
        obstacle_pct: fraction of cells to block
        seed: Optional  seed
    returns:
        NxN numpy array where 0=free, 1=obstacle
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    grid = np.zeros((n, n), dtype=int)
    num_obstacles = int(n * n * obstacle_pct)
    all_cells = list(range(n * n))
    obstacle_indices = random.sample(all_cells, num_obstacles)

    for idx in obstacle_indices:
        r, c = divmod(idx, n)
        grid[r][c] = OBSTACLE

    return grid

def place_start_goal(grid: np.ndarray, seed: int = None):
    """place start and goal at random free cells.
    args:
        grid: NxN numpy array
        seed: optional  seed
    returns:
        (start, goal) as (row, col) tuples
    """
    if seed is not None:
        random.seed(seed)

    free_cells = list(zip(*np.where(grid == FREE)))
    if len(free_cells) < 2:
        raise ValueError("Grid has fewer than 2 free cells — cannot place start and goal.")

    start, goal = random.sample(free_cells, 2)
    return tuple(start), tuple(goal)

#graph construction
def build_graph(grid: np.ndarray) -> nx.Graph:
    """build an undirected graph from traversable cells.
    args:
        grid: NxN numpy array
    returns:
        NetworkX graph with 4-connected neighbors
    """
    n = grid.shape[0]
    G = nx.Graph()

    for r in range(n):
        for c in range(n):
            if grid[r][c] == FREE:
                G.add_node((r, c))
                for dr, dc in DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == FREE:
                        G.add_edge((r, c), (nr, nc))

    return G

def is_reachable(G: nx.Graph, start: tuple, goal: tuple) -> bool:
    """check if goal is reachable from start.
    args:
        G: NetworkX graph
        start: start node
        goal: goal node
    returns:
        True if path exists
    """
    if start not in G or goal not in G:
        return False
    return nx.has_path(G, start, goal)

def get_neighbors(grid: np.ndarray, pos: tuple) -> list:
    """return valid 4-cardinal neighbors.
    args:
        grid: NxN numpy array
        pos: current position
    returns: lst of valid neighbor tuples
    """
    n = grid.shape[0]
    r, c = pos
    neighbors = []
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == FREE:
            neighbors.append((nr, nc))
    return neighbors
