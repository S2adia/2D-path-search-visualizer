from collections import deque

def reconstruct_path(parent: dict, goal: tuple) -> list:
    """trace parent map from goal to start
    args:
        parent: Node to predecessor mapping
        goal: Goal node
    returns: path from start to goal
    """
    if goal not in parent:
        return []
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]

    return list(reversed(path))

# BFS
def bfs(graph, start: tuple, goal: tuple):
    """Breadth-First Search with FIFO frontier.
    yields:
        (current, visited, frontier, parent) at each step
    args:
        graph: NetworkX graph
        start: Start node
        goal: Goal node
    """
    queue = deque([start])
    visited = set()
    parent = {start: None}

    while queue:
        current = queue.popleft()

        if current in visited:
            continue
        visited.add(current)

        # Yield current state
        yield current, visited, list(queue), parent

        if current == goal:
            return

        for neighbor in graph.neighbors(current):
            if neighbor not in visited and neighbor not in parent:
                parent[neighbor] = current
                queue.append(neighbor)


# Algorithm Registry 
ALGORITHMS = {
    "BFS": bfs,
}


def get_algorithm(name: str):
    """retrieve search algorithm by name.
    args:
        name: 'BFS'
    returns: generator function
    """
    name = name.upper()
    if name not in ALGORITHMS:
        raise ValueError("Only BFS is available in this version.")
    return ALGORITHMS[name]