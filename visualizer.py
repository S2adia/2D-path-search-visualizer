import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np


#color Palette
COLORS = {
    "free": "#F0F0F0",  #light grey—open cell
    "obstacle": "#2C2C2C",  # dark grey — blocked cell
    "visited": "#90CAF9",  # soft blue—explored
    "frontier": "#FFF176",  # yellow—queued nodes
    "path": "#FF7043",  # orange-red —final solution path
    "start": "#66BB6A",  # green—start node
    "goal": "#EF5350",  # red — goal node
    "agent": "#AB47BC",  # purple — current agent position
    "grid_line": "#BDBDBD",  # grid lines
    # Tree node colors
    "tree_node": "#90CAF9",
    "tree_current": "#AB47BC",
    "tree_path": "#FF7043",
    "tree_edge": "#546E7A",
    "tree_edge_path": "#FF7043",
}


# grid drawing
def draw_grid(
    ax,
    grid,
    start,
    goal,
    agent_pos,
    visited,
    frontier=None,
    path=None,
    step_count=0,
    states_explored=0,
):
    """render the 2D grid.
    args:
        ax: Matplotlib axes
        grid: NxN numpy array
        start: start position
        goal: goal position
        agent_pos: current agent position
        visited: set of explored nodes
        frontier: list of queued nodes
        path: final solution path
        step_count: number of steps
        states_explored: total states visited
    """
    ax.clear()
    n = grid.shape[0]
    frontier = frontier or []
    path = path or []
    path_set = set(map(tuple, path))
    for r in range(n):
        for c in range(n):
            cell = (r, c)
            #determine cell color
            if grid[r][c] == 1:
                color = COLORS["obstacle"]
            elif cell in path_set:
                color = COLORS["path"]
            elif cell == tuple(start):
                color = COLORS["start"]
            elif cell == tuple(goal):
                color = COLORS["goal"]
            elif cell == tuple(agent_pos):
                color = COLORS["agent"]
            elif cell in visited:
                color = COLORS["visited"]
            elif tuple(cell) in [tuple(f) for f in frontier]:
                color = COLORS["frontier"]
            else:
                color = COLORS["free"]
            rect = mpatches.FancyBboxPatch(
                (c + 0.05, n - r - 1 + 0.05),
                0.90,
                0.90,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor=COLORS["grid_line"],
                linewidth=0.8,
            )
            ax.add_patch(rect)
            #label start/goal/agent
            label = ""
            fontsize = max(6, 14 - n // 3)
            if cell == tuple(start):
                label = "S"
            elif cell == tuple(goal):
                label = "G"
            elif cell == tuple(agent_pos) and cell not in (tuple(start), tuple(goal)):
                label = "●"

            if label:
                ax.text(
                    c + 0.5,
                    n - r - 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    fontweight="bold",
                    color="white",
                    zorder=5,
                )
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"Grid View  |  Steps: {step_count}  |  Explored: {states_explored}",
        fontsize=10,
        pad=8,
        fontweight="bold",
    )

    #legend
    legend_items = [
        mpatches.Patch(color=COLORS["start"], label="Start (S)"),
        mpatches.Patch(color=COLORS["goal"], label="Goal (G)"),
        mpatches.Patch(color=COLORS["agent"], label="Agent"),
        mpatches.Patch(color=COLORS["visited"], label="Visited"),
        mpatches.Patch(color=COLORS["frontier"], label="Frontier"),
        mpatches.Patch(color=COLORS["path"], label="Path"),
        mpatches.Patch(color=COLORS["obstacle"], label="Obstacle"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper left",
        fontsize=6,
        framealpha=0.85,
        bbox_to_anchor=(0, -0.02),
    )
#tree drawing
def draw_tree(
    ax,
    parent_map: dict,
    current_node: tuple,
    start: tuple,
    path: list = None,
    max_nodes: int = 80,
):
    """render the search tree.
    args:
        ax: Matplotlib axes
        parent_map: Node to parent mapping
        current_node: Currently evaluated node
        start: Root node
        path: Final solution path
        max_nodes: Max nodes to render
    """
    ax.clear()

    if not parent_map:
        ax.set_title("Search Tree", fontsize=10, fontweight="bold")
        ax.axis("off")
        return

    path_set = set(map(tuple, path)) if path else set()
    #build tree from parent map
    T = nx.DiGraph()
    for child, par in parent_map.items():
        if par is not None:
            T.add_edge(par, child)
        else:
            T.add_node(child)

    #limit rendering for large trees
    if len(T.nodes) > max_nodes:
        T = _local_subtree(T, current_node, radius=2)

    if len(T.nodes) == 0:
        ax.axis("off")
        return

    #calculate tree depth for better spacing
    def get_depth(node, depth=0):
        children = [n for n in T.neighbors(node)]
        if not children:
            return depth
        return max(get_depth(child, depth + 1) for child in children)
    
    tree_depth = get_depth(start if start in T else list(T.nodes)[0])
    
    #adjust vertical gap based on tree depth
    vert_gap = max(0.15, min(0.3, 2.0 / (tree_depth + 1)))
    
    #calculate tree width for better horizontal spacing
    def get_max_width(node, level=0, level_counts=None):
        if level_counts is None:
            level_counts = {}
        level_counts[level] = level_counts.get(level, 0) + 1
        for child in T.neighbors(node):
            get_max_width(child, level + 1, level_counts)
        return level_counts
    
    level_counts = get_max_width(start if start in T else list(T.nodes)[0])
    max_width = max(level_counts.values()) if level_counts else 1
    
    #scale initial width based on maximum branching
    initial_width = min(10.0, max(2.0, max_width * 0.5))
    #layout
    pos = _hierarchy_pos(
        T, 
        root=start if start in T else list(T.nodes)[0],
        width=initial_width,
        vert_gap=vert_gap
    )

    #node color classification
    node_colors = []
    for node in T.nodes:
        if node == current_node:
            node_colors.append(COLORS["tree_current"])
        elif node in path_set:
            node_colors.append(COLORS["tree_path"])
        elif node == start:
            node_colors.append(COLORS["start"])
        else:
            node_colors.append(COLORS["tree_node"])

    #edge color classification
    edge_colors = []
    for u, v in T.edges:
        if u in path_set and v in path_set:
            edge_colors.append(COLORS["tree_edge_path"])
        else:
            edge_colors.append(COLORS["tree_edge"])

    #dynamic sizing based on tree complexity
    node_size = max(80, min(600, 1200 - len(T.nodes) * 8))
    font_size = max(4, min(8, 12 - len(T.nodes) // 15))
    edge_width = max(0.5, min(2.0, 3.0 - len(T.nodes) / 30))

    nx.draw(
        T,
        pos,
        ax=ax,
        node_color=node_colors,
        edge_color=edge_colors,
        node_size=node_size,
        font_size=font_size,
        font_color="#212121",
        with_labels=True,
        labels={node: f"{node[0]},{node[1]}" for node in T.nodes},
        width=edge_width,
        font_weight="bold",
    )

    ax.set_title("Search Tree  (root = Start ↓)", fontsize=10, fontweight="bold")
    ax.margins(0.1)  

#tree layout helpers
def _hierarchy_pos(
    G: nx.DiGraph,
    root,
    width=1.0,
    vert_gap=0.2,
    vert_loc=0,
    xcenter=0.5,
    pos=None,
    parent=None,
):
    
    #compute top-down hierarchical layout
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)

    children = [n for n in G.neighbors(root) if n != parent]
    if children:
        #increase width multiplier for better spacing
        dx = width / max(1, len(children) - 0.5)
        next_x = xcenter - width / 2 - dx / 2
        for child in children:
            next_x += dx
            pos = _hierarchy_pos(
                G,
                child,
                width=dx * len(children) * 0.8, 
                vert_gap=vert_gap,
                vert_loc=vert_loc - vert_gap,
                xcenter=next_x,
                pos=pos,
                parent=root,
            )
    return pos

def _local_subtree(T: nx.DiGraph, center, radius: int = 2) -> nx.DiGraph:
    #extract subgraph within radius hops of center
    undirected = T.to_undirected()
    if center not in undirected:
        return T
    nearby = nx.ego_graph(undirected, center, radius=radius).nodes
    return T.subgraph(nearby).copy()

#final Result Banner
def draw_result_banner(ax_grid, found: bool, path_length: int, states_explored: int):
    #overlay result message on grid after search completes
    msg = (
        f"Goal reached!  Path: {path_length} steps | Explored: {states_explored} states"
        if found
        else f"No path found (unsolvable)  |  Explored: {states_explored} states"
    )

    color = "#388E3C" if found else "#C62828"

    ax_grid.text(
        0.5,
        -0.08,
        msg,
        transform=ax_grid.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.9),
    )
