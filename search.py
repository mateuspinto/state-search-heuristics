import heapq
from math import dist

WALL = "X"
START_STATE = "S"
GOAL_STATE = "G"


def plan(map, algorithm="bfs", heuristic=None):
    """Loads a level, searches for a path between the given waypoints, and displays the result.

    Args:
        filename: The name of the text file containing the level.
        src_waypoint: The character associated with the initial waypoint.
        dst_waypoint: The character associated with the destination waypoint.

    """
    print(map)
    print("Algorithm:", algorithm)
    print("Heuristic:", heuristic)

    # Load the level from the file
    level = parse_level(map)

    # Retrieve the source and destination coordinates from the level.
    start = level["start"]
    goal = level["goal"]

    # Search for and display the path from src to dst.
    path = []
    visited = {}

    if algorithm == "bfs":
        path, visited = bfs(start, goal, level, transition_model)
    elif algorithm == "dfs":
        path, visited = dfs(start, goal, level, transition_model)
    elif algorithm == "ucs":
        path, visited = ucs(start, goal, level, transition_model)
    elif algorithm == "greedy":
        if heuristic == "euclidian":
            path, visited = greedy_best_first(
                start, goal, level, transition_model, h_euclidian
            )
        elif heuristic == "manhattan":
            path, visited = greedy_best_first(
                start, goal, level, transition_model, h_manhattan
            )
    elif algorithm == "astar":
        if heuristic == "euclidian":
            path, visited = a_star(start, goal, level, transition_model, h_euclidian)
        elif heuristic == "manhattan":
            path, visited = a_star(start, goal, level, transition_model, h_manhattan)

    return path, path_cost(path, level), visited


def parse_level(map):
    """Parses a level from a string.

    Args:
        level_str: A string containing a level.

    Returns:
        The parsed level (dict) containing the locations of walls (set), the locations of spaces
        (dict), and a mapping of locations to waypoints (dict).
    """
    start = None
    goal = None
    walls = set()
    spaces = {}

    for j, line in enumerate(map.split("\n")):
        for i, char in enumerate(line):
            if char == "\n":
                continue
            elif char == WALL:
                walls.add((i, j))
            elif char == START_STATE:
                start = (i, j)
                spaces[(i, j)] = 1.0
            elif char == GOAL_STATE:
                goal = (i, j)
                spaces[(i, j)] = 1.0
            elif char.isnumeric():
                spaces[(i, j)] = float(char)

    level = {"walls": walls, "spaces": spaces, "start": start, "goal": goal}

    return level


def path_cost(path, level):
    """Returns the cost of the given path.

    Args:
        path: A list of cells from the source to the goal.
        level: A loaded level, containing walls, spaces, and waypoints.

    Returns:
        The cost of the given path.
    """
    cost = 0
    for i in range(len(path) - 1):
        cost += cost_function(
            level,
            path[i],
            path[i + 1],
            level["spaces"][path[i]],
            level["spaces"][path[i + 1]],
        )

    return cost


# =============================
# Transition Model
# =============================


def cost_function(level, state1, state2, cost1, cost2):
    """Returns the cost of the edge joining state1 and state2.

    Args:
        state1: A source location.
        state2: A target location.

    Returns:
        The cost of the edge joining state1 and state2.
    """

    return dist(state1, state2) * ((cost1 + cost2) / 2)  # Using professor's formula


def transition_model(level, state1):
    """Provides a list of adjacent states and their respective costs from the given state.

    Args:
        level: A loaded level, containing walls, spaces, and waypoints.
        state: A target location.

    Returns:
        A list of tuples containing an adjacent sates's coordinates and the cost of
        the edge joining it and the originating state.

        E.g. from (0,0):
            [((0,1), 1),
             ((1,0), 1),
             ((1,1), 1.4142135623730951),
             ... ]
    """
    adj_states = {}
    # CHANGED: colocando da esq p dir, de cima p baixo. Fazer isso mudou nda
    POSSIBLE_MOVES = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ]  # Eight possible moves, as walking diagonally is allowed

    all_new_positions = [(x + state1[0], y + state1[1]) for x, y in POSSIBLE_MOVES]
    possible_new_positions = list(
        filter(
            lambda position: (
                position in level["spaces"]
            ),  # Must be inside the labyrinth
            all_new_positions,
        )
    )

    adj_states = {
        possible_new_position: cost_function(
            level,
            state1,
            possible_new_position,
            level["spaces"][state1],
            level["spaces"][possible_new_position],
        )
        for possible_new_position in possible_new_positions
    }  # Adding the movement cost of the new positions

    return adj_states.items()


def construct_path(visited, s, g):
    """Constructs the path from the start node to the goal node based on the visited nodes dictionary.

    Args:
        visited: A dictionary containing the visited nodes during a graph search.
        s: The start node represented as a tuple.
        g: The goal node represented as a tuple.

    Returns:
        A list representing the path from the start node to the goal node.
    """
    looking_back = g
    path = []

    while True:
        path.append(looking_back)
        if looking_back == s:
            break
        looking_back = visited[looking_back]
    path.reverse()  # It's not really necessary for the plot, since only the path and not the order matters for coloring the labyrinth. However, I wanted to make it more formal.

    print(path)
    return path


# =============================
# Uninformed Search Algorithms
# =============================


def bfs(s, g, level, adj):
    """Searches for a path from the source to the goal using the Breadth-First Search algorithm.

    Args:
        s: The source location.
        g: The goal location.
        level: The level containing the locations of walls, spaces, and waypoints.
        adj: A function that returns the adjacent cells and their respective costs from the given cell.

    Returns:
        A list of tuples containing cells from the source to the goal, and a dictionary
        containing the visited cells and their respective parent cells.
    """
    visited = {s: None}
    queue = [s]

    while queue:  # While there are still nodes to be visited
        current = queue.pop(0)

        if (
            current == g
        ):  # If the goal is reached, returns the real path and visited nodes
            return construct_path(visited, s, g), visited

        for neighbor, _ in adj(level, current):
            if neighbor not in visited:
                visited[neighbor] = current
                queue.append(neighbor)

    return [], visited  # If the goal is not reached, returns an empty path


def dfs(s, g, level, adj):
    """Searches for a path from the source to the goal using the Depth-First Search algorithm.
    Args:
        s: The source location.
        g: The goal location.
        level: The level containing the locations of walls, spaces, and waypoints.
        adj: A function that returns the adjacent cells and their respective costs from the given cell.

    Returns:
        A list of tuples containing cells from the source to the goal, and a dictionary containing the visited cells and their respective parent cells.
    """
    visited = {s: None}
    stack = [s]

    while stack:  # While there are still nodes to be visited
        current = stack.pop()

        if (
            current == g
        ):  # If the goal is reached, returns the real path and visited nodes
            return construct_path(visited, s, g), visited

        for neighbor, _ in adj(level, current):
            if neighbor not in visited:
                visited[neighbor] = current
                stack.append(neighbor)

    return [], visited  # If the goal is not reached, returns an empty path


class MinHeap:
    def __init__(self, initial_values):
        self.heap = initial_values
        heapq.heapify(self.heap)

    def __bool__(self):
        return bool(self.heap)

    def __str__(self):
        return str(self.heap)

    def __repr__(self) -> str:
        return self.__str__()

    def pop(self):
        return heapq.heappop(self.heap)

    def append(self, cost, node):
        for i, (c, n) in enumerate(self.heap):
            if n == node and c <= cost:
                return False
            if n == node:
                self.heap[i] = (
                    cost,
                    node,
                )
                heapq.heapify(self.heap)
                return True
        else:
            self.heap.append((cost, node))
            heapq.heapify(self.heap)
            return True


def ucs(s, g, level, adj):
    """Searches for a path from the source to the goal using the Uniform-Cost Search algorithm.

    Args:
        s: The source location.
        g: The goal location.
        level: The level containing the locations of walls, spaces, and waypoints.
        adj: A function that returns the adjacent cells and their respective costs from the given cell.

    Returns:
        A list of tuples containing cells from the source to the goal, and a dictionary containing the visited cells and their respective parent cells.
    """
    visited = {s: None}
    actual_best_costs = {s: 0}

    heap = MinHeap([(0, s)])

    while heap:  # While there are still nodes to be visited
        _, parent_node = heap.pop()
        if parent_node == g:
            return construct_path(visited, s, g), visited

        for child_node, movement_cost in adj(level, parent_node):
            new_cost = actual_best_costs[parent_node] + movement_cost
            current_cost = actual_best_costs.get(child_node, float("inf"))

            if (child_node not in visited) and (new_cost < current_cost):
                actual_best_costs[child_node] = new_cost
                visited[child_node] = parent_node
                heap.append(new_cost, child_node)

    return [], visited  # If the goal is not reached, returns an empty path


# ======================================
# Informed (Heuristic) Search Algorithms
# ======================================
def greedy_best_first(s, g, level, adj, h):
    """Searches for a path from the source to the goal using the Greedy Best-First Search algorithm.

    Args:
        s: The source location.
        g: The goal location.
        level: The level containing the locations of walls, spaces, and waypoints.
        adj: A function that returns the adjacent cells and their respective costs from the given cell.
        h: A heuristic function that estimates the cost from the current cell to the goal.

    Returns:
        A list of tuples containing cells from the source to the goal, and a dictionary containing the visited cells and their respective parent cells.
    """
    visited = {s: None}
    actual_best_costs = {s: 0}

    heap = MinHeap([(h(s, g), s)])

    while heap:  # While there are still nodes to be visited
        _, parent_node = heap.pop()
        if parent_node == g:
            return construct_path(visited, s, g), visited

        for child_node, movement_cost in adj(level, parent_node):
            new_cost = actual_best_costs[parent_node] + movement_cost
            current_cost = actual_best_costs.get(child_node, float("inf"))

            if (child_node not in visited) and (new_cost < current_cost):
                actual_best_costs[child_node] = new_cost
                visited[child_node] = parent_node
                heap.append(h(child_node, g), child_node)

    return [], visited  # If the goal is not reached, returns an empty path


def a_star(s, g, level, adj, h):
    """Searches for a path from the source to the goal using the A* algorithm.

    Args:
        s: The source location.
        g: The goal location.
        level: The level containing the locations of walls, spaces, and waypoints.
        adj: A function that returns the adjacent cells and their respective costs from the given cell.
        h: A heuristic function that estimates the cost from the current cell to the goal.

    Returns:
        A list of tuples containing cells from the source to the goal, and a dictionary containing the visited cells and their respective parent cells.
    """
    visited = {s: None}
    actual_best_costs = {s: 0}

    heap = MinHeap([(h(s, g), s)])

    while heap:  # While there are still nodes to be visited
        _, parent_node = heap.pop()
        if parent_node == g:
            return construct_path(visited, s, g), visited

        for child_node, movement_cost in adj(level, parent_node):
            new_cost = actual_best_costs[parent_node] + movement_cost
            current_cost = actual_best_costs.get(child_node, float("inf"))

            if (child_node not in visited) and (new_cost < current_cost):
                actual_best_costs[child_node] = new_cost
                visited[child_node] = parent_node
                heap.append(new_cost + h(child_node, g), child_node)

    return [], visited  # If the goal is not reached, returns an empty path


# ======================================
# Heuristic functions
# ======================================
def h_euclidian(s, g):
    """Estimates the cost from the current cell to the goal using the Euclidian distance.

    Args:
        s: The current location.
        g: The goal location.

    Returns:
        The estimated cost from the current cell to the goal.
    """

    return dist(s, g)


def h_manhattan(s, g):
    """Estimates the cost from the current cell to the goal using the Manhattan distance.

    Args:
        s: The current location.
        g: The goal location.

    Returns:
        The estimated cost from the current cell to the goal.
    """

    return abs(s[0] - g[0]) + abs(s[1] - g[1])
