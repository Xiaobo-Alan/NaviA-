# Copyright (c) Antfin, Inc. All rights reserved.

import copy
import heapq
import numpy as np
import torch
import torch_geometric as thg
from tqdm import tqdm
from model import Agent


def matrix_to_vector(assignment_matrix: np.ndarray) -> list:
    """
    Convert a one-hot assignment matrix into a vector representation.

    Args:
        assignment_matrix: 2D array of shape (n_units, n_servers), where each row has exactly one '1'.

    Returns:
        List of server IDs, indexed by unit ID.
    """
    return np.argmax(np.array(assignment_matrix), axis=1).tolist()


class State:
    """Represents a state in the search space."""

    def __init__(self, assignment_vec: list, value: float, step: int, prev_state):
        self.vec = assignment_vec  # Current assignment: unit_id -> server_id
        self.value = value  # Heuristic cost (lower is better)
        self.step = step  # Number of migration steps taken
        self.prev = prev_state  # Pointer to previous state (for path reconstruction)
        self.criticv = 0.0  # Placeholder for critic value (not used)

    def __lt__(self, other):
        """
        Define priority ordering in A* search:
        - Lower value (heuristic) comes first
        - If values are equal, prefer fewer steps
        """
        if self.value != other.value:
            return self.value < other.value
        return self.step < other.step

    def print(self):
        """Print the current state for debugging."""
        print(f"Assignment: {self.vec}, Value: {self.value}")


class SolverInfo:
    """Encapsulates problem instance data and maintains dynamic state during search."""

    def __init__(self, data: dict):
        # Problem dimensions
        self.NUNIT = 550  # Max number of units (padding size)
        self.NSERVER = 40  # Max number of servers
        self.NRES = 7  # Number of resource types (e.g., CPU, memory)

        # Load raw input data
        self.unit_capacity = np.array(data['u_matrix'], dtype=np.float64)  # (n_units, n_res)
        self.server_capacity = np.array(data['c_matrix'], dtype=np.float64)  # (n_servers, n_res)
        self.initial_assignment = np.array(data['x_matrix'], dtype=np.float64)  # Initial placement
        self.target_assignment = np.array(data['x_result'], dtype=np.float64)  # Target placement
        self.resource_tolerance = np.array(data['w_t'], dtype=np.float64)  # Resource scaling factors

        # Apply tolerance scaling to server capacities
        self.server_capacity *= self.resource_tolerance

        # Convert matrix assignments to vectors
        self.initial_assignment_vec = matrix_to_vector(self.initial_assignment)
        self.current_assignment_vec = matrix_to_vector(self.initial_assignment)
        self.target_assignment_vec = matrix_to_vector(self.target_assignment)

        # Derived dimensions
        self.n_unit = len(self.unit_capacity)
        self.n_server = len(self.server_capacity)
        self.n_res = len(self.server_capacity[0])

        self.unit_list = list(range(self.n_unit))
        self.server_list = list(range(self.n_server))

        # Precompute required resources under target assignment
        self.required_resources_per_server = np.zeros((self.n_server, self.n_res))
        for unit_id, server_id in enumerate(self.target_assignment_vec):
            self.required_resources_per_server[server_id] += self.unit_capacity[unit_id]

        # Compute max resource values across all units and servers (for normalization)
        all_capacities = np.concatenate((self.unit_capacity, self.server_capacity), axis=0)
        self.max_resource_values = np.max(all_capacities, axis=0)

        # Dynamic state variables
        self.used_resources = None  # Current usage per server
        self.unit_count_per_server = None  # Count of units on each server
        self.cur_state = None  # Current State object
        self.current_assignment_vec = None  # Current assignment vector
        self.need_update_graphobs = True  # Flag indicating observation needs update

        # Graph observation components (set in updategraphinfo)
        self.resource_usage_matrix = None
        self.degree_array = None  # In/out degree features
        self.node_not_on_position = None  # Binary flag: 1 if unit not at target

        # Feature matrices for GNN
        self.unit_features = None
        self.src_server_features = None  # Current server states
        self.dst_server_features = None  # Target server states

        # Edge index lists for HeteroData
        self.unit_to_src_edges = []
        self.src_to_unit_edges = []
        self.unit_to_dst_edges = []
        self.dst_to_unit_edges = []
        self.src_to_dst_edges = []

        # Action masking
        self.action_mask = None
        self.action_mask_unit = None
        self.edge_candidates = None  # Valid action candidates per unit

        # Cached graph observation
        self.graph_observation = None

    def set_state(self, state: State):
        """Set the current search state and update internal bookkeeping."""
        self.used_resources = np.zeros((self.n_server, self.n_res))
        self.unit_count_per_server = np.zeros((self.n_server,))
        self.cur_state = state
        self.current_assignment_vec = state.vec.copy()
        self.need_update_graphobs = True

        # Update used resources based on current assignment
        for unit_id, server_id in enumerate(state.vec):
            self.used_resources[server_id] += self.unit_capacity[unit_id]
            self.unit_count_per_server[server_id] += 1

    def is_action_valid(self, unit_id: int, target_server: int) -> bool:
        """
        Check if migrating unit_id to target_server is valid.

        Conditions:
        - Not staying on the same server
        - Does not violate capacity constraints

        Args:
            unit_id: ID of the unit to migrate
            target_server: Target server ID

        Returns:
            True if action is valid, False otherwise
        """
        current_server = self.cur_state.vec[unit_id]
        if current_server == target_server:
            return False
        new_usage = self.used_resources[target_server] + self.unit_capacity[unit_id]
        return np.all(new_usage <= self.server_capacity[target_server])

    def select_top_actions(self, agent, k1: int = 3, k2: int = 2) -> list:
        """
        Use the RL agent to select top-k valid actions.

        Args:
            agent: Trained RL agent with actor model
            k1: Number of top units to consider
            k2: Number of top servers per unit

        Returns:
            List of valid actions as tuples (unit_id, server_id)
        """
        obs = self.get_graph_observation()
        raw_action_list = agent.actor.eval_actlist([obs], [self.edge_candidates], k1, k2)
        valid_actions = []
        for act in raw_action_list:
            unit_idx = act[0].item()
            server_idx = act[1].item()
            if self.is_action_valid(unit_idx, server_idx):
                valid_actions.append((unit_idx, server_idx))
        return valid_actions

    def select_all_valid_actions(self) -> list:
        """
        Enumerate all valid migration actions (used for exhaustive baselines).

        Returns:
            List of all valid (unit_id, server_id) pairs
        """
        valid_actions = []
        for unit_id in self.unit_list:
            for server_id in self.server_list:
                if self.is_action_valid(unit_id, server_id):
                    valid_actions.append((unit_id, server_id))
        return valid_actions

    def heuristic_cost(self, assignment_vec: list) -> float:
        """
        Compute heuristic cost as the number of units not in their target positions.

        Args:
            assignment_vec: Current assignment vector

        Returns:
            Number of mismatched assignments
        """
        diff = np.not_equal(np.array(assignment_vec), np.array(self.target_assignment_vec))
        return float(np.sum(diff))

    def update_graph_info(self):
        """Update auxiliary graph-level features for GNN input construction."""
        self.resource_usage_matrix = self.used_resources.copy()
        self.degree_array = np.zeros((self.n_server, 2))  # (in-degree, out-degree) scaled
        self.node_not_on_position = np.zeros((self.n_unit,))

        for i in range(len(self.current_assignment_vec)):
            self.degree_array[self.current_assignment_vec[i], 0] += 0.1
            self.degree_array[self.target_assignment_vec[i], 1] += 0.1
            if self.current_assignment_vec[i] != self.target_assignment_vec[i]:
                self.node_not_on_position[i] = 1.0

        # Normalize and construct node features
        self.unit_features = np.zeros((self.n_unit, self.NRES + 2), dtype=np.float64)
        self.src_server_features = np.zeros((self.n_server, self.NRES * 2 + 2), dtype=np.float64)
        self.dst_server_features = np.zeros((self.n_server, self.NRES * 2 + 2), dtype=np.float64)

        for i, u in enumerate(self.unit_capacity):
            self.unit_features[i] = np.concatenate((
                u / self.max_resource_values,
                [self.node_not_on_position[i], 1.0]
            ))

        for i, c in enumerate(self.server_capacity):
            self.src_server_features[i] = np.concatenate((
                c / self.max_resource_values,
                (c - self.resource_usage_matrix[i]) / self.max_resource_values,
                [self.degree_array[i, 0], 0.0]
            ))
            self.dst_server_features[i] = np.concatenate((
                c / self.max_resource_values,
                (c - self.required_resources_per_server[i]) / self.max_resource_values,
                [self.degree_array[i, 1], 0.0]
            ))

        # Reset edge lists
        self.unit_to_src_edges = []
        self.src_to_unit_edges = []
        self.unit_to_dst_edges = []
        self.dst_to_unit_edges = []
        self.src_to_dst_edges = []

        for i in range(self.n_unit):
            current_server = self.current_assignment_vec[i]
            target_server = self.target_assignment_vec[i]
            self.unit_to_src_edges.append([i, current_server])
            self.src_to_unit_edges.append([current_server, i])
            self.unit_to_dst_edges.append([i, target_server])
            self.dst_to_unit_edges.append([target_server, i])

        for j in range(self.n_server):
            self.src_to_dst_edges.append([j, j])

        # Update action mask and candidates
        self.action_mask = np.zeros((self.NUNIT, self.NSERVER), dtype=np.int32)
        self.action_mask_unit = np.zeros((self.NUNIT,), dtype=np.int32)
        self.edge_candidates = [[] for _ in range(self.n_unit)]

        for i in range(self.n_unit):
            for nj in range(self.n_server):
                if self.is_action_valid(i, nj):
                    self.action_mask[i][nj] = 1
                    self.action_mask_unit[i] = 1
                    self.edge_candidates[i].append(nj)

    def list_to_edge_index(self, edge_list: list) -> torch.Tensor:
        """Convert Python list of edges to PyTorch Geometric edge index tensor."""
        if not edge_list:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    def get_graph_observation(self) -> thg.data.HeteroData:
        """
        Construct and return a heterogeneous graph observation for the current state.

        Returns:
            HeteroData object with node features and edge indices.
        """
        if self.need_update_graphobs:
            self.need_update_graphobs = False
            self.update_graph_info()
            data = thg.data.HeteroData()

            data['units'].x = torch.tensor(self.unit_features, dtype=torch.float)
            data['src_servers'].x = torch.tensor(self.src_server_features, dtype=torch.float)
            data['dst_servers'].x = torch.tensor(self.dst_server_features, dtype=torch.float)

            data['units', 'in', 'src_servers'].edge_index = self.list_to_edge_index(self.unit_to_src_edges)
            data['src_servers', 'contains', 'units'].edge_index = self.list_to_edge_index(self.src_to_unit_edges)
            data['units', 'moveto', 'dst_servers'].edge_index = self.list_to_edge_index(self.unit_to_dst_edges)
            data['dst_servers', 'contains', 'units'].edge_index = self.list_to_edge_index(self.dst_to_unit_edges)
            data['dst_servers', 'correspond', 'src_servers'].edge_index = self.list_to_edge_index(self.src_to_dst_edges)
            data['src_servers', 'rcorrespond', 'dst_servers'].edge_index = self.list_to_edge_index(
                self.src_to_dst_edges)

            self.graph_observation = data
        return self.graph_observation


def find_mismatched_unit(current_vec: list, target_vec: list) -> int:
    """
    Find the first unit that is not in its target position.

    Args:
        current_vec: Current assignment vector
        target_vec: Target assignment vector

    Returns:
        Index of first mismatched unit, or -1 if all match
    """
    for i in range(len(current_vec)):
        if current_vec[i] != target_vec[i]:
            return i
    return -1


def reconstruct_path(final_state: State, closed_set: dict) -> int:
    """
    Reconstruct the migration path from start to goal.

    Args:
        final_state: Reached goal state
        closed_set: Dictionary mapping state vectors to State objects

    Returns:
        Length of the migration path
    """
    path = []
    state = final_state
    while state.prev is not None:
        prev_state = closed_set[tuple(state.prev)]
        unit_id = find_mismatched_unit(state.vec, prev_state.vec)
        path.append((unit_id, prev_state.vec[unit_id], state.vec[unit_id]))  # (unit, src, dst)
        state = prev_state
    return len(path)


def Astar_search(
        data: dict,
        agent: Agent,
        timeout: float = 5.0,
        use_rl_policy: bool = False,
        k1: int = 3,
        k2: int = 2
) -> tuple:
    """
    Perform A* search guided by RL policy for action pruning.

    Args:
        data: Problem instance dictionary
        agent: Trained RL agent for action selection
        timeout: Time limit in seconds
        use_rl_policy: Whether to use RL-guided action selection (vs. full enumeration)
        k1: Top-k units from RL policy
        k2: Top-k servers per unit from RL policy

    Returns:
        Tuple of (path_length or status, expanded_nodes, branching_factor_sum, runtime)
    """
    start_time = time.time()
    solver_info = SolverInfo(data)

    # If already at target, no action needed
    if np.array_equal(solver_info.current_assignment_vec, solver_info.target_assignment_vec):
        return "UNCHANGED", 0, 0, 0.0

    open_set = []
    closed_set = {}  # Maps state tuple -> State object

    initial_state = State(solver_info.current_assignment_vec, 0, 0, None)
    heapq.heappush(open_set, initial_state)
    closed_set[tuple(initial_state.vec)] = initial_state

    expanded_count = 0
    branching_factor_sum = 0

    while open_set:
        if time.time() - start_time > timeout:
            return "TIMEOUT", expanded_count, branching_factor_sum, time.time() - start_time

        current_state = heapq.heappop(open_set)
        expanded_count += 1

        solver_info.set_state(current_state)

        # Select action candidates
        if use_rl_policy:
            action_list = solver_info.select_top_actions(agent, k1, k2)
        else:
            action_list = solver_info.select_all_valid_actions()

        branching_factor_sum += len(action_list)

        for unit_id, server_id in action_list:
            next_vec = current_state.vec.copy()
            next_vec[unit_id] = server_id

            # Skip if already visited with better step count
            next_key = tuple(next_vec)
            if next_key in closed_set:
                continue

            # Compute heuristic
            next_value = solver_info.heuristic_cost(next_vec)

            if next_value == 0:  # Goal reached
                final_state = State(next_vec, 0, current_state.step + 1, current_state.vec)
                path_length = reconstruct_path(final_state, closed_set)
                runtime = time.time() - start_time
                return path_length, expanded_count, branching_factor_sum, runtime

            next_state = State(next_vec, next_value, current_state.step + 1, current_state.vec)
            heapq.heappush(open_set, next_state)
            closed_set[next_key] = next_state

    return "INFEASIBLE", expanded_count, branching_factor_sum, time.time() - start_time
