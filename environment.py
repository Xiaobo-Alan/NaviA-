import ast
import json
import time
import os
import numpy as np
import gymnasium as gym
import torch
import torch_geometric as thg


def Astar_solve(
        u_matrix: np.ndarray,
        c_matrix: np.ndarray,
        x_matrix: np.ndarray,
        x_result: np.ndarray,
        timeout: float = 0.0,
        res_tolerance_transfer: list = None
) -> tuple:
    """
    Solve the migration path using an external A* solver.

    Args:
        u_matrix: Resource demands of units (n_units, n_res)
        c_matrix: Capacities of servers (n_servers, n_res)
        x_matrix: Initial assignment matrix (one-hot)
        x_result: Target assignment matrix (one-hot)
        timeout: Time limit for solving (passed to C++ solver)
        res_tolerance_transfer: Resource scaling factors per dimension

    Returns:
        (path: list or None, result_code: int)
            - path: List of migration steps [(unit_id, src_server, dst_server)]
            - result_code: 0 if success, non-zero otherwise
    """
    if res_tolerance_transfer is None:
        res_tolerance_transfer = [1.0] * 7

    astar_data = {
        'u_matrix': u_matrix.tolist(),
        'c_matrix': c_matrix.tolist(),
        'x_matrix': x_matrix.tolist(),
        'x_result': x_result.tolist(),
        'timeout': timeout,
        'w_t': res_tolerance_transfer
    }

    with open('astar_data.json', 'w') as f:
        json.dump(astar_data, f)

    start_time = time.time()
    os.system("./astar_path")
    end_time = time.time()

    lines = open('./astar.res', "r").readlines()
    result_code = int(lines[0])
    path = None

    if result_code == 0 and len(lines) > 1:
        try:
            path = ast.literal_eval(lines[1])
        except Exception:
            pass

    return path, result_code


class CloudConsolidationEnv(gym.Env):
    """
    Gym environment for cloud resource consolidation via VM migration.

    Observations are represented as heterogeneous graphs (using PyG), with nodes for:
        - Units (VMs): current state and target position
        - Source Servers: current load and degree
        - Destination Servers: target load and degree

    Action space: MultiDiscrete([n_unit, n_server]) ¡ú select which unit to migrate and where.
    """

    def __init__(self):
        super().__init__()

        # Training configuration
        self.running_mode = 'train'
        self.data_name = 'new_traindata'
        self.INVALID_DATA = []  # Known invalid problem instances (none used)

        # Reward design
        self.REWARD_EPISODE = 1.0  # Reward for completing the full migration
        self.REWARD_UNIT = 0.1  # Reward per unit placed correctly
        self.EPS = 1e-6  # Small epsilon for floating-point comparisons
        self.TIMEOUTSTEP = 100  # Maximum number of steps per episode

        # Difficulty adjustment parameters
        self.UPDATE_LEVEL_PROC = 3  # Successes needed to increase difficulty level

        # Problem dimensions
        self.NUNIT = 300  # Max number of units (for padding)
        self.NSERVER = 30  # Max number of servers
        self.NRES = 7  # Number of resource types (CPU, memory, etc.)

        # Define action and observation spaces
        self.action_space = gym.spaces.MultiDiscrete((self.NUNIT, self.NSERVER))
        self.observation_space = gym.spaces.Graph(
            node_space=gym.spaces.Box(low=0, high=1, shape=(self.NRES,), dtype=np.float32),
            edge_space=None,
            seed=42
        )

        # Initialize data placeholders
        self.data = []
        self.data_ans = []
        self.DATANUM = 0
        self.data_ptr = 0
        self.data_success = False
        self.data_retry_cnt = 0
        self.data_unsolved = np.ones((self.DATANUM,), dtype=np.short)
        self.data_proc = np.zeros((self.DATANUM,), dtype=np.short)
        self.data_level = np.ones((self.DATANUM,), dtype=np.short) * 24

        # Configure default dataset
        self.configure_env(
            running_mode='train',
            data_name='new_traindata',
            do_select=True,
            select_list=[]
        )

    def configure_env(self, running_mode: str = None, data_name: str = None,
                      do_select: bool = False, select_list: list = None):
        """
        Configure the dataset used by the environment.

        Args:
            running_mode: 'train' or 'eval'
            data_name: Name of the dataset file (without extension)
            do_select: Whether to use a subset of instances
            select_list: List of indices to include
        """
        if running_mode:
            self.running_mode = running_mode
        if data_name:
            self.data_name = data_name
            self.data_path = f"./data/{self.data_name}.txt"
            self.data_lines = open(self.data_path, "r").readlines()
            self.data = [ast.literal_eval(line) for line in self.data_lines]

            self.data_ans_path = f"./data/{self.data_name}.ans"
            self.data_ans_lines = open(self.data_ans_path, "r").readlines()
            self.data_ans = [ast.literal_eval(line) for line in self.data_ans_lines]

            if do_select and select_list:
                assert len(select_list) > 0
                self.data = [self.data[i] for i in select_list]
                self.data_ans = [self.data_ans[i] for i in select_list]

            self.DATANUM = len(self.data)
            self.data_ptr = 0
            self.data_success = False
            self.data_retry_cnt = 0
            self.data_unsolved = np.ones((self.DATANUM,), dtype=np.short)
            self.data_proc = np.zeros((self.DATANUM,), dtype=np.short)
            self.data_level = np.ones((self.DATANUM,), dtype=np.short) * 24

    def get_mapping(self, options=None):
        """
        Generate random permutations for unit/server IDs to augment input diversity.

        This helps prevent the agent from overfitting to fixed ID orders.
        """
        self.unit_mapping = np.arange(self.NUNIT, dtype=np.short)
        self.server_mapping = np.arange(self.NSERVER, dtype=np.short)

        if options is None:
            np.random.shuffle(self.unit_mapping)
            np.random.shuffle(self.server_mapping)

        self.unit_corr = np.argsort(self.unit_mapping)
        self.server_corr = np.argsort(self.server_mapping)

    def check_action_validity(self, unit_id: int, src_server: int, dst_server: int) -> bool:
        """
        Check if migrating unit_id from src_server to dst_server is valid.

        Valid if:
            - Not staying on same server
            - Does not violate capacity constraints
        """
        if src_server == dst_server:
            return False
        new_usage = self.r_mat[dst_server] + self.u_mat[unit_id]
        return np.all(new_usage <= self.c_mat[dst_server] + self.EPS)

    def update_action_mask(self):
        """
        Update the action mask indicating valid actions.
        Also constructs edge_candidates for potential pruning.
        """
        self.action_mask = np.zeros((self.NUNIT, self.NSERVER), dtype=np.int32)
        self.action_mask1 = np.zeros((self.NUNIT,), dtype=np.int32)
        self.edge_candidates = [[] for _ in range(self.n_unit)]

        for unit_id in range(self.n_unit):
            current_server = self.x_vec[unit_id]
            mapped_unit = self.unit_mapping[unit_id]
            for dst_server in range(self.n_server):
                mapped_dst = self.server_mapping[dst_server]
                if self.check_action_validity(unit_id, current_server, dst_server):
                    self.action_mask[mapped_unit][mapped_dst] = 1
                    self.action_mask1[mapped_unit] = 1
                    self.edge_candidates[mapped_unit].append(mapped_dst)

    def get_standard_action(self) -> list:
        """
        Get the next optimal action from the A* solver (used for imitation or curriculum).

        Returns:
            Standard action as [unit_id, src_server, dst_server], or None if unavailable.
        """
        std_path, res = Astar_solve(self.u_mat, self.c_mat, self.x_mat, self.x_res)
        if res == 0 and std_path and isinstance(std_path, list) and len(std_path) > 0:
            return std_path[0]
        return None

    def update_procedure_level(self, success: bool):
        """
        Adjust difficulty level based on success/failure.

        Increases level after consecutive successes; decreases after failures.
        """
        ptr = self.data_ptr
        if success:
            self.data_retry_cnt = 0
            self.data_proc[ptr] += 1
            if self.data_proc[ptr] >= self.UPDATE_LEVEL_PROC:
                self.data_success = True
                self.data_level[ptr] = min(self.data_level[ptr] * 2, self.NUNIT)
                self.data_proc[ptr] = 0
        else:
            self.data_proc[ptr] = 0
            self.data_retry_cnt += 0.5
            if self.data_retry_cnt > 20 and self.data_level[ptr] > 1:
                self.data_level[ptr] -= 1
                self.data_retry_cnt = 0

    def update_data(self, options=None):
        """
        Load and preprocess the current problem instance.
        Applies normalization, applies curriculum difficulty, and builds graph features.
        """
        raw_data = self.data[self.data_ptr]
        answer_path = self.data_ans[self.data_ptr]

        self.n_unit = len(raw_data['u_matrix'])
        self.n_server = len(raw_data['c_matrix'])
        self.n_res = len(raw_data['w_t'])

        self.u_mat = np.array(raw_data['u_matrix'], dtype=np.float64)
        self.c_mat = np.array(raw_data['c_matrix'], dtype=np.float64) * np.array(raw_data['w_t'])

        # Normalize resources
        max_values = np.max(np.concatenate((self.u_mat, self.c_mat)), axis=0)
        self.u_mat = self.u_mat / (max_values + self.EPS)
        self.c_mat = self.c_mat / (max_values + self.EPS)

        self.x_mat = np.zeros((self.n_unit, self.n_server))
        old_x = np.array(raw_data['x_matrix'])
        self.x_mat[:, :old_x.shape[1]] = old_x

        self.x_res = np.array(raw_data['x_result'])
        self.x_vec = self.matrix_to_vector(self.x_mat)
        self.x_res_vec = self.matrix_to_vector(self.x_res)

        # Apply curriculum difficulty: simulate partial solution
        if self.running_mode == 'train':
            expected_path_len = len(answer_path)
            self.PATH_LEN = expected_path_len
            difficulty_gap = expected_path_len - self.data_level[self.data_ptr]

            self.r_mat = np.zeros((self.n_server, self.n_res))
            for unit_id, server_id in enumerate(self.x_vec):
                self.r_mat[server_id] += self.u_mat[unit_id]

            for k in range(difficulty_gap):
                unit_id, src_server, dst_server = answer_path[k]
                assert self.check_action_validity(unit_id, src_server, dst_server), \
                    f"Invalid curriculum step {k} for instance {self.data_ptr}"
                self.x_vec[unit_id] = dst_server
                self.x_mat[unit_id, src_server] = 0
                self.x_mat[unit_id, dst_server] = 1
                self.r_mat[src_server] -= self.u_mat[unit_id]
                self.r_mat[dst_server] += self.u_mat[unit_id]

        # Build current state features
        self.r_mat = np.zeros((self.n_server, self.n_res))
        self.nr_mat = np.zeros((self.n_server, self.n_res))  # Required at target
        self.deg_arr = np.zeros((self.n_server, 2))  # Degree features
        self.node_not_on_pos = np.zeros((self.n_unit,))  # Misplaced units

        for unit_id in range(self.n_unit):
            src_server = self.x_vec[unit_id]
            dst_server = self.x_res_vec[unit_id]
            self.r_mat[src_server] += self.u_mat[unit_id]
            self.nr_mat[dst_server] += self.u_mat[unit_id]
            self.deg_arr[src_server, 0] += 0.1
            self.deg_arr[dst_server, 1] += 0.1
            if src_server != dst_server:
                self.node_not_on_pos[unit_id] = 1.0

        self.get_mapping(options)
        self.update_action_mask()
        assert self.action_mask1.sum() > 0, "No valid actions available at reset"

        # Prepare node features
        self.units_x = np.zeros((self.NUNIT, self.NRES + 2), dtype=np.float64)
        self.src_servers_x = np.zeros((self.NSERVER, self.NRES * 2 + 2), dtype=np.float64)
        self.dst_servers_x = np.zeros((self.NSERVER, self.NRES * 2 + 2), dtype=np.float64)

        for i, u in enumerate(self.u_mat):
            self.units_x[self.unit_mapping[i]] = np.concatenate([
                u, [self.node_not_on_pos[i], 1.0]
            ])
        for i, c in enumerate(self.c_mat):
            self.src_servers_x[self.server_mapping[i]] = np.concatenate([
                c, c - self.r_mat[i], [self.deg_arr[i, 0], 0.0]
            ])
        for i, c in enumerate(self.c_mat):
            self.dst_servers_x[self.server_mapping[i]] = np.concatenate([
                c, c - self.nr_mat[i], [self.deg_arr[i, 1], 0.0]
            ])

        # Build edge index lists
        self.u2s_edge_links = []
        self.s2u_edge_links = []
        self.u2d_edge_links = []
        self.d2u_edge_links = []
        self.sd_edge_links = []

        for i in range(self.n_unit):
            src_server = self.x_vec[i]
            dst_server = self.x_res_vec[i]
            mpi = self.unit_mapping[i]
            mp_src = self.server_mapping[src_server]
            mp_dst = self.server_mapping[dst_server]

            self.u2s_edge_links.append([mpi, mp_src])
            self.s2u_edge_links.append([mp_src, mpi])
            self.u2d_edge_links.append([mpi, mp_dst])
            self.d2u_edge_links.append([mp_dst, mpi])

        for j in range(self.n_server):
            mpj = self.server_mapping[j]
            self.sd_edge_links.append([mpj, mpj])

        self.action_seq = []

    @staticmethod
    def matrix_to_vector(x_matrix: np.ndarray) -> list:
        """Convert one-hot assignment matrix to vector of server IDs."""
        return np.argmax(x_matrix, axis=1).tolist()

    def array_to_edge_index(self, arr: list) -> torch.Tensor:
        """Convert list of edges to PyG-compatible edge index tensor."""
        if not arr:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(arr, dtype=torch.long).t().contiguous()

    @property
    def state(self) -> thg.data.HeteroData:
        """Generate current graph-structured observation."""
        data = thg.data.HeteroData()
        data['units'].x = torch.tensor(self.units_x, dtype=torch.float)
        data['src_servers'].x = torch.tensor(self.src_servers_x, dtype=torch.float)
        data['dst_servers'].x = torch.tensor(self.dst_servers_x, dtype=torch.float)

        data['units', 'in', 'src_servers'].edge_index = self.array_to_edge_index(self.u2s_edge_links)
        data['src_servers', 'contains', 'units'].edge_index = self.array_to_edge_index(self.s2u_edge_links)
        data['units', 'moveto', 'dst_servers'].edge_index = self.array_to_edge_index(self.u2d_edge_links)
        data['dst_servers', 'contains', 'units'].edge_index = self.array_to_edge_index(self.d2u_edge_links)
        data['dst_servers', 'correspond', 'src_servers'].edge_index = self.array_to_edge_index(self.sd_edge_links)
        data['src_servers', 'rcorrespond', 'dst_servers'].edge_index = self.array_to_edge_index(self.sd_edge_links)
        return data

    def reset(self, seed=None, options=None):
        """Reset environment to a new problem instance."""
        super().reset(seed=seed)

        if options is None:
            if self.data_success:
                self.data_success = False
                self.data_ptr = (self.data_ptr + 1) % self.DATANUM
                while self.data_ptr in self.INVALID_DATA:
                    self.data_ptr = (self.data_ptr + 1) % self.DATANUM
        else:
            self.data_ptr = options
            self.data_level[options] = self.NUNIT

        self.update_data(options)
        self.rewardsum = 0
        self.stepcount = 0
        return self.state, {}

    def step(self, action):
        """Execute one migration action."""
        self.stepcount += 1
        reward = 0.0

        unit_idx = self.unit_corr[action[0]]
        dst_server = self.server_corr[action[1]]

        if (unit_idx < self.n_unit and
                dst_server < self.n_server and
                self.action_mask[unit_idx, dst_server]):

            src_server = self.x_vec[unit_idx]
            mpi = self.unit_mapping[unit_idx]
            mp_src = self.server_mapping[src_server]
            mp_dst = self.server_mapping[dst_server]

            # Record action
            self.action_seq.append([unit_idx, src_server, dst_server])

            # Update resource usage
            self.r_mat[dst_server] += self.u_mat[unit_idx]
            self.r_mat[src_server] -= self.u_mat[unit_idx]
            self.x_vec[unit_idx] = dst_server
            self.x_mat[unit_idx, src_server] = 0
            self.x_mat[unit_idx, dst_server] = 1

            # Update degree and placement flags
            self.deg_arr[src_server, 0] -= 0.1
            self.deg_arr[dst_server, 0] += 0.1

            was_misplaced = self.node_not_on_pos[unit_idx]
            now_misplaced = int(dst_server != self.x_res_vec[unit_idx])
            self.node_not_on_pos[unit_idx] = now_misplaced

            if not now_misplaced and was_misplaced:
                reward += self.REWARD_UNIT
            elif now_misplaced and not was_misplaced:
                reward -= self.REWARD_UNIT

            # Update features
            self.units_x[mpi] = np.concatenate([self.u_mat[unit_idx], [now_misplaced, 1.0]])
            self.src_servers_x[mp_src] = np.concatenate([
                self.c_mat[src_server], self.c_mat[src_server] - self.r_mat[src_server],
                [self.deg_arr[src_server, 0], 0.0]
            ])
            self.src_servers_x[mp_dst] = np.concatenate([
                self.c_mat[dst_server], self.c_mat[dst_server] - self.r_mat[dst_server],
                [self.deg_arr[dst_server, 0], 0.0]
            ])
            self.dst_servers_x[mp_src] = np.concatenate([
                self.c_mat[src_server], self.c_mat[src_server] - self.nr_mat[src_server],
                [self.deg_arr[src_server, 1], 0.0]
            ])
            self.dst_servers_x[mp_dst] = np.concatenate([
                self.c_mat[dst_server], self.c_mat[dst_server] - self.nr_mat[dst_server],
                [self.deg_arr[dst_server, 1], 0.0]
            ])

            # Update edge links
            idx = unit_idx
            self.u2s_edge_links[idx] = [mpi, mp_dst]
            self.s2u_edge_links[idx] = [mp_dst, mpi]

        else:
            raise ValueError(f"Invalid action selected: unit={unit_idx}, dst={dst_server}")

        # Get next standard action (for training signal)
        self.ans_action = (self.get_standard_action()
                           if np.random.randint(0, 5) < self.data_retry_cnt
                              and self.running_mode == 'train'
                           else None)

        done = (self.x_vec == self.x_res_vec)
        truncated = (self.stepcount >= self.TIMEOUTSTEP or self.action_mask1.sum() == 0)

        self.update_action_mask()

        if done:
            reward += self.REWARD_EPISODE

        self.rewardsum += reward
        info = {}
        if done or truncated:
            self.update_procedure_level(done)
            info['final_info'] = [{
                'episode': {'r': self.rewardsum, 'l': self.stepcount},
                'unsolved_data_num': np.sum(self.data_unsolved) - len(self.INVALID_DATA),
                'data_ptr': self.data_ptr,
                'data_level': self.data_level[self.data_ptr],
                'path': self.action_seq,
            }]

        return self.state, reward, done, truncated, info
