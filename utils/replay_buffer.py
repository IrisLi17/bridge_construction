import random
import numpy as np
import torch
from collections import deque
import heapq


def unique(sorted_array):
    """
    More efficient implementation of np.unique for sorted arrays
    :param sorted_array: (np.ndarray)
    :return:(np.ndarray) sorted_array without duplicate elements
    """
    if len(sorted_array) == 1:
        return sorted_array
    left = sorted_array[:-1]
    right = sorted_array[1:]
    uniques = np.append(right != left, True)
    return sorted_array[uniques]


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """
        Build a Segment Tree data structure.

        https://en.wikipedia.org/wiki/Segment_tree

        Can be used as regular array that supports Index arrays, but with two
        important differences:

            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.

        :param capacity: (int) Total size of the array - must be a power of two.
        :param operation: (lambda (Any, Any): Any) operation for combining elements (eg. sum, max) must form a
            mathematical group together with the set of possible values for array elements (i.e. be associative)
        :param neutral_element: (Any) neutral element for the operation above. eg. float('-inf') for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation
        self.neutral_element = neutral_element

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """
        Returns result of applying `self.operation`
        to a contiguous subsequence of the array.

            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))

        :param start: (int) beginning of the subsequence
        :param end: (int) end of the subsequences
        :return: (Any) result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # indexes of the leaf
        idxs = idx + self._capacity
        self._value[idxs] = val
        if isinstance(idxs, int):
            idxs = np.array([idxs])
        # go up one level in the tree and remove duplicate indexes
        idxs = unique(idxs // 2)
        while len(idxs) > 1 or idxs[0] > 0:
            # as long as there are non-zero indexes, update the corresponding values
            self._value[idxs] = self._operation(
                self._value[2 * idxs],
                self._value[2 * idxs + 1]
            )
            # go up one level in the tree and remove duplicate indexes
            idxs = unique(idxs // 2)

    def __getitem__(self, idx):
        assert np.max(idx) < self._capacity
        assert 0 <= np.min(idx)
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=np.add,
            neutral_element=0.0
        )
        self._value = np.array(self._value)

    def sum(self, start=0, end=None):
        """
        Returns arr[start] + ... + arr[end]

        :param start: (int) start position of the reduction (must be >= 0)
        :param end: (int) end position of the reduction (must be < len(arr), can be None for len(arr) - 1)
        :return: (Any) reduction of SumSegmentTree
        """
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """
        Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum for each entry in prefixsum

        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.

        :param prefixsum: (np.ndarray) float upper bounds on the sum of array prefix
        :return: (np.ndarray) highest indexes satisfying the prefixsum constraint
        """
        if isinstance(prefixsum, float):
            prefixsum = np.array([prefixsum])
        assert 0 <= np.min(prefixsum)
        assert np.max(prefixsum) <= self.sum() + 1e-5
        assert isinstance(prefixsum[0], float)

        idx = np.ones(len(prefixsum), dtype=int)
        cont = np.ones(len(prefixsum), dtype=bool)

        while np.any(cont):  # while not all nodes are leafs
            idx[cont] = 2 * idx[cont]
            prefixsum_new = np.where(self._value[idx] <= prefixsum, prefixsum - self._value[idx], prefixsum)
            # prepare update of prefixsum for all right children
            idx = np.where(np.logical_or(self._value[idx] > prefixsum, np.logical_not(cont)), idx, idx + 1)
            # Select child node for non-leaf nodes
            prefixsum = prefixsum_new
            # update prefixsum
            cont = idx < self._capacity
            # collect leafs
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=np.minimum,
            neutral_element=float('inf')
        )
        self._value = np.array(self._value)

    def min(self, start=0, end=None):
        """
        Returns min(arr[start], ...,  arr[end])

        :param start: (int) start position of the reduction (must be >= 0)
        :param end: (int) end position of the reduction (must be < len(arr), can be None for len(arr) - 1)
        :return: (Any) reduction of MinSegmentTree
        """
        return super(MinSegmentTree, self).reduce(start, end)


class HashTable(object):
    def __init__(self):
        self._hash_dict = {}

    def _hashfunc(self, x):
        assert isinstance(x, dict)
        # mj_state = x['mj_state']
        # mj_qpos = mj_state.qpos
        # other_state = x['other_state']
        # cliff_pos = np.concatenate([other_state['cliff0_pos'], other_state['cliff1_pos']])
        # hashkey = tuple((mj_qpos * 100).astype(np.int)) + tuple((cliff_pos * 100).astype(np.int))
        # skyline = obs[-15:]  # TODO: Issue: skyline is not always available, put it into state maybe
        skyline = x['other_state']['skyline']
        cliffs = np.asarray([x['other_state']['cliff1_pos'][1] - x['other_state']['cliff0_pos'][1]])
        n_object = x['other_state']['cur_num_blocks']
        # object_size = [x['other_state']['object%d' % i][1] if i < n_object else 0 for i in range(7)]
        resolution = x['other_state']['cliff0_size'][1]
        hashkey = tuple(np.round(skyline / resolution).astype(np.int)) + tuple(np.round(cliffs / resolution * 2).astype(np.int)) + (n_object,)
        return hashkey

    def exist(self, x):
        hashkey = self._hashfunc(x)
        return hashkey in self._hash_dict

    def insert(self, x, entry):
        hashkey = self._hashfunc(x)
        if hashkey not in self._hash_dict:
            self._hash_dict[hashkey] = entry

    def delete(self, x):
        hashkey = self._hashfunc(x)
        # The key must be in the dict
        # del self._hash_dict[hashkey]
        return self._hash_dict.pop(hashkey)

    def update_step_to_go(self, x, step_to_go):
        hashkey = self._hashfunc(x)
        assert hashkey in self._hash_dict
        self._hash_dict[hashkey][0] = min(self._hash_dict[hashkey][0], step_to_go)

    def update_history(self, x, history):
        hashkey = self._hashfunc(x)
        assert hashkey in self._hash_dict
        if history is not None and len(history) < len(self._hash_dict[hashkey][1]):
            self._hash_dict[hashkey][1] = history

    def get(self, x):
        return self._hash_dict[self._hashfunc(x)]


class ReplayBuffer(object):
    def __init__(self, size: int, unique=False):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        # TODO: add hash table
        self.unique = unique
        self.hash_table = HashTable() if self.unique else None

    def __len__(self) -> int:
        return len(self._storage)

    @property
    def storage(self):
        """[(Union[np.ndarray, int], Union[np.ndarray, int], float, Union[np.ndarray, int], bool)]: content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self) -> int:
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples: int) -> bool:
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self) -> int:
        """
        Check whether the replay buffer is full or not.

        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, state: dict, next_obs: torch.Tensor, obs: torch.Tensor, reward: torch.Tensor,
            next_rnn_hxs=None, rnn_hxs=None, next_rnn_mask=None, rnn_mask=None):
        """
        add a new state to the buffer
        """

        data = (state, next_obs, obs, reward, next_rnn_hxs, rnn_hxs, next_rnn_mask, rnn_mask)
        if (not self.unique) or (not self.hash_table.exist(state)):
            if self._next_idx >= len(self._storage):
                if self.hash_table is not None:
                    self.hash_table.insert(state)
                self._storage.append(data)
            else:
                if self.hash_table is not None:
                    self.hash_table.delete(self._storage[self._next_idx][0])
                    self.hash_table.insert(state)
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def extend(self, state, next_obs, obs, reward, next_rnn_hxs=None, rnn_hxs=None, next_rnn_mask=None, rnn_mask=None):
        """
        add a new batch of transitions to the buffer

        Note: uses the same names as .add to keep compatibility with named argument passing
                but expects iterables and arrays with more than 1 dimensions
        """
        stored_idxes = []
        data_idx = 0
        if rnn_hxs is None:
            next_rnn_hxs = [None] * len(state)
            rnn_hxs = [None] * len(state)
            next_rnn_mask = [None] * len(state)
            rnn_mask = [None] * len(state)
        for data in zip(state, next_obs, obs, reward, next_rnn_hxs, rnn_hxs, next_rnn_mask, rnn_mask):
            if not self.unique or not self.hash_table.exist(data[0]):
                stored_idxes.append(data_idx)
                if self._next_idx >= len(self._storage):
                    if self.hash_table is not None:
                        self.hash_table.insert(data[0])
                    self._storage.append(data)
                else:
                    if self.hash_table is not None:
                        self.hash_table.delete(self._storage[self._next_idx][0])
                        self.hash_table.insert(data[0])
                    self._storage[self._next_idx] = data
                self._next_idx = (self._next_idx + 1) % self._maxsize
            data_idx += 1
        return stored_idxes

    def _encode_sample(self, idxes):
        state_buf, next_obs_buf, obs_buf, reward_buf, next_rnn_hxs_buf, rnn_hxs_buf, next_rnn_mask_buf, rnn_mask_buf = [], [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            state, next_obs, obs, reward, next_rnn_hxs, rnn_hxs, next_rnn_mask, rnn_mask = data
            state_buf.append(state)
            next_obs_buf.append(next_obs)
            obs_buf.append(obs)
            reward_buf.append(reward)
            next_rnn_hxs_buf.append(next_rnn_hxs)
            rnn_hxs_buf.append(rnn_hxs)
            next_rnn_mask_buf.append(next_rnn_mask)
            rnn_mask_buf.append(rnn_mask)
        next_obs_buf = torch.stack(next_obs_buf)
        obs_buf = torch.stack(obs_buf)
        reward_buf = torch.stack(reward_buf)
        if isinstance(next_rnn_hxs_buf[0], torch.Tensor):
            next_rnn_hxs_buf = torch.stack(next_rnn_hxs_buf)
            rnn_hxs_buf = torch.stack(rnn_hxs_buf)
            next_rnn_mask_buf = torch.stack(next_rnn_mask_buf)
            rnn_mask_buf = torch.stack(rnn_mask_buf)
        return state_buf, next_obs_buf, obs_buf, reward_buf, next_rnn_hxs_buf, rnn_hxs_buf, next_rnn_mask_buf, rnn_mask_buf

    def sample(self, batch_size: int):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :return:
            - state_batch: (list) batch of states
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, unique=False):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBuffer, self).__init__(size, unique)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, state, next_obs, obs, reward, next_rnn_hxs=None, rnn_hxs=None, next_rnn_mask=None, rnn_mask=None,
            priority=None):
        """
        add a new transition to the buffer
        """
        # Shortcut break if no need to insert
        if self.unique and self.hash_table.exist(state):
            return
        idx = self._next_idx
        super().add(state, next_obs, obs, reward, next_rnn_hxs, rnn_hxs, next_rnn_mask, rnn_mask)
        if priority is not None:
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)
        else:
            self._it_sum[idx] = self._max_priority ** self._alpha
            self._it_min[idx] = self._max_priority ** self._alpha

    def extend(self, state, next_obs, obs, reward, next_rnn_hxs=None, rnn_hxs=None, next_rnn_mask=None, rnn_mask=None,
               priority=None):
        """
        add a new batch of transitions to the buffer

        Note: uses the same names as .add to keep compatibility with named argument passing
            but expects iterables and arrays with more than 1 dimensions
        """
        idx = self._next_idx
        stored_idxes = super().extend(state, next_obs, obs, reward, next_rnn_hxs, rnn_hxs, next_rnn_mask, rnn_mask)
        if priority is not None:
            for stored_idx in stored_idxes:
                self._it_sum[idx] = priority[stored_idx] ** self._alpha
                self._it_min[idx] = priority[stored_idx] ** self._alpha
                idx = (idx + 1) % self._maxsize
                self._max_priority = max(self._max_priority, priority[stored_idx])
            assert idx == self._next_idx
            # _count = 0
            # while idx != self._next_idx:
            #     self._it_sum[idx] = priority[_count] ** self._alpha
            #     self._it_min[idx] = priority[_count] ** self._alpha
            #     idx = (idx + 1) % self._maxsize
            #     _count += 1
            # self._max_priority = max(self._max_priority, np.max(priority))
        else:
            while idx != self._next_idx:
                self._it_sum[idx] = self._max_priority ** self._alpha
                self._it_min[idx] = self._max_priority ** self._alpha
                idx = (idx + 1) % self._maxsize

    def _sample_proportional(self, batch_size):
        total = self._it_sum.sum(0, len(self._storage) - 1)
        # TODO(szymon): should we ensure no repeats?
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    def sample(self, batch_size: int, beta: float = 0):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :return:
            - state_batch: (list) batch of states
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)
        p_sample = self._it_sum[idxes] / self._it_sum.sum()
        weights = (p_sample * len(self._storage)) ** (-beta) / max_weight
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self.storage)
        self._it_sum[idxes] = priorities ** self._alpha
        self._it_min[idxes] = priorities ** self._alpha

        self._max_priority = max(self._max_priority, np.max(priorities))

'''
class PriorityQueue(object):
    def __init__(self, size, unique=False):
        self.storage = []
        self._maxsize = size
        self.unique = unique
        self.hash_table = HashTable() if unique else None
        self._is_sorted = True
        # self._next_idx = 0
        self._index = 0

    def __len__(self):
        return len(self.storage)

    def can_sample(self, n_samples: int) -> bool:
        return len(self) >= n_samples

    # def add(self, state: dict, next_obs: torch.Tensor, obs: torch.Tensor, reward: torch.Tensor,
    #         next_rnn_hxs=None, rnn_hxs=None, next_rnn_mask=None, rnn_mask=None):
    #     data = (state, next_obs, obs, reward, next_rnn_hxs, rnn_hxs, next_rnn_mask, rnn_mask)
    #     if (not self.unique) or (not self.hash_table.exist(state)):
    #         if len(self.storage) < self._maxsize:
    #             heapq.heappush(self.storage, )

    def extend(self, state, next_obs, obs, reward, time, next_rnn_hxs=None, rnn_hxs=None, next_rnn_mask=None, rnn_mask=None,
               priority=None, step_to_go=None, history=None):
        if rnn_hxs is None:
            next_rnn_hxs = [None] * len(state)
            rnn_hxs = [None] * len(state)
            next_rnn_mask = [None] * len(state)
            rnn_mask = [None] * len(state)
        if history is None:
            history = [None] * len(state)
        for idx, data in enumerate(zip(state, next_obs, obs, reward, next_rnn_hxs, rnn_hxs, next_rnn_mask, rnn_mask, step_to_go)):
            if not self.unique or not self.hash_table.exist(data[0]):
                # Rewrite in a ring buffer manner
                if len(self.storage) < self._maxsize:
                    # self.storage.append([priority[idx], data, time[idx]])
                    heapq.heappush(self.storage, (priority[idx], self._index, data, time[idx]))
                    self._index += 1
                    if self.hash_table is not None:
                        self.hash_table.insert(data[0], step_to_go[idx], history[idx])
                else:
                    # TODO: we need to apply some transformation on step_to_go as priority. Let's absorb step_to_go into priority
                    # if self.hash_table is not None:
                    #     self.hash_table.delete(self.storage[self._next_idx][1][0])
                    # self.storage[self._next_idx] = [priority[idx], data, time[idx]]
                    old_item = heapq.heappushpop(self.storage, (priority[idx], self._index, data, time[idx]))
                    if self.hash_table is not None and self.hash_table.exist(old_item[2][0]):  # Not necessarily delete
                        self.hash_table.delete(old_item[2][0])
                        self.hash_table.insert(data[0], step_to_go[idx])
                        self._index += 1
                # self._next_idx = (self._next_idx + 1) % self._maxsize
                # while len(self.storage) >= self._maxsize:
                #     old = self.storage.pop(0)
                #     if self.hash_table is not None:
                #         self.hash_table.delete(old[1][0])
                # self.storage.append([priority[idx], data])
                # TODO: bug, sometimes the new item is not inserted to queue
                if self.hash_table is not None:
                    # self.hash_table.insert(data[0])
                    assert len(self.hash_table._hash_dict) == len(self.storage)
            else:
                self.hash_table.update_step_to_go(data[0], step_to_go[idx])
                self.hash_table.update_history(data[0], history[idx])
        # self._is_sorted = False

    def _encode_sample(self, idxes):
        state_buf, next_obs_buf, obs_buf, reward_buf, next_rnn_hxs_buf, rnn_hxs_buf, next_rnn_mask_buf, \
            rnn_mask_buf, priority_buf, time_buf = [], [], [], [], [], [], [], [], [], []
        for idx in idxes:
            priority = self.storage[idx][0]
            priority_buf.append(priority)
            time = self.storage[idx][3]
            time_buf.append(time)
            state, next_obs, obs, reward, next_rnn_hxs, rnn_hxs, next_rnn_mask, rnn_mask, step_to_go = self.storage[idx][2]
            state_buf.append(state)
            next_obs_buf.append(next_obs)
            obs_buf.append(obs)
            reward_buf.append(reward)
            next_rnn_hxs_buf.append(next_rnn_hxs)
            rnn_hxs_buf.append(rnn_hxs)
            next_rnn_mask_buf.append(next_rnn_mask)
            rnn_mask_buf.append(rnn_mask)
        next_obs_buf = torch.stack(next_obs_buf)
        obs_buf = torch.stack(obs_buf)
        reward_buf = torch.stack(reward_buf)
        if isinstance(next_rnn_hxs_buf[0], torch.Tensor):
            next_rnn_hxs_buf = torch.stack(next_rnn_hxs_buf)
            rnn_hxs_buf = torch.stack(rnn_hxs_buf)
            next_rnn_mask_buf = torch.stack(next_rnn_mask_buf)
            rnn_mask_buf = torch.stack(rnn_mask_buf)
        return state_buf, next_obs_buf, obs_buf, reward_buf, next_rnn_hxs_buf, rnn_hxs_buf, next_rnn_mask_buf, \
               rnn_mask_buf, priority_buf, time_buf

    def sample(self, batch_size, beta=None, uniform=False):
        # Beta is just for compatibility
        if uniform:
            idxes = np.random.choice(np.arange(len(self.storage)), size=batch_size)
        else:
            # if not self._is_sorted:
            #     # Priority increasing order, so that low priority samples will be flushed out first
            #     # self.storage = sorted(self.storage, key=lambda v: v[0])
            #     self.storage.sort(key=lambda v: v[0])
            #     self._is_sorted = True
            #     # Reset next_idx if needed
            #     if len(self.storage) == self._maxsize:
            #         self._next_idx = 0
            self.storage.sort(key=lambda v: v[0])  # Remains a heap
            # idxes = np.arange(len(self.storage) - batch_size, len(self.storage))
            idxes = np.arange(len(self.storage) - 1, len(self.storage) - batch_size - 1, -1)
        encoded_samples = self._encode_sample(idxes)

        return tuple(list(encoded_samples) + [idxes])

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for i, data_id in enumerate(idxes):
            item = list(self.storage[data_id])
            item[0] = priorities[i]
            self.storage[data_id] = tuple(item)
        # self._is_sorted = False

    def force_sort(self):
        self.storage.sort(key=lambda v: v[0])
        # self._is_sorted = True

    def sync_step_to_go(self):
        assert self.hash_table is not None
        for idx in range(len(self.storage)):
            item = list(self.storage[idx])
            data = list(item[2])
            data[-1] = min(self.hash_table.get(data[0])[0], data[-1])
            data = tuple(data)
            self.storage[idx] = tuple([item[0], item[1], data, *item[3:]])

    def get_history(self, initial_state):
        assert self.hash_table is not None
        return self.hash_table.get(initial_state)[1]
'''


def ewma(new_priority, old_priority, n_encounter, decay=0.9):
    # n_encounter: correspond to old_priority
    old_total_weight = (1 - decay ** n_encounter) / (1 - decay)
    new_total_weight = decay * old_total_weight + 1
    old_ratio = decay * old_total_weight / new_total_weight
    new_ratio = 1 / new_total_weight
    return old_ratio * old_priority + new_ratio * new_priority


class PriorityQueue(object):
    REMOVED = '<removed>'

    def __init__(self, size, decay=0.9):
        self.storage = []
        self.decay = decay  # For updating priority
        self._maxsize = size
        self.hash_table = HashTable()
        self._is_heap = True
        self._index = 0
        self._n_valid = 0

    def __len__(self):
        return self._n_valid

    def can_sample(self, n_samples: int) -> bool:
        return len(self) >= n_samples

    def _add(self, state, priority, data, n_encounter=1):
        '''
        Add or update item
        :param state:
        :param priority:
        :param data: listu
        :return:
        '''
        if self.hash_table.exist(state):
            removed_entry = self._remove(state)
            old_priority = removed_entry[0]
            old_encounter = removed_entry[3]
            n_encounter = old_encounter + 1
            # print("old priority", old_priority, "new_priority", priority)
            priority = ewma(priority, old_priority, old_encounter, self.decay)
            # print("ewma priority", priority)
            # Can we instead update a running average of priority?
        self._index += 1
        entry = [priority, self._index, data, n_encounter, state]  # the last item is state or token for removed
        self.hash_table.insert(state, entry)  # entry should be shared by hash_table and storage
        heapq.heappush(self.storage, entry)
        self._n_valid += 1
        # Problem with maxsize. Need to track n_valid
        assert self._n_valid == len(self.hash_table._hash_dict)

        if self._n_valid > self._maxsize:
            self._pop_least()

    def _remove(self, state):
        '''
        Mark entry as removed since actual removing causes difficulty in remaining heap property
        :param state:
        :return:
        '''
        entry = self.hash_table.delete(state)  # the entry in hash_table is removed
        # DO NOT change priority
        # entry[2] = None
        entry[-1] = self.REMOVED  # The entry in storage should be marked as non-exist
        self._n_valid -= 1
        assert self._n_valid == len(self.hash_table._hash_dict)
        return entry

    def _pop_least(self):
        while len(self.storage):
            entry = heapq.heappop(self.storage)
            if entry[-1] is not self.REMOVED:
                self._n_valid -= 1
                self.hash_table.delete(entry[-1])
                assert self._n_valid == len(self.hash_table._hash_dict)
                return entry[-1]
        raise KeyError("Pop from an empty storage")

    def extend(self, state, next_obs, obs, reward, time, next_rnn_hxs=None, rnn_hxs=None, next_rnn_mask=None, rnn_mask=None,
               priority=None, step_to_go=None, actions=None, reset_actions=None, history=None):
        assert self._is_heap
        if rnn_hxs is None:
            next_rnn_hxs = [None] * len(state)
            rnn_hxs = [None] * len(state)
            next_rnn_mask = [None] * len(state)
            rnn_mask = [None] * len(state)
        if history is None:
            history = [None] * len(state)
        for idx, data in enumerate(zip(next_obs, obs, reward, next_rnn_hxs, rnn_hxs, next_rnn_mask, rnn_mask,
                                       step_to_go, actions, reset_actions)):
            self._add(state[idx], priority[idx], list(data) + [time[idx]])

    def _encode_sample(self, idxes):
        state_buf, next_obs_buf, obs_buf, reward_buf, next_rnn_hxs_buf, rnn_hxs_buf, next_rnn_mask_buf, \
        rnn_mask_buf, priority_buf, n_encounter_buf, time_buf = [], [], [], [], [], [], [], [], [], [], []
        action_buf, reset_action_buf = [], []
        for idx in idxes:
            priority = self.storage[idx][0]
            state = self.storage[idx][-1]
            priority_buf.append(priority)
            state_buf.append(state)
            next_obs, obs, reward, next_rnn_hxs, rnn_hxs, next_rnn_mask, rnn_mask, step_to_go, action, reset_action, time = \
                self.storage[idx][2]
            n_encounter_buf.append(self.storage[idx][3])
            next_obs_buf.append(next_obs)
            obs_buf.append(obs)
            reward_buf.append(reward)
            next_rnn_hxs_buf.append(next_rnn_hxs)
            rnn_hxs_buf.append(rnn_hxs)
            next_rnn_mask_buf.append(next_rnn_mask)
            rnn_mask_buf.append(rnn_mask)
            action_buf.append(action)
            reset_action_buf.append(reset_action)
            time_buf.append(time)
        next_obs_buf = np.stack(next_obs_buf)
        obs_buf = np.stack(obs_buf)
        reward_buf = np.stack(reward_buf)
        if isinstance(next_rnn_hxs_buf[0], np.ndarray):
            next_rnn_hxs_buf = np.stack(next_rnn_hxs_buf)
            rnn_hxs_buf = np.stack(rnn_hxs_buf)
            next_rnn_mask_buf = np.stack(next_rnn_mask_buf)
            rnn_mask_buf = np.stack(rnn_mask_buf)
        return state_buf, next_obs_buf, obs_buf, reward_buf, next_rnn_hxs_buf, rnn_hxs_buf, next_rnn_mask_buf, \
               rnn_mask_buf, action_buf, reset_action_buf, priority_buf, n_encounter_buf, time_buf

    def sample(self, batch_size, beta=None, uniform=False):
        # Beta is just for compatibility
        print("Before cleaning", len(self.storage))
        for idx in reversed(range(len(self.storage))):  # Reversed is important, otherwise the idx is incorrect
            if self.storage[idx][-1] == self.REMOVED:
                self.storage.pop(idx)
        assert self._n_valid == len(self.storage)
        if uniform:
            idxes = np.random.choice(np.arange(len(self.storage)), size=batch_size)
        else:
            self.storage.sort(key=lambda v: v[0])  # Remains a heap
            idxes = np.arange(len(self.storage) - 1, len(self.storage) - batch_size - 1, -1)
        encoded_samples = self._encode_sample(idxes)

        return tuple(list(encoded_samples) + [idxes])

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for i, data_id in enumerate(idxes):
            item = self.storage[data_id]
            item[0] = priorities[i]
        # Force sort must be called after all the priority update
        self._is_heap = False

    def force_sort(self):
        self.storage.sort(key=lambda v: v[0])
        self._is_heap = True
