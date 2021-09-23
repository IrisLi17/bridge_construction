import numpy as np
import heapq


class HashTable(object):
    def __init__(self):
        self._hash_dict = {}

    def _hashfunc(self, x):
        assert isinstance(x, dict)
        skyline = x['other_state']['skyline']
        cliffs = np.asarray([x['other_state']['cliff1_pos'][1] - x['other_state']['cliff0_pos'][1]])
        n_object = x['other_state']['cur_num_blocks']
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

    def extend(self, state, next_obs, obs, reward, time, next_rnn_hxs=None, rnn_hxs=None, next_rnn_mask=None,
               rnn_mask=None, priority=None, step_to_go=None, actions=None, reset_actions=None):
        assert self._is_heap
        if rnn_hxs is None:
            next_rnn_hxs = [None] * len(state)
            rnn_hxs = [None] * len(state)
            next_rnn_mask = [None] * len(state)
            rnn_mask = [None] * len(state)
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

    def sample(self, batch_size, uniform=False):
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
