import numpy as np
from torch_algorithms import logger
import os


def curriculum_callback(_locals, _globals):
    current_hard_ratio = _locals["self"].env.env_method('get_hard_ratio')[0]
    print('current hard ratio', current_hard_ratio)
    cur_update = _locals['j']
    detailed_sr = _locals["detailed_sr"]
    cur_max_blocks = int(_locals["self"].env.get_attr("cur_max_blocks")[0])
    print('cur_max_blocks', cur_max_blocks)
    num_blocks = int(_locals["self"].env.get_attr("num_blocks")[0])
    print('detailed_sr', [np.mean(item) for item in detailed_sr])
    is_adaptive_number = _locals["self"].env.get_attr("adaptive_number")[0]
    if _locals["j"] == 0:
        for n_obj in range(1, num_blocks + 1):
            _locals["self"].env.env_method('set_hard_ratio', 0.01, n_obj)
        _locals["self"].eval_env.env_method('set_success_rate', [1.0] * num_blocks)
    # Sync detailed sr
    if _locals["j"] > 0:
        _locals["self"].env.env_method('set_success_rate', [np.mean(item) for item in detailed_sr])
    # TODO: adaptively change hard ratio and max num of blocks
    if not is_adaptive_number:
        for obj_id in range(3, cur_max_blocks + 1, 2):
            if np.mean(detailed_sr[obj_id - 1]) > 0.6:
                if current_hard_ratio[obj_id - 1] < 0.9:
                    _locals["self"].env.env_method('set_hard_ratio', np.minimum(0.1 + current_hard_ratio[obj_id - 1], 1.0), obj_id)
            elif np.mean(detailed_sr[obj_id - 1]) < 0.3:
                _locals["self"].env.env_method('set_hard_ratio', np.maximum(current_hard_ratio[obj_id - 1] - 0.1, 0.01), obj_id)
    else:
        raise NotImplementedError
    # elif (is_adaptive_number and np.mean(detailed_sr[cur_max_blocks - 1]) > 0.6) \
    #         or ((not is_adaptive_number) and np.nanmean([np.mean(item) for item in detailed_sr]) > 0.6):
    #     if current_hard_ratio < 0.9:
    #         # _locals["self"].env.env_method('set_hard_ratio', np.minimum(1.5 * current_hard_ratio, 1.0))
    #         _locals["self"].env.env_method('set_hard_ratio', np.minimum(0.1 + current_hard_ratio, 1.0))
    #     else:
    #         if is_adaptive_number:
    #             _locals["self"].env.env_method("set_cur_max_objects", np.minimum(cur_max_blocks + 2, num_blocks))
    #             print('current max number of objects set to', _locals["self"].env.get_attr("cur_max_blocks")[0])
    #             if cur_max_blocks < num_blocks:
    #                 _locals["self"].env.env_method('set_hard_ratio', 0.01)
    # elif (is_adaptive_number and np.mean(detailed_sr[cur_max_blocks - 1]) < 0.3) \
    #         or ((not is_adaptive_number) and np.nanmean([np.mean(item) for item in detailed_sr]) < 0.3):
    #     # _locals["self"].env.env_method('set_hard_ratio', np.maximum(current_hard_ratio / 1.5, 0.01))
    #     _locals["self"].env.env_method('set_hard_ratio', np.maximum(current_hard_ratio - 0.1, 0.01))
    #     # if current_hard_ratio > 0.02:
    #     #     _locals["self"].env.env_method('set_hard_ratio', np.maximum(current_hard_ratio / 1.5, 0.01))
    #     # else:
    #     #     _locals["self"].env.env_method("set_cur_max_objects", max(cur_max_blocks - 1, 3))
    #     #     print('current max number of objects set to', _locals["self"].env.get_attr("cur_max_blocks")[0])
    #     #     if cur_max_blocks > 3:
    #     #         _locals["self"].env.env_method('set_hard_ratio', 1.0)
    cur_force_scale = _locals["self"].env.get_attr("cur_force_scale")[0]
    cur_stable_reward_coef = _locals["self"].env.get_attr("cur_stable_reward_coef")[0]
    print("Current force scale", cur_force_scale, "Current stable reward coef", cur_stable_reward_coef)
    if np.mean(detailed_sr[cur_max_blocks - 1]) > 0.6:
        force_scale = _locals["self"].env.get_attr("force_scale")[0]
        stable_reward_coef = _locals["self"].env.get_attr("stable_reward_coef")[0]
        _locals["self"].env.env_method('set_force_scale', np.minimum(cur_force_scale + 5, force_scale))
        _locals["self"].env.env_method('set_stable_reward_coef', np.minimum(cur_stable_reward_coef + 0.05, stable_reward_coef))

    if cur_update % 10 == 0:
        save_path = os.path.join(logger.get_dir(), "model_%d.pt" % (cur_update // 10))
        _locals["self"].save(save_path)


def always_hard_callback(_locals, _globals):
    _locals["self"].env.env_method('set_hard_ratio', 0.91)
    cur_update = _locals['j']
    if cur_update % 10 == 0:
        save_path = os.path.join(logger.get_dir(), "model_%d.pt" % (cur_update // 10))
        _locals["self"].save(save_path)
