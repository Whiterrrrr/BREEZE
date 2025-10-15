"""module for retrieving stand reward function from dm_control suite quadruped"""

from custom_dmc_tasks.quadruped import walk

reward_function = walk()._task.get_reward  # pylint: disable=protected-access
