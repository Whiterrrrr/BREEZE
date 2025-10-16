## Repository Structure
```
=======
BREEZE/
├── agents/                 # Algorithm implementations and configs (breeze, fb, cfb, cql, etc.)
│   ├── <algo>/
│   │   ├── agent.py        # Core training logic for the algorithm
│   │   ├── config.yaml     # Default hyperparameters used in the paper (edit to replicate ablations)
│   │   ├── saved_models/   # Default checkpoint directory
│   ├── base.py             # Common replay buffer utilities
│   ├── utils.py            # Agent-side helpers
│   └── workspaces.py       # Workspace classes orchestrating training loops
├── custom_dmc_tasks/       # Extensions to DM Control domains (jaco, point_mass_maze, quadruped, walker)
│   └── common/             # Shared wrappers and utilities for custom tasks
├── rewards/                # Reward shaping functions for evaluation tasks
├── main_offline.py         # Entry point for offline zero-shot RL training and evaluation
├── data_prepare.sh         # Convenience downloader for ExORL
├── exorl_reformatter.py    # Script to reshape ExORL datasets into a single NPZ
├── dmc.py                  # Environment wrappers adapted from the ExORL benchmark
├── utils.py                # Global helpers
├── requirements.txt        # Python dependencies (DM Control, Torch, hydra, tqdm, etc.)
├── LICENSE                 # MIT license
└── README.md               
```