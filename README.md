<div align="center">
  <h1>Behavior-regularized zero-shot RL with Expressivity Enhancement (BREEZE)</h1>
  <h3><em>Towards Robust Zero-shot Reinforcement Learning (NeurIPS 2025)</em></h3>
  
[Kexin Zheng](https://air-dream.netlify.app/author/kexin-zheng/)\*, [Lauriane Teyssier](https://arwen-c.github.io/)\*, [Yinan Zheng](https://github.com/ZhengYinan-AIR), Yu Luo, [Xianyuan Zhan](https://zhanzxy5.github.io/zhanxianyuan/)

</div>

## Table of Contents
- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Running Experiments](#running-experiments)
- [Acknowledgements](#acknowledgements)

## Overview
**BREEZE** is an upgraded FB-based framework that simultaneously enhances offline learning stability, policy extraction capability, and representation learning quality.

- BREEZE introduces behavioral regularization in zero-shot RL policy learning, transforming policy optimization into a stable in-sample learning paradigm.
- BREEZE extracts the policy using a task-conditioned diffusion model, enabling the generation of high-quality and multimodal action distributions in zero-shot RL settings.
- BREEZE employs expressive attention-based architectures for representation modeling to capture the complex relationships between environmental dynamics.

### Performance Showcase
BREEZE achieves the best or near-best returns with faster convergence and enhanced stability. 
<div align="center">
<image src="img/curves_quadruped_rnd.png" width=100%>
</div>
BREEZE within 400k steps can match or exceed baselines trained for 1M steps.
<div align="center">
<image src="img/performance.png" width=80%>
</div>

## Environment Setup
### Requirements
* Python 3.10

```bash
conda create -n breeze python=3.10
conda activate breeze
pip install -r requirements.txt
```

### Additional Dependencies
- The DM Control suite requires [Mujoco](https://mujoco.org/).
- Weights & Biases logging is enabled by default; set `WANDB_API_KEY` before launching experiments or pass `--wandb_logging False`.

### Data Preparation
All experiments rely on offline datasets released with ExORL. This repository includes scripts to download and reformat those datasets.

**ExORL Download & Reformat**
```bash
bash data_prepare.sh
```

### Repository Structure
We provide the repository structure in [repository_structure.md](docs/repository_structure.md).

## Running Experiments
The main entry point is `main_offline.py`, which takes the algorithm name, domain, and exploration policy that generated the dataset. Key flags:

```
usage: main_offline.py <algorithm> <domain_name> <exploration_algorithm> \
                       --eval_tasks TASK [TASK ...] [--train_task TASK]
                       [--seed INT] [--learning_steps INT]
                       [--z_inference_steps INT]
                       [--wandb_logging {True,False}]
```

- `algorithm`: one of `breeze`, `fb`, `cfb`, `vcfb`, `mcfb`, `cql`, `sac`, `td3`, `sf-lap`, `sf-hilp`.
- `domain_name`: DMC domain (`walker`, `quadruped`, `jaco`, `point_mass_maze`, ...).
- `exploration_algorithm`: dataset source tag (`proto`, `rnd`, `aps`, etc.).
- `--eval_tasks`: list of downstream tasks for zero-shot evaluation.

### Examples
```bash
# BREEZE on Quadruped with RND exploration data
python main_offline.py breeze quadruped rnd \
  --eval_tasks stand run walk jump \
  --seed 42 --learning_steps 1000000
```

Configuration defaults (network sizes, optimizers, diffusion settings, etc.) are stored in `agents/<algo>/config.yaml`. Override any value via CLI flags or by editing the YAML.

### Reproducing the Paper
We provide the domain-specific hyperparameters used in our experiments in [domain_specific_hyp.md](docs/domain_specific_hyp.md) to reproduce our result.


## Acknowledgements
We thank all the contributions of prior studies:
- This implementation is based on the [Zero-Shot Reinforcement Learning from Low Quality Data](https://enjeeneer.io/projects/zero-shot-rl/) codebase.

- The implementation of Diffusion model is based on [IDQL](https://arxiv.org/pdf/2304.10573)

## Citation
If you find this repository helpful, please cite:

```bibtex
@inproceedings{zheng2025towards,
  title={Towards Robust Zero-Shot Reinforcement Learning},
  author={Kexin Zheng and Lauriane Teyssier and Yinan Zheng and Yu Luo and Xianyuan Zhan},
  booktitle={NeurIPS},
  year={2025}
}
```

## License
This project is licensed under the MIT License. See `LICENSE` for the full text.
