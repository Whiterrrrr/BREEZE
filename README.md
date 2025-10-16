
<div align="center">
<h3>
  Robust Zero-Shot Reinforcement Learning
</h3>

*The 39th Annual Conference on Neural Information Processing Systems (NeurIPS), 2025*

</div>

This is the official implementation of framework Behavior-regularized zero-shot RL with expressivity enhancement (**BREEZE**). The implementation is based on [Zero-Shot Reinforcement Learning from Low Quality Data](https://enjeeneer.io/projects/zero-shot-rl/).

## Requirements
* Python 3.10

## Installation
```
conda create --name=breeze python=3.10
conda activate breeze
pip install -r requirements.txt
```

## Getting Started
### ExORL Data prepare
- Datasets Overview

| Data collection algorithm 	| Usage 	|
|:---:	|:---:	|
| [master](https://github.com/ZhengYinan-AIR/FISOR) 	| FISOR implementation for ``Point Robot``, ``Safety-Gymnasium`` and ``Bullet-Safety-Gym``; data quantity experiment; feasible region visualization. |
| [metadrive_imitation](https://github.com/ZhengYinan-AIR/FISOR/tree/metadrive_imitation) 	| FISOR implementation for ``MetaDrive``; data quantity experiment; imitation learning experiment. 	|

Download the dataset and reformat it:
```bash
bash data_prepare.sh
```



### ExORL Data prepare


```bash
python main_offline.py breeze walker rnd --eval_task stand run walk flip
```

### Bibtex

If you find our code and paper can help, please cite our paper as:

```commandline
@inproceedings{
zheng2025towards,
title={Towards Robust Zero-Shot Reinforcement Learning},
author={Kexin Zheng, Lauriane Teyssier, Yinan Zheng, Yu Luo, Xianyuan Zhan},
booktitle={In the Thirty-Ninth Conference on Neural Information Processing Systems (NeurIPS 2025)},
year={2025},
}
```

## License
MIT





