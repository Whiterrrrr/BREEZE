
<div align="center">
<h3>
  Robust Zero-Shot Reinforcement Learning
</h3>

*The 39th Annual Conference on Neural Information Processing Systems (NeurIPS), 2025*

</div>

This is the official implementation of framework Behavior-regularized zero-shot RL with expressivity enhancement (**BREEZE**). The implementation is based on [Zero-Shot Reinforcement Learning from Low Quality Data](https://enjeeneer.io/projects/zero-shot-rl/).

## Requirements
* Python 3.8

## Installation
```
conda create --name=breeze python=3.8
conda activate breeze
pip install -r requirements.txt
```

### Getting Started
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





