# Relation-aware Graph Attention Network for Visual Question Answering

This repository is the implementation of [Relation-aware Graph Attention Network for Visual Question Answering](https://arxiv.org/abs/1903.12314).

![Overview of ReGAT](misc/regat_overview.jpg)

This repository is based on and inspired by @hengyuan-hu's [work](https://github.com/hengyuan-hu/bottom-up-attention-vqa) and @Jin-Hwa Kim's [work](https://github.com/jnhwkim/ban-vqa). We sincerely thank for their sharing of the codes.

## Prerequisites

1. Install [PyTorch](http://pytorch.org/)

## Training and Evaluating

```bash
python .\main_modify.py --config config/butd_vqa.json
```

## Citation

```text
@article{li2019relation,
  title={Relation-aware Graph Attention Network for Visual Question Answering},
  author={Li, Linjie and Gan, Zhe and Cheng, Yu and Liu, Jingjing},
  journal={ICCV},
  year={2019}
}
```

## License

MIT License
