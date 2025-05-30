# Open-Loop Reinforcement Learning

This is the official repository to the paper "A Pontryagin Perspective on Reinforcement Learning" by Onno Eberhard, Claire Vernade, and Michael Muehlebach (published at L4DC 2025).
All algorithms discussed in the paper are included here.

## Installation
```
pip install git+https://github.com/onnoeberhard/pontryagin
```

## Examples
Once installed, you can run the examples in the `examples` folder, e.g.
```
python examples/pendulum.py
``` 
On first run it might take a few minutes. If it hangs after the plot shows, try to change the file to have `plot=False` in the function call.

## Citation
If you use this code in your research, please cite our paper:
```bibtex
@inproceedings{eberhard-2025-pontryagin,
  title = {A Pontryagin Perspective on Reinforcement Learning},
  author = {Eberhard, Onno and Vernade, Claire and Muehlebach, Michael},
  booktitle = {Proceedings of the Sixth Annual Learning for Dynamics \& Control Conference},
  year = {2025},
  series = {Proceedings of Machine Learning Research},
  volume = {283},
  url = {https://arxiv.org/abs/2405.18100}
}
```

If there are any problems, or if you have a question, don't hesitate to open an issue here on GitHub.
