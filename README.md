
# TMPI: Tiled Multiplane Images

![TMPI results on gargoyles](imgs/a.gif) ![TMPI results on monastery](imgs/b.gif) ![TMPI results on Paris](imgs/c.gif)
        

PyTorch implementation of the ICCV 2023 paper.

[Tiled Multiplane Images for Practical 3D Photography](https://arxiv.org/abs/2309.14291)<br>
[Numair Khan](https://nkhan2.github.io)<sup>1</sup>, Douglas Lanman<sup>1</sup>, [Lei Xiao](https://leixiao-ubc.github.io/)<sup>1</sup><br>
<sup>1</sup>Reality Labs Research<br>

## Setup
The code has been tested in the following setup
* Linux (Ubuntu 20.04.04/Fedora 34)
* Python 3.7
* PyTorch 1.10.2
* CUDA 11.3

We recommend running the code in a Conda environment: after cloning the repo, run the following commands from the base directory:
```
$ conda env create -f environment.yml
$ conda activate tmpi
```

Run the initialization script to download model checkpoints and set up the monocular depth estimator; we use [DPT](https://github.com/isl-org/DPT):

```
$ sh ./initialize.sh
```

## Running the Code
To execute 3D photos from the test images provided in the `test_data` folder run:

```
$ python ./run.py
```

The results are written to the `output` folder.


To execute the code on your own images use:

```
$ python ./run.py --indir=/PATH_TO_INPUT_IMAGES_DIR --outdir=/PATH_TO_OUTPUT_DIR
```

By default, the code uses OpenGL to render the tiled multiplane images efficiently. We notice that on some implementations this may cause flickering around a small number of tile edges. To avoid this (or in case your system does not have OpenGL installed) the differentiable PyTorch renderer we use for training may be utilized by providing the `--pytorch_renderer` flag. 

```
$ python ./run.py --pytorch_renderer --indir=/PATH_TO_INPUT_IMAGES_DIR --outdir=/PATH_TO_OUTPUT_DIR
```
Note, however, that this will run much slower.

While we use DPT as the depth estimator, the method can work with any depth input. To use a different source, we suggest replacing the DPT depth loader on L87 of the `dataset.py` file with the desired (inverse) depth input.


## Citation
If you find our work useful for your research, please cite the following paper:

```
@article{khan2023tcod,
  title={Tiled Multiplane Images for Practical 3D Photography},
  author={Numair Khan, Eric Penner, Douglas Lanman, Lei Xiao},
  journal={International Conference on Computer Vision (ICCV)},
  year={2023},
}
```

## License
Our source code is CC-BY-NC licensed, as found in the LICENSE file.

