# Direct2.5: Diverse Text-to-3D Generation via Multi-view 2.5D Diffusion
Yuanxun Lu<sup>1</sup>, Jingyang Zhang<sup>2</sup>, Shiwei Li<sup>2</sup>, Tian Fang<sup>2</sup>, David McKinnon<sup>2</sup>, Yanghai Tsin<sup>2</sup>, Long Quan<sup>3</sup>, Xun Cao<sup>1</sup>, Yao Yao<sup>1*</sup>  
<sup>1</sup>Nanjing University, <sup>2</sup>Apple, <sup>3</sup>HKUST

### [Project Page](https://nju-3dv.github.io/projects/direct25/) | [Paper](https://arxiv.org/abs/2311.15980) | [Weights](#pre-trained-models) 
This is the official implementation of Direct2.5, a text-to-3D generation system that can produce diverse, Janus-free, and high-fidelity 3D content in only __10 seconds__.

<p align="center">
  <img width="70%" src="https://nju-3dv.github.io/projects/direct25/resources/pipeline.png"/>
</p>


## Environment Setup

### Installation

- This project is successfully tested on Ubuntu 20.04 with PyTorch 2.1 (Python 3.10). We recommend creating a new environment and install necessary dependencies:

  ```
  conda create -y -n py310 python=3.10
  conda activate py310
  pip install torch==2.1.0 torchvision
  pip install -r requirements.txt
  ```

* Install `torch-scatter` and `pytorch3d` according to your cuda versions. For example, you could run this if you have `cu118`:

  ```
  # torch-scatter
  pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
  # pytorch3d
  conda install -c fvcore -c iopath -c conda-forge fvcore iopath
  conda install -c bottler nvidiacub
  pip install fvcore iopath
  pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html
  ```

* Install `nvdiffrast`following the official [instruction](https://nvlabs.github.io/nvdiffrast/#installation)

* (Optional) Install `Real-ESRGAN` if you use super-resolution:

  ```
  pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git
  ```

### Pre-trained models

* Download the checkpoints: [mvnormal_finetune_sd21b](https://docs-assets.developer.apple.com/ml-research/models/cvpr24-direct/mvnormal_finetune_sd21b.zip) and [mvrgb_normalcond_sd21b](https://docs-assets.developer.apple.com/ml-research/models/cvpr24-direct/mvrgb_normalcond_sd21b.zip)

- Replace the ckpt path in the config file `configs/default-256-iter.yaml` by your `$ckpt_folder`

  ```
  L4 and L5:
  mvnorm_pretrain_path: '$ckpt_folder/mvnormal_finetune_sd21b'
  mvtexture_pretrain_path: '$ckpt_folder/mvrgb_normalcond_sd21b'
  ```


## Inference

- Run the following command to generate a few examples. Note that the generation variances are quite large, so it may need a few tries.

  ```
  python text-to-3d.py --num_repeats 2 --prompts configs/example_prompts.txt
  ```

  Results can be found under the `results` folder. The corresponding prompts could be in found in the txt file.

  You could also try single prompt using flag like `-prompts "A bald eagle carved out of wood"`

- You can opt to upsample the images with RealESRGAN before texturing by modifying `L17` of the configuration file. This will additionally cost around 1 second. 


## License
This sample code is released under the [LICENSE](LICENSE) terms.

## Citation
```
@article{lu2024direct2,
  title={Direct2.5: Diverse Text-to-3d Generation via Multi-view 2.5D Diffusion},
  author={Lu, Yuanxun and Zhang, Jingyang and Li, Shiwei and Fang, Tian and McKinnon, David and Tsin, Yanghai and Quan, Long and Cao, Xun and Yao, Yao},
  journal={Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
