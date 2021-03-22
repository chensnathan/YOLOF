# You Only Look One-level Feature (YOLOF), CVPR2021
A simple, fast, and efficient object detector **without** FPN.

- This repo provides a neat implementation for YOLOF based on Detectron2. A [`cvpods`](https://github.com/Megvii-BaseDetection/cvpods) version can be 
  found in [https://github.com/megvii-model/YOLOF](https://github.com/megvii-model/YOLOF).

> [**You Only Look One-level Feature**](https://arxiv.org/abs/2103.09460),            
> Qiang Chen, Yingming Wang, Tong Yang, Xiangyu Zhang, Jian Cheng, Jian Sun

![image](images/yolof.png)

## Getting Started

- Our project is developed on [detectron2](https://github.com/facebookresearch/detectron2). Please follow the official detectron2 [installation](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). 
- Install `mish-cuda` to speed up the training and inference when using `CSPDarkNet-53` as the backbone (**optional**)
  ```shell
  git clone https://github.com/thomasbrandon/mish-cuda
  cd mish-cuda
  python setup.py build install
  cd ..
  ```
- Install `YOLOF` by:
  ```python
  python setup.py develop
  ```
- Then link your dataset path to `datasets`
  ```shell
  cd datasets/
  ln -s /path/to/coco coco
  ```
- Download the pretrained model in [OneDrive](https://1drv.ms/u/s!AgM0VtBH3kV9imGxZX3n_TMQGtbP?e=YMgpGJ) or in the [Baidu Cloud](https://pan.baidu.com/s/1BSOncRYq6HeCQ8q2hrWowA) with code `qr6o` to train with the **CSPDarkNet-53** backbone (**optional**)
  ```shell
  mkdir pretrained_models
  # download the `cspdarknet53.pth` to the `pretrained_models` directory
  ```
- Train with `yolof`
  ```python
  python ./tools/train_net.py --num-gpus 8 --config-file ./configs/yolof_R_50_C5_1x.yaml
  ```
- Test with `yolof`
  ```python
  python ./tools/train_net.py --num-gpus 8 --config-file ./configs/yolof_R_50_C5_1x.yaml --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
  ```
- Note that there might be API changes in future detectron2 releases that 
make the code incompatible.
  
## Main results

The models listed below can be found in [this onedrive link](https://1drv.ms/u/s!AgM0VtBH3kV9imGxZX3n_TMQGtbP?e=YMgpGJ) or in the [BaiduCloud link](https://pan.baidu.com/s/1BSOncRYq6HeCQ8q2hrWowA) with code `qr6o`. 
The FPS is tested on a 2080Ti GPU.
More models will be available in the near future.

| Model                                     |  COCO val mAP |  FPS  |
|-------------------------------------------|---------------|-------|
| YOLOF_R_50_C5_1x                          |  37.7         |   36  |
| YOLOF_R_50_DC5_1x                         |  39.2         |   23  |
| YOLOF_R_101_C5_1x                         |  39.8         |   23  |
| YOLOF_R_101_DC5_1x                        |  40.5         |   17  |
| YOLOF_X_101_64x4d_C5_1x                   |  42.2         |   11  |
| YOLOF_CSP_D_53_DC5_3x                     |  41.2         |   41  |
| YOLOF_CSP_D_53_DC5_9x                     |  42.8         |   41  |
| YOLOF_CSP_D_53_DC5_9x_stage2_3x           |  43.2         |   41  |

- Note that, the speed reported in this repo is 2~3 FPS faster than the one 
  reported in the cvpods version.


## Citation

If you find this project useful for your research, please use the following BibTeX entry.

    @inproceedings{chen2021you,
      title={You Only Look One-level Feature},
      author={Chen, Qiang and Wang, Yingming and Yang, Tong and Zhang, Xiangyu and Cheng, Jian and Sun, Jian},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
      year={2021}
    }
