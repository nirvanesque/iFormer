## Main Results on ImageNet with Pretrained Models
We train MaskR-CNN models on the COCO 2017 dataset for 12 epochs using standard training settings from the MMDetection toolkit.

| Model      | Params(M) | Latency(ms) | $AP^{box}$ | $AP^{mask}$ | Ckpt. | Core ML | Log |
|:-----------|:---------:|:-----------:|:----------:|:-----------:|:-----:|:-------:|:---:|
| iFormer-M  |    8.9    |    4.00     |    40.8    |    37.9     |       |         |     |
| iFormer-L  |   14.7    |    6.60     |    42.2    |    39.1     |       |         |     |
| iFormer-L2 |   24.5    |    9.06     |    44.6    |    41.1     |       |         |     |
## Getting Started
### Requirements
Please see the official [MMDetection](https://github.com/open-mmlab/mmdetection) for installation.

Here we provide our installation process:

Download [mmcv](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
```bash
pip install mmcv-2.1.0-cp38-cp38-manylinux1_x86_64.whl
```
Download the MMDetection source code and compile from the source.
```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout cfd5d3a985b0249de009b67d04f37263e11cdf3d
pip install -v .
```
Our specific configuration details:
```bash
mmcv                      2.1.0
mmdet                     3.3.0
mmengine                  0.10.5
mmsegmentation            1.2.2 
```
You can check your environmental setting by
```bash
python mmdet/utils/collect_env.py
```

<details>
<summary>
Ours MMDetection Environment
</summary>

```
sys.platform: linux
Python: 3.8.16 (default, Mar  2 2023, 03:21:46) [GCC 11.2.0]
CUDA available: True
MUSA available: False
numpy_random_seed: 2147483648
GPU 0,1,2: NVIDIA A800-SXM4-80GB
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 11.7, V11.7.99
PyTorch: 2.1.2+cu118
PyTorch compiling details: PyTorch built with:
  - GCC 9.3
  - C++ Version: 201703
  - CPU capability usage: AVX512
  - CUDA Runtime 11.8
  - NVCC architecture flags: -gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_90,code=sm_90
  - CuDNN 8.7
  - Magma 2.6.1
TorchVision: 0.16.2+cu118
OpenCV: 4.5.5
MMEngine: 0.10.5
MMDetection: 3.3.0+cfd5d3a
```
</details>

Verify the installation
```bash
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```
You will see a new image demo.jpg on your ./outputs/vis folder, where bounding boxes are plotted on cars, benches, etc.

### Data Preparation
Prepare COCO 2017 dataset according to the instructions in [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md#test-existing-models-on-standard-datasets).

### Training
```bash
cd iFormer/detection
sh dist_train.sh configs/mask_rcnn_iformer_m_fpn_1x_coco.py 3 --work-dir=./output/coco_m_0
```
### Evaluation
```bash
cd iFormer/detection
checkpoint_path=your checkpoint path
sh dist_test.sh configs/mask_rcnn_iformer_m_fpn_1x_coco.py $checkpoint_path 3 --work-dir=./output/coco_m_0
```