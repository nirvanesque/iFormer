## Main Results on ImageNet with Pretrained Models
 We train Semantic FPN models on the ADE20K dataset for 40,000 iterations using standard training settings from the MMSegmentation toolkit. The input images are cropped to a resolution of 512×512 during training.

| Model      | Params(M) | Latency(ms) | mIoU |                                                   Ckpt.                                                   |                                                   Log                                                    |
|:-----------|:---------:|:-----------:|:----:|:---------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------:|
| iFormer-M  |    8.9    |    4.00     | 42.4 | [40000 iters](https://github.com/ChuanyangZheng/iFormer/releases/download/v0.9/iFormer_m_iter_40000.pth)  | [40000 iters](https://github.com/ChuanyangZheng/iFormer/releases/download/v0.9/iFormer_m_iter_40000.log) |
| iFormer-L  |   14.7    |    6.60     | 44.5 |                                              [40000 iters](https://github.com/ChuanyangZheng/iFormer/releases/download/v0.9/iFormer_l_iter_40000.pth)  | [40000 iters](https://github.com/ChuanyangZheng/iFormer/releases/download/v0.9/iFormer_l_iter_40000.log) |
| iFormer-L2 |   24.5    |    9.06     | 46.2 | [40000 iters](https://github.com/ChuanyangZheng/iFormer/releases/download/v0.9/iFormer_l2_iter_40000.pth) |                                             [40000 iters](https://github.com/ChuanyangZheng/iFormer/releases/download/v0.9/iFormer_l2_iter_40000.log)   |
* As detection, we have lost some original checkpoints corresponding to the performance reported in the paper due to certain errors. As a result, the performance may not be consistent with those presented in the paper. In some cases, our reproduced results surpass those reported in the paper.
## Getting Started
### Requirements
Please see the official [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) for installation.

Here we provide ours installation process:

Download [mmcv](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
```bash
pip install mmcv-2.1.0-cp38-cp38-manylinux1_x86_64.whl
```
Download the MMSegmentation source code and compile from the source.
```bash
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
git checkout b040e147adfa027bbc071b624bedf0ae84dfc922
pip install -v -e .
```
Our specific configuration details:
```bash
mmcv                      2.1.0
mmdet                     3.3.0
mmengine                  0.10.5
mmsegmentation            1.2.2 
```
### Data Preparation
Prepare the challenging ADE20K dataset according to the instructions in [MMSeg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets).

### Training
```bash
cd iFormer/segmentation
sh tools/dist_train.sh configs/sem_fpn/fpn_iformer_m_ade20k_40k.py 8 --work-dir=./output/seg_m_0
```
### Evaluation
```bash
cd iFormer/detection
checkpoint_path=your checkpoint path
sh tools/dist_test.sh configs/sem_fpn/fpn_iformer_m_ade20k_40k.py $check_path 1 --work-dir=./output/seg_m_0
```
