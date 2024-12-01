checkpoint_path=iFormer_m_epoch_12.pth
sh dist_test.sh configs/mask_rcnn_iformer_m_fpn_1x_coco.py $checkpoint_path 8 --work-dir=iFormer_m_epoch_12