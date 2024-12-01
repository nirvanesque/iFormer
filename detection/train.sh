output=nohup.out
nohup sh dist_train.sh configs/mask_rcnn_iformer_m_fpn_1x_coco.py 8 --work-dir=./output/iformer_m_coco > $output 2>&1 &
echo $output