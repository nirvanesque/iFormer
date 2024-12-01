# nohup your_command > output.log 2>&1 & to avoid SignalException
output=nohup.out
nohup sh tools/dist_train.sh configs/sem_fpn/fpn_iformer_m_ade20k_40k.py 3 --work-dir=iformer_m_seg > $output 2>&1 &
echo $output
