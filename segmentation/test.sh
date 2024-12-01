check_path=iFormer_m_iter_40000.pth
sh tools/dist_test.sh configs/sem_fpn/fpn_iformer_m_ade20k_40k.py $check_path 8 --work-dir=iFormer_m_iter_40000