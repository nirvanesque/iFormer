data_path=imagenet path
python -m torch.distributed.launch --nproc_per_node=1  \
    main.py  \
    --model iFormer_m \
    --input_size 224 \
    --num_workers 16 \
    --layer_scale_init_value 0 \
    --finetune iFormer_m.pth \
    --eval true \
    --data_path $data_path