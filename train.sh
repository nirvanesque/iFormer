python -m torch.distributed.launch --nproc_per_node=8  \
    main.py  \
    --cfg-path configs/iFormer_m.yaml