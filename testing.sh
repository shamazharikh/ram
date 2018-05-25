python main.py --is_train=False --ckpt_dir=./ckpt/densenet121_model --patch_size=32 --num_patches=1 \
--conv=True --model=densenet121 --device=0 --data_dir=./../data/slices --best=True

python main.py --is_train=False --ckpt_dir=./ckpt/densenet121_model --patch_size=32 --num_patches=2 \
--conv=True --model=densenet121 --device=0 --data_dir=./../data/slices --best=True
