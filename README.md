# math2latex

## Environment

- python 3.8.13
- torch 2.0.1
- torchvision 0.15.2
- opencv-python
- onnx

## How to use?

First, download the datasets in the `dataset` dir, and modify the dir name:

```
├── dataset
│   ├── data_mix
│   │   ├── dev
│   │   ├── test
│   │   ├── train
│   │   ├── dev_ids.txt
│   │   ├── test_ids.txt
│   │   └── train_ids.txt
│   └── data_pure
│       ├── dev
│       ├── test
│       ├── train
│       ├── dev_ids.txt
│       ├── test_ids.txt
│       └── train_ids.txt
```
 

If you want to train the model:

```sh
python -u main.py \
    --task "pure" \ # options: ["pure", "mix"]
    --batch_size 64 \
    --n_epochs 30 \
    --lr 0.001 \
    --model "ResnetTransformer" \
    --dim 256 \
    --n_layers 3 \
    --n_heads 4 \
    --img_size 224 \
    --dropout 0.2 \
    --seed 2023 
```

or run the script:

```sh
nohup ./scripts/pure.py > pure_log.txt &
# nohup ./scripts/mix.py > mix_log.txt & # for mix dataset
```

If you want to sample an image for testing(you should modify the image path in the src code):

```sh
python main.py --sample True
```

Remember modify the settings like `batch_size` in `scripts/main.sh`

If you failed to run the script, maybe you should:

```sh
chmod -R 777 scripts/main.sh
```
