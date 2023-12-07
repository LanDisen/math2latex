# math2latex

## Environment

- python 3.8.13
- torch 2.0.1
- torchvision 0.15.2

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
nohup ./scripts/main.py > log.txt &
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

## TODO

- [ ] YOLO for math formula detection
