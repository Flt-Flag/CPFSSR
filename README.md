# CPFSSR: Combined Permuted Self-Attention and Fast Fourier Transform-Based Network for Stereo Image Super-Resolution

Wenwu Luo, Jing Wu, Feng Huang, Yunxiang Li

## Environment
- [PyTorch >= 1.11](https://pytorch.org/)
- [BasicSR == 1.3.5](https://github.com/XPixelGroup/BasicSR/blob/master/INSTALL.md) 


### Installation
Install Pytorch first.
Then,
```
pip install -r requirements.txt
python setup.py develop
```

## How To Inference
**Stereo Image Super Resolution**
``` 
python inference/inference_cpfssr.py
```

## How To Test
**Stereo Image Super Resolution**
- Then run the follwing codes (taking `CPFSSR_SSRx4.yml` as an example):
```
python cpfssr/test.py -opt options/test/CPFSSR_SSRx4.yml
```

## How To Train
- Refer to `./options/train` for the configuration file of the model to train.
- The training command is like
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 cpfssr/train.py -opt options/train/train_CPFSSR_SSRx4_scratch.yml --launcher pytorch
```

The training logs and weights will be saved in the `./experiments` folder.

## Acknowledgement
This project is mainly based on [BasicSR](https://github.com/XPixelGroup/BasicSR)
