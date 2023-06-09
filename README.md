# simple-stable-diffusion-model
A full system and models that use Diffusion

for training model, you may need a machine with GPU engine

## Prerequisites

```
1. docker-compose #
2. python
3. img2dataset #https://github.com/rom1504/img2dataset
4. cuda for ubuntu #https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
```

## Download training data and validation data
```
$ pull-data.sh
```



## Prepare  env
```
$ make prepare

$ export CUDA_MODULE_LOADING="LAZY"

```

## Choose stack for fine-tuning
In this project, I use 2 approach, 1 with pytorch base on diffusers
```
https://ngwaifoong92.medium.com/how-to-fine-tune-stable-diffusion-using-lora-85690292c6a8
```
and another one via Kiras
```
https://keras.io/examples/generative/finetune_stable_diffusion/
```
