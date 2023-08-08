# simple-stable-diffusion-model
A full system and models that use Diffusion

for training model, you may need a machine with GPU engine

## Disclaimers.
Open Source Disclaimer. 
Notwithstanding anything in the Agreement to the contrary, khoa-nguyendang does not make any representation or warranty with respect to any OSS or free software that may be included in or accompany the Service. khoa-nguyendang HEREBY DISCLAIMS ANY AND ALL LIABILITY TO DEMAND PARTNER OR ANY THIRD PARTY RELATED TO ANY SUCH SOFTWARE THAT MAY BE INCLUDED IN OR ACCOMPANY THE SERVICE.

## Prerequisites

```
1. docker-compose #
2. python
3. img2dataset #https://github.com/rom1504/img2dataset
4. cuda for ubuntu #https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
5. conda virtual env
```

## Download training data and validation data
```
img2dataset --url_list /k/AI_Data/CC3M/cc3m_vn.tsv --input_format "tsv" --url_col "url" --caption_col "caption" --output_format webdataset\
 --output_folder /k/AI_Data/CC3M/dataset --processes_count 16 --thread_count 64 --image_size 512
```



## Prepare  env
```
$ make prepare

conda install 

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

