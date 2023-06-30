# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export OUTPUT_DIR="/mnt/k/AI_Models/SD/"
# export HUB_MODEL_ID="pokemon-lora"
# export DATASET_NAME="lambdalabs/pokemon-blip-captions"
export DATASET_DIR="/mnt/k/AI_Data/CC3M/dataset/"
export CACHE_DIR="/mnt/k/AI_Data/CC3M/_cache/"
export DISABLE_TELEMETRY=YES

accelerate launch --mixed_precision="fp16"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_DIR \
  --cache_dir=$CACHE_DIR \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=500 \
  --validation_prompt="artist" \
  --seed=1337