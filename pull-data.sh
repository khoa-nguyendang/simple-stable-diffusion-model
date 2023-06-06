wget https://storage.cloud.google.com/gcc-data/Train/GCC-training.tsv?_ga=2.191230122.-1896153081.1529438250 training.tsv
wget https://storage.cloud.google.com/gcc-data/Validation/GCC-1.1.0-Validation.tsv?_ga=2.141047602.-1896153081.1529438250 validation.tsv

sed -i '1s/^/caption\turl\n/' training.tsv
sed -i '1s/^/caption\turl\n/' validation.tsv

img2dataset --url_list training.tsv --input_format "tsv"\
        --url_col "url" --caption_col "caption" --output_format webdataset\
        --output_folder _dataset/training --processes_count 16 --thread_count 64 --image_size 256\
        --enable_wandb True

img2dataset --url_list validation.tsv --input_format "tsv"\
        --url_col "url" --caption_col "caption" --output_format webdataset\
        --output_folder _dataset/validation --processes_count 16 --thread_count 64 --image_size 256\
        --enable_wandb True