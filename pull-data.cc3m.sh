img2dataset --url_list $0 --input_format "tsv" --url_col "url" --caption_col "caption" --output_format webdataset --output_folder $1 --processes_count 16 --thread_count 64 --image_size 512