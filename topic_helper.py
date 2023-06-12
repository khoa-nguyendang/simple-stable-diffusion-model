import csv
import os
import shutil

destination_path = "/f/AI_Data/training"
summary_path = "/f/AI_Data/summary_small.tsv"

def append_text_to_file(file_path, text):
  """Appends text to a file.

  Args:
    file_path: The path to the file.
    text: The text to be appended to the file.
  """

  with open(file_path, "a") as f:
    f.write(text + "\n")

def append_record_to_tsv(data : list, file: str):
    # Create a file object for the TSV file.
    with open(file, "a") as csvfile:
        # Create a writer object for the TSV file.
        writer = csv.writer(csvfile, delimiter='\t')
        # Append multiple lines to the TSV file.
        writer.writerow(data)

def clone_file(file_source: str, des_dir: str):
    # Get the source path
    file_name = os.path.basename(file_source)
    new_file = os.path.join(des_dir, file_name)
    if not os.path.exists(new_file):
        append_text_to_file("/home/anhcoder/repos/github.com/khoa-nguyendang/simple-stable-diffusion-model/exceptions.txt", new_file)
        return
    if not os.path.exists(new_file):
        shutil.copy(file_source,new_file )


def proc_tsv(file_path: str):
    with open(file_path, "r") as tsv_file:
        # Create a csv reader object
        reader = csv.reader(tsv_file, delimiter="\t")

        # Iterate over the rows in the TSV file
        for row in reader:
            # file_name = clone_file(row[0], destination_path)
            file_name = os.path.basename(row[0])
            if file_name == "000567195.jpg":
                return
            append_record_to_tsv(["training_car/" + file_name, row[1]], summary_path)


import pandas as pd


def proc_tsv_v2(file_path: str):
    data_frame = pd.read_csv(file_path, delimiter='\t')
    mask = data_frame.caption.str.contains(' car ') \
    | data_frame.caption.str.contains(' cars ') \
    | data_frame.caption.str.contains(' vehicle ') \
    | data_frame.caption.str.contains(' vehicles ') \
    | data_frame.caption.str.contains(' bus ') \
    | data_frame.caption.str.contains(' buses ') \
    | data_frame.caption.str.contains(' vance ') \
    | data_frame.caption.str.contains(' vances ') \
    | data_frame.caption.str.contains(' auto ') \
    | data_frame.caption.str.contains(' autos ')
    vehicles = data_frame.loc[mask]
    vehicles.to_csv('vehicles.tsv', sep="\t", index=False)

def clone_dataset(file_path: str):
    df = pd.read_csv(file_path, delimiter='\t')
    root_path = "/k/AI_Data/Diffusions/"
    next_path = "/k/AI_Data/Diffusions/vehicles/"
    for index, row in df.iterrows():
        clone_file(root_path+row['image_path'], next_path)


#proc_tsv_v2("/k/AI_Data/Diffusions/training_full.tsv") 
clone_dataset("/k/AI_Data/Diffusions/vehicles.tsv") 


            