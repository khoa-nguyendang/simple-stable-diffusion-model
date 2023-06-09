import csv
import os
import shutil

destination_path = "/f/AI_Data/training"
summary_path = "/f/AI_Data/summary_small.tsv"

def append_record_to_tsv(data : list, file: str):
    # Create a file object for the TSV file.
    with open(file, "a") as csvfile:
        # Create a writer object for the TSV file.
        writer = csv.writer(csvfile, delimiter='\t')
        # Append multiple lines to the TSV file.
        writer.writerow(data)

def clone_file(file_source: str, des_dir: str) -> str:
    # Get the source path
    file_name = os.path.basename(file_source)

    # Create the destination directory if it doesn't exist
    if not os.path.exists(des_dir):
        os.mkdir(des_dir)

    shutil.copy(file_source, os.path.join(des_dir, file_name))
    return file_name


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
            append_record_to_tsv(["training/" + file_name, row[1]], summary_path)



proc_tsv("/g/AI_Data/_dataset/summary.tsv") 
            