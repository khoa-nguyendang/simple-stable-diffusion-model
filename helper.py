# this file support to lookup all images that downloaded via img2dataset
# some files were missing during download, this helper will indexing them and write new line into csv file 

import csv
import os
import traceback


def get_file_content(file_path):
  """Get the content of a file.

  Args:
    file_path: The path to the file.

  Returns:
    The content of the file as a string.
  """

  with open(file_path, "r") as f:
    content = f.read()

  return content

def append_text_to_file(file_path, text):
  """Appends text to a file.

  Args:
    file_path: The path to the file.
    text: The text to be appended to the file.
  """

  with open(file_path, "a") as f:
    f.write(text + "\n")

def exception_to_string(exception):
  """Converts an exception to a string.

  Args:
    exception: The exception to be converted.

  Returns:
    A string representation of the exception.
  """

  exception_string = ""
  exception_string += "Exception type: " + str(type(exception)) + "\n"
  exception_string += "Exception message: " + str(exception) + "\n"
  exception_string += "Exception traceback: " + traceback.format_exc()

  return exception_string


def convert_csv_to_tsv(csvPath: str, tsvPath: str, maximum: int):
    with open(csvPath,'r') as csvin, open(tsvPath, 'w') as tsvout:
        csvin = csv.reader(csvin)
        tsvout = csv.writer(tsvout, delimiter='\t')
        counter = 0
        for row in csvin:
            if counter > maximum:
               return
            tsvout.writerow(row)
            counter = counter + 1

def append_records_to_tsv(data : list, file: str):
    # Create a file object for the TSV file.
    with open(file, "a") as csvfile:
        # Create a writer object for the TSV file.
        writer = csv.writer(csvfile, delimiter='\t')
        # Append multiple lines to the TSV file.
        writer.writerows(data)

def append_records_to_csv(data : list, file: str):
    # Create a file object for the CSV file.
    with open(file, "a") as csvfile:
        # Create a writer object for the CSV file.
        writer = csv.writer(csvfile)
        # Append multiple lines to the CSV file.
        writer.writerows(data)

def produce_records(path):
    try:
        records = list()
        for root, dirs, files in os.walk(path, topdown=False):
            if len(dirs) > 0:
                for name in dirs:
                    produce_records(name)
            
            for f in files:
                if f.endswith(".txt"):
                    img = f.replace(".txt", ".jpg")
                    records.append([os.path.join(root, img), get_file_content(os.path.join(root, f))])
            append_records_to_tsv(records, "/g/AI_Data/_dataset/summary.tsv")
    except Exception as e:
       print(e)
       append_text_to_file("/home/anhcoder/repos/github.com/khoa-nguyendang/simple-stable-diffusion-model/exceptions.txt", exception_to_string(e))

#produce_records("/g/AI_Data/_dataset/training")

#convert_csv_to_tsv("/g/AI_Data/_dataset/summary.csv", "/g/AI_Data/_dataset/summary.tsv", 3000000)