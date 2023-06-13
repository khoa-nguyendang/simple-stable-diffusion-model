import asyncio
import multiprocessing
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from os import listdir

import cv2
import pandas as pd
import requests
from PIL import Image

download_folder: str = "/f/AI_Data/vehicles/"
vehicle_tsv_path: str = "/f/AI_Data/vehicles_urls.tsv"
log_file: str = "/home/anhcoder/repos/github.com/khoa-nguyendang/simple-stable-diffusion-model/ex.txt"

def append_text_to_file(file_path, text):
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

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    return wrapped

@background
def download(url: str, file_path: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            append_text_to_file(log_file, f'{r.status_code}\t{url}')
            return
        with open(file_path, "wb") as f:
            f.write(r.content)
    except Exception as ex:
        append_text_to_file(log_file, f'{500}\t{url}')
        # append_text_to_file(log_file, exception_to_string(ex))

def download_v2(url: str, file_path: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            append_text_to_file(log_file, f'{r.status_code}\t{url}')
            return
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
    except Exception as ex:
        append_text_to_file(log_file, f'{500}\t{url}')
        # append_text_to_file(log_file, exception_to_string(ex))

def proc_tsv_v3(file_path: str):
    data_frame = pd.read_csv(file_path, delimiter='\t')
    mask = data_frame.caption.str.contains('car ') \
    | data_frame.caption.str.contains('cars ') \
    | data_frame.caption.str.contains(' vance ') \
    | data_frame.caption.str.contains(' vances ') \
    | data_frame.caption.str.contains('racing ')
    vehicles = data_frame.loc[mask]
    vehicles['image_path'] = vehicles.index.map('{:09d}.jpg'.format)
    vehicles = vehicles[['image_path', 'caption', 'url']]
    vehicles.to_csv('vehicles_urls.tsv', sep="\t", index=False)


def make_dataset(file_path: str):
    df = pd.read_csv(file_path, delimiter='\t')
    for index, row in df.iterrows():
        download(row["url"], f'{download_folder}{row["image_path"]}')

def make_dataset_v2(file_path: str):
    df = pd.read_csv(file_path, delimiter='\t')
    download_parallel(df)

def download_parallel(df):
    success = 0
    fail = 0
    pool = ThreadPoolExecutor(max_workers=1000)
    futures = [pool.submit(download_v2,  row["url"], f'{download_folder}{row["image_path"]}') for index, row in df.iterrows()]
    for future in as_completed(futures):
        result = future.result()


def produce_records(path, tsv_file_path):
    try:
        records = list()
        for root, dirs, files in os.walk(path, topdown=False):
            for f in files:
                if f.endswith(".jpg"):
                    records.append(f)
        
        print("success scan folder: ", root)

        df = pd.read_csv(tsv_file_path, delimiter='\t')
        vehicles = df[df['image_path'].isin(records)]
        vehicles = vehicles[['image_path', 'caption', 'url']]
        vehicles.to_csv('vehicles_clean_vn.tsv', sep="\t", index=False)
    except Exception as e:
       print(e)
       append_text_to_file("/home/anhcoder/repos/github.com/khoa-nguyendang/simple-stable-diffusion-model/exceptions.txt", exception_to_string(e))


def detect_bad_imgs(path) -> list:
    lst = list()
    for root, dirs, files in os.walk(path, topdown=False):
            for f in files:
                try:
                    img = Image.open(path +f) # open the image file
                    img.verify() # verify that it is, in fact an image
                    cv2.imread(path +f)
                except (IOError, SyntaxError) as e:
                    append_text_to_file("/home/anhcoder/repos/github.com/khoa-nguyendang/simple-stable-diffusion-model/exceptions.txt", f)
                    lst.append(f)
                    try:
                        os.remove(path + f)
                    except Exception as ex:
                        print(ex)
    
    return lst
    
def detect_bad_imgs_opencv(path) -> list:
    lst = list()
    for root, dirs, files in os.walk(path, topdown=False):
            for f in files:
                try:
                    img = cv2.imread(path + f)
                    if img is None:
                        append_text_to_file("/home/anhcoder/repos/github.com/khoa-nguyendang/simple-stable-diffusion-model/invalid.txt", f)
                        continue
                    height, width, channels = img.shape
                    if channels != 3:
                        append_text_to_file("/home/anhcoder/repos/github.com/khoa-nguyendang/simple-stable-diffusion-model/ex_channel.txt", f)
                        continue
                    if height != 512 and width != 512:
                        append_text_to_file("/home/anhcoder/repos/github.com/khoa-nguyendang/simple-stable-diffusion-model/ex.txt", f)
                        new_imng =  cv2.resize(img, (512, 512))
                        cv2.imwrite(path + f, new_imng)
                except (IOError, SyntaxError) as e:
                    append_text_to_file("/home/anhcoder/repos/github.com/khoa-nguyendang/simple-stable-diffusion-model/exceptions.txt", f)
                    lst.append(f)
                    try:
                        os.remove(path + f)
                    except Exception as ex:
                        print('exception at file: ' + f)
                        print(ex)
    
    return lst

def produce_training_file(metadata_path, training_path, output_training_path):
    data_frame = pd.read_csv(metadata_path, delimiter='\t')
    bad_images = detect_bad_imgs_opencv(training_path)
    # bad_images = detect_bad_imgs(training_path)
    print(f'bad images amount: {len(bad_images)}')
    vehicles = data_frame[~data_frame['image_path'].isin(bad_images)]
    vehicles = vehicles[['image_path', 'caption', 'url']]
    vehicles.to_csv(output_training_path, sep="\t", index=False)
    


metadata_path = "/k/AI_Data/vehicles_vn.tsv"
training_path = "/f/AI_Data/vehicles/"
output_training_path = "/f/AI_Data/vehicles_vn.tsv"


#produce_training_file(metadata_path, training_path, output_training_path)
# proc_tsv_v3("/k/AI_Data/Diffusions/training_url.tsv")



#test download single file
#download("http://lh6.ggpht.com/-IvRtNLNcG8o/TpFyrudaT6I/AAAAAAAAM6o/_11MuAAKalQ/IMG_3422.JPG?imgmax=800", f'{download_folder}000000000.jpg')


#download multiple file
# make_dataset(vehicle_tsv_path)
#make_dataset_v2(vehicle_tsv_path)


#produce_records("/f/AI_Data/vehicles", "/home/anhcoder/repos/github.com/khoa-nguyendang/simple-stable-diffusion-model/vehicles_urls.tsv")

