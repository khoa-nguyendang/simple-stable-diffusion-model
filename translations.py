import csv

from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "VietAI/envit5-translation"
# inputs: english caption path
english_caption_path = "/k/AI_Data/CC3M/cc3m_en.tsv"
# outputs: vietnamese caption save path
vietnamese_caption_path = "/k/AI_Data/CC3M/cc3m_vn.tsv"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda')

vietnamese_caption_f = open(vietnamese_caption_path, 'w', newline='', encoding="utf-8")
field_name = ['url',  'caption']
writer = csv.DictWriter(vietnamese_caption_f, fieldnames=field_name, delimiter='\t')
writer.writeheader()

# Chọn batch_size nha: [1, 64]
batch_size = 64

count = 0
batch_count = 0
inputs = []
urls = []
with open(english_caption_path, 'r', encoding="utf-8") as tsvfile:
    reader = csv.DictReader(tsvfile, delimiter='\t')
    for row in tqdm(reader):
        try:
            caption = row['caption']
            url = row['url']
            inputs.append(f"en: {caption}")
            urls.append(url)
            batch_count += 1
            if batch_count % batch_size == 0:
                batch_count = 0
                outputs = model.generate(tokenizer(inputs, return_tensors="pt", padding=True).input_ids.to('cuda'), max_length=77)
                vietnamese_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for vietnamese, url_ in zip(vietnamese_batch, urls):
                    vietnamese = " ".join(vietnamese.split(' ')[1:]).strip()
                    writer.writerow({'url': url_, 'caption': vietnamese})
                    count += 1

                inputs = []
                urls = []
                
            if count % 10 == 0:
                vietnamese_caption_f.flush()
                count = 1
        except Exception as ex:
            print(ex)
            vietnamese_caption_f.flush()
            count = 1
            pass

vietnamese_caption_f.close()      