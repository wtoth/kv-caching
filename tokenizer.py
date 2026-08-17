import numpy as np
import pandas as pd
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("gpt2")

def tokenize_data(input_path, output_path):
    data = pd.read_parquet(input_path)
    text = "".join(data["text"].tolist())

    ids = tokenizer.encode(text).ids
    print(f"{input_path}: {len(ids):,} tokens")

    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(output_path)

#files = ["test-00000-of-00001.parquet", "validation-00000-of-00001.parquet"]
files = ["train-00000-of-00001.parquet", "test-00000-of-00001.parquet", "validation-00000-of-00001.parquet"]
data_path = "raw_data/wikitext-2-raw-v1"

for file in files:
    tokenize_data(input_path=f'{data_path}/{file}', output_path=f'data/{file.split("-")[0]}.bin')