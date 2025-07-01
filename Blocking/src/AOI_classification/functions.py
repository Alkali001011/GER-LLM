import os
from io import open
import numpy as np
import config
import wget
import zipfile


def textualize_block(row, cols):

    text = ''
    for i, col in enumerate(cols):
        if i != len(cols) - 1:
            text += 'COL ' + str(cols[i].split('_')[1]) + ' VAL ' + str(row[i]) + ' '

    return text


def tokenize_block(row, cols):

    toks = []
    for i, col in enumerate(cols):
        if i != len(cols) - 1:
            for tok in str(row[i]).split():
                toks.append(tok.replace('_', ' '))

    return toks


def load_glove_model():
    print("Loading Glove Model")
    glove_model = {}
    file_path = config.glove_folder+config.glove_file+str(config.glove_size)+'d'+config.glove_path_suffix
    #           "glove/glove.6B.100d.txt"
    if not os.path.isfile(file_path):
        print("Downloading Glove Embeddings...")
        _ = wget.download(config.glove_url, out=config.glove_folder)
        with zipfile.ZipFile(config.glove_folder+config.glove_zip, 'r') as zip_ref:
            print("Unzipping Glove Embeddings...")
            zip_ref.extractall(config.glove_folder)

    with open(file_path, 'r', encoding="utf8") as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model
