import argparse
import pandas as pd
import glob
import os
import csv
from functools import partial

from sacremoses import MosesTokenizer, MosesDetokenizer
from pythainlp.tokenize import word_tokenize


th_word_space_tokenize = partial( word_tokenize, engine='newmm', keep_whitespace=True)

def th_detokenize(line):
    detok = ''.join(list(map(lambda x: ' ' if x == '' else x, line.split(' '))))
    return detok

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('split_directory', type=str)

    args = parser.parse_args()

    file_paths = glob.glob(os.path.join(args.split_directory, '*.test.csv'))
    assert len(file_paths) == 1

    test_filepath = file_paths[0]

    print(f'Read csv file from {test_filepath}')
    test_df = pd.read_csv(test_filepath, encoding='utf-8')

    
    en_tokenizer = MosesTokenizer(lang='en')
    en_detokenizer = MosesDetokenizer(lang='en')

    test_df['en'] = test_df['en'].apply(lambda x: en_detokenizer.detokenize(en_tokenizer.tokenize(x)))
    test_df['th'] = test_df['th'].apply(lambda x: ' '.join(th_word_space_tokenize(x))).apply(th_detokenize)

    test_df[['en']].to_csv(os.path.join(args.split_directory, 'test.detok.en'), encoding='utf-8', sep="\t", index=False, header=False, escapechar="\\", quotechar="", quoting=csv.QUOTE_NONE)
    test_df[['th']].to_csv(os.path.join(args.split_directory, 'test.detok.th'), encoding='utf-8', sep="\t", index=False, header=False, escapechar="\\", quotechar="", quoting=csv.QUOTE_NONE)
    
    print('Done writing test set into text files.')
