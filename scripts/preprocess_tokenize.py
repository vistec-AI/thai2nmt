import argparse
import os
import time
import csv
import glob
import shutil
import re
import html
import uuid
from pathlib import Path
from functools import partial

from mosestokenizer import MosesTokenizer
from pythainlp.tokenize import word_tokenize as newmm_word_tokenizer

import pandas as pd
import sentencepiece as spm


en_moses_word_tokenize = MosesTokenizer('en')
newmm_word_tokenize_no_space = partial(
    newmm_word_tokenizer, keep_whitespace=False)
newmm_word_tokenize_with_space = partial(
    newmm_word_tokenizer, keep_whitespace=True)

TOKENIZER = {
    'newmm_space': newmm_word_tokenize_with_space,
    'newmm': newmm_word_tokenize_no_space,
    'moses': en_moses_word_tokenize
}


def write_txt(path, sentences):

    with open(path, 'w', encoding='utf-8') as f:
        for sent in sentences:
            f.write(str(sent) + '\n')


def train_spm(train_dataset_path, side, vocab_size, spm_out_dir, lower):
    """
        side :str -- is either "src" or "tgt"
    """

    df = pd.read_csv(train_dataset_path, encoding='utf-8')

    if lower:
        df[side] = df[side].apply(lambda x: x.lower())

    sentences = df[side].tolist()

    train_dataset_path = Path(train_dataset_path)
    current_dir = train_dataset_path.cwd()


    print('\nCurrent directory:', current_dir)

    case = 'uncased' if lower else 'cased'
    spm_file_prefix = f'spm.{side}.v-{vocab_size}.{case}'

    if not os.path.exists('spm_out_dir'):
        print(f'\nCreate a directory at `{spm_out_dir}`.')
        os.makedirs(spm_out_dir, exist_ok=True)

    spm_model_path = os.path.join(spm_out_dir, f'{spm_file_prefix}.model')

    if not os.path.exists(spm_model_path):

        temp_filepath = f'./temp/temp.sentences.{uuid.uuid1()}.{side}'

        print(f'\nWrite sentences to a temporary file at : {temp_filepath}')
        write_txt(temp_filepath, sentences)

        print('\nSentencePiece model is not found.')
        print(
            f'\nBegin training SentencePiece model, filename: {spm_file_prefix}.model')

        spm.SentencePieceTrainer.Train(
            f'--input={temp_filepath} --character_coverage=1.0 --model_prefix={spm_file_prefix} --vocab_size={vocab_size}')

        spm_model_path = f'./{spm_file_prefix}.model'
        spm_vocab_path = f'./{spm_file_prefix}.vocab'

        print(f'Move vocab and model files to {spm_out_dir}')
        spm_model_path = shutil.move(spm_model_path, os.path.join(
            spm_out_dir, f'{spm_file_prefix}.model'))
        spm_vocab_path = shutil.move(spm_vocab_path, os.path.join(
            spm_out_dir, f'{spm_file_prefix}.vocab'))

        print('\nDone')

        # remove tempfile
        print('\nRemove temp file')
        os.remove(temp_filepath)
    else:
        spm_model_path = os.path.join(spm_out_dir, f'{spm_file_prefix}.model')
        spm_vocab_path = os.path.join(spm_out_dir, f'{spm_file_prefix}.vocab')
        print('SentencePiece model was found.')

    print(f'Begin loading SentencePiece model from {spm_model_path}')
    model = spm.SentencePieceProcessor()

    print(f'Done loading SPM model')

    model.Load(spm_model_path)

    return model


def main(split_dataset_dir,
         src_lang, tgt_lang,
         src_uncased, tgt_uncased,
         src_tokenizer, tgt_tokenizer,
         src_spm_vocab_size, tgt_spm_vocab_size,
         out_dir, spm_out_dir):

    if src_tokenizer == 'spm':
        # Train SentencePiece model on train set

        file_paths = glob.glob(os.path.join(args.split_dataset_dir, '*.csv'))
        file_paths = list(filter(lambda x: 'train' in x, file_paths))

        assert len(file_paths) == 1
        train_filepath = file_paths[0]

        src_spm_model = train_spm(
            train_filepath, src_lang, src_spm_vocab_size, spm_out_dir, lower=src_uncased)

        _src_tokenizer = partial(src_spm_model.EncodeAsPieces)
    else:
        # for `newmm` or `moses`
        _src_tokenizer = TOKENIZER[src_tokenizer]

    if tgt_tokenizer == 'spm':
        # Train SentencePiece model on train set
        file_paths = glob.glob(os.path.join(args.split_dataset_dir, '*.csv'))
        file_paths = list(filter(lambda x: 'train' in x, file_paths))
        assert len(file_paths) == 1
        train_filepath = file_paths[0]

        tgt_spm_model = train_spm(
            train_filepath, tgt_lang, tgt_spm_vocab_size, spm_out_dir, lower=tgt_uncased)

        _tgt_tokenizer = partial(tgt_spm_model.EncodeAsPieces)
    else:
        # for `newmm` or `moses`
        _tgt_tokenizer = TOKENIZER[tgt_tokenizer]

    print(f'\nTokenizer (source): {src_tokenizer}')
    print(f'Tokenizer (target): {tgt_tokenizer}\n\n')

    file_paths = glob.glob(os.path.join(args.split_dataset_dir, '*.csv'))

    for file_path in file_paths:

        lang_pair, name, split = Path(file_path).stem.split('.')

        df = pd.read_csv(file_path, encoding='utf-8')

        print(
            f'\nPerfom tokenization for {src_lang} ({src_tokenizer}, src_uncased={src_uncased}) and {tgt_lang} ({tgt_tokenizer}, tgt_uncased={tgt_uncased}) of the {split} set.')

        df[src_lang] = df[src_lang].apply(str)
        df[tgt_lang] = df[tgt_lang].apply(str)

        if src_uncased:
            df[src_lang] = df[src_lang].apply(lambda x: x.lower())
        if tgt_uncased:
            df[src_lang] = df[src_lang].apply(lambda x: x.lower())

        src_tokens = df[src_lang].apply(_src_tokenizer)
        tgt_tokens = df[tgt_lang].apply(_tgt_tokenizer)

        if not os.path.exists(out_dir):
            print(f'Create a directiony at `{out_dir}`.')
            os.makedirs(out_dir, exist_ok=True)

        src_out_path = os.path.join(out_dir, f'{split}.{src_lang}')
        tgt_out_path = os.path.join(out_dir, f'{split}.{tgt_lang}')

        print(
            f'\n - Write tokenized result of {src_lang} langauge of the {split} set, to {src_out_path}')

        write_tokenized_result(src_tokens, src_out_path)

        print(
            f'\n - Write tokenized result of {tgt_lang} langauge of the {split} set, to {tgt_out_path}')

        write_tokenized_result(tgt_tokens, tgt_out_path)

    print('\n\nDone.')

    if src_lang == 'en' and src_tokenizer == 'moses':
        # close Moses tokenizer
        _src_tokenizer.close()
    if tgt_lang == 'en' and src_tokenizer == 'moses':
        # close Moses tokenizer
        _tgt_tokenizer.close()


def write_tokenized_result(series, path):

    with open(path, 'w', encoding='utf-8') as f:

        for tokens in series.tolist():

            line = ' '.join(tokens)
            f.write(f"{line}\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='./dataset/tokenized')
    parser.add_argument('--spm_out_dir', type=str, default='./dataset/spm')

    parser.add_argument('--split_dataset_dir', type=str,
                        default='./dataset/split')
    parser.add_argument('--src_lang', type=str, default='en')
    parser.add_argument('--tgt_lang', type=str, default='th')

    parser.add_argument('--src_uncased', action='store_true',
                        help='Whether to lowercase all characters')
    parser.add_argument('--tgt_uncased', action='store_true',
                        help='Whether to lowercase all characters')

    parser.add_argument('--src_spm_vocab_size', default=10000,
                        type=int, help='Vocab size for the source langauge')
    parser.add_argument('--tgt_spm_vocab_size', default=10000,
                        type=int, help='Vocab size for the target langauge')

    parser.add_argument('--src_tokenizer', type=str, default='newmm',
                        help='Select a tokenizer to tokenize segments of the source language, [\'newmm\', \'spm\', \'moses\']')
    parser.add_argument('--tgt_tokenizer', type=str, default='newmm',
                        help='Select a tokenizer to tokenze segments of the target language, [\'newmm\', \'spm\', \'moses\']')

    args = parser.parse_args()

    main(args.split_dataset_dir,
         args.src_lang, args.tgt_lang,
         args.src_uncased, args.tgt_uncased,
         args.src_tokenizer, args.tgt_tokenizer,
         args.src_spm_vocab_size, args.tgt_spm_vocab_size,
         args.out_dir, args.spm_out_dir)
