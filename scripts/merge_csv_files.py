import argparse
import os
import glob
import time
import csv

from pathlib import Path
from collections import Counter

from tqdm.auto import tqdm

import pandas as pd


def main(csv_directory, out_directory):

    csv_files = glob.glob(os.path.join(csv_directory, '*.csv'))
    df_list = {}
    sent_pairs_counter = Counter()

    for filepath in csv_files:

        filename = Path(filepath).stem
        df_list[filename] = pd.read_csv(filepath)
        sent_pairs_counter[filename] = df_list[filename].shape[0]

    print('\nDataset Statisitcs:\n')
    print('-' * 30)
    print('')

    for k, v in sent_pairs_counter.most_common(len(sent_pairs_counter)):

        print(f'Sub-dataset: {k:30}, # Sentence pairs: {v:10,}')

    print('')
    print(f'\nTotal: {sum(sent_pairs_counter.values()):10,}')

    write_to_txt(out_directory, df_list)


def write_to_txt(out_directory, df_list):
    if not os.path.exists(out_directory):
        print(f'\nCreate a directory at: `{out_directory}`')
        os.makedirs(out_directory, exist_ok=True)

    out_path = os.path.join(out_directory, 'en-th.merged.csv')
    print(f'\n\nBegin writing file in txt format to: `{out_path}`.\n')

    merged_item_ids = []

    for dataset_name, df in df_list.items():

        for index, _ in df.iterrows():
            sentence_id = f'{index}:{dataset_name}'
            merged_item_ids.append(sentence_id)

    merged_en_texts = pd.concat([df.en_text for _, df in df_list.items()]).apply(
        lambda x: str(x).strip())
    merged_th_texts = pd.concat([df.th_text for _, df in df_list.items()]).apply(
        lambda x: str(x).strip())

    merged_en_texts_is_duplicated = merged_en_texts.duplicated(
        keep=False).tolist()
    merged_th_texts_is_duplicated = merged_th_texts.duplicated(
        keep=False).tolist()

    with open(out_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['sentence_id', 'en', 'th',
                         'is_en_uniq', 'is_th_uniq'])

        for index, sentence_id in tqdm(enumerate(merged_item_ids), total=len(merged_item_ids)):

            is_en_uniq = not merged_en_texts_is_duplicated[index]
            is_th_uniq = not merged_th_texts_is_duplicated[index]

            en, th = merged_en_texts.iloc[index].replace(
                '\n', ''), merged_th_texts.iloc[index].replace('\n', '')

            writer.writerow([sentence_id, en, th, is_en_uniq, is_th_uniq])

    print('\nDone merging csv files into a txt file.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'csv_dir', help='Directory that stored the dataset in .csv format')
    parser.add_argument('--out_dir', help='Directory that stored merged dataset in .txt format',
                        default='./dataset/merged')

    args = parser.parse_args()

    csv_directory = args.csv_dir
    out_directory = args.out_dir

    main(csv_directory, out_directory)
