# split train/val/test
import argparse
import os
import time
import csv

import numpy as np
import pandas as pd


def print_sub_dataset_dist(series):

    N = sum(series.values)
    for dataset, count in series.items():
        print(f'{dataset:25}: {count:8,} ( {float(count/N*100):5.2f}% )')


def main(path_merged_csv, out_dir, train_ratio, val_ratio, test_ratio, seed, stratify):

    df = pd.read_csv(path_merged_csv, encoding='utf-8', engine='python')
    df.is_en_uniq.astype(bool)
    df.is_th_uniq.astype(bool)

    df['dataset'] = df['sentence_id'].apply(lambda x: x.split(':')[-1])
    train_df, val_df, test_df = None, None, None

    N = df.shape[0]

    print(f'SEED value: {seed}')
    
    print('\nSummary: Number of segment pairs for each sub-dataset and percentage\n')

    print_sub_dataset_dist(df['dataset'].value_counts())

    print('')

    # print(df.dtypes)
    if val_ratio != 0:
        n_val = int(N * val_ratio)
        n_test = int(N * test_ratio)

        print(
            f'\nRatio (train, val, test) : ({train_ratio:2}, {val_ratio:2}, {test_ratio:2})')

        if stratify:
            N_valid_set = df[(df['is_en_uniq'] == True) & (df['is_th_uniq'] == True)].shape[0]
            val_test_df = df[(df['is_en_uniq'] == True) & (df['is_th_uniq'] == True)] \
                .groupby('dataset', group_keys=False) \
                .apply(lambda x: x.sample(n=int(np.rint((n_val + n_test) * len(x) / N_valid_set)), random_state=seed)) \
                .sample(frac=1).reset_index(drop=True)
        else:
            val_test_df = df[(df['is_en_uniq'] == True) & (
                df['is_th_uniq'] == True)].sample(n=n_val + n_test, random_state=seed)

        val_test_ids = val_test_df.sentence_id.tolist()

        if stratify:
            val_df = val_test_df.groupby('dataset', group_keys=False) \
                .apply(lambda x: x.sample(n=int(np.rint((n_val) * len(x) / val_test_df.shape[0])), random_state=seed)) \
                .sample(frac=1).reset_index(drop=True)
        else:
            val_df = val_test_df.sample(n=n_val, random_state=seed)

        val_ids = val_df.sentence_id.tolist()

        test_df = val_test_df[val_test_df['sentence_id'].isin( val_ids) == False]
        train_df = df[df['sentence_id'].isin(val_test_ids) == False]

        print('\nDone spliting train/val/test set')

        print( f'\nRatio (train, val, test): ({train_ratio:2}, {val_ratio:2}, {test_ratio:2})')
        print(f'Number of segment pairs (train, val, test): {train_df.shape[0]:6,} | {val_df.shape[0]:6,} | {test_df.shape[0]:6,}')

    if val_ratio == 0:
        print(f'\nRatio (train, test): ({train_ratio:2}, {test_ratio:2})')
        n_test = int(N * test_ratio)
        if stratify:
            N_valid_set = df[(df['is_en_uniq'] == True) & (
                df['is_th_uniq'] == True)].shape[0]

            test_df = df[(df['is_en_uniq'] == True) & (df['is_th_uniq'] == True)] \
                .groupby('dataset', group_keys=False) \
                .apply(lambda x: x.sample(n=int(np.rint(n_test * len(x) / N_valid_set)), random_state=seed)) \
                .sample(frac=1).reset_index(drop=True)
        else:
            test_df = df[(df['is_en_uniq'] == True) & (df['is_th_uniq'] == True)].sample(n=n_test, random_state=seed)

        test_ids = test_df.sentence_id.tolist()

        train_df = df[df['sentence_id'].isin(test_ids) == False]

        print('\nDone spliting train/test set')

        print(f'\nRatio (train, test): ({train_ratio:2}, {test_ratio:2})')
        print(
            f'Number of paris (train, test): {train_df.shape[0]:6,} | {test_df.shape[0]:6,}')

    print('\nSummary (train set): Number of sampled segment pairs for each sub-dataset\n')

    print_sub_dataset_dist(train_df['dataset'].value_counts())

    if val_ratio != 0.0:
        print('\nSummary (val set): Number of sampled segment pairs for each sub-dataset\n')
        print_sub_dataset_dist(val_df['dataset'].value_counts())

    print('\nSummary (test set): Number of sampled segment pairs for each sub-dataset\n')

    print_sub_dataset_dist(test_df['dataset'].value_counts())

    if not os.path.exists(out_dir):
        print(f'\nCreate a directory at: `{out_dir}`')
        os.makedirs(out_dir, exist_ok=True)

    print(f'\n\nStart writing output files to `{out_dir}`')

    train_df = train_df.drop(columns=['dataset'])
    test_df = test_df.drop(columns=['dataset'])

    stratify_opt = ''
    if stratify == True:
        stratify_opt = '_stratified'

    if val_ratio != 0.0:

        val_df = val_df.drop(columns=['dataset'])
        val_df.to_csv(os.path.join(
            out_dir, f'en-th.merged{stratify_opt}.val.csv'), encoding='utf-8')

    train_df.to_csv(os.path.join(
        out_dir, f'en-th.merged{stratify_opt}.train.csv'), encoding='utf-8') 
    test_df.to_csv(os.path.join(
        out_dir, f'en-th.merged{stratify_opt}.test.csv'), encoding='utf-8')

    
    print('\nDone writing files.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('path_merged_csv', type=str)

    parser.add_argument('train_ratio', type=float)
    parser.add_argument('test_ratio', type=float)
    parser.add_argument('--stratify', action='store_true', default=False)
    parser.add_argument('--val_ratio', type=float, default=0.0)
    parser.add_argument('--out_dir', type=str, default='./dataset/split')
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

    if args.val_ratio != 0:
        assert args.train_ratio + args.val_ratio + args.test_ratio == 1.0
    else:
        assert args.train_ratio + args.val_ratio + args.test_ratio == 1.0

    main(args.path_merged_csv, args.out_dir, args.train_ratio,
         args.val_ratio, args.test_ratio, args.seed, args.stratify)
