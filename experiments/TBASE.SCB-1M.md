# Experiment 1: TBASE.SCB-1M


In this experiment, we train Transformer BASE model on `scb-mt-en-th-2020` version 1.0 (Available at: https://github.com/vistec-AI/dataset-releases/releases/tag/scb-mt-en-th-2020_v1.0). The total number of English-Thai segment pairs is 1,001,752.


The Transformer BASE model used in this experiment consists of 6 encoder and 6 decoder blocks, 512 embedding dimensions, and 2,048 feed forward hidden units. The dropout rate is set to 0.3. The embedding of decoder input and output are shared. Maximum number of tokens per mini-batch is 9,750. The optimizer is Adam with initial learning rate of 1e-7 and weight decay rate of 0.0. The learning rate has an inverse squared schedule with warmup for the first 4,000 updates. Label smoothing of 0.1 is applied during training. The criteria for selecting the best model checkpoint is label-smoothed cross entropy loss. 


We train each model on 1 NVIDIA Tesla V100 GPU (as a part of [DGX-1](https://images.nvidia.com/content/pdf/dgx1-v100-system-architecture-whitepaper.pdf)) with mixed-precision training (fp16) and gradient accumulation for 16 steps.

<br/>

## Experiment setup

1. Package Installlation


    1.1 Install required Python packages via `pip install`

    ```
    pip install -r requirements.txt
    ```

    1.2 Install [Fairseq Toolkit](https://github.com/pytorch/fairseq/tree/master/fairseq) from source. 


    ```
    bash scripts/install_fairseq.sh y
    ```

    Note: When the first argument is specified as `y`, it will install [apex](https://github.com/NVIDIA/apex), an extension for mixed-precision training in Pytorch, designed for host machine with GPUs. If specify this argument as `n`, it will only install `fairseq.` The default value is `n`. 

    In our experiment, we install `apex` librery. This may take serveral minutes to complete the installation.

    1.3 Install [SentencePiece](https://github.com/google/sentencepiece) from source (required root permission).

    You can follow the installation manual from https://github.com/google/sentencepiece#c-from-source or from our code snippet below.

    Install required tools and libraries for building SentencePiece

    ```
    apt-get install cmake build-essential pkg-config libgoogle-perftools-dev
    ```
    
    Build SentencePiece (C++) from source

    ```
    bash scripts/install_sentencepiece.sh
    ```

2. Download Dataset

    Download `scb-mt-en-th-2020` dataset - version 1.0 from the following script. 
    
    ```
    bash scripts/download_dataset.scb_mt.sh 1.0
    ```   

    Note: The first argument indicates the version of `scb-mt-en-th-2020` dataset. (default value is `1.0`) 

<br/>

## Data Preprocessing

1. Perform text cleaning and filtering

    ```
    python ./scripts/clean_text.py ./dataset/raw/scb-mt-en-th-2020 \
        --unicode_norm NFKC \
        --out_dir ./dataset/cleaned/scb-mt-en-th-2020
    ```

2. Merge csv files into txt file.

    ```
    python ./scripts/merge_csv_files.py ./dataset/cleaned/scb-mt-en-th-2020/ \
        --out_dir ./dataset/merged/scb-mt-en-th-2020/
    ```

3. Split the dataset into train/val/test set with the ratio 80/10/10

    ```
    python ./scripts/split_dataset.py ./dataset/merged/scb-mt-en-th-2020/en-th.merged.csv \
        0.8 \
        0.1 \
        --val_ratio 0.1 \
        --stratify \
        --seed 2020 \
        --out_dir ./dataset/split/scb-mt-en-th-2020
    ```

    As this script splits train/val/test set differently each time, we provide our version of train/val/test split in order to reproduce our experiment. This can be download via the following script.

    ```
    bash scripts/download_dataset_split.scb-1m.sh
    ```

4. Create a copy to detokenized segments of the test set for model evaluation.


    ```
    python scripts/write_detok_test_set.py ./dataset/split/scb-mt-en-th-2020
    ```

5. Perform text preprocessing for th→en

    newmm→moses

    ```bash
    python ./scripts/preprocess_tokenize.py \
        --out_dir ./dataset/tokenized/scb-mt-en-th-2020/th-en/newmm-moses/ \
        --spm_out_dir ./dataset/spm/scb-mt-en-th-2020/th-en \
        --split_dataset_dir ./dataset/split/scb-mt-en-th-2020 \
        --src_lang th \
        --tgt_lang en \
        --src_tokenizer newmm \
        --tgt_tokenizer moses
    ```

    newmm→spm

    ```bash
    python ./scripts/preprocess_tokenize.py \
        --out_dir ./dataset/tokenized/scb-mt-en-th-2020/th-en/newmm-spm/ \
        --spm_out_dir ./dataset/spm/scb-mt-en-th-2020/th-en \
        --split_dataset_dir ./dataset/split/scb-mt-en-th-2020 \
        --src_lang th \
        --tgt_lang en \
        --tgt_spm_vocab_size 16000 \
        --src_tokenizer newmm \
        --tgt_tokenizer spm
    ```

    spm→moses

    ```bash
    python ./scripts/preprocess_tokenize.py \
        --out_dir ./dataset/tokenized/scb-mt-en-th-2020/th-en/spm-moses/ \
        --spm_out_dir ./dataset/spm/scb-mt-en-th-2020/th-en \
        --split_dataset_dir ./dataset/split/scb-mt-en-th-2020 \
        --src_lang th \
        --tgt_lang en \
        --src_spm_vocab_size 16000 \
        --src_tokenizer spm \
        --tgt_tokenizer moses
    ```

    spm→spm

    ```bash
    python ./scripts/preprocess_tokenize.py \
        --out_dir ./dataset/tokenized/scb-mt-en-th-2020/th-en/spm-spm/ \
        --spm_out_dir ./dataset/spm/scb-mt-en-th-2020/th-en \
        --split_dataset_dir ./dataset/split/scb-mt-en-th-2020 \
        --src_lang th \
        --tgt_lang en \
        --src_spm_vocab_size 16000 \
        --tgt_spm_vocab_size 16000 \
        --src_tokenizer spm \
        --tgt_tokenizer spm
    ```

5. Perform text preprocessing for en→th

    moses→newmm

    ```bash
    python ./scripts/preprocess_tokenize.py \
        --out_dir ./dataset/tokenized/scb-mt-en-th-2020/en-th/moses-newmm_space/ \
        --spm_out_dir ./dataset/spm/scb-mt-en-th-2020/en-th \
        --split_dataset_dir ./dataset/split/scb-mt-en-th-2020 \
        --src_lang en \
        --tgt_lang th \
        --src_tokenizer moses \
        --tgt_tokenizer newmm_space
    ```

    moses→spm

    ```bash
    python ./scripts/preprocess_tokenize.py \
        --out_dir ./dataset/tokenized/scb-mt-en-th-2020/en-th/moses-spm/ \
        --spm_out_dir ./dataset/spm/scb-mt-en-th-2020/en-th \
        --split_dataset_dir ./dataset/split/scb-mt-en-th-2020 \
        --src_lang en \
        --tgt_lang th \
        --tgt_spm_vocab_size 16000 \
        --src_tokenizer moses \
        --tgt_tokenizer spm
    ```

    spm→newmm

    ```bash
    python ./scripts/preprocess_tokenize.py \
        --out_dir ./dataset/tokenized/scb-mt-en-th-2020/en-th/spm-newmm_space/ \
        --spm_out_dir ./dataset/spm/scb-mt-en-th-2020/en-th \
        --split_dataset_dir ./dataset/split/scb-mt-en-th-2020 \
        --src_lang en \
        --tgt_lang th \
        --src_spm_vocab_size 16000 \
        --src_tokenizer spm \
        --tgt_tokenizer newmm_space
    ```

    spm→spm

    ```bash
    python ./scripts/preprocess_tokenize.py \
        --out_dir ./dataset/tokenized/scb-mt-en-th-2020/en-th/spm-spm/ \
        --spm_out_dir ./dataset/spm/scb-mt-en-th-2020/en-th \
        --split_dataset_dir ./dataset/split/scb-mt-en-th-2020 \
        --src_lang en \
        --tgt_lang th \
        --src_spm_vocab_size 16000 \
        --tgt_spm_vocab_size 16000 \
        --src_tokenizer spm \
        --tgt_tokenizer spm
    ```

6. Binarize tokenized segments in train/val/test with `fairseq-preprocess` via the following script.


    ```
    bash ./scripts/fairseq_preprocess.train_val_test.sh th en 130000 130000 ./dataset/tokenized/scb-mt-en-th-2020/th-en/newmm-moses ./dataset/binarized/scb-mt-en-th-2020/th-en/newmm-moses/130000-130000/
    ```

    ```
    bash ./scripts/fairseq_preprocess.train_val_test.sh th en 130000 16000 ./dataset/tokenized/scb-mt-en-th-2020/th-en/newmm-spm ./dataset/binarized/scb-mt-en-th-2020/th-en/newmm-spm/130000-16000/
    ```

    ```
    bash ./scripts/fairseq_preprocess.train_val_test.sh th en 16000 130000 ./dataset/tokenized/scb-mt-en-th-2020/th-en/spm-moses ./dataset/binarized/scb-mt-en-th-2020/th-en/spm-moses/16000-130000/
    ```

    ```
    bash ./scripts/fairseq_preprocess.train_val_test.sh th en 32000 32000 ./dataset/tokenized/scb-mt-en-th-2020/th-en/spm-spm ./dataset/binarized/scb-mt-en-th-2020/th-en/spm-spm/32000-joined/ --joined-dictionary
    ```

    ```
    bash ./scripts/fairseq_preprocess.train_val_test.sh en th 130000 130000 ./dataset/tokenized/scb-mt-en-th-2020/en-th/moses-newmm_space ./dataset/binarized/scb-mt-en-th-2020/en-th/moses-newmm_space/130000-130000/
    ```

    ```
    bash ./scripts/fairseq_preprocess.train_val_test.sh en th 130000 16000 ./dataset/tokenized/scb-mt-en-th-2020/en-th/moses-spm ./dataset/binarized/scb-mt-en-th-2020/en-th/moses-spm/130000-16000/
    ```

    ```
    bash ./scripts/fairseq_preprocess.train_val_test.sh en th 16000 130000 ./dataset/tokenized/scb-mt-en-th-2020/en-th/spm-newmm_space ./dataset/binarized/scb-mt-en-th-2020/en-th/spm-newmm_space/16000-130000/
    ```

    ```
    bash ./scripts/fairseq_preprocess.train_val_test.sh en th 32000 32000 ./dataset/tokenized/scb-mt-en-th-2020/en-th/spm-spm ./dataset/binarized/scb-mt-en-th-2020/en-th/spm-spm/32000-joined/ --joined-dictionary
    ```

<br/>

## Model Training


Train Transformer BASE model via the following script: `scripts/fairseq_train.transformer_base.single_gpu.fp16.sh`

Note: The first argument indicate the ID of GPU. In this case, we train each model on 1 GPU (`GPU_ID`: 0-7).


### 1. Train models for th→en direction

1.1 moses→newmm

```bash
bash ./scripts/fairseq_train.transformer_base.single_gpu.fp16 0 ./dataset/binarized/scb-mt-en-th-2020/th-en/newmm-moses/130000-130000/ scb-mt-en-th-2020/th-en/newmm-moses/130000-130000 9750 150
```

1.2 moses→spm

```bash
bash ./scripts/fairseq_train.transformer_base.single_gpu.fp16 1 ./dataset/binarized/scb-mt-en-th-2020/th-en/newmm-spm/130000-16000/ scb-mt-en-th-2020/th-en/newmm-spm/130000-16000 9750 150
```

1.3 spm→newmm

```bash
bash ./scripts/fairseq_train.transformer_base.single_gpu.fp16 2 ./dataset/binarized/scb-mt-en-th-2020/th-en/spm-moses/16000-130000/ scb-mt-en-th-2020/th-en/spm-moses/16000-130000 9750 150
```

1.4 spm→spm

```bash
bash ./scripts/fairseq_train.transformer_base.single_gpu.fp16 3 ./dataset/binarized/scb-mt-en-th-2020/th-en/spm-spm/32000-joined/ scb-mt-en-th-2020/th-en/spm-spm/32000-joined 9750 150
```

### 2. Train models for en→th direction

2.1 newmm→moses

```bash
bash ./scripts/fairseq_train.transformer_base.single_gpu.fp16 4 ./dataset/binarized/scb-mt-en-th-2020/en-th/moses-newmm_space/130000-130000/ scb-mt-en-th-2020/en-th/moses-newmm_space/130000-130000 9750 150
```

2.2 newmm→spm

```bash
bash ./scripts/fairseq_train.transformer_base.single_gpu.fp16 5  ./dataset/binarized/scb-mt-en-th-2020/en-th/moses-spm/130000-16000/ scb-mt-en-th-2020/en-th/moses-spm/130000-16000 9750 150

```

2.3 spm→moses

```bash
bash ./scripts/fairseq_train.transformer_base.single_gpu.fp16 6 ./dataset/binarized/scb-mt-en-th-2020/en-th/spm-newmm_space/16000-130000/ scb-mt-en-th-2020/en-th/spm-newmm_space/16000-130000 9750 150
```

2.4 spm→spm

```bash
bash ./scripts/fairseq_train.transformer_base.single_gpu.fp16 7 ./dataset/binarized/scb-mt-en-th-2020/en-th/spm-spm/32000-joined/ scb-mt-en-th-2020/en-th/spm-spm/32000-joined 9750 150
```



<br/>

## Model Evaluation


### 1. Evaluate models on `scb-mt-en-th-2020` test set. 

The total number of segment pairs is 100,177.

Note: The first argument indicate the ID of GPU. In this case, we train each model on 1 GPU (`GPU_ID`: 0-7).

    
#### 1.1 Evaluate models on th→en direction

1.1.1 newmm→moses

```bash
CUDA_VISIBLE_DEVICES=0 bash ./scripts/evaluate_model.test_set.fp16.sh \
./checkpoints/scb-mt-en-th-2020/th-en/newmm-moses/130000-130000/checkpoint_best.pt \
./dataset/binarized/scb-mt-en-th-2020/th-en/newmm-moses/130000-130000 \
./dataset/tokenized/scb-mt-en-th-2020/th-en/newmm-moses/test.th \
th \
en \
word \
./dataset/split/scb-mt-en-th-2020/test.detok.en \
./translation_results/scb-mt-en-th-2020/th-en/newmm-moses/130000-130000/checkpoint_best \
20000 \
4
```

1.1.2 newmm→spm

```bash
CUDA_VISIBLE_DEVICES=0 bash ./scripts/evaluate_model.test_set.fp16.sh \
./checkpoints/scb-mt-en-th-2020/th-en/newmm-spm/130000-16000/checkpoint_best.pt \
./dataset/binarized/scb-mt-en-th-2020/th-en/newmm-spm/130000-16000 \
./dataset/tokenized/scb-mt-en-th-2020/th-en/newmm-spm/test.th \
th \
en \
sentencepiece \
./dataset/split/scb-mt-en-th-2020/test.detok.en \
./translation_results/scb-mt-en-th-2020/th-en/newmm-spm/130000-16000/checkpoint_best \
20000 \
4
```

1.1.3 spm→moses

```bash
CUDA_VISIBLE_DEVICES=0 bash ./scripts/evaluate_model.test_set.fp16.sh \
./checkpoints/scb-mt-en-th-2020/th-en/spm-moses/16000-130000/checkpoint_best.pt \
./dataset/binarized/scb-mt-en-th-2020/th-en/spm-moses/16000-130000 \
./dataset/tokenized/scb-mt-en-th-2020/th-en/spm-moses/test.th \
th \
en \
word \
./dataset/split/scb-mt-en-th-2020/test.detok.en \
./translation_results/scb-mt-en-th-2020/th-en/spm-moses/16000-130000/checkpoint_best \
20000 \
4
```

1.1.4 spm→spm

```bash
CUDA_VISIBLE_DEVICES=0 bash ./scripts/evaluate_model.test_set.fp16.sh \
./checkpoints/scb-mt-en-th-2020/th-en/spm-spm/32000-joined/checkpoint_best.pt \
./dataset/binarized/scb-mt-en-th-2020/th-en/spm-spm/32000-joined \
./dataset/tokenized/scb-mt-en-th-2020/th-en/spm-spm/test.th \
th \
en \
sentencepiece \
./dataset/split/scb-mt-en-th-2020/test.detok.en \
./translation_results/scb-mt-en-th-2020/th-en/spm-spm/32000-joined/checkpoint_best \
20000 \
4
```


#### 1.2 Evaluate models on en→th direction


1.2.1 moses→newmm_space

```bash
CUDA_VISIBLE_DEVICES=0 bash ./scripts/evaluate_model.test_set.fp16.sh \
./checkpoints/scb-mt-en-th-2020/en-th/moses-newmm_space/130000-130000/checkpoint_best.pt \
./dataset/binarized/scb-mt-en-th-2020/en-th/moses-newmm_space/130000-130000 \
./dataset/tokenized/scb-mt-en-th-2020/en-th/moses-newmm_space/test.en \
en \
th \
word \
./dataset/split/scb-mt-en-th-2020/test.detok.th \
./translation_results/scb-mt-en-th-2020/en-th/moses-newmm_space/130000-130000/checkpoint_best \
20000 \
4    
```

1.2.2 moses→spm

```bash
CUDA_VISIBLE_DEVICES=0 bash ./scripts/evaluate_model.test_set.fp16.sh \
./checkpoints/scb-mt-en-th-2020/en-th/moses-spm/130000-16000/checkpoint_best.pt \
./dataset/binarized/scb-mt-en-th-2020/en-th/moses-spm/130000-16000 \
./dataset/tokenized/scb-mt-en-th-2020/en-th/moses-spm/test.en \
en \
th \
sentencepiece \
./dataset/split/scb-mt-en-th-2020/test.detok.th \
./translation_results/scb-mt-en-th-2020/en-th/moses-spm/130000-16000/checkpoint_best \
20000 \
4
```

1.2.3 spm→newmm_space

```bash
CUDA_VISIBLE_DEVICES=0 bash ./scripts/evaluate_model.test_set.fp16.sh \
./checkpoints/scb-mt-en-th-2020/en-th/spm-newmm_space/16000-130000/checkpoint_best.pt \
./dataset/binarized/scb-mt-en-th-2020/en-th/spm-newmm_space/16000-130000 \
./dataset/tokenized/scb-mt-en-th-2020/en-th/spm-newmm_space/test.en \
en \
th \
word \
./dataset/split/scb-mt-en-th-2020/test.detok.th \
./translation_results/scb-mt-en-th-2020/en-th/spm-newmm_space/16000-130000/checkpoint_best \
20000 \
4
```

1.2.4 spm→spm

```bash
CUDA_VISIBLE_DEVICES=0 bash ./scripts/evaluate_model.test_set.fp16.sh \
./checkpoints/scb-mt-en-th-2020/en-th/spm-spm/32000-joined/checkpoint_best.pt \
./dataset/binarized/scb-mt-en-th-2020/en-th/spm-spm/32000-joined \
./dataset/tokenized/scb-mt-en-th-2020/en-th/spm-spm/test.en \
en \
th \
sentencepiece \
./dataset/split/scb-mt-en-th-2020/test.detok.th \
./translation_results/scb-mt-en-th-2020/en-th/spm-spm/32000-joined/checkpoint_best \
20000 \
4
```


### 2. Evaluate models on Thai-Englush IWSLT 2015 test sets (tst2010-2013). 

The total number of segment pairs is 4,242.

#### 2.1 Evaluate models on th→en direction

2.1.1 newmm→moses

```bash
CUDA_VISIBLE_DEVICES=0 bash ./scripts/evaluate_model.iwslt2015.fp16.sh \
./checkpoints/scb-mt-en-th-2020/th-en/newmm-moses/130000-130000/checkpoint_best.pt \
./dataset/binarized/scb-mt-en-th-2020/th-en/newmm-moses/130000-130000 \
th \
en \
word \
word \
./iwslt_2015/test/tst2010-2013_th-en.th \
./iwslt_2015/test/tst2010-2013_th-en.en \
./translation_results/scb-mt-en-th-2020@eval_on@iwslt2015/th-en/newmm-moses/130000-130000/checkpoint_best \
1000 \
4 
```

2.1.2 newmm→spm

```bash
CUDA_VISIBLE_DEVICES=0 bash ./scripts/evaluate_model.iwslt2015.fp16.sh \
./checkpoints/scb-mt-en-th-2020/th-en/newmm-spm/130000-16000/checkpoint_best.pt \
./dataset/binarized/scb-mt-en-th-2020/th-en/newmm-spm/130000-16000 \
th \
en \
word \
sentencepiece \
./iwslt_2015/test/tst2010-2013_th-en.th \
./iwslt_2015/test/tst2010-2013_th-en.en \
./translation_results/scb-mt-en-th-2020@eval_on@iwslt201/th-en/newmm-spm/130000-16000/checkpoint_best \
20000 \
4
```

2.1.3 spm→moses

```bash
CUDA_VISIBLE_DEVICES=0 bash ./scripts/evaluate_model.iwslt2015.fp16.sh \
./checkpoints/scb-mt-en-th-2020/th-en/spm-moses/16000-130000/checkpoint_best.pt \
./dataset/binarized/scb-mt-en-th-2020/th-en/spm-moses/16000-130000 \
th \
en \
sentencepiece \
word \
./dataset/split/scb-mt-en-th-2020/test.detok.en \
./translation_results/scb-mt-en-th-2020@eval_on@iwslt201/th-en/spm-moses/16000-130000/checkpoint_best \
20000 \
4 \
./dataset/spm/scb-mt-en-th-2020/th-en/spm.th.v-16000.cased.model
```

2.1.4 spm→spm

```bash
CUDA_VISIBLE_DEVICES=0 bash ./scripts/evaluate_model.iwslt2015.fp16.sh \
./checkpoints/scb-mt-en-th-2020/th-en/spm-spm/32000-joined/checkpoint_best.pt \
./dataset/binarized/scb-mt-en-th-2020/th-en/spm-spm/32000-joined \
th \
en \
sentencepiece \
sentencepiece \
./iwslt_2015/test/tst2010-2013_th-en.th \
./iwslt_2015/test/tst2010-2013_th-en.en \
./translation_results/scb-mt-en-th-2020@eval_on@iwslt201/th-en/spm-spm/32000-joined/checkpoint_best \
20000 \
4 \
./dataset/spm/scb-mt-en-th-2020/th-en/spm.th.v-16000.cased.model
```


#### 2.2 Evaluate models on en→th direction


Pretokenize Thai target sentences with PyThaiNLP's **newmm** dictionary-based word tokenizer with the following script.

```
python ./scripts/th_newmm_space_tokenize.py \
./iwslt_2015/test/tst2010-2013_th-en.th \
./iwslt_2015/test/tst2010-2013_th-en.th.ref.tok
```

2.2.1 moses→newmm_space

```bash
CUDA_VISIBLE_DEVICES=0 bash ./scripts/evaluate_model.iwslt2015.fp16.sh \
./checkpoints/scb-mt-en-th-2020/en-th/moses-newmm_space/130000-130000/checkpoint_best.pt \
./dataset/binarized/scb-mt-en-th-2020/en-th/moses-newmm_space/130000-130000 \
en \
th \
word \
word \
./iwslt_2015/test/tst2010-2013_th-en.en \
./iwslt_2015/test/tst2010-2013_th-en.th \
./translation_results/scb-mt-en-th-2020@eval_on@iwslt201/en-th/moses-newmm_space/130000-130000/checkpoint_best \
20000 \
4    
```

2.2.2 moses→spm

```bash
CUDA_VISIBLE_DEVICES=0 bash ./scripts/evaluate_model.iwslt2015.fp16.sh \
./checkpoints/scb-mt-en-th-2020/en-th/moses-spm/130000-16000/checkpoint_best.pt \
./dataset/binarized/scb-mt-en-th-2020/en-th/moses-spm/130000-16000 \
en \
th \
word \
sentencepiece \
./iwslt_2015/test/tst2010-2013_th-en.en \
./iwslt_2015/test/tst2010-2013_th-en.th \
./translation_results/scb-mt-en-th-2020@eval_on@iwslt201/en-th/moses-spm/130000-16000/checkpoint_best \
20000 \
4
```

2.2.3 spm→newmm_space

```bash
CUDA_VISIBLE_DEVICES=0 bash ./scripts/evaluate_model.iwslt2015.fp16.sh \
./checkpoints/scb-mt-en-th-2020/en-th/spm-newmm_space/16000-130000/checkpoint_best.pt \
./dataset/binarized/scb-mt-en-th-2020/en-th/spm-newmm_space/16000-130000 \
en \
th \
sentencepiece \
word \
./iwslt_2015/test/tst2010-2013_th-en.en \
./iwslt_2015/test/tst2010-2013_th-en.th \
./translation_results/scb-mt-en-th-2020@eval_on@iwslt201/en-th/spm-newmm_space/16000-130000/checkpoint_best \
20000 \
4
sentencepiece \
./dataset/spm/scb-mt-en-th-2020/en-th/spm.en.v-16000.cased.model
```

2.2.4 spm→spm

```bash
CUDA_VISIBLE_DEVICES=0 bash ./scripts/evaluate_model.iwslt2015.fp16.sh \
./checkpoints/scb-mt-en-th-2020/en-th/spm-spm/32000-joined/checkpoint_best.pt \
./dataset/binarized/scb-mt-en-th-2020/en-th/spm-spm/32000-joined \
en \
th \
sentencepiece \
sentencepiece \
./iwslt_2015/test/tst2010-2013_th-en.en \
./iwslt_2015/test/tst2010-2013_th-en.th \
./translation_results/scb-mt-en-th-2020@eval_on@iwslt201/en-th/spm-spm/32000-joined/checkpoint_best \
20000 \
4 \
./dataset/spm/scb-mt-en-th-2020/en-th/spm.en.v-16000.cased.model
```
