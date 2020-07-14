# 1. Specify path to NMT model checkpoint

MODEL_PATH=$1

# 2. Specify directory to the source / target dictionary of the NMT model (e.g, `binarized/`)

MODEL_BIN_DIR=$2

# 3. Specify path to the source sentence `tokenized` according to the NMT model dictionary

# if model use word-level token, then tokenized source with word-level tokenizer (spaced-separated) and save as text file
# if model use subword-level token, then tokenized source with BPE tokenizer (e.g. SentencePiece, subword-nmt) (spaced-separated) 

SRC_TEST_FILE_PATH=$3

# 4. Specify source and target lanugae ( `th` or `en`)
SRC_LANG=$4
TGT_LANG=$5

# 5. Specify token type of target language (`sentencenpiece`, 'word')

TGT_TOKEN=$6

# 6. Specify target path of the detokenized test sentences

TGT_DETOK_TEST_FILE_PATH=$7

# 7. Specify directory to store the translation result of the specified model checkpoint (from 1.)

TRANSLATION_RESULT_DIR=$8

# 8. Specify batch size 

BATCH_SIZE=$9

# 9. Specify BEAM WIDTH

BEAM_WIDTH=${10}

mkdir -p $TRANSLATION_RESULT_DIR

echo "Begin translation with max tokens: ${BATCH_SIZE}"

cat $SRC_TEST_FILE_PATH \
    | fairseq-interactive $MODEL_BIN_DIR \
      --task translation \
      --source-lang $SRC_LANG --target-lang $TGT_LANG \
      --path $MODEL_PATH \
      --buffer-size 2500 \
      --fp16 \
      --max-tokens $BATCH_SIZE \
      --beam $BEAM_WIDTH \
    > $TRANSLATION_RESULT_DIR/hypo.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.out

echo "Done translation"

# 3. Select only the hypothesis from the text file generated from fairseq-interactive
#    regex(`^H`) (only line starts with H)

grep ^H $TRANSLATION_RESULT_DIR/hypo.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.out | cut -f3 \
    > $TRANSLATION_RESULT_DIR/hypo.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.sys

# 4. If target langauge is `en`, then perform detokenization with Moses

if [ "$TGT_TOKEN" = "word" ]
then 
    if [ "$TGT_LANG" = "en" ]
    then
        sacremoses detokenize -j 32 -l en < $TRANSLATION_RESULT_DIR/hypo.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.sys > $TRANSLATION_RESULT_DIR/hypo.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.detok.sys

        # 5a. Compute BLEU score with sacrebleu 
        cat $TRANSLATION_RESULT_DIR/hypo.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.detok.sys | sacrebleu $TGT_DETOK_TEST_FILE_PATH

    elif [ "$TGT_LANG" = "th" ]
    then
        # 5a. Compute BLEU score with bleu

        # detokenize with python script
        echo "Detokenized translation result"
        python ./scripts/th_newmm_space_detokenize.py $TRANSLATION_RESULT_DIR/hypo.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.sys $TRANSLATION_RESULT_DIR/hypo.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.detok.sys

        # Tokenize `$TRANSLATION_RESULT_DIR/hypo.$SRC_LANG-$TGT_LANG.$TGT_LANG.detok.sys` and `$TGT_DETOK_TEST_FILE_PATH`
        # with pythainlp CLI
        python ./scripts/th_newmm_space_tokenize.py $TRANSLATION_RESULT_DIR/hypo.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.detok.sys $TRANSLATION_RESULT_DIR/hypo.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.detok.sys.tok
        python ./scripts/th_newmm_space_tokenize.py $TGT_DETOK_TEST_FILE_PATH $TGT_DETOK_TEST_FILE_PATH.tok
        
        fairseq-score --sys $TRANSLATION_RESULT_DIR/hypo.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.detok.sys.tok --ref $TGT_DETOK_TEST_FILE_PATH.tok

    else
        echo "target language is invalid"
    fi
elif [ "$TGT_TOKEN" = "sentencepiece" ]
then
    if [ "$TGT_LANG" = "en" ]
    then
        python ./scripts/remove_bpe.py $TRANSLATION_RESULT_DIR/hypo.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.sys $TRANSLATION_RESULT_DIR/hypo.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.sys.removed_bpe
            
        cat $TRANSLATION_RESULT_DIR/hypo.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.sys.removed_bpe | sacrebleu $TGT_DETOK_TEST_FILE_PATH
    
    elif [ "$TGT_LANG" = "th" ]
    then
        python ./scripts/remove_bpe.py $TRANSLATION_RESULT_DIR/hypo.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.sys $TRANSLATION_RESULT_DIR/hypo.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.sys.removed_bpe         
         
        python ./scripts/th_newmm_space_tokenize.py $TRANSLATION_RESULT_DIR/hypo.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.sys.removed_bpe $TRANSLATION_RESULT_DIR/hypo.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.sys.removed_bpe.tok

        python ./scripts/th_newmm_space_tokenize.py $TGT_DETOK_TEST_FILE_PATH $TGT_DETOK_TEST_FILE_PATH.tok
                
        fairseq-score --sys $TRANSLATION_RESULT_DIR/hypo.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.sys.removed_bpe.tok --ref $TGT_DETOK_TEST_FILE_PATH.tok
    fi
else
    echo "target token type is invalid"
fi
