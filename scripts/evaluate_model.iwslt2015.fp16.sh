# 1. Specify path to NMT model checkpoint

MODEL_PATH=$1

# 2. Specify directory to the source / target dictionary of the NMT model (e.g, `binarized/`)

MODEL_BIN_DIR=$2

# 3. Specify source and target lanugae ( `th` or `en`)

SRC_LANG=$3
TGT_LANG=$4

# 4. Specify token type of target language (`sentencenpiece`, 'word')

SRC_TOKEN=$5
TGT_TOKEN=$6

# 5. Specify target path of the detokenized source and reference test sentences

TEST_SRC_PATH=$7
TEST_REF_PATH=$8

# 6. Specify directory to store the translation result of the specified model checkpoint (from 1.)

TRANSLATION_RESULT_DIR=$9

# 7. Specify batch size 

aarMAX_SENTENCES=${10}

# 8. Specify BEAM WIDTH

BEAM_WIDTH=${11}

# 9. Specify vocab size of BPE tokens in case of SRC_TOKEN is `sentencepiece`

SRC_BPE_MODEL_PATH=${12}

mkdir -p $TRANSLATION_RESULT_DIR

if [ "$SRC_TOKEN" = "sentencepiece" ]
then
    echo "SPM encode from spm model ${SRC_BPE_MODEL_PATH}"
    
    spm_encode --model=$SRC_BPE_MODEL_PATH --output_format=piece < $TEST_SRC_PATH > $TEST_SRC_PATH.src.$SRC_LANG.$SRC_TOKEN.tok

fi

if [ "$SRC_TOKEN" = "word" ]
then
    if [ "$SRC_LANG" = "th" ]
    then
        python ./scripts/th_newmm_space_tokenize.py $TEST_SRC_PATH $TEST_SRC_PATH.src.$SRC_LANG.$SRC_TOKEN.tok
    fi

    if [ "$SRC_LANG" = "en" ]
    then
        python ./scripts/en_moses_tokenize.py $TEST_SRC_PATH $TEST_SRC_PATH.src.$SRC_LANG.$SRC_TOKEN.tok
        
    fi

fi

echo "Begin translation,  max sentences per mini-batch: ${MAX_SENTENCES}"
if [ "$TGT_TOKEN" = "sentencepiece" ]
then
    cat $TEST_SRC_PATH.src.$SRC_LANG.$SRC_TOKEN.tok \
    | fairseq-interactive $MODEL_BIN_DIR \
        --task translation \
        --source-lang $SRC_LANG --target-lang $TGT_LANG \
        --path $MODEL_PATH \
        --buffer-size 128 \
        --fp16 \
        --max-sentences $MAX_SENTENCES \
        --beam $BEAM_WIDTH --remove-bpe=sentencepiece \
        > $TRANSLATION_RESULT_DIR/hypo.IWSLT2015.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.out
else
    cat $TEST_SRC_PATH.src.$SRC_LANG.$SRC_TOKEN.tok \
    | fairseq-interactive $MODEL_BIN_DIR \
        --task translation \
        --source-lang $SRC_LANG --target-lang $TGT_LANG \
        --path $MODEL_PATH \
        --buffer-size 128 \
        --fp16 \
        --max-sentences $MAX_SENTENCES \
        --beam $BEAM_WIDTH \
        > $TRANSLATION_RESULT_DIR/hypo.IWSLT2015.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.out
fi
echo "Done translation"

# 3. Select only the hypothesis from the text file generated from fairseq-interactive
#    regex(`^H`) (only line starts with H)


grep ^H $TRANSLATION_RESULT_DIR/hypo.IWSLT2015.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.out | cut -f3 \
    > $TRANSLATION_RESULT_DIR/hypo.IWSLT2015.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.sys

# 4. If target langauge is `en`, then perform detokenization with Moses

if [ "$TGT_TOKEN" = "word" ]
then
    if [ "$TGT_LANG" = "en" ]
    then
        sacremoses detokenize -j 32 -l en < $TRANSLATION_RESULT_DIR/hypo.IWSLT2015.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.sys > $TRANSLATION_RESULT_DIR/hypo.IWSLT2015.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.detok.sys

        # 5a. Compute BLEU score with sacrebleu 
        cat $TRANSLATION_RESULT_DIR/hypo.IWSLT2015.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.detok.sys | sacrebleu $TEST_REF_PATH

    elif [ "$TGT_LANG" = "th" ]
    then
        # 5b. Compute BLEU score with bleu

        # detokenize with python script
        echo "Detokenized translation result"
        python ./scripts/th_newmm_space_detokenize.py $TRANSLATION_RESULT_DIR/hypo.IWSLT2015.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.sys $TRANSLATION_RESULT_DIR/hypo.IWSLT2015.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.sys.detok

        # Tokenize `$TRANSLATION_RESULT_DIR/hypo.$SRC_LANG-$TGT_LANG.$TGT_LANG.detok.sys` and `$TGT_DETOK_TEST_FILE_PATH`
        # with pythainlp CLI
        python ./scripts/th_newmm_space_tokenize.py $TRANSLATION_RESULT_DIR/hypo.IWSLT2015.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.sys.detok $TRANSLATION_RESULT_DIR/hypo.IWSLT2015.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.sys.detok.tok
        
        fairseq-score --sys $TRANSLATION_RESULT_DIR/hypo.IWSLT2015.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.sys.detok.tok --ref $TEST_REF_PATH.ref.tok
    else
        echo "target language is invalid"
    fi
elif [ "$TGT_TOKEN" = "sentencepiece" ]
then
    if [ "$TGT_LANG" = "en" ]
    then

        cat $TRANSLATION_RESULT_DIR/hypo.IWSLT2015.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.sys | sacrebleu $TEST_REF_PATH
    
    elif [ "$TGT_LANG" = "th" ]
    then
         
        python ./scripts/th_newmm_space_tokenize.py $TRANSLATION_RESULT_DIR/hypo.IWSLT2015.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.sys $TRANSLATION_RESULT_DIR/hypo.IWSLT2015.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.sys.tok
                
        fairseq-score --sys $TRANSLATION_RESULT_DIR/hypo.IWSLT2015.beam-$BEAM_WIDTH.$SRC_LANG-$TGT_LANG.$TGT_LANG.sys.tok --ref $TEST_REF_PATH.ref.tok
    fi
else
    echo "target token type is invalid"
fi
