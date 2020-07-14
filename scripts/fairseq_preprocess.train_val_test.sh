SRC=$1
TGT=$2
SRC_TOK=$3
TGT_TOK=$4
INPUT_FILE=$5
OUT_DIR=$6

OPT=$7

echo "SRC_LANG = '$SRC', TGT_LANG = '$TGT'"
echo "SRC max tokens = '$SRC_TOK', TGT max tokens = '$TGT_TOK'"

echo ""
echo "Preporcess input files from '${INPUT_FILE}'"
echo ""
echo "Craete a directory at '${OUT_DIR}' to stored binarized data"
echo "option = '$OPT'"

mkdir -p $OUT_DIR
echo ''
echo "Start running 'fairseq-preprocess'"

fairseq-preprocess --source-lang $SRC --target-lang $TGT \
    --trainpref $INPUT_FILE/train \
    --validpref $INPUT_FILE/val \
    --testpref $INPUT_FILE/test \
    --destdir $OUT_DIR \
    --workers 30 \
    --nwordssrc $SRC_TOK \
    --nwordstgt $TGT_TOK $OPT

echo "Done."