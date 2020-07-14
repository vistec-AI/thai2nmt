MODEL_DIR=$1
N_CKP=$2

echo "Perform model averging from checkpoint in the directory: '$MODEL_DIR'"
echo "Number of last N checkpoints is '$N_CKP'"

python ../fairseq/scripts/average_checkpoints.py \
    --input $MODEL_DIR \
    --output $MODEL_DIR/checkpoint_avg.last-$N_CKP.pt \
    --num-epoch-checkpoints $N_CKP

echo "Writing the averaged model checkpoint to: '$MODEL_DIR/checkpoint_avg.last-$N_CKP.pt'"
echo "Done."