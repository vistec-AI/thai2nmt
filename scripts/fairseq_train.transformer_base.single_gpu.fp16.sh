GPU_ID=$1
BINARIZED_DIR=$2
OUT_DIR=$3
MAX_TOKEN=$4
MAX_EPOCH=$5

echo "GPU ID: '$GPU_ID'"
echo "MAX_TOKENS: '$MAX_TOKEN'"
echo "Binarized data directory: '$BINARIZED_DIR'"
echo "Output directory to store model checkpoints:  '$OUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU_ID fairseq-train $BINARIZED_DIR \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens $MAX_TOKEN \
    --save-dir ./checkpoints/$OUT_DIR \
    --tensorboard-logdir ./checkpoints/$OUT_DIR/log \
    --update-freq 16 \
    --fp16 \
    --keep-last-epochs 25 \
    --max-epoch $MAX_EPOCH \
    --num-workers 0
