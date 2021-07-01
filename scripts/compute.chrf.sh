HYPO_SYS_PATH=$1
REF_PATH=$2

CHRF_BETA=${3:-3}
CHRF_CHAR_ORDER=${4:-6}


echo "Hypothesis file path is $HYPO_SYS_PATH"
echo "Reference file path is $REF_PATH"
echo "CHRF_BETA: $CHRF_BETA"
echo "CHRF_CHAR_ORDER: $CHRF_CHAR_ORDER"


sed 's/<unk>//g' $HYPO_SYS_PATH > ${HYPO_SYS_PATH}.replace_unk

# Lowercase
sed 's/.*/\L&/g' $HYPO_SYS_PATH.replace_unk > $HYPO_SYS_PATH.replace_unk.uncased
sed 's/.*/\L&/g' $REF_PATH > $REF_PATH.uncased


echo "[Cased chrF]"

python ${PWD}/scripts/chrF++.py \
--reference $REF_PATH \
--hypothesis ${HYPO_SYS_PATH}.replace_unk \
--beta $CHRF_BETA \
--ncorder $CHRF_CHAR_ORDER \
--nworder 0

echo ""
echo "[Uncased chrF]"

python ${PWD}/scripts/chrF++.py \
--reference $REF_PATH.uncased \
--hypothesis ${HYPO_SYS_PATH}.replace_unk.uncased \
--beta $CHRF_BETA \
--ncorder $CHRF_CHAR_ORDER \
--nworder 0
