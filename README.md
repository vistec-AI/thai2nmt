# thai2nmt: English-Thai Machine Translation Models

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/vistec-AI/thai2nmt/blob/master/LICENSE)


This repository includes code to reproduce our experiments on Thai-English NMT models and scripts to download the datasets (`scb-mt-en-th-2020`, `mt-opus` and `scb-mt-en-th-2020+mt-opus`) along with the train/validation/test split that we used in the experiments. 

Our experiments are listed below.

- [Experiment #1 TBASE.SCB-1M](./experiments/TBASE.SCB-1M.md) -- Transformer BASE models trained on [scb-mt-en-th-2020 v1.0](https://github.com/vistec-AI/dataset-releases/releases/tag/scb-mt-en-th-2020_v1.0)
  

- [Experiment #2 TBASE.MT-OPUS](./experiments/TBASE.MT-OPUS.md) -- Transformer BASE models trained on English-Thai datasets listed in [Open Parallel Corpus (OPUS)](http://opus.nlpl.eu/)
  

- [Experiment #3 TBASE.SCB-1M+MT-OPUS](./experiments/TBASE.SCB-1M+MT-OPUS.md) -- Transformer BASE models trained on English-Thai [scb-mt-en-th-2020 v1.0](https://github.com/vistec-AI/dataset-releases/releases/tag/scb-mt-en-th-2020_v1.0) and datasets listed in [Open Parallel Corpus (OPUS)](http://opus.nlpl.eu/)
  

<br>
<br>

BibTeX entry and citation info

```text
@Article{Lowphansirikul2021,
    author={Lowphansirikul, Lalita
            and Polpanumas, Charin
            and Rutherford, Attapol T.
            and Nutanong, Sarana},
    title={A large English--Thai parallel corpus from the web and machine-generated text},
    journal={Language Resources and Evaluation},
    year={2021},
    month={Mar},
    day={30},
    issn={1574-0218},
    doi={10.1007/s10579-021-09536-6},
    url={https://doi.org/10.1007/s10579-021-09536-6}
```
