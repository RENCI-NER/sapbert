# Babel SapBERT (RENCI fork of SapBERT)

This repository is a fork of [cambridgeltl/sapbert](https://github.com/cambridgeltl/sapbert), adapted by [RENCI](https://renci.org/) to train and run inference with a SapBERT model on synonym data from [Babel](https://github.com/NCATSTranslator/Babel), the NCATS Biomedical Data Translator identifier-normalization resource. Trained models are available on Hugging Face at [RenaissanceComputing/babel-sapbert-models](https://huggingface.co/collections/RenaissanceComputing/babel-sapbert-models), with each SapBERT model versioned per Babel data release. The table below lists the Babel data release and code release used for each model.

| Model | Babel data release | Code release |
|---|---|---|
| [SapBERT-PubMedBERT-BabelSynonyms-v2025-01-23](https://huggingface.co/RenaissanceComputing/SapBERT-PubMedBERT-BabelSynonyms-v2025-01-23) | [Babel 2025-01-23](https://github.com/NCATSTranslator/Babel/tree/main/releases/2025jan23) | [`v1.0.0`](https://github.com/RENCI-NER/sapbert/releases/tag/v1.0.0) |
| SapBERT-PubMedBERT-BabelSynonyms-v2026-07-22 (upcoming) | [Babel 2026-07-22](https://github.com/NCATSTranslator/Babel/tree/main/releases/2026jul22) | [`v1.0.0`](https://github.com/RENCI-NER/sapbert/releases/tag/v1.0.0) |

## Changes from upstream

**Training data preparation** (`training_data/babel/`)
- `create_training_data.py` converts Babel SapBERT training exports (Anatomy, Cell, Disease, Gene, Protein, DrugChemicalConflated, OrganismTaxon, and other Biolink categories) into SapBERT's `concept_id||name_1||name_2` format. It lowercases and strips names and drops pairs that are identical under case-insensitive comparison. The result is about 26 million synonym pairs for the Babel 2025-01-23 release.
- `create_babel_name_id_types.py` and `concatenate_babel_name_id_types.py` build a single name → CURIE → Biolink type mapping, using canonical/preferred labels, to serve as the inference/lookup dictionary.

**Training** (`train/`, `src/`)
- The data loader reads a directory of training files and is optimized for memory use at this scale.
- `train.py` has more robust error handling and supports loading custom base models.
- `pretrain.sh` holds the configuration used to train the released models. It fine-tunes from `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` for 1 epoch with batch size 256, learning rate 2e-5, max length 25, `ms_loss` with miner (margin 0.2), CLS aggregation, and AMP.

**Inference** (`inference/`)
- `inference_on_babel_data_gpu.py` streams the name-id-type mapping in chunks, generates embeddings on GPU, and writes chunked compressed `.npz` embedding files with matching metadata CSVs.
- Additional scripts support PubMed/MeSH inference, CURIE normalization, prediction comparison, and splitting embedding files.

**Environment**
- A `Dockerfile` based on `nvcr.io/nvidia/nemo:22.01` is included, and `requirements.txt` is updated.

Sections below the divider are the original upstream README and describe the original SapBERT models and workflow.

---

# SapBERT: Self-alignment pretraining for BERT

**\[news | 22 Aug 2021\]** SapBERT is integrated into NVIDIA's deep learning toolkit NeMo as its [entity linking module](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/entity_linking.html) (thank you NVIDIA!). You can play with it in this [google colab](https://colab.research.google.com/github/NVIDIA/NeMo/blob/v1.0.2/tutorials/nlp/Entity_Linking_Medical.ipynb).

--------

This repo holds code, data, and pretrained weights for **(1)** the **SapBERT** model presented in our NAACL 2021 paper: [*Self-Alignment Pretraining for Biomedical Entity Representations*](https://www.aclweb.org/anthology/2021.naacl-main.334.pdf); **(2)** the **cross-lingual SapBERT** and a cross-lingual biomedical entity linking benchmark (**XL-BEL**) proposed in our ACL 2021 paper: [*Learning Domain-Specialised Representations for Cross-Lingual Biomedical Entity Linking*](https://arxiv.org/pdf/2105.14398.pdf).

![front-page-graph](/misc/sapbert_front_graphs_v6.png?raw=true)




## Huggingface Models

### English Models: [\[SapBERT\]](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext) and [\[SapBERT-mean-token\]](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token)
Standard SapBERT as described in [\[Liu et al., NAACL 2021\]](https://www.aclweb.org/anthology/2021.naacl-main.334.pdf). Trained with UMLS 2020AA (English only), using `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` as the base model. For [\[SapBERT\]](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext), use `[CLS]` (before pooler) as the representation of the input; for [\[SapBERT-mean-token\]](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token), use mean-pooling across all tokens.

### Cross-Lingual Models: [\[SapBERT-XLMR\]](https://huggingface.co/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR) and [\[SapBERT-XLMR-large\]](https://huggingface.co/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large)
Cross-lingual SapBERT as described in [\[Liu et al., ACL 2021\]](https://arxiv.org/pdf/2105.14398.pdf). Trained with UMLS 2020AB (all languages), using `xlm-roberta-base`/`xlm-roberta-large` as the base model. Use `[CLS]` (before pooler) as the representation of the input.

## Environment
The code is tested with python 3.8, torch 1.7.0 and huggingface transformers 4.4.2. Please view `requirements.txt` for more details.

## Embedding Extraction with SapBERT

The following script converts a list of strings (entity names) into embeddings.
```python
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel  

tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")  
model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda()

# replace with your own list of entity names
all_names = ["covid-19", "Coronavirus infection", "high fever", "Tumor of posterior wall of oropharynx"] 

bs = 128 # batch size during inference
all_embs = []
for i in tqdm(np.arange(0, len(all_names), bs)):
    toks = tokenizer.batch_encode_plus(all_names[i:i+bs], 
                                       padding="max_length", 
                                       max_length=25, 
                                       truncation=True,
                                       return_tensors="pt")
    toks_cuda = {}
    for k,v in toks.items():
        toks_cuda[k] = v.cuda()
    cls_rep = model(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
    all_embs.append(cls_rep.cpu().detach().numpy())

all_embs = np.concatenate(all_embs, axis=0)
```

Please see [inference/inference_on_snomed.ipynb](https://github.com/cambridgeltl/sapbert/blob/main/inference/inference_on_snomed.ipynb) for a more extensive inference example.

## Train SapBERT
Extract training data from UMLS as insrtructed in `training_data/generate_pretraining_data.ipynb` (we cannot directly release the training file due to licensing issues).

Run:
```bash
>> cd train/
>> ./pretrain.sh 0,1 
```
where `0,1` specifies the GPU devices. 

For finetuning on your customised dataset, generate data in the format of 
```
concept_id || entity_name_1 || entity_name_2
...
```
where `entity_name_1` and `entity_name_2` are synonym pairs (belonging to the same concept `concept_id`) sampled from a given labelled dataset. If one concept is associated with multiple entity names in the dataset, you could traverse all the pairwise combinations.

For cross-lingual SAP-tuning with general domain parallel data (muse, wiki titles, or both), the data can be found in `training_data/general_domain_parallel_data/`. An example script: `train/xling_train.sh`. 

## Evaluate SapBERT
For evaluation (both monlingual and cross-lingual), please view `evaluation/README.md` for details. `evaluation/xl_bel/` contains the XL-BEL benchmark proposed in [\[Liu et al., ACL 2021\]](https://arxiv.org/pdf/2105.14398.pdf).

## Citations
SapBERT: 
```bibtex
@inproceedings{liu2021self,
	title={Self-Alignment Pretraining for Biomedical Entity Representations},
	author={Liu, Fangyu and Shareghi, Ehsan and Meng, Zaiqiao and Basaldella, Marco and Collier, Nigel},
	booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
	pages={4228--4238},
	month = jun,
	year={2021}
}
```
Cross-lingual SapBERT and XL-BEL:
```bibtex
@inproceedings{liu2021learning,
	title={Learning Domain-Specialised Representations for Cross-Lingual Biomedical Entity Linking},
	author={Liu, Fangyu and Vuli{\'c}, Ivan and Korhonen, Anna and Collier, Nigel},
	booktitle={Proceedings of ACL-IJCNLP 2021},
	pages = {565--574},
	month = aug,
	year={2021}
}
```

## Acknowledgement
Parts of the code are modified from [BioSyn](https://github.com/dmis-lab/BioSyn). We appreciate the authors for making BioSyn open-sourced.

## License
SapBERT is MIT licensed. See the [LICENSE](LICENSE) file for details.
