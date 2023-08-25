# syntax_data_preprocessing
Data Preprocessing for syntax-enhanced RE models.

:star: This codebase provides scripts that turn raw WordPiece-tokenized data to ready-to-use data for syntax-enhanced models as described in the paper:
https://academic.oup.com/database/article/doi/10.1093/database/baac070/6675625. Changes are made including replacing BioBERT with PubMedBERT.

There are four models either dependency-syntax-enhanced or constituency-syntax-enhanced:
- CE-PubMedBERT (constituency)
- CT-PubMedBERT (constituency)
- Late-fusion (dependency)
- MTS-PubMedBERT (dependency)

:star: Pre-processing described here takes WordPiece-tokenized data as input. To perform the first tokenization, it suffices to load the PubMedBERT tokenizer:
```
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
```
Refer to the README file in the preprocessing folder for subsequent steps (if you want to use your own corpus).

🔨 You can skip pre-processing and download data for syntax-enhanced models (BB-Rel, ChemProt, DrugProt) here:

You can also generate these data using our pre-processing scripts (Refer to /preprocessing/).
