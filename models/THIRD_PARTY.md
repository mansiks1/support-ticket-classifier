# Artifact attribution

DistilBERT base weights and tokenizer: Hugging Face / DistilBERT authors,
https://huggingface.co/distilbert/distilbert-base-uncased,
revision 12040accade4e8a0f71eabdb258fecc2e7e948be, Apache License 2.0.
See LICENSE-APACHE-2.0.txt. Modification: fine-tuned 77-class head and encoder
on BANKING77 with two epochs; weights and calibration differ from upstream.

Training data: BANKING77, Iñigo Casanueva, Tadas Temcinas, Daniela Gerz,
Matthew Henderson and Ivan Vulić (2020), Efficient Intent Detection with Dual
Sentence Encoders, https://arxiv.org/abs/2003.04807,
https://github.com/PolyAI-LDN/task-specific-datasets, CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/.
Changes: deduplication, development splitting and model training. No endorsement
by the original authors is implied. Raw dataset is not bundled.

Original service code is MIT. MIT does not relicense the source data or base model.
