# Semi-Automated Grading of Free-Text Computer Science Answers

Please find the research paper here:
[Semi-Automated Grading System - paper](https://github.com/MZSFighters/auto-grading-of-computer-science-answers/raw/main/2433079.pdf)

**Very Important Note:** The hyperparameter tuning graphs, which were omitted from the paper's Appendix due to LaTeX image file limitations, are available in this repository. You can find them in the Results/Sampling folder. There exists multiple folders for different parameters that were fine-tuned.

The code and results for each component of the semi-automated pipeline are organised into their respective folders:
- Fine-tuning (Embeddings)
- Clustering
- Sampling

![Semi-automated Grading System Pipeline](https://github.com/MZSFighters/auto-grading-of-computer-science-answers/raw/main/system_pipeline.png)

Most of the code can be executed using the provided Python files, as long as the file paths are correctly maintained within the existing folder structure.

**Note:** Some parts of the code cannot be run because the bert_epoch_48 and gpt2_epoch_50 models are not included in this repository. This is due to GitHub's file size limitations, as the model tensors are too large to store here.
