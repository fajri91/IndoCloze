# IndoCloze

## About
Although commonsense reasoning is a key component of natural language understanding (NLU), previous studies have focused predominantly on English, leaving open the question of how the findings generalize to other languages, such as Indonesian. In this paper, we follow the Story Cloze Test framework of Mostafazadeh et al. (2016) in evaluating story understanding in Indonesian, by constructing a four-sentence story with one correct ending and one incorrect ending.

## Paper
Fajri Koto, Timothy Baldwin, and Jey Han Lau. [_Cloze Evaluation for Deeper Understanding of Commonsense Stories in Indonesian_](https://aclanthology.org/2022.csrr-1.2.pdf). 
In In Proceedings of Commonsense Representation and Reasoning Workshop 2022 (**CSRR at ACL 2022**), Dublin, Ireland. **[Best Paper Award]**

## Dataset

We examine content-based summarization evaluation from the aspects of precision and recall, in the form of focus and coverage to 
compare system-generated summaries to groundtruth summaries.

<img src="https://github.com/fajri91/eval_picts/blob/master/indocloze.png.png" width="400">

## MTurk 

* We use the customized [direct assessment](https://github.com/ygraham/direct-assessment) method for annotation.
* We use Amazon Mechanical Turk for annotation. You can find the MTurk user interface at `mturk/html`.
* Jupyter notebooks number 0-3 are used to pre- and post-process the MTurk annotation
* In this repository, we only provide annotation process for ID, FR, TR, ZH, RU, DE, ES. Annotation process for EN will be released seperately
because the data is from the [FFCI paper](https://arxiv.org/pdf/2011.13662.pdf).

## Data (Annotation Result)

* You can find **all** annotation result in folder `resulting_data`.
* The provided scores are the normalized z-score.

## Traditional metrics (ROUGE, METEOR, BLEU)

* You can use jupyter notebooks number 5-6 to compute traditional metrics and its Pearson and Spearman correlations.
* Please note that for ZH and RU you need to convert all character/word to its latin form by using jupyter notebook `5. transform_RU_and_ZH.ipynb`.

## BERTScore and MoverScore

* We already provide all of the output of BERTScore and MoverScore in folder `bert_score` and `mover_score`, respectively
* You can use jupyter notebook number 7-8 to compute its Pearson and Spearman correlations.
