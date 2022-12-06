# IndoCloze

## About
Although commonsense reasoning is a key component of natural language understanding (NLU), previous studies have focused predominantly on English, leaving open the question of how the findings generalize to other languages, such as Indonesian. In this paper, we follow the Story Cloze Test framework of Mostafazadeh et al. (2016) in evaluating story understanding in Indonesian, by constructing a four-sentence story with one correct ending and one incorrect ending.

## Paper
Fajri Koto, Timothy Baldwin, and Jey Han Lau. [_Cloze Evaluation for Deeper Understanding of Commonsense Stories in Indonesian_](https://aclanthology.org/2022.csrr-1.2.pdf). 
In In Proceedings of Commonsense Representation and Reasoning Workshop 2022 (**CSRR at ACL 2022**), Dublin, Ireland. **[Best Paper Award]**

## Dataset

A story in our dataset consists of four-sentence premise, one-sentence correct ending, and one-sentence incorrect ending. In total, we have created 2,325 Indonesian stories with the train/dev/test split 1,000/200/1,135. Please see some examples of our data below, and note that the English translation is only for the illustratrive purposes.

<img src="https://github.com/fajri91/eval_picts/blob/master/indocloze.png" width="850">

You can find the Indonesian dataset in `Data/data_id`. In the experiment, we also utilize Mostafazadeh et al. (2016) English datasets. Please contact the authors to obtain this dataset and put them in `Data/data_en`. 

## Experiments

Please first install the requirement by `pip install -r requirements.txt`
* For classification, you can run `classification_*.py` 
* For generation, please check folder `generation/`
