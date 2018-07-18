---
layout: post
title:  "Datasets for Similarity Learning"
date:   2018-07-18 19:24:25 +1924
categories: jekyll update
---

# Dataset Description for Similarity Learning

Dataset Name | Link | Suggested Metrics | Some Papers that use the dataset | Brief Description
-- | -- | -- | -- | --
WikiQA | <ul><li>[Dataset](https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip)</li><li>[Paper](https://aclweb.org/anthology/D15-1237)</li></ul> | <ul><li>MAP</li><li>MRR</li></ul> | <ul><li>SeqMatchSeq</li><li>BiMPM</li><li>QA-Transfer</li></ul> | Question-Candidate_Answer1_to_N-Relevance1_to_N
SQUAD 2.0 | [Website](https://rajpurkar.github.io/SQuAD-explorer/) | <ul><li>Exact Match</li><li>F1</li></ul> | QA-Transfer | Question-Context-Answer_Range_in_context
Quora Duplicate Question Pairs | <ul><li>gensim-data(quora-duplicate-questions)</li><li>[Quora Official](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)</li><li>[Kaggle](https://www.kaggle.com/c/quora-question-pairs)</li></ul> | Accuracy, F1 | BiMPM(88%), | Q1-Q2-DuplicateProbablity
Sem Eval 2016 Task 3A | genism-data(semeval-2016-2017-task3-subtaskA-unannotated) | <ul><li>MAP</li><li>AvgRecall</li><li>MRR</li><li>P</li><li>R</li><li>F1</li><li>Acc</li> | QA-Transfer | Question-Comment-SimilarityProbablity
MovieQA | <ul><li>[Paper](http://movieqa.cs.toronto.edu/static/files/CVPR2016_MovieQA.pdf)</li><li>[Website](http://movieqa.cs.toronto.edu/home/)</li></ul> | Accuracy | QA-Transfer | Plot-Question-Candidate_Answers
InsuranceQA | [Website](https://github.com/shuzi/insuranceQA) | Accuracy | QA-Transfer | Question-Ground_Truth_Answer-Candidate_answer
SNLI | <ul><li>[Paper](https://nlp.stanford.edu/pubs/snli_paper.pdf)</li><li>[Website](https://nlp.stanford.edu/projects/snli/)</li></ul> | Accuracy | QA-Transfer | Text-Hypothesis-Judgement
TRECQA | https://aclweb.org/aclwiki/Question_Answering_(State_of_the_art), https://github.com/castorini/data/tree/master/TrecQA, http://cs.jhu.edu/~xuchen/packages/jacana-qa-naacl2013-data-results.tar.bz2 | <ul><li>MAP</li><li>MRR</li></ul> | BiMPM(MAP:0.802, MRR:0.875) | Question-Candidate_Answer1_to_N-relevance1_to_N
SICK | [Website](http://clic.cimec.unitn.it/composes/sick.html) | Accuracy | QA-Transfer | sent1-sent2-entailment_label-relatedness_score


More dataset info can be found at the [SentEval](https://github.com/facebookresearch/SentEval) repo.


Model/Paper | Link
----------- | ----
SeqMatchSeq | https://arxiv.org/pdf/1611.01747.pdf
QA-Transfer | http://aclweb.org/anthology/P17-2081
BiMPM | https://arxiv.org/pdf/1702.03814.pdf

## Some useful examples

### SQUAD
![SQUAD](https://raw.githubusercontent.com/aneesh-joshi/aneesh-joshi.github.io/master/_posts/images/squad.png)

### SQUAD, WikiQA, SemEval, SICK
![alt](https://raw.githubusercontent.com/aneesh-joshi/aneesh-joshi.github.io/master/_posts/images/dataset_description.png)

### MovieQA and InsuranceQA
![alt](https://raw.githubusercontent.com/aneesh-joshi/aneesh-joshi.github.io/master/_posts/images/MovieQA&InsuranceQA.png)
