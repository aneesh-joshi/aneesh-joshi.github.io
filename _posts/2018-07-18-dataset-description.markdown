---
layout: post
title:  "Datasets for Similarity Learning"
date:   2018-07-18 19:24:25 +1924
categories: jekyll update
---

# Dataset Description for Similarity Learning

<table>
<thead>
<tr>
<th>Dataset Name</th>
<th>Link</th>
<th>Suggested Metrics</th>
<th>Some Papers that use the dataset</th>
<th>Brief Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>WikiQA</td>
<td><ul><li><a href="https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip" rel="nofollow">Dataset</a></li><li><a href="https://aclweb.org/anthology/D15-1237" rel="nofollow">Paper</a></li></ul></td>
<td><ul><li>MAP</li><li>MRR</li></ul></td>
<td><ul><li>SeqMatchSeq(<code>MULT</code> MAP=0.74, MRR=0.75)</li><li>BiMPM(MAP=0.71, MRR=0.73)</li><li>QA-Transfer(<code>SQUAD*</code> MAP=0.83, MRR=84.58, P@1=75.31)</li></ul></td>
<td>Question-Candidate_Answer1_to_N-Relevance1_to_N</td>
</tr>
<tr>
<td>SQUAD 2.0</td>
<td><a href="https://rajpurkar.github.io/SQuAD-explorer/" rel="nofollow">Website</a></td>
<td><ul><li>Exact Match</li><li>F1</li></ul></td>
<td>QA-Transfer(for pretraining)</td>
<td>Question-Context-Answer_Range_in_context</td>
</tr>
<tr>
<td>Quora Duplicate Question Pairs</td>
<td><ul><li>gensim-data(quora-duplicate-questions)</li><li><a href="https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs" rel="nofollow">Quora Official</a></li><li><a href="https://www.kaggle.com/c/quora-question-pairs" rel="nofollow">Kaggle</a></li></ul></td>
<td>Accuracy, F1</td>
<td><ul><li>BiMPM(Acc=88.17%)</li><ul></ul></ul></td>
<td>Q1-Q2-DuplicateProbablity</td>
</tr>
<tr>
<td>Sem Eval 2016 Task 3A</td>
<td>genism-data(semeval-2016-2017-task3-subtaskA-unannotated)</td>
<td><ul><li>MAP</li><li>AvgRecall</li><li>MRR</li><li>P</li><li>R</li><li>F1</li><li>Acc</li></ul></td>
<td>QA-Transfer(<code>SQUAD*</code> MAP=80.2, MRR=86.4, P@1=89.1)</td>
<td>Question-Comment-SimilarityProbablity</td>
</tr>
<tr>
<td>MovieQA</td>
<td><ul><li><a href="http://movieqa.cs.toronto.edu/static/files/CVPR2016_MovieQA.pdf" rel="nofollow">Paper</a></li><li><a href="http://movieqa.cs.toronto.edu/home/" rel="nofollow">Website</a></li></ul></td>
<td>Accuracy</td>
<td>SeqMatchSeq(<code>SUBMULT+NN</code> test=72.9%, dev=72.1%)</td>
<td>Plot-Question-Candidate_Answers</td>
</tr>
<tr>
<td>InsuranceQA</td>
<td><a href="https://github.com/shuzi/insuranceQA">Website</a></td>
<td>Accuracy</td>
<td>SeqMatchSeq(<code>SUBMULT+NN</code> test1=75.6%, test2=72.3%, dev=77%)</td>
<td>Question-Ground_Truth_Answer-Candidate_answer</td>
</tr>
<tr>
<td>SNLI</td>
<td><ul><li><a href="https://nlp.stanford.edu/pubs/snli_paper.pdf" rel="nofollow">Paper</a></li><li><a href="https://nlp.stanford.edu/projects/snli/" rel="nofollow">Website</a></li></ul></td>
<td>Accuracy</td>
<td><ul><li>QA-Transfer(for pretraining)</li><li>SeqMatchSeq(<code>SUBMULT+NN</code> train=89.4%, test=86.8%)</li><li>BiMPM()<code>Ensemble</code> Acc=88.8%)</li></ul></td>
<td>Text-Hypothesis-Judgement</td>
</tr>
<tr>
<td>TRECQA</td>
<td><ul><li><a href="https://aclweb.org/aclwiki/Question_Answering_(State_of_the_art)" rel="nofollow">https://aclweb.org/aclwiki/Question_Answering_(State_of_the_art)</a></li><li><a href="https://github.com/castorini/data/tree/master/TrecQA">https://github.com/castorini/data/tree/master/TrecQA</a></li><li><a href="http://cs.jhu.edu/%7Exuchen/packages/jacana-qa-naacl2013-data-results.tar.bz2" rel="nofollow">http://cs.jhu.edu/~xuchen/packages/jacana-qa-naacl2013-data-results.tar.bz2</a></li></ul></td>
<td><ul><li>MAP</li><li>MRR</li></ul></td>
<td>BiMPM(MAP:0.802, MRR:0.875)</td>
<td>Question-Candidate_Answer1_to_N-relevance1_to_N</td>
</tr>
<tr>
<td>SICK</td>
<td><a href="http://clic.cimec.unitn.it/composes/sick.html" rel="nofollow">Website</a></td>
<td>Accuracy</td>
<td>QA-Transfer(Acc=88.2)</td>
<td>sent1-sent2-entailment_label-relatedness_score</td>
</tr></tbody></table>



More dataset info can be found at the [SentEval](https://github.com/facebookresearch/SentEval) repo.

Links to Papers:
- [SeqMatchSeq](https://arxiv.org/pdf/1611.01747.pdf)
- [QA-Transfer](http://aclweb.org/anthology/P17-2081)
- [BiMPM](https://arxiv.org/pdf/1702.03814.pdf)

## Some useful examples

### SQUAD
![SQUAD](https://raw.githubusercontent.com/aneesh-joshi/aneesh-joshi.github.io/master/_posts/images/squad.png)

### SQUAD, WikiQA, SemEval, SICK
![alt](https://raw.githubusercontent.com/aneesh-joshi/aneesh-joshi.github.io/master/_posts/images/dataset_description.png)

### MovieQA and InsuranceQA
![alt](https://raw.githubusercontent.com/aneesh-joshi/aneesh-joshi.github.io/master/_posts/images/MovieQA&InsuranceQA.png)
