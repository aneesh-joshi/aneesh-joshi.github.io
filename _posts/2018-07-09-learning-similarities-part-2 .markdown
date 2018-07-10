---
layout: post
title:  "Learning Similarities - Part 2"
date:   2018-07-09 18:21:25 +0730
categories: jekyll update
---

<p>I am writing this blog to chronicle my activities and progress as I go about my project for Google Summmer of Code(GSOC), 2018.</p>

So, it's been almost a month since my last update and now I have a clearer idea of my project. GSOC has been a great learning experience for me. I have majorly developed myself and the project in two domains:

<h2>Software Development:</h2>
<p>Gensim isn't only a research repository, it is an engineering toolbox whose code adheres to the best practices and any code change requires careful deliberation before merging. Any proposed change should be justified in its usefulness compared to the cost.</p>
<p>Being an undergrad student, I was used to writing flimsy code which got the job slopily done. In the past few weeks, with the guidance of my mentors, I made several changes in my code to make it worthy of the gensim standard. These changes primarily included:</p>
<ol>
1. doctests
2. doctrings
3. Code refactoring
4. Streamable inputs
5. gensim-data integration
</ol>

<p>Initially, it felt like an unnecessary burden for my project but as I wrote more test cases, I noticed bugs in my code and failures at edge cases. Writing docstrings felt important but not that important; my docstring lacked the needed rigour. But I got used to it over time. Recently, I was reading someone else's code and I got irritated seeing the lack of docstrings. "What is this varaible?! is it a list or a list of lists! ugh!" I can only imagine Ivan's irritation. I realised the truth in the PEP8's statement : "Code is read more times than it is written"</p>

<h2>Research</h2>
<p>After a certain point, the codebase had reached an acceptable place. Now it was time to work on parameter tuning to get the best possible model. From our initial evaluation, we had found the DRMM_TKS model to be the best and it had already been implemented. Now, I had to tune it. This was probably the most annoying and difficult part of the project. It was made worse because:</p>

1. It takes almost 2 minutes to train on one epoch.
2. The word embedding matrix for 300 dimensions is almost a GB in size (uncompressed) and almost fills the RAM completely.

<p>These two points meant I had to wait a minute before all the data was loaded into memory and the data preprocessing was done. Once that was over, my RAM was fully siezed, it's previous contents pushed into the SWAP space. Then I wait while fiddling with my phone nervously checking the performance metrics hoping that the validation score improves considerably. Once training was complete, I had to wait till RAM was cleared and the SWAP was unloaded. Add a HDD with say 40 MB per second read/write and you get a very slowly iterating process of parameter tuning</p>
<p>Luckily, towards the later stages I discovered [Google Colab](https://colab.research.google.com/) which allows free training on GPUs (with certain limits)</p>
<p>While tuning the models, our goal was to beat the Word2Vec baseline (each document is an average of its word's vectors). It was made harder because Word2Vec is so good by itself. For justifying a collecting data for a supervised model, the model should perform a good deal better. Going primarily on the MAP score, Word2Vec scored 0.57 on the WikiQA dataset, which is close to some other supervised models.</p>
<p>After extensive training, I got a model which **consistently** beat the Word2Vec baseline. But the results weren't **too great** :(</p>
<img src="https://raw.githubusercontent.com/aneesh-joshi/aneesh-joshi.github.io/master/_posts/images/ranged%20benchmarks%20mz.PNG" />
<p>Alas, the model we made is better but not so much better. This still needs more validation but the results probably won't change so much considering the highest ever recordrded WikiQA score is 0.65 and my model gets 0.62. Even if I manged to get 0.67, which is a 0.1 improvement, it wouldn't be considerably more to justify practical usage</p>

<h2>Future Work</h2>
There still needs to be some more validation on the above results which is what I am working on for the next few days. I will update this post with more details and the code as soon as possible. Now, it remains to be decided whether we should publish the code with the incremental improvement so it can help a few people or shelve this project and work on something which can be merged in the next few weeks.

<h2>Conclusion</h2>
Research is hard. It won't always lead to positive results. Sometimes a negative result is an important result as well. Edison went through 1600 metals before finding the right one. I have to admit, I am bit disappointed but that is the way of science and things. Hopefully my **reoproducible** results will help someone else down this path.