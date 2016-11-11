#Intro
Code to reproduce experiments from the following paper:

Michal Lukasik, P. K. Srijith, Duy Vu, Kalina Bontcheva, Arkaitz Zubiaga, Trevor Cohn. Hawkes Processes for Continuous Time Sequence Classification: an Application to Rumour Stance Classification in Twitter.
In Proceedings of the 54th annual meeting of the Association for Computational Linguistics, ACL 2015. 

#Dataset
The dataset that the experiments were run on are Twitter rumours from the PHEME datasets
(Zubiaga et al. 2016, Analysing how people orient to and spread rumours in social media by looking at conversational threads. PLoS ONE, 11(3):1–29, 03.).
In the 'data' folder we provide processed data files used for experiments. If you would like to access the raw dataset please contact Zubiaga et al.

Each datafile is in a tab-seperated column format, where rows correspond to tweets. Consecutive columns correspond to the following pieces of information describing a tweet:
* rumourid - a unique identifier describing the rumour that a tweet is part of (manually annotated)
* tweetid - a unique identitifier of a tweet (corresponds to the 'id' field in the json format from Twitter; the ids are mapped to consecutive integer values)
* infectingtweetid - a unique identitifier of an infecting tweet (corresponds to 'in_reply_to_status_id' field in the json format from Twitter, or tweetid if the 'in_reply_to_status_id' field is absent; the ids are mapped to consecutive integer values)
* sdqclabel - a support/deny/question/comment tweet label (manually annotated)
* time - timestamp when a tweet happened (corresponds to the 'created_at' field in the json format from Twitter)
* tokenslen - number of unique tokens in a tweet (extracted from the 'text' field in the json format from Twitter)
* list-of-tokens-and-counts - the rest of the line consists space seperated token-count pairs, where a token-count pair is in format "token:count". E.g. "token1:count1 token2:count2" (extracted from the 'text' field in the json format from Twitter)

#Dependencies
You need to install the following Python libraries: 
* numpy at version 1.9.1 
* matplotlib at version 1.4.2
* scikit-learn at version 0.15.2
* scipy at version 0.15.1

#Running
To reproduce the experiments, run script RUN_ALL.sh

This will run the experiments, generate the resulting text files in appropriate new folders 
and gather the metric values for different settings.

Alternatively, you can access the parallelized version of the experiments by calling submit_dataset_emailres.sh script with appropriate parameters. 
Our parallelization is specific for the Iceberg server from the University of Sheffield. 

The experiments for the Gaussian Process baseline were run using code at github.com/mlukasik/rumour-classification.

#Closing remarks
If you find this code useful, please let us know (m dot lukasik at sheffield dot ac dot uk) and cite our paper.

This work was partially supported by the European Union under Grant Agreement No. 611233 PHEME (http://www.pheme.eu).

