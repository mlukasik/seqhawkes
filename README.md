#Intro
Code to reproduce experiments from the following paper:

Michal Lukasik, P. K. Srijith, Duy Vu, Kalina Bontcheva, Arkaitz Zubiaga, Trevor Cohn. Hawkes Processes for Continuous Time Sequence Classification: an Application to Rumour Stance Classification in Twitter.
In Proceedings of the 54th annual meeting of the Association for Computational Linguistics, ACL 2015. 

#Dataset
The dataset that the experiments were run on are Twitter rumours from the PHEME datasets
(Zubiaga et al. 2016, Analysing how people orient to and spread rumours in social media by looking at conversational threads. PLoS ONE, 11(3):1–29, 03.).
In folder data, we provide processed data files used for experiments. If you would like to access the raw dataset please contact Zubiaga et al.

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

