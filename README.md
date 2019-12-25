# TD-based analysis on ADRs

This repo is for the paper "An analysis of adverse drug reactions identified in nursing notes using reinforcement learning." This is only for review or providing details to readers.

The contents are as follows:
* Additional details for the paper are in addpendix.md
* The evaluation codes used for the experiments (Table 4)

Experimental method | Execution file |
---- | ---- |
TD-based logistic regression | eval_TD_model.py  |
Naive Bayes | eval_NB_model.py |
SVM_Linear, SVM_RBF | eval_svm_model.py |
Text-CNN | eval_cnn_model.py |
Text-CNN with pretrained embedding | eval_cnn_model_embedding.py |
LSTM | eval_LSTM_model.py |
LSTM with pretrained embedding | eval_LSTM_model_embedding.py |  

The implementation code we used for Text-CNN can be found in this [repo](https://github.com/dennybritz/cnn-text-classification-tf).
