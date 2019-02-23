
# Fine-tuning classifer with BERT
Based on the https://github.com/google-research/bert/blob/master/run_classifier.py

## File Format
Tab seperated value file. It looks like as follows:
<label> <\t> <seq1_token1> <seq1_token2> ... [<\t>] [<seq2_token1>] [<seq2_token2>] ...

Second sequence is optinonal. Only must be specified for sequence pair tasks.

Additionaly, you should contain a label file named label.txt in the data dir,
one line for one label name.


Just run as follows
$ ./oh_my_so
