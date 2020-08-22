## Propensity score matching for feature selection

This script calculates significance scores for text features using the method described in:

Michael J. Paul. [Feature Selection as Causal Inference: Experiments with Text Classification](https://www.aclweb.org/anthology/K17-1018/). 21st Conference on Computational Natural Language Learning (CoNLL 2017). Vancouver, August 2017.

### Input format

The input should be a text file containing one document per line. On each line, the first token should be a binary integer label (0 or 1). The remaining tokens are the word tokens of the document. The whitespace-separated string tokens will be read as-is, so any preprocessing like punctuation removal and lowercasing should be done before using this script.

### Output format

The output will be written to a file with the same name as the input, with ".out" appended to the filename. Each line of the file contains a word followed by the log of the p-value calculated by the script. The words are sorted by their log-p-values, where lower values (i.e., more negative) indicate higher significance.

### Running the script

The script takes three command line arguments. The first is the name of the input file. The second is the regularization parameter, $\lambda$ in the paper. I recommend a value of $1$. The third is the threshold for matching, $\tau$ in the paper. A very high value like $100000$ is functionally as if there is no threshold. 

The command to run the script will thus look something like:

`python propensity.py myfile.txt 1.0 100000`

and the output in this example will be written to `myfile.txt.out`.

This thing is quite slow to run, and it doesn't scale to large numbers of features. For bag of words experiments, I prune the vocabulary so that the size is only a few thousand word types. Improving the efficiency is something that will help make this more useful.
