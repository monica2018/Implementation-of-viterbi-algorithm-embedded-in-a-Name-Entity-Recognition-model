# Viterbi-Algorithm-embedded-in-a-Named-Entity-Recognition-Model
**Implementation of viterbi algorithm using batch computation**

The Viterbi algorithm is a dynamic programming algorithm for finding the most likely sequence of hidden states—called the Viterbi path—that results in a sequence of observed events, especially in the context of Markov information sources and hidden Markov models (HMM). It computes the best sequence (and its score) given the transition and emission scores.

Mehul Gupta offers a great explaination to the Viterbi algorithm in his post <a href="https://medium.com/data-science-in-your-pocket/pos-tagging-using-hidden-markov-models-hmm-viterbi-algorithm-in-nlp-mathematics-explained-d43ca89347c4">POS Tagging using Hidden Markov Models (HMM) & Viterbi algorithm in NLP mathematics explained</a>.

I've implemented a vitervi algorithm using **batch computation** from scratch, which avoids nested loop and makes the implementation much faster. Have a look at the script <em>viterbi.py</em> and feel free to play with it or test it using <em>viterbi_test.py</em>. 
