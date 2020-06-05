import numpy as np
import copy
def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.
    B - batch size
    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an BxNxL array
    - Transition scores (Yp -> Yc), as an BxLxL array
    - Start transition scores (S -> Y), as an BxLx1 array
    - End transition scores (Y -> E), as an BxLx1 array

    You have to return a tuple (scores, y_pred), where:
    - scores (B): scores of the best predicted sequences
    - y_pred (BxN): an list of list of integers representing the best sequence.
    """
    B = start_scores.shape[0]
    L = start_scores.shape[1]
    N = emission_scores.shape[1]
    
    assert end_scores.shape[1] == L
    assert trans_scores.shape[1] == L
    assert trans_scores.shape[2] == L
    assert emission_scores.shape[2] == L
    
    # Reshape the input arrays to make sure that the array shapes are as desired
    emission_scores = emission_scores.reshape(B,N,L)
    trans_scores = trans_scores.reshape(B,L,L)
    start_scores = start_scores.reshape(B,L,1)
    end_scores = end_scores.reshape(B,L,1)
    
    # score set to 0
    scores = np.zeros(B)
    y_pred = np.zeros((B, N), dtype=int)

    # Creat matrix to store max values of each transition for each label, as well as their backpointers
    inter = np.zeros((B, L, N))
    backpointers = np.zeros((B, L, N-1))
    
    inter[:,:,0]=(emission_scores[:,0,:].reshape(B,L,1)+start_scores).reshape(B,L)

    # update the inter matrix using broadcasting following the updating rules
    for i in range(1,N):
        mltway = inter[:,:,i-1].reshape(B,1,L)+np.transpose(trans_scores,(0,2,1))+emission_scores[:,i,:].reshape(B,L,1)
        inter[:,:,i] = np.max(mltway, axis = 2) # Take the row-wise largest values and update the inter matrix
        backpointers[:,:,i-1] = np.argmax(mltway, axis = 2) # Take the position of row-wise largest values and update the backpointers matrix

    scores = np.max(inter[:,:,-1].reshape(B,L,1)+end_scores,axis=1).reshape(B,) # Adding the end_scores and get the max scores for the whole seqence
    last = np.argmax(inter[:,:,-1].reshape(B,L,1)+end_scores,axis=1).reshape(B,) # Get the last value of y_pred seqence for all batches

    y_pred[:,-1] = last

    # Update the y_pred using backpointer matrix
    for i in range(N-2,-1,-1):
        y_pred[:,i] = backpointers[np.arange(B),last,np.repeat(i,B)]
        last = copy.copy(y_pred[:,i])
        
    return (scores, y_pred.tolist())
