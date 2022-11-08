import numpy as np

#Author: Max Althaus


def SVM_loss(S, y, d=1):
    """
    Compute multiclass SVM loss and derivative for a minibatch of scores.

    Parameters:
        - S: Matrix of scores with shape (N, K) with number of samples N and classes K.
        - y: Vector with ground truth labels that has length N.
        - d: Margin hyperparameter.

    Returns:
        - L: Total loss.
        - dS: Partial derivative of L with respect to S.

    """
    ############################################################
    ###                  START OF YOUR CODE                  ###
    ############################################################

    # LOSS

    # calculate distance part
    # ? wie funtkioniert diese Syntax?
    L = S - S[np.arange(len(S)), y].reshape(len(S),1) + d

    # get max from the distance part
    L = np.where(L<0, 0, L)

    # remove k=y_i from the sum
    L[np.arange(len(S)), y] = 0

    # get over each row
    L = np.sum(L, axis=1)

    # get the mean of the losses
    L = 1/S.shape[0] * np.sum(L)

    # DERIVATIVE

    # compute conditiom
    dS = S - S[np.arange(len(S)), y].reshape(len(S),1) + d
    dS = np.where(dS > 0, 1, 0)

    # compute elements where k=y
    dS[np.arange(len(S)), y] = 0
    dS[np.arange(len(S)), y] = - np.sum(dS, axis=1)

    # compute partial derivative
    dS = 1 / S.shape[0] * dS

    ############################################################
    ###                    END OF YOUR CODE                  ###
    ############################################################
    return L, dS

def cross_entropy_loss(S, y):
    """
    Compute cross-entropy loss and derivative for a minibatch of scores.

    Parameters:
        - S: Matrix of scores with shape (N, K) with number of samples N and classes K.
        - y: Vector with ground truth labels that has length N.

    Returns:
        - L: Total loss.
        - dS: Partial derivative of L with respect to S.

    """
    ############################################################
    ###                  START OF YOUR CODE                  ###
    ############################################################

    c = np.max(S, axis=1)[:, None]    

    # compute losses
    L = - S[np.arange(len(S)), y] + np.log(np.sum(np.exp(S), axis=1))

    # compulte sum of losses
    L = 1 / S.shape[0] * np.sum(L)

    
    # compute derivatives
    dS = np.sum(np.exp(S-c))
    dS = np.exp(S-c) / dS 

    dS[np.arange(len(S)), y] -= 1

    dS = 1 / S.shape[0] * dS


    ############################################################
    ###                    END OF YOUR CODE                  ###
    ############################################################
    return L, dS


