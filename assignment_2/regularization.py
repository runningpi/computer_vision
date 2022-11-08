import numpy as np

# Author: Max Althaus

def L1_reg(W, r):
    """
    Compute L1 regularization loss for weights in the given parameter matrix.

    The last row in W is assumed to be the bias.
    Regularization is only applied to the weights and not to the bias.

    Parameters:
        - W: Parameter matrix with shape (D+1, K) with input dimension D and number of classes K.
        - r: Regularization strength.

    Returns:
        - R: Regularization loss.
        - dW: Partial derivatives of L with respect to W.

    """
    ############################################################
    ###                  START OF YOUR CODE                  ###
    ############################################################

    # regular. loss
    # remove last line
    R = W[:-1, :]

    # get the sum and multiply with r
    R = np.sum(np.abs(R)) * r

    # dW:
    # set bias to 0 
    dW = np.append(W[:-1, :], np.zeros(W.shape[1])).reshape(W.shape)

    # get derivative
    dW = np.where(dW < 0, -1, dW)
    dW = np.where(dW > 0, 1, dW)

    # apply r
    dW = dW * r

    ############################################################
    ###                   END OF YOUR CODE                   ###
    ############################################################
    return R, dW



def L2_reg(W, r):
    """
    Compute L2 regularization loss for weights in the given parameter matrix.

    The last row in W is assumed to be the bias.
    Regularization is only applied to the weights and not to the bias.

    Parameters:
        - W: Parameter matrix with shape (D+1, K) with input dimension D and number of classes K.
        - r: Regularization strength.

    Returns:
        - R: Regularization loss.
        - dW: Partial derivatives of L with respect to W.

    """
    ############################################################
    ###                  START OF YOUR CODE                  ###
    ############################################################

    # regular. loss
    # remove last line
    R = W[:-1, :]

    # get the sum and multiply with r
    R = np.sum(np.square(np.abs(R))) * r

    # dW:
    # set bias to 0 
    dW = np.append(W[:-1, :], np.zeros(W.shape[1])).reshape(W.shape)

    # apply r
    dW = dW 

    ############################################################
    ###                   END OF YOUR CODE                   ###
    ############################################################
    return R, dW





