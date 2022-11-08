import numpy as np



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

    R = None

    dW = None

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

    R = None

    dW = None

    ############################################################
    ###                   END OF YOUR CODE                   ###
    ############################################################
    return R, dW





