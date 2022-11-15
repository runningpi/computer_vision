import numpy as np
from .module import Module



__all__ = ['ReLU', 'Sigmoid', 'Tanh']



class ReLU(Module):

    def forward(self, x):
        """
        Apply the ReLU activation function to input values.

        Inputs:
            - x (np.array): Input features.

        Returns:
            - out (np.array): Output features.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        self.x = x
        # replace < 0 values with 0
        out = np.where(x<0, 0, x)

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return out


    def backward(self, out_grad):
        """
        Compute the gradient with respect to the inputs.

        Parameters:
            - out_grad (np.array): Gradient with respect to module output.

        Returns:
            - in_grad (np.array): Gradient with respect to module input.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        # replace <0 values with 0 and >= 0 values with 1 and multiply with loss
        in_grad = np.multiply(out_grad, np.where(self.x >= 0, 1, np.where(self.x<0, 0, self.x)))

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return in_grad



class Sigmoid(Module):

    def forward(self, x):
        """
        Apply the sigmoid activation function to input values.

        Inputs:
            - x (np.array): Input features.

        Returns:
            - out (np.array): Output features.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        # compute sigmoid
        out = 1 / (1+ np.exp(- x))

        # safe sigmoid in class variable
        self.sigmoid = out

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return out


    def backward(self, out_grad):
        """
        Compute the gradient with respect to the inputs.

        Parameters:
            - out_grad (np.array): Gradient with respect to module output.

        Returns:
            - in_grad (np.array): Gradient with respect to module input.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        # compute derivative of the layer outputs with respect to the layer inputs
        dSigmoid = np.multiply(self.sigmoid, 1-self.sigmoid)

        # compute derivative of loss with respect to the layer inputs
        in_grad = np.multiply(dSigmoid, out_grad)

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return in_grad



class Tanh(Module):

    def forward(self, x):
        """
        Apply the tanh activation function to input values.

        Inputs:
            - x (np.array): Input features.

        Returns:
            - out (np.array): Output features.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        out = np.divide((np.exp(x)- np.exp(-x)), (np.exp(x) + np.exp(-x)))

        self.tanh = out

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return out


    def backward(self, out_grad):
        """
        Compute the gradient with respect to the inputs.

        Parameters:
            - out_grad (np.array): Gradient with respect to module output.

        Returns:
            - in_grad (np.array): Gradient with respect to module input.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        # compute derivative of layer outputs with respect to layer inputs
        in_grad = 1 - np.square(self.tanh)


        # compute derivative of loss with respect to layer inputs
        in_grad = np.multiply(out_grad, in_grad)

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return in_grad


