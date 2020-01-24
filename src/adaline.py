import math
from .model import Model
from .utils import ft_dot


class Adaline(Model):

    def __init__(self, lr):
        self.W = None
        self.lr = lr
        self.performance = []
    
    def _activation(self, X):
        # Linear model output: y = W.T * X + b
        net_input = ft_dot(self.W[1:], X) + self.W[0]
        return 1 / (1 + math.exp(-net_input))

    def predict(self, X):
        """
        Mapping the net input to binary class value using sigmoid activation function
        """
        return 1 if self._activation(X) > 0.5 else 0
    
    def _eval_epoch(self, X, y, epoch):
        num_missclass = 0

        for xi, yi in zip(X, y):
            num_missclass += int(self.predict(xi) != int(yi))

        return num_missclass
    
    def _train_epoch(self, X, y, epoch, mode, verbose):
        epoch_errors = []

        for xi, yi in zip(X, y):
            error = yi - self._activation(xi)
            epoch_errors.append(error)

            if mode == 'stochastic':
                self.W[0] += self.lr * error
                self.W[1:] += self.lr * error * xi

        if mode == 'batch':
            self.W[0] += self.lr * sum(epoch_errors)
            self.W[1:] += self.lr * ft_dot(X, epoch_errors)

        num_missclass = self._eval_epoch(X, y, epoch)
        if verbose and epoch % 10 == 0:
            print('Epoch {}: {} errors'.format(epoch, num_missclass))
        self.performance.append((epoch, num_missclass, self.W[1:], self.W[0]))
        
        return epoch_errors
