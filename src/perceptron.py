from .model import Model
from .utils import ft_dot


class Perceptron(Model):
    
    def __init__(self, lr):
        self.W = None
        self.lr = lr
        self.performance = []

    def predict(self, X):
        """
        Mapping the net input to binary class value using unit (heaviside) step function
        """
        net_input = ft_dot(self.W[1:], X) + self.W[0]
        return 1 if net_input > 0.0 else 0
    
    def _train_epoch(self, X, y, epoch, mode, verbose):
        epoch_errors = 0

        for xi, yi in zip(X, y):
            update = self.lr * (yi - self.predict(xi))
            self.W[0] += update
            self.W[1:] += update * xi
            epoch_errors += int(update != 0.0)

        if verbose and epoch % 10 == 0:
            print('Epoch {}: {} errors'.format(epoch, epoch_errors))
        self.performance.append((epoch, epoch_errors, self.W[1:], self.W[0]))
        
        return epoch_errors
