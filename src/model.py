import random
from abc import ABC, abstractmethod


class Model(ABC):

    def __init__(self, lr):
        self.W = None
        self.lr = lr
        self.performance = []

    def evaluate_accuracy(self, X, y):
        num_correct = 0

        for xi, yi in zip(X, y):
            num_correct += int(self.predict(xi) == int(yi))
        
        accuracy = num_correct / len(y)
        print('[.] Model accuracy: {0:.3f}'.format(accuracy))
        
        return accuracy

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def _train_epoch(self, X, y, epoch, mode, verbose):
        pass

    def train(self, X, y, epochs, mode='batch', verbose=False, seed=None):
        if epochs < 0:
            raise ValueError('invalid number of training epochs')
        
        if mode != 'stochastic' and mode != 'batch':
            raise ValueError('invalid training mode')

        if seed:
            random.seed(seed)
            
        if self.W is None:
            self.W = [0.0001 * random.uniform(-1, 1) for i in range(X.shape[1] + 1)]

        epoch = 0
        while True:
            epoch_errors = self._train_epoch(X, y, epoch, mode, verbose)
            epoch += 1
            if epochs != 0 and epoch == epochs:
                break
            elif epochs == 0 and epoch_errors == 0:
                break
        
        return self.performance
