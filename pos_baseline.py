# This script uses Theano's Logistic Regression Algorithm To Evaluate Part of Speech (POS Tagging) on Real World Data Set (Wall Street Journal)
# and Decision Tress as sanity check.


import os, sys
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier


class LogisticRegression:
    """Generative Probabilistic Class for Real World Data Set Using Theano"""

    def __init__(self):
        pass

    # define learning rate = 0.1, momentum = 0.99, batch_size=100 and epochs=6
    def fit(self, X, Y, V=None, K=None, D=50, lr=1e-1, mu=0.99, batch_sz=100, epochs=6):
        if V is None:
            V = len(set(X))
        if K is None:
            K = len(set(Y))
        N = len(X)

        # initialize weights
        W = np.random.randn(V, K) / np.sqrt(V + K)
        b = np.zeros(K)
        self.W = theano.shared(W)
        self.b = theano.shared(b)
        self.params = [self.W, self.b]

        # encode tokens and PoS tags as theano integer vectors
        thX = T.ivector("X")
        thY = T.ivector("Y")

        py_x = T.nnet.softmax(self.W[thX] + self.b)
        prediction = T.argmax(py_x, axis=1)

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value() * 0) for p in self.params]
        self.cost_predict_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            allow_input_downcast=True,
        )

        updates = [
            (p, p + mu * dp - lr * g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [(dp, mu * dp - lr * g) for dp, g in zip(dparams, grads)]
        train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            updates=updates,
            allow_input_downcast=True,
        )

        costs = []
        n_batches = N // batch_sz
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            print("epoch:", i)
            for j in range(n_batches):
                Xbatch = X[j * batch_sz : (j * batch_sz + batch_sz)]
                Ybatch = Y[j * batch_sz : (j * batch_sz + batch_sz)]

                c, p = train_op(Xbatch, Ybatch)
                costs.append(c)
                if j % 200 == 0:
                    print(
                        "i:",
                        i,
                        "j:",
                        j,
                        "n_batches:",
                        n_batches,
                        "cost:",
                        c,
                        "error:",
                        np.mean(p != Ybatch),
                    )
        plt.plot(costs)
        plt.show()

    # define accuracy score
    def score(self, X, Y):
        _, p = self.cost_predict_op(X, Y)
        return np.mean(p == Y)

    # define f1 score, which is the harmonic mean of precision and recall
    def f1_score(self, X, Y):
        _, p = self.cost_predict_op(X, Y)
        return f1_score(Y, p, average=None).mean()


# abstract reading of train and test data
def get_data(split_sequences=False):
    if not os.path.exists("pos_data"):
        print("Create a folder in your local directory called 'pos_data'")
        exit()
    elif not os.path.exists("pos_data/train_real.txt"):
        print("Please make sure train.txt is in pos_data/train.txt")
        exit()
    elif not os.path.exists("pos_data/test_real.txt"):
        print("Please make sure test.txt is not in pos_data/test.txt")
        exit()

    # dictionary with token indices of vocabulary
    word2idx = {}

    # dictionary with PoS tag indices
    tag2idx = {}
    word_idx = 0
    tag_idx = 0
    Xtrain = []
    Ytrain = []
    # create placeholder for updates of features (words or tokens) and target (PoS tags)
    currentX = []
    currentY = []
    # load train data
    for line in open("pos_data/train_real.txt"):
        line = line.rstrip()
        if line:
            r = line.split()
            word, tag, _ = r
            if word not in word2idx:
                word2idx[word] = word_idx
                word_idx += 1
            currentX.append(word2idx[word])

            if tag not in tag2idx:
                tag2idx[tag] = tag_idx
                tag_idx += 1
            currentY.append(tag2idx[tag])
        elif split_sequences:
            Xtrain.append(currentX)
            Ytrain.append(currentY)
            currentX = []
            currentY = []

    if not split_sequences:
        Xtrain = currentX
        Ytrain = currentY

    # load and score test data
    Xtest = []
    Ytest = []
    currentX = []
    currentY = []
    for line in open("pos_data/test_real.txt"):
        line = line.rstrip()
        if line:
            r = line.split()
            word, tag, _ = r
            if word in word2idx:
                currentX.append(word2idx[word])
            else:
                currentX.append(word_idx)  # use this as unknown
            currentY.append(tag2idx[tag])
        elif split_sequences:
            Xtest.append(currentX)
            Ytest.append(currentY)
            currentX = []
            currentY = []
    if not split_sequences:
        Xtest = currentX
        Ytest = currentY

    return Xtrain, Ytrain, Xtest, Ytest, word2idx


def main():
    Xtrain, Ytrain, Xtest, Ytest, word2idx = get_data()

    # convert to numpy arrays
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)

    # convert Xtrain to indicator matrix
    N = len(Xtrain)
    V = len(word2idx) + 1
    print("vocabulary size:", V)
    Xtrain_indicator = np.zeros((N, V))
    Xtrain_indicator[np.arange(N), Xtrain] = 1

    # decision tree for sanity check
    dt = DecisionTreeClassifier()

    # logistic regression model
    model = LogisticRegression()
    model.fit(Xtrain, Ytrain, V=V)
    print("logistic regression training done")
    print("logistic regression training score:", model.score(Xtrain, Ytrain))
    print("logistic regression training f1:", model.f1_score(Xtrain, Ytrain))
    print("logistic regression test f1:", model.f1_score(Xtest, Ytest))

    # if the features are not indicators
    dt.fit(Xtrain.reshape(N, 1), Ytrain)
    print("decision tree train score:", dt.score(Xtrain.reshape(N, 1), Ytrain))
    p = dt.predict(Xtrain.reshape(N, 1))
    print("decision tree train f1:", f1_score(Ytrain, p, average=None).mean())

    # with indicator -- substantially slower than the other algorithms
    dt.fit(Xtrain_indicator, Ytrain)
    print("dt score:", dt.score(Xtrain_indicator, Ytrain))


if __name__ == "__main__":
    main()
