
class SimpleLinearRegression:

    def __init__(self, lr = 0.003, n_iters=9000):
        self.__lr = lr #learning rate
        self.__n_iters = n_iters # number of iterations
        
        self.__w = 0 #weight
        self.__b = 0 #bias
        
        self.__X = []
        self.__y = []

    def __gradient_descent(self):
        f = lambda x: self.__w * x + self.__b 
        n = len(self.__X)

        w_gradient = 0
        b_gradient = 0

        for i in range(n):
            w_gradient += -(2/n) * (self.__y[i] - f(self.__X[i])) * self.__X[i]
            b_gradient += -(2/n) * (self.__y[i] - f(self.__X[i]))
        
        self.__w -= w_gradient * self.__lr
        self.__b -= b_gradient * self.__lr


    def fit(self, X: list, y: list):
        self.__w = 0
        self.__b = 0

        self.__X = X
        self.__y = y

        for _ in range(self.__n_iters):
            self.__gradient_descent()

    def predict(self, X: list):
        f = lambda x: self.__w * x + self.__b 

        return [f(x) for x in X]