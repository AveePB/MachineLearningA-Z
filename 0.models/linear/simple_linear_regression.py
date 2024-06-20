
class SimpleLinearRegressor:

    def __init__(self, lr = 0.003, n_iters=9000):
        self.__lr = lr #learning rate
        self.__n_iters = n_iters # number of iterations
        
        self.__coeff = 0 #weight
        self.__inter = 0 #bias
        
        self.__X = []
        self.__y = []

    def __gradient_descent(self):
        f = lambda x: self.__coeff * x + self.__inter 
        n = len(self.__X)

        coeff_gradient = 0
        inter_gradient = 0

        for i in range(n):
            coeff_gradient += -(2/n) * (self.__y[i] - f(self.__X[i])) * self.__X[i]
            inter_gradient += -(2/n) * (self.__y[i] - f(self.__X[i]))
        
        self.__coeff -= coeff_gradient * self.__lr
        self.__inter -= inter_gradient * self.__lr


    def fit(self, X: list, y: list):
        self.__coeff = 0
        self.__inter = 0

        self.__X = X
        self.__y = y

        for _ in range(self.__n_iters):
            self.__gradient_descent()

    def predict(self, X: list):
        f = lambda x: self.__coeff * x + self.__inter 

        return [f(x) for x in X]