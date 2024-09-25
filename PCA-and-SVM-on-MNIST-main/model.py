import numpy as np
from tqdm import tqdm


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
    
    def fit(self, X) -> None:
        # fit the PCA model
        covar = np.cov(X.T)
        Eval,Evac = np.linalg.eigh(covar)
        k_Evac = Evac[:,-self.n_components:]
        self.components = k_Evac
        # return k_Evac
        # raise NotImplementedError
    
    def transform(self, X) -> np.ndarray:
        # transform the data
        return np.matmul(self.components.T,X.T).T
        raise NotImplementedError

    def fit_transform(self, X) -> np.ndarray:
        # fit the model and transform the data
        self.fit(X)
        return self.transform(X)


class SupportVectorModel:
    def __init__(self) -> None:
        self.w = None
        self.b = None
    
    def _initialize(self, X) -> None:
        # initialize the parameters
        n,d = X.shape
        self.w = np.zeros(d)
        self.b = 0

    def fit(
            self, X, y, 
            learning_rate: float,
            num_iters: int,
            C: float = 1.0,
    ) -> None:
        self._initialize(X)
        
        # fit the SVM model using stochastic gradient descent
        for i in tqdm(range(1, num_iters + 1)):
            # sample a random training example
            n,d = X.shape
            # for j in range(0,n):
            index = np.random.choice(n)
            xj,yj = X[index],y[index]
            if(yj*(np.dot(self.w,xj)+ self.b) < 1):
                self.w = self.w + learning_rate*(C*yj*xj - self.w)
                self.b = self.b + learning_rate * C*yj    # update in else?
            # raise NotImplementedError
    
    def predict(self, X) -> np.ndarray:
        # make predictions for the given data
        return np.matmul(X,self.w) + self.b   # add np.sign
        raise NotImplementedError

    def accuracy_score(self, X, y) -> float:
        # compute the accuracy of the model (for debugging purposes)
        return np.mean(self.predict(X) == y)


class MultiClassSVM:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.models = []
        for i in range(self.num_classes):
            self.models.append(SupportVectorModel())
    
    def fit(self, X, y, **kwargs) -> None:
        # first preprocess the data to make it suitable for the 1-vs-rest SVM model
        C,learning_rate,num_iters = kwargs.values()
        for i in range(len(self.models)):
            y_new = np.where(y==i,1,-1)
            self.models[i].fit(X,y_new,learning_rate,num_iters,C)
        # then train the 10 SVM models using the preprocessed data for each class

        # raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        # pass the data through all the 10 SVM models and return the class with the highest score
        pred = np.zeros((X.shape[0],self.num_classes))
        for i in range(len(self.models)):
            pred[:,i] = self.models[i].predict(X)
        return (np.argmax(pred,axis=1))
        raise NotImplementedError

    def accuracy_score(self, X, y) -> float:
        return np.mean(self.predict(X) == y)
    
    def precision_score(self, X, y) -> float:
        y_pred = self.predict(X)
        precision = 0
        for i in range(len(np.unique(y))):
            t_p = np.sum(((y == i) & (y_pred == i)))
            f_p = np.sum(((y != i) & (y_pred == i)))
            if(t_p + f_p == 0):
                precision += 0
            else:
                precision += t_p / (t_p+f_p)
        return precision / len(np.unique(y))
        raise NotImplementedError
    
    def recall_score(self, X, y) -> float:
        y_pred = self.predict(X)
        recall = 0
        for i in range(len(np.unique(y))):
            t_p = np.sum(((y == i) & (y_pred == i)))
            f_n = np.sum(((y == i) & (y_pred != i)))
            if(t_p + f_n == 0):
                recall += 0
            else:
                recall += t_p / (t_p+f_n)
        return recall / len(np.unique(y))
        raise NotImplementedError
    
    def f1_score(self, X, y) -> float:
        pre_score = self.precision_score(X,y)
        rec_score = self.recall_score(X,y)
        return 2*((pre_score*rec_score)/(pre_score+rec_score))
        raise NotImplementedError
