class KNN:
    def __init__(self , K):
        self.K = K
        self.X_train= None
        self.Y_train = None
        pass
    
    def fit(self , X , Y):
        self.X_train= X
        self.Y_train = Y
        pass
    
    def predict(self , X):
        distances = [self._distance(X, x) for x in self.X_train]
        k_nearest = sorted(range(len(distances)), key=lambda i: distances[i])[:self.K]
        k_labels = [self.Y_train.iloc[i] for i in k_nearest]

        return max(set(k_labels), key=k_labels.count)
    
    def evaluate(self):
        pass
    
    def grid_search(self):
        pass