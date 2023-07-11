import numpy as np 

class KNN :
    def __init__(self , k ) :
        self.k = k 
        
    def fit(self , X , Y): 
        self.X_train = X
        self.Y_train = Y

    def predict(self , list_of_new_datas): 
        Y_list = []
        for x in list_of_new_datas :
            
            distances = []
            for x_train in self.X_train :
                d = self.euclidean_distance(x , x_train) # euclidean_distance between new data & each of the TRAIN DATA
                distances.append(d)

            nearest_neighbour = np.argsort(distances)[0 : self.k]
            result = np.bincount(self.Y_train[nearest_neighbour])
            y = np.argmax(result)
            Y_list.append(y)
        return Y_list 
    def euclidean_distance(self , x1 , x2) :
        return np.sqrt(np.sum((x1 - x2)**2 ))
    def evaluate(self , X , Y ) : 
        Y_predicted = self.predict(X)     
        accuracy = np.sum(Y_predicted == Y)  / len(Y) 
        return accuracy