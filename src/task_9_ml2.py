"""
Task 9 - Machine Learning - Cross valiadation.
"""
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from task_8_ml_1_iris import iris_dataset
from task_7_titanic import titanic_data
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

class DataProvider:
    """
    Dataset handler for bank dataset
    """

    def __init__(self, X, y):
        
        self.X = X
        self.y = y
        self.__fit(X,y)
        
    def __fit(self, X, y):
      
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)
        self.data_encoder = OrdinalEncoder(dtype=np.int32)
        self.data_encoder.fit(X)

    def get_data(self):
        """
        get nicely encoded data
        """
        y_enc = self.label_encoder.transform(self.y)
        X_enc = self.data_encoder.transform(self.X)
        return X_enc, y_enc

class EvaluateClassifier:
    """
    Teach and cv test classifier.
    """
    
    def __init__(self, cls, X, y):
        self.cls = cls
        self.X = X
        self.y = y
    
    def prepare_cls_not_lazy(self, n_folds=5)-> tuple[BaseEstimator,list]:
        
        """
        Prepares classifier, proper way.

        Parameters:
        - Classifier, X data, y data

        Returns:
        - Classifier in provided type, list with cross valiadation results.
        """

        trained_classifier = self.cls.fit(self.X, self.y)
        skf = StratifiedKFold(n_folds)
        scores = []
        for train_index, test_index in skf.split(self.X, self.y):
            X_train = self.X.iloc[train_index,:]
            y_train = self.y.iloc[train_index].values
            X_test = self.X.iloc[test_index,:]
            y_test = self.y.iloc[test_index].values
            self.cls.fit(X_train, y_train)
            y_pred = self.cls.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))


        return trained_classifier, scores
    
    def prepare_cls_lazy(self, n_folds=5, measure='accuracy'):
        """
        Prepares classifier, lazy way.

        Parameters:
        - Classifier, X data, y data, n_folds, measure accuracy type

        Returns:
        - Classifier in provided type, list with cross valiadation results.
        """

        trained_classifier = self.cls.fit(self.X, self.y)
        scores = cross_val_score(self.cls, self.X, self.y, cv=n_folds, scoring=measure)
        return trained_classifier, scores

def prepare_data_iris():
    """
    Prepares X and y data for classifiers with iris dataset and splits to train and test data.
    """
    
    iris_data = iris_dataset()
    
    X = iris_data.iloc[:, 1:4]
    y = iris_data.iloc[:,0]

    X_train, X_test, y_train, y_test = train_test_split(X ,y , train_size=0.8, shuffle=False)
    return X_train, y_train, X_test, y_test

def prepare_data_titanic(X,y):
    """
    Prepares X and y data for classifiers with titanic dataset and splits to train and test data.
    """
 
    X_train, X_test, y_train, y_test = train_test_split(X ,y , train_size=0.8, shuffle=True)
    return X_train, y_train, X_test, y_test




if __name__ == "__main__":


    # DATASET CHOISE (0 - iris , 1 - titanic)
    dataset_index = 1
    
    
    if dataset_index==0:
        X_train, y_train, X_test, y_test = prepare_data_iris()
   
    if dataset_index==1:
        
        df = titanic_data()
        X_not_encoded = df[['Sex','Age', 'Pclass', 'Fare', 'Cabin', 'Embarked', 'SibSp', 'Parch']].copy()
        y_not_encoded = df.iloc[:,1]
        X_not_encoded.loc[:,'Age'] = X_not_encoded['Age'].fillna(X_not_encoded['Age'].median())
        X_not_encoded.loc[:,'Cabin'] = X_not_encoded['Cabin'].fillna('')
        X_not_encoded.loc[:,'Embarked'] = X_not_encoded['Cabin'].fillna('')
        
        data_provider = DataProvider(X=X_not_encoded, y=y_not_encoded)
        X, y = data_provider.get_data()
        
        # print(X[:5])
        # print(y)
        X_train, y_train, X_test, y_test = prepare_data_titanic(X=X,y=y)   

    # CLASSIFIER CHOISE (0 - KNN, 1 - DT)
    classifier_index = 2    
    
   
    if classifier_index == 0:
        cls = KNeighborsClassifier()
    
    if classifier_index == 1:
        cls = DecisionTreeClassifier()
    if classifier_index == 2:
        # Create a pipeline that first standarises the data and then applies the KNeighborsClassifier
        cls = Pipeline([
        ('scaler', StandardScaler()),  # Feature scaling
        ('knn', KNeighborsClassifier())  # KNN classifier
        ])
        
    mean = []
    for i  in range(10):
        evl = EvaluateClassifier(cls= cls, X= X_train, y= y_train)
        trained_classifier, scores = evl.prepare_cls_lazy()
        # print(scores)
        predictions = trained_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        mean.append(accuracy)
    mean = np.array(mean)    
    print(np.mean(mean))


    
