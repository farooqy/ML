
# coding: utf-8

# # from sklearn.ensemble import RandomForestRegressor

# In[37]:

from numpy import genfromtxt, savetxt
import graphlab
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# In[38]:

def load_data():
    #using graphlab
    dataset = graphlab.SFrame('SAT11_HAND-ai-perf.csv')
    return dataset


# In[39]:

def load_data_two():
    #using genfromtxt numpy
    dataset = genfromtxt('SAT11_HAND-ai-perf.csv')
    return dataset


# In[40]:

def get_train_test_data(dataset):#same function as train_test_data
    #using the random split graphlab
    data =[]
    train_data, test_data = dataset.random_split(.8, seed=0)
    data.append(train_data), data.append(test_data)
    print train_data
    return data


# In[41]:

def load_features():
    features = graphlab.SFrame('SAT11_HAND-features.txt')
    return features


# In[47]:

def build_model_one():
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    return model


# In[62]:

def build_model_two(train_data, n_target, test_data):
    #using the graphlab linear regression
    model = graphlab.linear_regression.create(train_data,target=n_target,
                                             features=n_target,#error invalid type
                                             validation_set=test_data)
    return model


# In[59]:

def train_test_data(dataset):#Same function as get_train_test_data
    #using train_test_split of randomforest
    # Error ValueError: Found input variables
    #with inconsistent numbers of samples: [0, 296]
    data= []
    n_features = load_features()
    x_train, x_test, y_train, y_test = train_test_split(
    dataset, n_features, test_size=0.2, random_state=0)
    data.append(x_train), data.append(x_test), data.append(y_train), data.append(y_test)
    return data


# In[60]:

def main():
    data_set = load_data()
    n_features = load_features()
    train_test = get_train_test_data(data_set) #From the graphlab split
    train_data = train_test[0]
    test_data = train_test[1]
    #model_one = build_model_one()#From scikit learn randomforestregressor
    #fit_stat = model_one.fit(train_data, n_features)
    #predicted_set_one = model_one.predict(test_data)
    #Build model two from graphlab, linear_regression
    model_two = build_model_two(train_data, n_features, test_data)   
    #to get the error and the root mean square error
    error, rmse = model_two.evaluate(test_data)
    predicted_set_two = model_two.predict(test_data)
    print "Error: % \n RMSE: %" %error, rmse
    #print "SET ONE: ", predicted_set_one
    print "SET TWO: ", predicted_set_two


# In[61]:

if __name__=='__main__' :
    main()


# In[ ]:



