
# coding: utf-8

# In[38]:

from sklearn.ensemble import RandomForestRegressor


# In[23]:

from numpy import genfromtxt, savetxt
import graphlab
from sklearn.model_selection import train_test_split


# In[16]:

def load_data():
    dataset = graphlab.SFrame('SAT11_HAND-cv.txt')
    return dataset


# In[50]:

def get_train_test_data(dataset):
    #using the random split graphlab
    data =[]
    train_data, test_data = dataset.random_split(.8, seed=0)
    data.append(train_data), data.append(test_data)
    print train_data
    return data


# In[27]:

def load_features():
    features =graphlab.SFrame('SAT11_HAND-features.txt')
    return features


# In[40]:

def build_model():
    model =  RandomForestRegressor(n_estimators=10, random_state=0)
    return model


# In[31]:

def train_test_data(dataset):
    #using train_test_split of randomforest
    # Error ValueError: Found input variables
    #with inconsistent numbers of samples: [0, 296]
    data_sets = []
    n_features = load_features()
    x_train, x_test, y_train, y_test = train_test_split(
    dataset, n_features, test_size=0.2, random_state=0)
    data.append(x_train), data.append(x_test), data.append(y_train), data.append(y_test)
    return data


# In[42]:

def main():
    data_set = load_data()
    n_features = load_features()
    train_test = get_train_test_data(data_set)
    train_data = train_test[0]
    test_data = train_test[1]
    model = build_model()
    fit_stat = model.fit(train_data, n_features)
    predicted_set = model.predict(x_test)
    print predicted_set
    


# In[ ]:



