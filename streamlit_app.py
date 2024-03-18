# import necessary packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
### *********** THE MODEL ******************
# random seed
train=pd.read_csv("D:/Data/Urineflowmeterdata.csv")
np.unique(train['class'],return_counts=True)
features=train.loc[:, 'Pr':'SNO']

classes= train['class']

le=LabelEncoder()
classes= le.fit_transform(classes)



sc=StandardScaler()
columns_to_scale=['Pr',	'Frate',	'Favrg',	'Time',	'Vtotal',	'Fmax'	,'Tmax',	'SNO']
train[columns_to_scale]=sc.fit_transform(train[columns_to_scale])

y=classes
X=train[columns_to_scale]
X_train,X_test,y_train, y_test= train_test_split(X , y,test_size=0.1,stratify=classes,random_state=40)


rf_classifier = RandomForestClassifier()
param_grid = {'n_estimators': [50, 100,],'max_depth': [None, 10,],'min_samples_split': [2, 5],'min_samples_leaf': [1,]}
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=10, scoring='neg_log_loss', n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best Hyperparameters:", grid_search.best_params_)
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")  # Accuracy: 0.91

# save the model to disk
pickle.dump(best_rf_model, open("rf_model.sav", 'wb'))





## ******************* THE web APP ************************
# title and description:
st.title('Classifying  ')
st.markdown('Toy model to play with the iris flowers dataset and classify the three Species into \
     (setosa, versicolor, virginica) based on their sepal/petal \
    and length/width.')

# features sliders for the four plant features:
st.header("Plant Features")
col1, col2 = st.columns(2)

with col1:
    st.text("Sepal characteristics")
    sepal_l = st.slider('Sepal lenght (cm)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)

with col2:
    st.text("Pepal characteristics")
    petal_l = st.slider('Petal lenght (cm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)

st.text('')

# prediction button
if st.button("Predict type of Iris"):
    result = clf.predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    st.text(result[0])


st.text('')
st.text('')
st.markdown(
    '`Initial code was developed by` [santiviquez](https://twitter.com/santiviquez) | \
         `Code modification and update by:` [Mohamed Alie](https://github.com/Kmohamedalie/iris-streamlit/tree/main)')
