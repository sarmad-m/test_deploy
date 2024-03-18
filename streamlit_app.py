# import necessary packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
### *********** THE MODEL ******************
# random seed
train=pd.read_csv("Urineflowmeterdata.csv")
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
pickle.dump(best_rf_model, open("rf_model.pkl", 'wb'))





## ******************* THE web APP ************************


# Load the saved model
model = pickle.load(open("rf_model.pkl", "rb"))

# Define a function to preprocess new input data
def preprocess_data(data):
    le = LabelEncoder()
    data["class"] = le.fit_transform(data["class"])

    scaler = StandardScaler()
    columns_to_scale = ['Pr', 'Frate', 'Favrg', 'Time', 'Vtotal', 'Fmax', 'Tmax', 'SNO']
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    return data.drop("class", axis=1)  # Drop 'class' column for prediction

# Create a Streamlit app
st.title("Urine Flowmeter Prediction App")

# Collect user input
with st.form("form"):
    Pr = st.slider("Pr", min_value=0, max_value=100, step=1)
    Frate = st.number_input("Frate")
    Favrg = st.number_input("Favrg")
    Time = st.number_input("Time")
    Vtotal = st.number_input("Vtotal")
    Fmax = st.number_input("Fmax")
    Tmax = st.number_input("Tmax")
    SNO = st.number_input("SNO")
    submitted = st.form_submit_button("Predict")

if submitted:
    # Preprocess input data
    new_data = pd.DataFrame([[Pr, Frate, Favrg, Time, Vtotal, Fmax, Tmax, SNO]], columns=['Pr', 'Frate', 'Favrg', 'Time', 'Vtotal', 'Fmax', 'Tmax', 'SNO'])
    processed_data = preprocess_data(new_data)

    # Make prediction
    prediction = model.predict(processed_data)[0]

    # Display the predicted class
    st.write("Predicted Class:", prediction)
