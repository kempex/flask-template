import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df=pd.read_csv("C:/Users/Bharath/Downloads/diabetes.csv")

img=Image.open("D:/PYTHON/DIabetes with Streamlit/diabetes-stats-report-724px.png")
st.title('Diabetes Analysis')
st.image(img)

st.markdown('This project helps to predict the diabetes of a patient by getting their input. Here I have used RandomForestClassifier to train the data and I have displayed the result using streamlit.')
st.divider()

st.subheader('Description')
st.write(df.describe())

st.subheader('Data Visualization')
st.area_chart(df)

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

#Standardization
from sklearn.preprocessing import StandardScaler
std=StandardScaler()
X_train=std.fit_transform(X_train)
X_test=std.transform(X_test)
dummy='''
# Model Building
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

#Saving the Model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
'''

def user():
    st.sidebar.title('Enter Input')
    preg=st.sidebar.slider('Pregnancies',25,0)
    glu=st.sidebar.slider('Glucose',200,0)
    bp=st.sidebar.slider('Blood Pressure',200,0)
    skin=st.sidebar.slider('Skin Thickness',100,0)
    ins=st.sidebar.slider('Insulin',900,0)
    bmi=st.sidebar.slider('BMI',100,0)
    dpf=st.sidebar.slider('Diabetes Pedigree Function',2.42,0.078)
    age=st.sidebar.slider('Age',100,18)

    userinput={'Pregnancies':preg,'Glucose':glu,'BloodPressure':bp,'SkinThickness':skin,'Insulin':ins,'BMI':bmi,'DiabetesPedigreeFunction':dpf,
                'Age':age}
    user_input=pd.DataFrame(userinput,index=[0])
    return user_input

#Loading Model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
y_pred=model.predict(X_test)

st.subheader('Output')
user_input=user()
user_output=model.predict(user_input)
st.divider()
if user_output==1:
    st.write('Result : You have Diabetes')
else:
    st.write('Result : You do not have Diabetes')

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)*100

st.write('Accuracy :',accuracy)