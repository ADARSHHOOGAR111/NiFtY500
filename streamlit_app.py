import pandas as pd
df=pd.read_csv("processed_data.csv")
from sklearn.model_selection import train_test_split
X=df.drop(columns='Close')
y=df['Close']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=42)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression
step1=ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(drop='first'),[])],remainder="passthrough")
step2=LinearRegression()
pipe=Pipeline([
    ('step1',step1),
    ('step2',step2)
])
pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)
r2_score(y_test,y_pred)
import streamlit as st
import pickle5 as pickle
import numpy as np
#df=pickle.load(open('df.pkl','rb'))
#pipe=pickle.load(open('pipe.pkl','rb'))

st.header("TNIFTY %_")

volume=st.number_input("Volume",step=1)
open=st.number_input("Open",step=1)
high=st.number_input("High",step=1)
low=st.number_input("Low",step=1)
if st.button("Predict"):
  query=np.array([[open,high,low,volume]])
  op=pipe.predict(query)
  st.subheader("Closing Predicted Price: "+str(round(op[0])))
