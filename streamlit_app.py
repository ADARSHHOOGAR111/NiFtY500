# streamlit
%%writefile app.py
import streamlit as st
import pickle
import numpy as np
df=pickle.load(open('df.pkl','rb'))
pipe=pickle.load(open('pipe.pkl','rb'))

st.header("TNIFTY %_")

volume=st.number_input("Volume",step=1)
open=st.number_input("Open",step=1)
high=st.number_input("High",step=1)
low=st.number_input("Low",step=1)
if st.button("Predict"):
  query=np.array([[open,high,low,volume]])
  op=pipe.predict(query)
  st.subheader("Closing Predicted Price: "+str(round(op[0])))
