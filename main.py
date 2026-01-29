import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import streamlit as st

def generate_sample(n=100):
    np.random.seed(101)
    size=np.random.normal(1400,50,n)
    price=size*50 + np.random.normal(0,50,n)
    return pd.DataFrame({'size':size,'price':price})

def train_model():
    df=generate_sample(n=100)
    x=df[['size']]
    y=df['price']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    model=LinearRegression()
    model.fit(x_train,y_train)
    return model

def main():
    st.title("Simple Linear Regression Model")
    st.write("Enter you house size")

    model = train_model()

    size=st.number_input("House Size",min_value=500,max_value=2000,value=1500)

    if st.button("Predict Price"):
        predicted_price = model.predict(pd.DataFrame([[size]], columns=['size']))
        st.success(f"Predicted Price = ${np.round(predicted_price[0],2)}")

        df=generate_sample(n=100)
        plt.figure(figsize=(5,6))
        plt.title("Size VS House Price")
        plt.scatter(df['size'],df['price'],label='Other Houses')
        plt.xlabel('House Size')
        plt.ylabel('House price')
        plt.scatter(size,predicted_price,color='red',label='Your House')
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)

if __name__=="__main__":
    main()