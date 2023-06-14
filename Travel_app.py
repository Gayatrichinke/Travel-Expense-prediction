import streamlit as st
import pickle
import pandas as pd

# Load the linear regression model from the .pkl file
with open('travel_expense_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.set_page_config(page_title='Travel Expense Prediction', page_icon='✈️')

def predict_expense(data):
    # Perform the prediction using the loaded model
    prediction = model.predict(data)
    return prediction[0]


def main():
    st.title("Travel Expense Prediction")

    # Input fields
    From_loc = st.text_input('From Location')
    To_loc = st.text_input('To Location')
    distance = st.number_input('Distance', min_value=0.0, step=0.1)
    days = st.number_input('Days', min_value=1, step=1)
    

    # Predict button
    if st.button("Predict"):
        # Create a DataFrame with the input values
        data = pd.DataFrame({'distance': [distance],
                             'days': [days]})

        # Perform the prediction
        expense_prediction = predict_expense(data)
        st.header('Prediction of Travel Expense')
        st.write('The predicted Travel expense is:', expense_prediction)


if __name__ == '__main__':
    main()
