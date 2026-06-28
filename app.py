import streamlit as st
import pandas as pd
import joblib

# Load the trained model and the expected column structure
model = joblib.load('Model/student_lr_model.pkl')
model_columns = joblib.load('Model/model_columns.pkl')

st.title("Student Performance Prediction System")
st.write("Enter the student's demographic and academic details below to predict their final average score.")

# Create user input fieldsmatching the original dataset
gender = st.selectbox("Gender", ["female", "male"])
race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
education = st.selectbox("Parental Level of Education", ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])
lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])

# Create a prediction button
if st.button("Predict Average Score"):
    
    # 1. Store the user inputs in a dataframe
    input_data = pd.DataFrame({
        'gender': [gender],
        'race/ethnicity': [race],
        'parental level of education': [education],
        'lunch': [lunch],
        'test preparation course': [test_prep]
    })
    
    # 2. Encode the categorical inputs (just like in Phase 3)
    input_encoded = pd.get_dummies(input_data)
    
    # 3. Align the new encoded columns with the ones the model was trained on
    # This adds missing columns with 0s and ensures the exact same order.
    input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    # 4. Make the prediction
    prediction = model.predict(input_aligned)[0]
    
    # 5. Display the result
    st.success(f"Predicted Final Average Score: {prediction:.2f} / 100")

    st.subheader("How was this score calculated?")
# Extract the coefficients (weights) from the trained model
coefficients = model.coef_
features = model_columns

# Create a dataframe of the weights and display it as a bar chart
importance_df = pd.DataFrame({
    'Feature': features,
    'Impact on Score': coefficients
}).sort_values(by='Impact on Score', ascending=False)

# st.bar_chart(importance_df.set_index('Feature'))
st.bar_chart(importance_df.set_index('Feature'), horizontal=True)