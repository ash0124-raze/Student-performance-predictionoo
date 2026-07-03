#Demography Based----------------------------------------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# import joblib

# # Load the trained model and the expected column structure
# model = joblib.load('Model/student_lr_model.pkl')
# model_columns = joblib.load('Model/model_columns.pkl')

# st.title("Student Performance Prediction System")
# st.write("Enter the student's demographic and academic details below to predict their final average score.")

# # Create user input fieldsmatching the original dataset
# gender = st.selectbox("Gender", ["female", "male"])
# race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
# education = st.selectbox("Parental Level of Education", ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])
# lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
# test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])

# # Create a prediction button
# if st.button("Predict Average Score"):
    
#     # 1. Store the user inputs in a dataframe
#     input_data = pd.DataFrame({
#         'gender': [gender],
#         'race/ethnicity': [race],
#         'parental level of education': [education],
#         'lunch': [lunch],
#         'test preparation course': [test_prep]
#     })
    
#     # 2. Encode the categorical inputs (just like in Phase 3)
#     input_encoded = pd.get_dummies(input_data)
    
#     # 3. Align the new encoded columns with the ones the model was trained on
#     # This adds missing columns with 0s and ensures the exact same order.
#     input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)
    
#     # 4. Make the prediction
#     prediction = model.predict(input_aligned)[0]
    
#     # 5. Display the result
#     st.success(f"Predicted Final Average Score: {prediction:.2f} / 100")

#     st.subheader("How was this score calculated?")
# # Extract the coefficients (weights) from the trained model
# coefficients = model.coef_
# features = model_columns

# # Create a dataframe of the weights and display it as a bar chart
# importance_df = pd.DataFrame({
#     'Feature': features,
#     'Impact on Score': coefficients
# }).sort_values(by='Impact on Score', ascending=False)


# # st.bar_chart(importance_df.set_index('Feature'))
# st.bar_chart(importance_df.set_index('Feature'), horizontal=True)



#Academic Based ----------------------------------------------------------------------------------------------------------


# import streamlit as st
# import pandas as pd
# import joblib

# # Load the new trained model
# model = joblib.load('Model/academic_rf_model.pkl')
# model_columns = joblib.load('Model/academic_columns.pkl')

# st.title("Student Performance Prediction System")
# st.write("Enter the student's academic indicators below to predict their final score (0-20 scale).")

# # Create user input sliders based on academic metrics
# st.subheader("Internal Assessments")
# g1_score = st.slider("First Internal Assessment (G1)", min_value=0, max_value=20, value=10)
# g2_score = st.slider("Second Internal Assessment (G2)", min_value=0, max_value=20, value=10)

# st.subheader("Attendance & Habits")
# absences = st.slider("Number of Absences", min_value=0, max_value=93, value=5)
# study_time = st.selectbox("Weekly Study Time", [1, 2, 3, 4], format_func=lambda x: ["<2 hours", "2 to 5 hours", "5 to 10 hours", ">10 hours"][x-1])
# failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])

# if st.button("Predict Final Grade"):
    
#     # Store the user inputs in a dataframe
#     input_data = pd.DataFrame({
#         'absences': [absences],
#         'G1': [g1_score],
#         'G2': [g2_score],
#         'studytime': [study_time],
#         'failures': [failures]
#     })
    
#     # Make the prediction
#     prediction = model.predict(input_data)[0]
    
#     # Cap the prediction between 0 and 20 (the school's grading scale)
#     prediction = max(0, min(20, prediction))
    
#     st.success(f"Predicted Final Grade: {prediction:.2f} / 20")

#     st.subheader("What influenced this prediction?")

# # Random Forest uses 'feature_importances_' instead of 'coef_'
# importances = model.feature_importances_

# # Create a dataframe to hold the feature names and their importance scores
# importance_df = pd.DataFrame({
#     'Feature': model_columns,
#     'Importance (%)': importances
# }).sort_values(by='Importance (%)', ascending=True) 

# # Display the horizontal bar chart
# st.bar_chart(importance_df.set_index('Feature'), horizontal=True)


import streamlit as st
import pandas as pd
import joblib

# ------------------------
# Page Configuration
# ------------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# ------------------------
# Load Model
# ------------------------
model = joblib.load("Model/academic_rf_model.pkl")
model_columns = joblib.load("Model/academic_columns.pkl")

# ------------------------
# Custom CSS
# ------------------------
st.markdown("""
<style>

.main{
    background-color:#f5f7fb;
}

.title{
    text-align:center;
    font-size:40px;
    color:#1565C0;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.box{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 3px 10px rgba(0,0,0,0.15);
}

</style>
""", unsafe_allow_html=True)

# ------------------------
# Header
# ------------------------

st.markdown("<p class='title'>🎓 Student Performance Prediction System</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict a student's final academic grade using Machine Learning.</p>", unsafe_allow_html=True)

st.write("")

# ------------------------
# Input Layout
# ------------------------

left, right = st.columns(2)

with left:

    st.markdown("### 📘 Internal Assessments")

    g1 = st.slider(
        "First Internal Assessment (G1)",
        0,
        20,
        10
    )

    g2 = st.slider(
        "Second Internal Assessment (G2)",
        0,
        20,
        10
    )

with right:

    st.markdown("### 📅 Attendance & Study")

    absences = st.slider(
        "Number of Absences",
        0,
        93,
        5
    )

    studytime = st.selectbox(
        "Weekly Study Time",
        [1,2,3,4],
        format_func=lambda x:["<2 hrs","2-5 hrs","5-10 hrs",">10 hrs"][x-1]
    )

    failures = st.selectbox(
        "Previous Failures",
        [0,1,2,3]
    )

st.divider()

# ------------------------
# Prediction
# ------------------------

if st.button("🚀 Predict Final Grade", use_container_width=True):

    input_df = pd.DataFrame({
        "absences":[absences],
        "G1":[g1],
        "G2":[g2],
        "studytime":[studytime],
        "failures":[failures]
    })

    prediction = model.predict(input_df)[0]

    prediction = max(0,min(20,prediction))

    st.success("Prediction Completed Successfully!")

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Predicted Grade",
        f"{prediction:.2f}/20"
    )

    percentage = prediction*5

    c2.metric(
        "Percentage",
        f"{percentage:.1f}%"
    )

    if prediction>=16:
        status="Excellent ⭐"
    elif prediction>=12:
        status="Good ✅"
    elif prediction>=8:
        status="Average ⚠️"
    else:
        status="Needs Improvement ❌"

    c3.metric(
        "Performance",
        status
    )

    st.divider()

    st.subheader("📈 Feature Importance")

    importance_df = pd.DataFrame({
        "Feature":model_columns,
        "Importance":model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    st.bar_chart(
        importance_df.set_index("Feature")
    )

    st.divider()

    st.subheader("💡 Recommendation")

    if prediction>=16:
        st.success(
            "Excellent performance. Maintain the same consistency in attendance and internal assessments."
        )

    elif prediction>=12:
        st.info(
            "Good performance. Improving attendance and study time can further increase the final grade."
        )

    elif prediction>=8:
        st.warning(
            "Average performance. Focus on improving internal assessment scores and reducing absences."
        )

    else:
        st.error(
            "Student is at risk. Immediate academic support and regular attendance are strongly recommended."
        )






