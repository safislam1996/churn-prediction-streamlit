import streamlit as st
import pickle


st.title("""
    # Bank Churn Rate
    This model checks whether a person will leave the bank or not
""")

@st.cache(allow_output_mutation=True)
def load_model():
    model_file='./models/pipeline.bin'
    
    with open(model_file,'rb') as f:    
        pipeline=pickle.load(f)
    return pipeline

pipeline=load_model()


gender=st.selectbox(
    'Gender',
    ('Male','Female')
)

geography=st.selectbox(
    'Geography',
    ('France','Germany','Spain')
)

age=st.slider(
    'Age',
    min_value=18,
    max_value=100,
    value=30
)
has_cr_card=st.selectbox(
    'HasCrCard',
    ('Yes','No')
)

credit_score=st.slider(
    'CreditScore',
    min_value=350,
    max_value=850,
    value=500
)
tenure=st.slider(
    'Tenure',
    min_value=0,
    max_value=10,
    value=5
)
balance=st.slider(
    'Balance',
    min_value=0,
    max_value=30000,
    value=15000
)
numOfProduct=st.slider(
    'NumOfProducts',
    min_value=1,
    max_value=4,
    value=1
)
estimatedSalary=st.slider(
    'EstimatedSalary',
    min_value=0,
    max_value=20000,
    value=5000
)

isActiveMember=st.selectbox(
    'IsActiveMember',
    ('Yes','No')
)

customer={
    'CreditScore':float(credit_score),
    'Geography': str(geography),
    'Gender':str(gender),
    'Age': int(age),
    'Tenure':int(tenure),
    'Balance':float(balance),
    'NumOfProducts':int(numOfProduct),
    'HasCrCard': str(has_cr_card)=='Yes',
    'IsActiveMember':str(isActiveMember)=='Yes',
    'EstimatedSalary':float(estimatedSalary)
}

pred=pipeline.predict_proba(customer)[0,1]
pred=float(pred)
title_col,result_col=st.columns(2)

title_col.write("Probability of leaving the bank is: ")
result_col.write(f"{pred*100:.2f}%")