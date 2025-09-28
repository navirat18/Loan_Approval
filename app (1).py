import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder


import gradio as gr

# Load data
data = pd.read_csv('loan_data.csv')
X = data.drop(['Loan_Approved','ID'], axis=1)   # ID dropped from training
y = data['Loan_Approved']

# Encode target
LE = LabelEncoder()
y_n = LE.fit_transform(y)

# Train/test split
xtrain, xtest, ytrain, ytest = train_test_split(X, y_n, test_size=0.2, random_state=42)

# Train model
classi = RandomForestClassifier(n_estimators=100, random_state=42)
classi.fit(xtrain, ytrain)

# Prediction function
def laa(Age, Income, Loan_Amount, Credit_Score, Employement_Years):
    dd = classi.predict([[Age, Income, Loan_Amount, Credit_Score, Employement_Years]])
    ss = LE.inverse_transform(dd)[0]
    return f"The loan approval status is: {ss}"

# Gradio interface
interface = gr.Interface(
    fn=laa,
    inputs=[
        gr.Number(label="Age", value=22),
        gr.Number(label="Income", value=1200),
        gr.Number(label="Loan_Amount", value=1200),
        gr.Number(label="Credit_Score", value=700),
        gr.Number(label="Employement_Years", value=5),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Loan Approval Prediction",
    description="Enter applicant details to predict whether the loan will be approved."
)

interface.launch(inline=True)
