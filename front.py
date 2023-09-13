
import joblib

import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder

import streamlit as st

st.set_page_config(page_title="Attrition Predictor", page_icon=":performing_arts:", layout="centered")

def main():
    st.title("Upload CSV File")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        st.write("Uploaded file: ")
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

        st.write("Who will resign?: ")

        # 전처리 => kafka 혹은 spark
        object_columns = df.select_dtypes(include=['object']).columns
        object_columns = object_columns.tolist()
        df[object_columns] = OrdinalEncoder().fit_transform(df[object_columns])

        # 예측 => mlflow
        model = joblib.load('./model/lgbm_clf_1692024379.pkl')
        y_pred = model.predict(df)
        result_df = df.iloc[np.where(y_pred == 1)[0]]
        st.dataframe(result_df)


if __name__ == "__main__":
    main()