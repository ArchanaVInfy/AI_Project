import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.write("""
# Predict income so that they can be used to identify who can donate for Hospitals
""")

# Loading the original dataset
data = pd.read_csv("cup98LRN.txt",sep=",")


# Display the original dataset with a heading
st.write("## Original Dataset Preview")
st.dataframe(data.head())

# Button to show all features in the original data
show_all_features = st.button("Show All Features")

# Display all features when the button is clicked
if show_all_features:
    st.write('## All Features in the Original Data')
    st.write("Column Names:", data.columns.tolist())



# Streamlit app
def main():
    st.write("# Data after pre-processing")


    # Loading the pre-processed and scaled dataset
    df_copy = pd.read_csv("df_copy.csv", low_memory=False)
    df_standardized = pd.read_csv("df_standardized.csv", low_memory=False)
    # Button to print the pre-processed and scaled dataset
    print_data_button = st.button("Show Data After Pre-Processing")

    # Print the dataset when the button is clicked
    if print_data_button:
        st.write("## Data After Pre-Processing")
        st.write("##dfscaled")
        
    featured_data = pd.read_csv("featured_data.csv", low_memory=False)
    pca = PCA(n_components=6)
    df_pca = pca.fit_transform(df_standardized)
    y = featured_data[['INCOME_encoded']]
    X = df_pca
    from sklearn.model_selection import train_test_split
    Xpca_train,Xpca_test,ypca_train,ypca_test = train_test_split(X,y,test_size=0.2,random_state=42)
    # Convert DataFrame target variables to 1D arrays


        # Button to calculate and display classification metrics
    calculate_metrics_button = st.button("Calculate Classification Metrics")

    if calculate_metrics_button:
        st.write("## Classification Metrics")

     
        # Assuming Xpca_train, Xpca_test, ypca_train, ypca_test are already defined

        # Convert DataFrame target variables to 1D arrays
        ypca_train = ypca_train.values.ravel()
        ypca_test = ypca_test.values.ravel()

         # Instantiate classifiers
        rf_clf = RandomForestClassifier(random_state=42)
        svm_clf = SVC(kernel='linear', probability=True)
        gb_clf = GradientBoostingClassifier(random_state=42)

        # Train classifiers
        rf_clf.fit(Xpca_train, ypca_train)
        svm_clf.fit(Xpca_train, ypca_train)
        gb_clf.fit(Xpca_train, ypca_train)

        # Make predictions
        rf_preds = rf_clf.predict(Xpca_test)
        svm_preds = svm_clf.predict(Xpca_test)
        gb_preds = gb_clf.predict(Xpca_test)

        # Calculate metrics for each classifier
        def calculate_metrics(y_true, y_pred):
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            return accuracy, precision, recall, f1

        rf_accuracy, rf_precision, rf_recall, rf_f1 = calculate_metrics(ypca_test, rf_preds)
        svm_accuracy, svm_precision, svm_recall, svm_f1 = calculate_metrics(ypca_test, svm_preds)
        gb_accuracy, gb_precision, gb_recall, gb_f1 = calculate_metrics(ypca_test, gb_preds)

        st.write("### Random Forest")
        st.write("Accuracy:", rf_accuracy)
        st.write("Precision:", rf_precision)
        st.write("Recall:", rf_recall)
        st.write("F1 Score:", rf_f1)

        st.write("### SVM")
        st.write("Accuracy:", svm_accuracy)
        st.write("Precision:", svm_precision)
        st.write("Recall:", svm_recall)
        st.write("F1 Score:", svm_f1)

        st.write("### Random Forest")
        st.write("Accuracy:", gb_accuracy)
        st.write("Precision:", gb_precision)
        st.write("Recall:", gb_recall)
        st.write("F1 Score:", gb_f1)

        # Create a Voting Classifier ensemble
        ensemble_clf = VotingClassifier(estimators=[('rf', rf_clf), ('svm', svm_clf), ('gb', gb_clf)], voting='hard')
        ensemble_clf.fit(Xpca_train, ypca_train)

        # Make predictions using the ensemble
        ensemble_preds = ensemble_clf.predict(Xpca_test)

        # Calculate metrics for the ensemble
        ensemble_accuracy, ensemble_precision, ensemble_recall, ensemble_f1 = calculate_metrics(ypca_test, ensemble_preds)

        st.write("### Ensemble Classifier")
        st.write("Accuracy:", ensemble_accuracy)
        st.write("Precision:", ensemble_precision)
        st.write("Recall:", ensemble_recall)
        st.write("F1 Score:", ensemble_f1)


if __name__ == "__main__":
    main()
