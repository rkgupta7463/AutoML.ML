from flask import Flask, render_template, request, send_file, flash,abort,Response
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)

from sklearn.svm import SVR,SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report,mean_absolute_error
from sklearn.model_selection import cross_val_score
import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os
from ydata_profiling import ProfileReport
import io
# import langchain
from bs4 import BeautifulSoup
## PandasAI
import pandasai
from pandasai.smart_dataframe import SmartDataframe
from pandasai.llm import OpenAI
from Secret_key.constants import openai_key
import matplotlib.pyplot as plt
import re

# Instantiate a LLM
llm = OpenAI(api_token=openai_key)

# Define the directory where your model files are stored
MODEL_DIRECTORY = '/home/AutoML/'


app = Flask(__name__)

# Set the maximum file size to 5MB (5 * 1024 * 1024 bytes)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB



# preprocessing function
# Define a function to extract numerical values from the target column
def extract_numerical_value(text):
    # Use regular expressions to extract the numerical value
    match = re.search(r'(\d+\.\d+)', str(text))
    if match:
        return float(match.group(1))  # Convert the extracted text to a float
    else:
        return None  # Return None if no match is found

# Modify the preprocessing function to correctly convert the target column to numeric format


def preprocessing(df, **args):
    if any(df.isnull().sum()):
        print("Null value exists in the dataframe!")

        # Identify columns with object dtype and numeric dtype
        object_columns = df.select_dtypes(include=['object']).columns
        numeric_columns = df.select_dtypes(include=['int', 'float']).columns

        # Fill null values in object columns with mode
        for col in object_columns:
            mode_val = df[col].mode().iloc[0]  # Get the mode value
            df[col].fillna(mode_val, inplace=True)

        # Impute missing values in numeric columns with mean
        for col in numeric_columns:
            imputer = SimpleImputer(strategy='mean')
            df[col] = imputer.fit_transform(df[col].values.reshape(-1, 1))

        # Identify object columns that contain numeric values with commas
        columns_with_commas = [
            col for col in object_columns if df[col].str.contains(',').any()]

        # Remove commas from columns with numeric values
        for col in columns_with_commas:
            df[col] = df[col].str.replace(',', '')

        # Convert target column to numeric format
        if 'target_column' in args:
            target_column = args['target_column']
            if target_column in object_columns:
                # Use the extract_numerical_value function to convert values
                # like 'X.XX Lakh' to numeric format
                df[target_column] = df[target_column].apply(
                    extract_numerical_value)

        # Identify object columns with all numerical values and try to convert them
        for col in object_columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except ValueError:
                pass  # Ignore columns that can't be converted to numeric

        return df
    else:
        return df



def prediction(df, target_column, problem_type, test_size=0.3, random_state=42):
    X = df.drop(columns=[target_column])
    # print("Before",X)
    y = df[target_column]
    X=pd.get_dummies(X)
    # print("After",X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    categorical_cols = X_train.select_dtypes(
        include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()


    if problem_type == 'Classification':
        if y.dtype == 'object':
            lb = LabelEncoder()
            y_train = lb.fit_transform(y_train)
            y_test = lb.transform(y_test)
        else:
            pass

        best_model, best_model_name, best_accuracy, report = classify(
            X_train, y_train, X_test, y_test)
        return best_model, best_model_name, best_accuracy, report

    elif problem_type == 'Regression':
        best_model, best_model_name, best_mse,best_mae,best_rmse, predictions = regress(
            X_train, y_train, X_test, y_test)
        return best_model, best_model_name, best_mse,best_mae,best_rmse, predictions

# Modify the classify and regress functions to return the trained model object and its name
def classify(X_train_final, y_train, X_test_final, y_test):
    algorithms = {
        "RandomForestClassifier": RandomForestClassifier(),
        "AdaBoostClassifier": AdaBoostClassifier(),
        "GradientBoostingClassifier": GradientBoostingClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "LogisticRegression": LogisticRegression(),
        "XGBClassifier": XGBClassifier()
    }

    best_model = None
    best_model_name = ""
    best_accuracy = 0
    best_report = None

    for name, model in algorithms.items():
        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name
            best_report = classification_report(
                y_test, y_pred, output_dict=True)

    return best_model, best_model_name, best_accuracy, best_report


# Modify the regression function to use a pipeline with an imputer

def regress(X_train_final, y_train, X_test_final, y_test):
    algorithms = {
        "RandomForestRegressor": RandomForestRegressor(),
        "AdaBoostRegressor": AdaBoostRegressor(),
        "GradientBoostingRegressor": GradientBoostingRegressor(),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "LinearRegression": LinearRegression(),
        "XGBRegressor": XGBRegressor()
    }

    best_model = None
    best_model_name = ""
    best_mse = float("inf")
    best_mae = float("inf")
    best_rmse = float("inf")
    best_predictions = None

    for name, model in algorithms.items():
        # Identify columns with missing values in training data
        missing_cols = X_train_final.columns[X_train_final.isnull().any()]

        # Create a list of transformers for the pipeline
        transformers = []

        # Add the model to the pipeline
        transformers.append(('model', model))

        # Create the pipeline
        pipeline = Pipeline(transformers)

        pipeline.fit(X_train_final, y_train)
        y_pred = pipeline.predict(X_test_final)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        if mse < best_mse:
            best_mse = mse
            best_mae = mae
            best_rmse = rmse
            best_model = pipeline
            best_model_name = name
            best_predictions = y_pred

    return best_model, best_model_name, best_mse, best_mae, best_rmse, best_predictions


global_df = None
# Flask routes

@app.route('/', methods=['GET', 'POST'])
def home():
    problem_type = None
    target_column = None  # Initialize target_column
    try:
        if request.method == 'POST':
            if 'csvFile' not in request.files:
                return "No file part"

            file = request.files['csvFile']

            if file.filename == '':
                return "No selected file"

            # Check the file format by its extension
            file_extension = file.filename.split('.')[-1]
            if file_extension.lower() not in ['csv', 'xlsx', 'xls']:
                return "Invalid file format. Please upload a CSV or Excel file."

            # Check the file size
            if file.content_length > app.config['MAX_CONTENT_LENGTH']:
                return "File size exceeds the limit of 5MB."

            # Read the file based on its extension
            if file_extension.lower() == 'csv':
                df = pd.read_csv(file)
            else:
                # For Excel files (xlsx or xls), you can use pd.read_excel
                df = pd.read_excel(file)
            top_5 = df
            # Perform preprocessing and check if it's successful
            print("original dataset:-\n",df)
            preprocessed_df = preprocessing(df)
            global global_df
            global_df = preprocessed_df
            # Perform preprocessing and check if it's successful
            print("After preprocessed dataset:-\n",preprocessed_df)
            
            if preprocessed_df is not None:
                if 'problemType' in request.form and 'targetcol' in request.form and 'name' in request.form:
                    problem_type = request.form['problemType']
                    target_column = request.form['targetcol']

                    # Check if the target column exists in the DataFrame
                    if target_column not in preprocessed_df.columns:
                        return f"Target column '{target_column}' does not exist in the dataset. Please check your dataset."

                    # Check if the selected problem type matches the data type of the target column
                    if problem_type == 'Classification' and preprocessed_df[target_column].dtype not in ['object', 'int64', 'float64']:
                        return f"Selected problem type is Classification, but the target column '{target_column}' contains non-categorical data. Please select the correct problem type."
                    elif problem_type == 'Regression' and preprocessed_df[target_column].dtype not in ['int64', 'float64']:
                        return f"Selected problem type is Regression, but the target column '{target_column}' contains non-numeric data. Please select the correct problem type."

                    project_title = request.form['name']

                    if problem_type == 'Classification':
                        result = prediction(
                            preprocessed_df, target_column, problem_type)
                        if isinstance(result, tuple) and len(result) == 4:
                            best_model, best_model_name, best_accuracy, report = result

                            # Save the best classification model
                            model_filename = f"{project_title}_best_classification_model.joblib"
                            joblib.dump(best_model, model_filename)

                            # Provide a download link for the model
                            model_link = f"/download/{model_filename}"
                            # Pass the best model name
                            return render_template('home.html', table=top_5, problem_type=problem_type, target_column=target_column, project_title=project_title, prediction_report=report, accuracy=best_accuracy * 100, model_link=model_link, best_model_name=best_model_name)

                    elif problem_type == 'Regression':
                        best_model, best_model_name, best_mse,best_mae,best_rmse, predictions = prediction(
                            preprocessed_df, target_column, problem_type)
                        # Save the best regression model
                        model_filename = f"{project_title}_best_regression_model.joblib"
                        joblib.dump(best_model, model_filename)

                        # Provide a download link for the model
                        model_link = f"/download/{model_filename}"

                        # Pass the best model name and mse
                        return render_template('home.html', table=top_5, problem_type=problem_type, target_column=target_column, project_title=project_title, mse=best_mse,mae=best_mae,rmse=best_rmse, predictions=predictions, model_link=model_link, best_model_name=best_model_name)
                    else:
                        return f"Other ML Algorithms is not implemented till, Only Regression and Classification problems can be by this application. In future we will implement all possible ML Type's Algorithms📢.Sorry for this🙏🚫, we couldn't help you😒🚫.Best of luck for next time👍🤞"
        return render_template('home.html', table=None, error_message=None, prediction_report=None, problem_type=problem_type, target_column=target_column)
    except Exception as e:
        error_message = str(e)
        return render_template('home.html', table=None, error_message=error_message, prediction_report=None, problem_type=problem_type, target_column=target_column)



##PandasAI function
@app.route("/chat-with-dataset",methods=['GET','POST'])
def chat_datasets():
    question=None
    error_message=None
    table=None
    if request.method=="POST":
        question=request.form.get('question')
    try:
        global global_df  # Access the global DataFrame
        if global_df is None:
            return Response("No dataset available. Please upload a CSV file first.")
        df = SmartDataframe(global_df, config={"llm": llm})
        if question:
            result = df.chat(question)
            # Print the result (you can modify this part to use the result as needed)
            print(result)
                # Return a response (you can customize this response)
            return render_template("chat.html",table=global_df,result=result,question=question)
            # return Response("Chat with dataset: " + result)
        
    except Exception as e:
        # Handle any exceptions here and return an appropriate response
        error_message = "Somethings went wrong😒😔!"#str(e)
    return render_template("chat.html",table=global_df,message=error_message)
        # return Response("An error occurred: " + error_message, status=500)


##reports generating about of dataset function
report_html = None
@app.route("/dataset's reports", methods=['GET', 'POST'])
# Initialize report_html as a global variable
def generate_and_display_profile():
    global report_html  # Use the global report_html variable
    profile = None
    report_generated = False  # Add a variable to track report generation

    if request.method == 'POST':
        if 'csvFile' not in request.files:
            return "No file part"

        file = request.files['csvFile']

        if file.filename == '':
            return "No selected file"

        if not file.filename.endswith('.csv'):
            return "Invalid file format. Please upload a CSV file."

        # Check the file size
        if file.content_length > app.config['MAX_CONTENT_LENGTH']:
            return "File size exceeds the limit of 5MB."

        df = pd.read_csv(file)
        # Create a Pandas Profiling Report
        profile = ProfileReport(df, title="DataSet's Report")

        # Convert the report to HTML as a string
        report_html = profile.to_html()
        report_generated = True  # Set report_generated to True when the report is generated

    if report_html:
        # Option 1: Render the report HTML
        return render_template("generated_report.html", report_html=report_html, report_generated=report_generated)

    # Option 2: Provide a download link for the generated report
    return render_template('datasets_report.html', report_generated=report_generated)


@app.route('/download_report')
def download_report():
    global report_html  # Use the global report_html variable
    if report_html is not None:
        # Create an in-memory file object to store the HTML content
        report_file = io.BytesIO()
        report_file.write(report_html.encode('utf-8'))
        report_file.seek(0)

        # Return the HTML file as an attachment for download
        return send_file(report_file, as_attachment=True, download_name='report.html')

    return "Report not found"


## this is for production based function
# @app.route('/download/<model_filename>')
# def download_model(model_filename):
#     # Build the full path to the model file
#     model_path = os.path.join(MODEL_DIRECTORY, model_filename)

#     # Check if the file exists
#     if not os.path.exists(model_path):
#         abort(404)

#     # Serve the model file for download as an attachment
#     return send_file(model_path, as_attachment=True)



## This is for Local system function
@app.route('/download/<model_filename>')
def download_model(model_filename):
    return send_file(model_filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True,port=3355)
