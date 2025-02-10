import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.model_selection import RandomizedSearchCV
#from scipy.stats import uniform, randint
from xgboost import XGBClassifier  
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_curve, roc_auc_score, roc_curve, accuracy_score, confusion_matrix, classification_report, recall_score, precision_score, f1_score
from flask import Flask, redirect, url_for, request, render_template, jsonify, send_file, make_response, Response
import json
from pyngrok import ngrok
import os
import io
from werkzeug.utils import secure_filename
import base64
from datetime import datetime
from tabulate import tabulate
from time import time
import shap
from io import BytesIO
import matplotlib

matplotlib.use('agg')

static_folder = 'static'
template_folder = 'templates'
app = Flask(__name__,
            template_folder=template_folder,
            static_folder=static_folder)


uploaded_df = None  
X = None
y = None
X_train_data = None
X_train = None
X_test = None
y_train = None
y_test = None
X_train_scaled = None
X_test_scaled = None
LR_train_duration = None
DT_train_duration = None
XGB_train_duration = None
X_test_data = None

'''
param_distributions = {
    'Logistic Regression': {
        'C': uniform(0.1, 99.9),  
        'solver': ['liblinear', 'lbfgs']
    },
    'Decision Tree': {
        'max_depth': [None] + list(randint(1, 51).rvs(10)),  
        'min_samples_split': randint(2, 21)  
    },
    'XGBoost': {
        'n_estimators': randint(50, 1001),  
        'learning_rate': uniform(0.01, 0.29)  
    }
}
'''

param_grids = {
    'Logistic Regression': {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    },
    'Decision Tree': {
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2, 5],
        'min_samples_split': [2, 5, 10]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    }
}

tuned_params = {
    'Logistic Regression': {},
    'Decision Tree': {},
    'XGBoost': {}
}

LR_model = LogisticRegression(max_iter=100000)

DT_model = DecisionTreeClassifier(random_state=42)

XGB_model = XGBClassifier()



@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload():
    global uploaded_df

    if 'file' not in request.files:
        return jsonify({"message": "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"message": "No selected file"})

    if file:
        uploaded_df = pd.read_csv(file, encoding='utf-8')

        return redirect(url_for("konten"))


@app.route('/konten')
def konten():
    return render_template("konten.html")

@app.route('/download_guide')
def download_guide():
    path = 'static/guide/Panduan.txt'
    return send_file(path, as_attachment=True)

@app.route('/view_dataframe')
def view_dataframe():
    global uploaded_df

    df_head = uploaded_df.head()
    df_tail = uploaded_df.tail()

    df_head_html = df_head.to_html()
    df_tail_html = df_tail.to_html()

    dataframe_html = render_template('df.html',
                                     df_head_html=df_head_html,
                                     df_tail_html=df_tail_html)

    return jsonify({"message": dataframe_html})


@app.route('/info_dataset')
def info_dataset():
    global uploaded_df

    try:
        info = {
            'column': uploaded_df.columns.tolist(),
            'dtype': uploaded_df.dtypes.tolist(),
            'non_null_count': uploaded_df.notnull().sum().tolist()
        }

        dtype_counts = uploaded_df.dtypes.value_counts().to_dict()

        app.logger.info(f'Extracted info: {info}')
        app.logger.info(f'Data types summary: {dtype_counts}')

        table_html = render_template('info_data.html', info=info, dtype_counts=dtype_counts)

        app.logger.info(f'Generated HTML: {table_html}')

        return jsonify({"message": table_html})

    except Exception as e:
        app.logger.error(f'Error in info_dataset: {e}')
        return jsonify({"message": f"An error occurred: {e}"}), 500


@app.route('/check_nan')
def check_nan():
    global uploaded_df

    nan_counts = uploaded_df.isna().sum()
    nan = nan_counts.to_string()
    nan_html = render_template('cek_nan.html', nan=nan)

    return jsonify({"message": nan_html})


@app.route('/remove_nan')
def remove_nan():
    global uploaded_df

    cleaned_df = uploaded_df.dropna()
    removed_rows_count = len(uploaded_df) - len(cleaned_df)
    uploaded_df = cleaned_df  

    return jsonify(
        {"message": f"{removed_rows_count} rows with NaN values removed."})


@app.route('/check_numeric')
def check_numeric():
    global uploaded_df

    try:
        for column_name in uploaded_df.columns:
            if uploaded_df[column_name].dtype == 'object':
                return jsonify({"message": "Data not numeric"})
        return jsonify({"message": "Data is already numeric."})
    except Exception as e:
        return jsonify({"message": f"Failed to check the data: {str(e)}"})


'''@app.route('/change_data_type')
def change_data_type():
    global uploaded_df

    try:
        # Melakukan one-hot encoding pada kolom-kolom dengan tipe 'object'
        uploaded_df = pd.get_dummies(uploaded_df, columns=uploaded_df.select_dtypes(include=['object']).columns)

        return jsonify({
            "message": "Successfully applied One-Hot Encoding to string columns."
        })
    except Exception as e:
        return jsonify({"message": f"Failed to apply One-Hot Encoding: {str(e)}"})'''


@app.route('/change_data_type')
def change_data_type():
    global uploaded_df

    try:
        if uploaded_df.isnull().values.any():
            return jsonify({"message": "Please remove NaN values before applying Label Encoding."})
            
        for column_name in uploaded_df.columns:
            if uploaded_df[column_name].dtype == 'object':
                label_encoder = LabelEncoder()
                encoded_values = label_encoder.fit_transform(
                    uploaded_df[column_name])
                uploaded_df[column_name] = encoded_values.astype('float64')
        return jsonify({
            "message":
            "Successfully applied Label Encoding to string columns."
        })
    except Exception as e:
        return jsonify(
            {"message": f"Failed to apply Label Encoding: {str(e)}"})


def generate_correlation_heatmap(file_path):
    global uploaded_df

    plt.figure(figsize=(35, 35))
    correlation_matrix = uploaded_df.corr(method='spearman').round(5)
    heatmap = sns.heatmap(data=correlation_matrix,
                          annot=True,
                          cmap='coolwarm',
                          linewidths=0.7)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, horizontalalignment='right')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.title("Correlation Matrix for Numeric Features (Spearman)", size=20)

    plt.savefig(file_path, format='png')
    plt.close()
  


@app.route('/download_correlation_matrix')
def download_correlation_matrix():

    tmp_file = '/tmp/correlation_matrix.png'  

    try:
        generate_correlation_heatmap(tmp_file)

        response = make_response(send_file(tmp_file, mimetype='image/png'))
        response.headers['Content-Disposition'] = 'attachment; filename=correlation_matrix.png'
        return response

    except Exception as e:
        return jsonify({"message": f"Failed to generate or download the correlation matrix: {str(e)}"})


@app.route('/remove_column')
def remove_column():
    global uploaded_df
    column_name = request.args.get('column')

    if column_name not in uploaded_df.columns:
        return jsonify({"message": f"Column '{column_name}' not found."})

    uploaded_df.drop([column_name], inplace=True,
                     axis=1)  
    return jsonify(
        {"message": f"Column '{column_name}' removed successfully."})


@app.route('/set_target')
def set_target():
    global uploaded_df, X, y
    target_column = request.args.get('column')

    if target_column not in uploaded_df.columns:
        return jsonify({"message": f"Column '{target_column}' not found."})

    X = uploaded_df.drop([target_column],
                         axis=1)  
    y = uploaded_df[target_column]  

    return jsonify(
        {"message": f"Target column set to '{target_column}' successfully."})


@app.route('/split_data')
def split_data():
    global X, y, X_train, X_test, y_train, y_test

    test_size_str = request.args.get('test_size')
    if test_size_str is None or not test_size_str.strip():
        return jsonify({
            "message":
            "Test size or target column not set. Please set test size or target column first."
        })

    try:
        test_size = float(test_size_str) / 100
        if test_size <= 0 or test_size >= 1:
            raise ValueError(
                "Test size must be a percentage between 0 and 100.")
    except ValueError:
        return jsonify({
            "message":
            "Invalid test size. Test size must be a numeric value between 1 and 99."
        })

    if y is None:
        return jsonify({
            "message":
            "Target column not set. Please set the target column first."
        })

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=42)
    return jsonify({
        "message":
        f"Data successfully split. Train size: {len(X_train)}, Test size: {len(X_test)}"
    })


@app.route('/normalize_data')
def normalize_data():
    global uploaded_df, X_train, X_test, X_train_scaled, X_test_scaled

    # Check if data is numeric
    for column_name in uploaded_df.columns:
        if uploaded_df[column_name].dtype == 'object':
            return jsonify({"message": "Data not numeric. Please change data type to numeric first before normalizing data."})

    if X_train is None or X_test is None:
        return jsonify(
            {"message": "Data not split yet. Please split the data first before normalizing data."})

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return jsonify({"message": "Data normalized successfully."})

@app.route('/check_normalize')
def check_data_status():
    global X_train_scaled, X_test_scaled
    
    try:
        if X_train_scaled is None or X_test_scaled is None:
            return jsonify({"message": "Data not scaled. Please normalize data first before performing Hyperparameter Tuning."})
        else: 
            return jsonify({"message": "Data is already normalize."})
            
    except Exception as e:
        return jsonify({"message": f"Failed to check the data: {str(e)}"})


'''
@app.route('/random_search')
def random_search():
    global X_train_scaled, y_train, LR_model, DT_model, XGB_model, param_distributions

    results = {}

    models = {
        'Logistic Regression': LR_model,
        'Decision Tree': DT_model,
        'XGBoost': XGB_model
    }

    for model_name, model in models.items():
        param_dist = param_distributions.get(model_name, {})
        if not param_dist:
            continue

        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
        random_search.fit(X_train_scaled, y_train)

        best_params = random_search.best_params_
        best_score = random_search.best_score_

        results[model_name] = {
            'best_params': best_params,
            'best_score': best_score
        }

    return jsonify(results)
'''

@app.route('/grid_search')
def grid_search():
    global X_train_scaled, y_train, LR_model, DT_model, XGB_model, param_grids

    results = {}  

    ordered_models = [
        ('Logistic Regression', LR_model),
        ('Decision Tree', DT_model),
        ('XGBoost', XGB_model)
    ]

    for model_name, model in ordered_models:
        param_grid = param_grids.get(model_name, {})
        if not param_grid:
            continue

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        best_score_percent = "{:.2f}".format(best_score * 100)

        results[model_name] = {
            'best_params': best_params,
            'best_score': f"{best_score_percent}%"
        }

    return Response(json.dumps(results), mimetype='application/json')

@app.route('/tuning')
def tuning_page():     
    return render_template('tuning.html')

@app.route('/set_tuning_params', methods=['POST'])
def set_tuning_params():
    global tuned_params
    params = request.json
    tuned_params['Logistic Regression'] = params.get('logistic_regression')
    tuned_params['Decision Tree'] = params.get('decision_tree')
    tuned_params['XGBoost'] = params.get('xgboost')
    return jsonify({"message": "Tuning parameters set successfully."})

@app.route('/train_logistic_regression')
def train_logistic_regression():
    global X_train, X_test, X_train_scaled, y_train, LR_model, LR_train_duration, tuned_params, X_train_data

    if X_train is None or X_test is None:
        return jsonify(
            {"message": "Data not split yet. Please split the data first before Logistic Regression training."})

    X_train_data = X_train_scaled if X_train_scaled is not None else X_train

    C = tuned_params.get('Logistic Regression', {}).get('C')
    solver = tuned_params.get('Logistic Regression', {}).get('solver')

    lr_params = {}

    if C is not None:
        lr_params['C'] = C
    if solver is not None:
        lr_params['solver'] = solver

    LR_model = LogisticRegression(**lr_params)

    start_time = time()  

    LR_model.fit(X_train_data, y_train)

    end_time = time()  
    train_duration = end_time - start_time  
    LR_train_duration = train_duration

    train_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({
        "message":
        f"Logistic Regression Models trained successfully. Training completed at {train_time}. Training duration: {LR_train_duration:.2f} seconds."
    })


@app.route('/train_decision_tree')
def train_decision_tree():
    global X_train, X_test, X_train_scaled, y_train, DT_model, DT_train_duration, tuned_params, X_train_data

    if X_train is None or X_test is None:
        return jsonify(
            {"message": "Data not split yet. Please split the data first before Decision Tree training."})

    X_train_data = X_train_scaled if X_train_scaled is not None else X_train

    max_depth = tuned_params.get('Decision Tree', {}).get('max_depth')
    min_samples_split = tuned_params.get('Decision Tree', {}).get('min_samples_split')
    min_samples_leaf = tuned_params.get('Decision Tree', {}).get('min_samples_leaf')

    dt_params = {}

    if max_depth is not None:
        dt_params['max_depth'] = max_depth
    if min_samples_split is not None:
        dt_params['min_samples_split'] = min_samples_split
    if min_samples_leaf is not None:
        dt_params['min_samples_leaf'] = min_samples_leaf

    DT_model = DecisionTreeClassifier(**dt_params)

    start_time = time()  

    DT_model.fit(X_train_data, y_train)

    end_time = time() 
    train_duration = end_time - start_time 
    DT_train_duration = train_duration

    train_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({
        "message":
        f"Decision Tree Models trained successfully. Training completed at {train_time}. Training duration: {DT_train_duration:.2f} seconds."
    })


@app.route('/train_xgboost')
def train_xgboost():
    global X_train, X_test, X_train_scaled, y_train, XGB_model, XGB_train_duration, tuned_params, X_train_data

    if X_train is None or X_test is None:
        return jsonify(
            {"message": "Data not split yet. Please split the data first before XGBoost training."})

    X_train_data = X_train_scaled if X_train_scaled is not None else X_train

    learning_rate = tuned_params.get('XGBoost', {}).get('learning_rate')
    n_estimators = tuned_params.get('XGBoost', {}).get('n_estimators')

    xgb_params = {}

    if learning_rate is not None:
        xgb_params['learning_rate'] = learning_rate
    if n_estimators is not None:
        xgb_params['n_estimators'] = n_estimators

    XGB_model = XGBClassifier(**xgb_params)

    start_time = time()  

    XGB_model.fit(X_train_data, y_train)

    end_time = time()  
    train_duration = end_time - start_time  
    XGB_train_duration = train_duration

    train_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return jsonify({
        "message":
        f"XGBoost Models trained successfully. Training completed at {train_time}. Training duration: {XGB_train_duration:.2f} seconds."
    })


@app.route('/view_model_results')
def view_model_results():
    global LR_model, DT_model, XGB_model, X_train, X_test, y_test, X_test_scaled, LR_train_duration, DT_train_duration, XGB_train_duration, X_test_data, LR_y_pred, DT_y_pred, XGB_y_pred

    if X_train is None or X_test is None:
        return jsonify(
            {"message": "Data not split yet. Please split the data first before view the results."})

    if not hasattr(LR_model, "coef_") or not hasattr(
            DT_model, "tree_") or not hasattr(XGB_model, "get_booster"):
        return jsonify({"message": "models have not been trained yet."})

    X_test_data = X_test_scaled if X_test_scaled is not None else X_test
    

    LR_y_pred = LR_model.predict(X_test_data)
    LR_accuracy = accuracy_score(y_test, LR_y_pred)
    LR_accuracy_percent = "{:.2f}%".format(LR_accuracy * 100)
    LR_recall = recall_score(y_test, LR_y_pred, average='weighted')
    LR_recall_percent = "{:.2f}%".format(LR_recall * 100)
    LR_precision = precision_score(y_test, LR_y_pred, average='weighted')
    LR_precision_percent = "{:.2f}%".format(LR_precision * 100)
    LR_f1 = f1_score(y_test, LR_y_pred, average='weighted')
    LR_f1_percent = "{:.2f}%".format(LR_f1 * 100)
    LR_support = y_test.size
    LR_conf_matrix = confusion_matrix(y_test, LR_y_pred)
    LR_class_report = classification_report(y_test, LR_y_pred)
    LR_TN, LR_FP, LR_FN, LR_TP = LR_conf_matrix.ravel()
    LR_support_negative = LR_TN + LR_FP
    LR_support_positive = LR_FN + LR_TP
    LR_Predicted_Negative = LR_TN
    LR_Actual_Negative = LR_FP + LR_TN
    LR_Acc = LR_accuracy*100
    LR_Accp = "{:.2f}%".format(LR_Acc)

    DT_y_pred = DT_model.predict(X_test_data)
    DT_accuracy = accuracy_score(y_test, DT_y_pred)
    DT_accuracy_percent = "{:.2f}%".format(DT_accuracy * 100)
    DT_recall = recall_score(y_test, DT_y_pred, average='weighted')
    DT_recall_percent = "{:.2f}%".format(DT_recall * 100)
    DT_precision = precision_score(y_test, DT_y_pred, average='weighted')
    DT_precision_percent = "{:.2f}%".format(DT_precision * 100)
    DT_f1 = f1_score(y_test, DT_y_pred, average='weighted')
    DT_f1_percent = "{:.2f}%".format(DT_f1 * 100)
    DT_support = y_test.size
    DT_conf_matrix = confusion_matrix(y_test, DT_y_pred)
    DT_class_report = classification_report(y_test, DT_y_pred)
    DT_TN, DT_FP, DT_FN, DT_TP = DT_conf_matrix.ravel()
    DT_support_negative = DT_TN + DT_FP
    DT_support_positive = DT_FN + DT_TP
    DT_Predicted_Negative = DT_TN
    DT_Actual_Negative = DT_FP + DT_TN
    DT_Acc = DT_accuracy*100
    DT_Accp = "{:.2f}%".format(DT_Acc)

    XGB_y_pred = XGB_model.predict(X_test_data)
    XGB_accuracy = accuracy_score(y_test, XGB_y_pred)
    XGB_accuracy_percent = "{:.2f}%".format(XGB_accuracy * 100)
    XGB_recall = recall_score(y_test, XGB_y_pred, average='weighted')
    XGB_recall_percent = "{:.2f}%".format(XGB_recall * 100)
    XGB_precision = precision_score(y_test, XGB_y_pred, average='weighted')
    XGB_precision_percent = "{:.2f}%".format(XGB_precision * 100)
    XGB_f1 = f1_score(y_test, XGB_y_pred, average='weighted')
    XGB_f1_percent = "{:.2f}%".format(XGB_f1 * 100)
    XGB_support = y_test.size
    XGB_conf_matrix = confusion_matrix(y_test, XGB_y_pred)
    XGB_class_report = classification_report(y_test, XGB_y_pred)
    XGB_TN, XGB_FP, XGB_FN, XGB_TP = XGB_conf_matrix.ravel()
    XGB_support_negative = XGB_TN + XGB_FP
    XGB_support_positive = XGB_FN + XGB_TP
    XGB_Predicted_Negative = XGB_TN
    XGB_Actual_Negative = XGB_FP + XGB_TN
    XGB_Predicted_Positive = XGB_TP
    XGB_Acc = XGB_accuracy*100
    XGB_Accp = "{:.2f}%".format(XGB_Acc)

    conclusions = {
        "accuracy": "",
        "recall": "",
        "precision": "",
        "f1_score": ""
    }

    if LR_accuracy_percent > DT_accuracy_percent and LR_accuracy_percent > XGB_accuracy_percent:
        conclusions["accuracy"] = "Logistic Regression"
    elif DT_accuracy_percent > LR_accuracy_percent and DT_accuracy_percent > XGB_accuracy_percent:
        conclusions["accuracy"] = "Decision Tree"
    else:
        conclusions["accuracy"] = "XGBoost"

    if LR_recall_percent > DT_recall_percent and LR_recall_percent > XGB_recall_percent:
        conclusions["recall"] = "Logistic Regression"
    elif DT_recall_percent > LR_recall_percent and DT_recall_percent > XGB_recall_percent:
        conclusions["recall"] = "Decision Tree"
    else:
        conclusions["recall"] = "XGBoost"

    if LR_precision_percent > DT_precision_percent and LR_precision_percent > XGB_precision_percent:
        conclusions["precision"] = "Logistic Regression"
    elif DT_precision_percent > LR_precision_percent and DT_precision_percent > XGB_precision_percent:
        conclusions["precision"] = "Decision Tree"
    else:
        conclusions["precision"] = "XGBoost"

    if LR_f1_percent > DT_f1_percent and LR_f1_percent > XGB_f1_percent:
        conclusions["f1_score"] = "Logistic Regression"
    elif DT_f1_percent > LR_f1_percent and DT_f1_percent > XGB_f1_percent:
        conclusions["f1_score"] = "Decision Tree"
    else:
        conclusions["f1_score"] = "XGBoost"

    conclusion = " ".join([f"{key}: {value}" for key, value in conclusions.items()])


    def format_confusion_matrix(matrix):
        table = tabulate(matrix,
                         tablefmt="html")
        table = table.replace('<table>', '<table id="confusion-matrix-table">')
        return table

    result_string = result_string = f"""
    <!DOCTYPE html>
<html>
<head>
    <title>Model Evaluation Results</title>
    <style> 
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
        }}
        .model-container {{
            display: flex;
            flex-direction: column; /* Mengatur arah flex menjadi kolom */
            margin-bottom: 50px;
            overflow-x: auto; /* Menambahkan scroll horizontal */
        }}
        .model-sections {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
            overflow-x: auto;
            padding-bottom: 20px;
        }}
        .model-section {{
            flex: 1;
            margin-right: 20px;
            border: 2px solid #ccc;
            padding: 20px;
            border-radius: 16px;
            background-color: white;
            min-width: 500px; /* Menambahkan lebar minimum untuk setiap section */
            margin-top: 55px;
        }}
        h2 {{
            color: #2E86C1;
        }}
        p.conc {{
            border: 2px solid #ccc;
            padding: 20px;
            border-radius: 15px;
            background-color: white;
            margin-right: 10px;
        }}
        .conclusion {{
            text-align: left;
            justify-content: flex-start;
            margin-right: 10px;
            background: white;
            padding: 20px;
            border-radius: 16px;
        }}
    </style>
</head>
<body>
    <div class="model-container">
        <div class="model-sections">
            <div class="model-section">
                <h2 style="margin-bottom:20px;">Logistic Regression</h2>
                <p><strong>Training Duration:</strong> {LR_train_duration:.2f} seconds</p>
                <p><strong>Accuracy:</strong> {LR_accuracy_percent}</p>
                <p><strong>Recall:</strong> {LR_recall_percent}</p>
                <p><strong>Precision:</strong> {LR_precision_percent}</p>
                <p><strong>F1 Score:</strong> {LR_f1_percent}</p>
                <p><strong>Support:</strong> {LR_support}</p>
                <p style="margin-top:30px;"><strong>Confusion Matrix:</strong></p>
                {format_confusion_matrix(LR_conf_matrix)}
                <p style="margin-top:30px;"><strong>Classification Report:</strong></p>
                <pre>{LR_class_report}<br></pre>
                <p style="margin-top:30px;">
                <p style="margin-top:30px;">
                <p style="margin-top:30px;">
                    Dari <span style="font-weight: bold;">{LR_support}</span> data pengujian yang terdiri dari <span style="font-weight: bold;">{LR_support_negative}</span> kelas negatif dan <span style="font-weight: bold;">{LR_support_positive}</span> kelas positif, model memiliki akurasi sebesar <span style="font-weight: bold;">{LR_Accp}</span> dalam melakukan prediksi benar. Sebanyak <span style="font-weight: bold;">{LR_TN}</span> dari total <span style="font-weight: bold;">{LR_support_negative}</span> kelas negatif dapat diprediksi benar oleh model. Sedangkan dari kelas positif, <span style="font-weight: bold;">{LR_support_positive}</span> data, model mampu memprediksi benar sebanyak <span style="font-weight: bold;">{LR_TP}</span> kelas positif.
                </p>
            </div>

            <div class="model-section">
                <h2 style="margin-bottom:20px;">Decision Tree</h2>
                <p><strong>Training Duration:</strong> {DT_train_duration:.2f} seconds</p>
                <p><strong>Accuracy:</strong> {DT_accuracy_percent}</p>
                <p><strong>Recall:</strong> {DT_recall_percent}</p>
                <p><strong>Precision:</strong> {DT_precision_percent}</p>
                <p><strong>F1 Score:</strong> {DT_f1_percent}</p>
                <p><strong>Support:</strong> {DT_support}</p>
                <p style="margin-top:30px;"><strong>Confusion Matrix:</strong></p>
                {format_confusion_matrix(DT_conf_matrix)}
                <p style="margin-top:30px;"><strong>Classification Report:</strong></p>
                <pre>{DT_class_report}<br></pre>
                <p style="margin-top:30px;">
                    Dari <span style="font-weight: bold;">{DT_support}</span> data pengujian yang terdiri dari <span style="font-weight: bold;">{DT_support_negative}</span> kelas negatif dan <span style="font-weight: bold;">{DT_support_positive}</span> kelas positif, model memiliki akurasi sebesar <span style="font-weight: bold;">{DT_Accp}</span> dalam melakukan prediksi benar. Sebanyak <span style="font-weight: bold;">{DT_TN}</span> dari total <span style="font-weight: bold;">{DT_support_negative}</span> kelas negatif dapat diprediksi benar oleh model. Sedangkan dari kelas positif, <span style="font-weight: bold;">{DT_support_positive}</span> data, model mampu memprediksi benar sebanyak <span style="font-weight: bold;">{DT_TP}</span> kelas positif.
                </p>
            </div>

            <div class="model-section">
                <h2 style="margin-bottom:20px;">XGBoost</h2>
                <p><strong>Training Duration:</strong> {XGB_train_duration:.2f} seconds</p>
                <p><strong>Accuracy:</strong> {XGB_accuracy_percent}</p>
                <p><strong>Recall:</strong> {XGB_recall_percent}</p>
                <p><strong>Precision:</strong> {XGB_precision_percent}</p>
                <p><strong>F1 Score:</strong> {XGB_f1_percent}</p>
                <p><strong>Support:</strong> {XGB_support}</p>
                <p style="margin-top:30px;"><strong>Confusion Matrix:</strong></p>
                {format_confusion_matrix(XGB_conf_matrix)}
                <p style="margin-top:30px;"><strong>Classification Report:</strong></p>
                <pre>{XGB_class_report}<br></pre>
                <p style="margin-top:30px;">
                    Dari <span style="font-weight: bold;">{XGB_support}</span> data pengujian yang terdiri dari <span style="font-weight: bold;">{XGB_support_negative}</span> kelas negatif dan <span style="font-weight: bold;">{XGB_support_positive}</span> kelas positif, model memiliki akurasi sebesar <span style="font-weight: bold;">{XGB_Accp}</span> dalam melakukan prediksi benar. Sebanyak <span style="font-weight: bold;">{XGB_TN}</span> dari total <span style="font-weight: bold;">{XGB_support_negative}</span> kelas negatif dapat diprediksi benar oleh model. Sedangkan dari kelas positif, <span style="font-weight: bold;">{XGB_support_positive}</span> data, model mampu memprediksi benar sebanyak <span style="font-weight: bold;">{XGB_TP}</span> kelas positif.
                </p>
            </div>
        </div>
        <div class="conclusion">
            <p style="margin-bottom:0px; font-weight:bold;">Kesimpulan:</p>
            <p>
                Model yang terbaik untuk setiap metrik evaluasi adalah: 
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Accuracy</th>
                        <th>Recall</th>
                        <th>Precision</th>
                        <th>F1-Score</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>{conclusions['accuracy']}</td>
                        <td>{conclusions['recall']}</td>
                        <td>{conclusions['precision']}</td>
                        <td>{conclusions['f1_score']}</td>
                    </tr>
                </tbody>
            </table>
        </div> 
    </div>
</body>
</html>

    """

    return jsonify({"message": result_string})


@app.route('/view_shap')
def view_shap():
    global XGB_model, X_test, X_test_scaled

    if XGB_model is None or X_test_scaled is None:
        return jsonify({"message": "Model or test data not available."})

    feature_names = X_test.columns.tolist()

    # Initialize the SHAP explainer
    explainer = shap.Explainer(XGB_model)

    # Calculate SHAP values
    shap_values = explainer(X_test_scaled)
    
    # Create a summary plot
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, features=X_test_scaled,                     
    feature_names=feature_names, plot_type="bar")
    plt.title("SHAP Summary Plot for XGBoost")

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    
    result_string = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SHAP Summary Plot</title>
    </head>
    <body>
        <h2>SHAP Summary Plot for XGBoost</h2>
        <img src="data:image/png;base64,{image_base64}" alt="SHAP Summary Plot">
    </body>
    </html>
    """

    return jsonify({"message": result_string})

@app.route('/check_feature_testing_data_lr')
def check_feature_testing_data_lr():
    global X_test_data

    try:
        if X_test_data is None: 
            return jsonify({"message": "Feature testing data is not available. Please click view result button before download predicted LR CSV."})
        else:
            return jsonify({"message": "Feature testing data are available."})

    except Exception as e:
        return jsonify({"message": f"Failed to check the data: {str(e)}"})

@app.route('/check_feature_testing_data_dt')
def check_feature_testing_data_dt():
    global X_test_data

    try:
        if X_test_data is None: 
            return jsonify({"message": "Feature testing data is not available. Please click view result button before download predicted DT CSV."})
        else:
            return jsonify({"message": "Feature testing data are available."})
            
    except Exception as e:
        return jsonify({"message": f"Failed to check the data: {str(e)}"})

@app.route('/check_feature_testing_data_xgb')
def check_feature_testing_data_xgb():
    global X_test_data

    try:
        if X_test_data is None: 
            return jsonify({"message": "Feature testing data is not available. Please click view result button before download predicted XGB CSV."})
        else:
            return jsonify({"message": "Feature testing data are available."})

    except Exception as e:
        return jsonify({"message": f"Failed to check the data: {str(e)}"})

@app.route('/download_lr_predictions')
def download_lr_predictions():
    global X_test_data

    lr_y_pred = LR_model.predict(X_test_data)
    
    LR_output = io.BytesIO()
    LR_result_df = pd.DataFrame(X_test_data)
    LR_result_df['Predicted'] = lr_y_pred
    LR_result_df.to_csv(LR_output, index=False)
    LR_output.seek(0)

    LR_test_results_csv = LR_output

    return send_file(LR_test_results_csv, mimetype='text/csv', as_attachment=True, download_name='LR_test_results.csv')

@app.route('/download_dt_predictions')
def download_dt_predictions():
    global X_test_data

    dt_y_pred = DT_model.predict(X_test_data)

    DT_output = io.BytesIO()
    DT_result_df = pd.DataFrame(X_test_data)
    DT_result_df['Predicted'] = dt_y_pred
    DT_result_df.to_csv(DT_output, index=False)
    DT_output.seek(0)

    DT_test_results_csv = DT_output

    return send_file(DT_test_results_csv, mimetype='text/csv', as_attachment=True, download_name='DT_test_results.csv')

@app.route('/download_xgb_predictions')
def download_xgb_predictions():
    global X_test_data

    xgb_y_pred = XGB_model.predict(X_test_data)

    XGB_output = io.BytesIO()
    XGB_result_df = pd.DataFrame(X_test_data)
    XGB_result_df['Predicted'] = xgb_y_pred
    XGB_result_df.to_csv(XGB_output, index=False)
    XGB_output.seek(0)

    XGB_test_results_csv = XGB_output

    return send_file(XGB_test_results_csv, mimetype='text/csv', as_attachment=True, download_name='XGB_test_results.csv')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Gunakan port dari environment variable
    app.run(host='0.0.0.0', port=port)  # Wajib pakai host='0.0.0.0' agar bisa diakses di Fly.io
    
'''
if __name__ == '__main__':
    ngrok.set_auth_token('2fdDpcQcNjj615RdcLkAwWubpyY_73aL3VdcrHEzN7dSJ2Ut1')

    url = ngrok.connect(5000)
    print("Public URL:", url)
    app.run()
    '''