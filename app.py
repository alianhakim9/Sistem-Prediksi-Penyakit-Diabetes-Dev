from flask import Flask,request, url_for, redirect, render_template  ## importing necessary libraries
import pickle  ## pickle for loading model(Diabetes.pkl)
import pandas as pd  ## to convert the input data into a dataframe for giving as a input to the model

app = Flask(__name__)

model_file = open('diabetes.pkl', 'rb')
model = pickle.load(model_file, encoding='bytes')

@app.route('/')
def index():
    return render_template('index.html', insurance_cost=0)

@app.route('/predict', methods=['POST'])
def predict():
    #int_features=[float(x) for x in request.form.values()]
    text1 = request.form['1']      ## Fetching each input field value one by one
    text2 = request.form['2'] 
    text3 = request.form['3']
    text4 = request.form['4']
    text5 = request.form['5']
    text6 = request.form['6']
    text7 = request.form['7']
    text8 = request.form['8']
 
    row_df = pd.DataFrame([pd.Series([text1,text2,text3,text4,text5,text6,text7,text8])])  ### Creating a dataframe using all the values
    print(row_df)
    prediction=model.predict_proba(row_df) ## Predicting the output
    output='{0:.{1}f}'.format(prediction[0][1], 2)    ## Formating output

    if output>str(0.5):
        return render_template('index.html',pred='Anda berpeluang terkena diabetes.\nKemungkinan terkena Diabetes adalah {}'.format(output)) ## Returning the message for use on the same index.html page
    else:
        return render_template('index.html',pred='Anda aman.\n Kemungkinan menderita diabetes adalah {}'.format(output)) 


if __name__ == '__main__':
    app.run(debug=True)