# Installing necessary Libraries, Packages, and Functions
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from datetime import datetime, date, timedelta


# Creating the flask app to establish a connection between the HTML and the python model code
flask_app = Flask(__name__)

# Defining a Flask route for the root URL of the web application
@flask_app.route("/")
def home():
    """
    Loading homepage HTML
    """

    return render_template("home.html")


@flask_app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
    return redirect(url_for('home'))


@flask_app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(flask_app.config['UPLOAD_PATH'], filename)


@flask_app.route("/analysis_hub", methods=["POST"])
def predict():
    """
    Defining Processing and Rendering logic for to tranform the
    input data sent by the html forms into formatted array values
    to create predictions for display on the HTML GUI
    """

    # obtaining predicted values through form request sent to NSure_model and formatting them to work with the model's format
    df = pd.read_csv(request.files['file'], delimiter=",")

    ph_color_class = {0:"square_ph0",
                      1:"square_ph1",
                      2:"square_ph2",
                      3:"square_ph3",
                      4:"square_ph4",
                      5:"square_ph5",
                      6:"square_ph6",
                      7:"square_ph7",
                      8:"square_ph8",
                      9:"square_ph9",
                      10:"square_ph10",
                      11:"square_ph11",
                      12:"square_ph12",
                      13:"square_ph13",
                      14:"square_ph14"}

    def fancy_dict(*args):
        'Pass in a list of tuples, which will be key/value pairs'
        ret = {}
        for k,v in args:
            for i in k:
                ret[i] = v
        return ret
    sm_color_class = fancy_dict((range(0,30), 'square_sm_es'),
                                (range(30,40), 'square_sm_s'),
                                (range(40,60), 'square_sm_o'),
                                (range(60,70), 'square_sm_e'),
                                (range(70,101), 'square_sm_saturation'))
    
    # returning values calculated using the ML model back to the HTML website
    return render_template(
        "analysis.html",
        ph1="{}".format(df.iloc[0,0:16].values[0]),
        ph2="{}".format(df.iloc[0,0:16].values[1]),
        ph3="{}".format(df.iloc[0,0:16].values[2]),
        ph4="{}".format(df.iloc[0,0:16].values[3]),
        ph5="{}".format(df.iloc[0,0:16].values[4]),
        ph6="{}".format(df.iloc[0,0:16].values[5]),
        ph7="{}".format(df.iloc[0,0:16].values[6]),
        ph8="{}".format(df.iloc[0,0:16].values[7]),
        ph9="{}".format(df.iloc[0,0:16].values[8]),
        ph10="{}".format(df.iloc[0,0:16].values[9]),
        ph11="{}".format(df.iloc[0,0:16].values[10]),
        ph12="{}".format(df.iloc[0,0:16].values[11]),
        ph13="{}".format(df.iloc[0,0:16].values[12]),
        ph14="{}".format(df.iloc[0,0:16].values[13]),
        ph15="{}".format(df.iloc[0,0:16].values[14]),
        ph16="{}".format(df.iloc[0,0:16].values[15]),
        ph_avg="{}".format(round(np.mean(df.iloc[0,0:16].values),2)),
        ph1_class= "{}".format(ph_color_class[round(df.iloc[0,0:16].values[0])]),
        ph2_class= "{}".format(ph_color_class[round(df.iloc[0,0:16].values[1])]),
        ph3_class= "{}".format(ph_color_class[round(df.iloc[0,0:16].values[2])]),
        ph4_class= "{}".format(ph_color_class[round(df.iloc[0,0:16].values[3])]),
        ph5_class= "{}".format(ph_color_class[round(df.iloc[0,0:16].values[4])]),
        ph6_class= "{}".format(ph_color_class[round(df.iloc[0,0:16].values[5])]),
        ph7_class= "{}".format(ph_color_class[round(df.iloc[0,0:16].values[6])]),
        ph8_class= "{}".format(ph_color_class[round(df.iloc[0,0:16].values[7])]),
        ph9_class= "{}".format(ph_color_class[round(df.iloc[0,0:16].values[8])]),
        ph10_class= "{}".format(ph_color_class[round(df.iloc[0,0:16].values[9])]),
        ph11_class= "{}".format(ph_color_class[round(df.iloc[0,0:16].values[10])]),
        ph12_class= "{}".format(ph_color_class[round(df.iloc[0,0:16].values[11])]),
        ph13_class= "{}".format(ph_color_class[round(df.iloc[0,0:16].values[12])]),
        ph14_class= "{}".format(ph_color_class[round(df.iloc[0,0:16].values[13])]),
        ph15_class= "{}".format(ph_color_class[round(df.iloc[0,0:16].values[14])]),
        ph16_class= "{}".format(ph_color_class[round(df.iloc[0,0:16].values[15])]),
        sm1="{}".format((df.iloc[0,16:32].values)[0]),
        sm2="{}".format((df.iloc[0,16:32].values)[1]),
        sm3="{}".format((df.iloc[0,16:32].values)[2]),
        sm4="{}".format((df.iloc[0,16:32].values)[3]),
        sm5="{}".format((df.iloc[0,16:32].values)[4]),
        sm6="{}".format((df.iloc[0,16:32].values)[5]),
        sm7="{}".format((df.iloc[0,16:32].values)[6]),
        sm8="{}".format((df.iloc[0,16:32].values)[7]),
        sm9="{}".format((df.iloc[0,16:32].values)[8]),
        sm10="{}".format((df.iloc[0,16:32].values)[9]),
        sm11="{}".format((df.iloc[0,16:32].values)[10]),
        sm12="{}".format((df.iloc[0,16:32].values)[11]),
        sm13="{}".format((df.iloc[0,16:32].values)[12]),
        sm14="{}".format((df.iloc[0,16:32].values)[13]),
        sm15="{}".format((df.iloc[0,16:32].values)[14]),
        sm16="{}".format((df.iloc[0,16:32].values)[15]),
        sm_avg="{}".format(round(np.mean(df.iloc[0,16:32].values),2)),
        sm1_class= "{}".format(sm_color_class[(df.iloc[0,16:32].values)[0]]),
        sm2_class= "{}".format(sm_color_class[(df.iloc[0,16:32].values)[1]]),
        sm3_class= "{}".format(sm_color_class[(df.iloc[0,16:32].values)[2]]),
        sm4_class= "{}".format(sm_color_class[(df.iloc[0,16:32].values)[3]]),
        sm5_class= "{}".format(sm_color_class[(df.iloc[0,16:32].values)[4]]),
        sm6_class= "{}".format(sm_color_class[(df.iloc[0,16:32].values)[5]]),
        sm7_class= "{}".format(sm_color_class[(df.iloc[0,16:32].values)[6]]),
        sm8_class= "{}".format(sm_color_class[(df.iloc[0,16:32].values)[7]]),
        sm9_class= "{}".format(sm_color_class[(df.iloc[0,16:32].values)[8]]),
        sm10_class= "{}".format(sm_color_class[(df.iloc[0,16:32].values)[9]]),
        sm11_class= "{}".format(sm_color_class[(df.iloc[0,16:32].values)[10]]),
        sm12_class= "{}".format(sm_color_class[(df.iloc[0,16:32].values)[11]]),
        sm13_class= "{}".format(sm_color_class[(df.iloc[0,16:32].values)[12]]),
        sm14_class= "{}".format(sm_color_class[(df.iloc[0,16:32].values)[13]]),
        sm15_class= "{}".format(sm_color_class[(df.iloc[0,16:32].values)[14]]),
        sm16_class= "{}".format(sm_color_class[(df.iloc[0,16:32].values)[15]]),
        t_minus0="{}".format((df.iloc[0,32:56].values)[23]),
        t_minus1="{}".format((df.iloc[0,32:56].values)[22]),
        t_minus2="{}".format((df.iloc[0,32:56].values)[21]),
        t_minus3="{}".format((df.iloc[0,32:56].values)[20]),
        t_minus4="{}".format((df.iloc[0,32:56].values)[19]),
        t_minus5="{}".format((df.iloc[0,32:56].values)[18]),
        t_minus6="{}".format((df.iloc[0,32:56].values)[17]),
        t_minus7="{}".format((df.iloc[0,32:56].values)[16]),
        t_minus8="{}".format((df.iloc[0,32:56].values)[15]),
        t_minus9="{}".format((df.iloc[0,32:56].values)[14]),
        t_minus10="{}".format((df.iloc[0,32:56].values)[13]),
        t_minus11="{}".format((df.iloc[0,32:56].values)[12]),
        t_minus12="{}".format((df.iloc[0,32:56].values)[11]),
        t_minus13="{}".format((df.iloc[0,32:56].values)[10]),
        t_minus14="{}".format((df.iloc[0,32:56].values)[9]),
        t_minus15="{}".format((df.iloc[0,32:56].values)[8]),
        t_minus16="{}".format((df.iloc[0,32:56].values)[7]),
        t_minus17="{}".format((df.iloc[0,32:56].values)[6]),
        t_minus18="{}".format((df.iloc[0,32:56].values)[5]),
        t_minus19="{}".format((df.iloc[0,32:56].values)[4]),
        t_minus20="{}".format((df.iloc[0,32:56].values)[3]),
        t_minus21="{}".format((df.iloc[0,32:56].values)[2]),
        t_minus22="{}".format((df.iloc[0,32:56].values)[1]),
        t_minus23="{}".format((df.iloc[0,32:56].values)[0]),
        t_avg="{}".format(round(np.mean(df.iloc[0,32:56].values),2)),
        t_year="{}".format(df.iloc[0,56]),
        t_month="{}".format(df.iloc[0,57]),
        t_day="{}".format(df.iloc[0,58]),
        t_hour="{}".format(df.iloc[0,59]),
        current_year="{}".format(datetime.now().year),
        current_month="{}".format(datetime.now().month),
        current_day="{}".format(datetime.now().day),


    )

# Enabling debug functionality during development of the web application
if __name__ == "__main__":
    flask_app.run(debug=True)