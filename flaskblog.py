from flask import Flask, redirect, url_for, request, render_template
from werkzeug import secure_filename
import matplotlib.pyplot as plt 
import os 

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template("home.html")

@app.route('/', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      print(type(f))
      return render_template('home.html')

#@app.route('/')
#def show_index():
#    full_filename = "C:\\Users\\Nabilla Abraham\\Desktop\\hackxplore\\static\\output.jpg"
#    return render_template("home.html", user_image = full_filename)

if __name__ == '__main__' :
    app.run(debug=True)
