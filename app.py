from flask import Flask, render_template, redirect, url_for
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', show_image=False)

@app.route('/run_facemesh')
def run_facemesh():
    subprocess.run(["python", "facemesh.py"])
    return redirect(url_for('show_result'))

@app.route('/thakkalimeter')
def show_result():
    return render_template('index.html', show_image=True)

if __name__ == '__main__':
    app.run(debug=True)
