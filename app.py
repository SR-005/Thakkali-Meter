from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/value", methods=["POST"])
def value_me():
    subprocess.Popen(["python", "facemesh.py"])
    return render_template("index.html", launched=True)

if __name__ == "__main__":
    app.run(debug=True)
