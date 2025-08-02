from flask import Flask, render_template, redirect, url_for,request
from facemesh import generate_frames
import subprocess
import os

app = Flask(__name__)

def generate_roast(count):
    if count <= 1:
        return "You got one tomato? Face reveal cancelled. 🍅🚫"
    elif count == 2:
        return "Tiny face, massive disappointment. 🥲"
    elif count == 3:
        return "That’s... a modest face. Respectfully. 😐"
    elif count == 4:
        return "Mid sauce potential. Not bad, not thakkali. 😶"
    elif count == 5:
        return "Respectable! Sauce-worthy face detected. 🍝"
    elif count == 6:
        return "Thakkali certified. Saucy and proud. 🍅💅"
    elif count >= 7:
        return "Tomato overlord. Ketchup bow down. 🔥🍅👑"

@app.route('/')
def home():
    return render_template('index.html', show_image=False)

@app.route('/run_facemesh')
def run_facemesh():
    count = generate_frames()
    return redirect(url_for('show_result', count=count))

@app.route('/thakkalimeter')
def show_result():
    count = int(request.args.get("count", 0))

    # Optional roast comment
    roast = ""
    if count == 0:
        roast = "Bro you got anti-tomato energy 🧅"
    elif count < 5:
        roast = "You could barely fill a ketchup packet 😬"
    elif count < 10:
        roast = "Medium rare tomato head 🍅"
    else:
        roast = "Certified Tomato Warehouse 🏭"

    return render_template('index.html', show_image=True, count=count, roast=roast)

if __name__ == '__main__':
    app.run(debug=True)
