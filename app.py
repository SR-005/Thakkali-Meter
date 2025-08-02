from flask import Flask, render_template, request, send_file
import os
import uuid
from thakkalimeter import estimate_tomatoes

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            output_path = os.path.join(RESULT_FOLDER, filename)
            count, _ = estimate_tomatoes(filepath, tomato_path="static/tomato.png", save_path=output_path)

            return render_template("index.html", tomato_count=count, result_img=output_path)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
