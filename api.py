from flask import Flask, render_template, url_for, request
import re
from sc_model import SC
import torch

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/process', methods=["POST"])
def process():
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        rawtext = model.predict(rawtext)


    return render_template("index.html", results=rawtext, rawtext=request.form['rawtext'])


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    model = SC.load("model.bin").to(device)
    app.run(port ='8881')
