from pathlib import Path

# -*- coding: UTF-8 -*-
from flask import Flask, redirect, render_template, url_for, request, flash
from flask import session
import uuid
from flask_cors import CORS, cross_origin
import os

app = Flask(__name__)
cors = CORS(app)
print(app.root_path)
video_folder = os.path.join(app.root_path, "input")
os.makedirs(video_folder, exist_ok=True)
app.config['VIDEO_FOLDER'] = video_folder
app.secret_key = uuid.uuid4().hex

statements = [
    ["Sex ist für mich nebensächlich."],
    ["Ich finde klassische Geschlechterrollen bei Mann und Frau wichtig.", "Der Mann sollte schon meistens die Initiative übernehmen."],
    ["Ich möchte auch viel Zeit mit Freunden ohne meinen Partner verbringen."],
    ["Eifersucht und ein bisschen Kontrolle ist ein Ausdruck von Liebe."],
    ["Paare die Kinder bekommen, haben keinen Spaß mehr im Leben."],
    ["statement6 text und 6.jpeg in image folder legen"]
]

@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        username = request.form['usernameInput']
        session["username"] = username
        session["uuid"] = uuid.uuid4()
        return redirect(url_for('start', name=username))
    elif request.method == 'GET':
        return render_template("index.html")


@app.route("/start/<name>", methods=['POST', 'GET'])
def start(name):
    if request.method == 'GET':
        return render_template('waitRoom.html', name=session.get("username"))
    elif request.method == 'POST':
        return redirect(url_for('statement', id=1))


@app.route("/statement/<id>", methods=['POST', 'GET'])
def statement(id):
    if request.method == 'GET':
        session["statement_id"] = id
        statement = statements[int(id) - 1]
        last = False
        if int(id) == len(statements):
            last = True
        print(last)
        return render_template(f'statement.html', last=last, statement_id=id, statements=statement, name=session.get("username"))
    elif request.method == 'POST':
        return redirect(url_for('result'))


@app.route("/result/", methods=['POST', 'GET'])
def result():
    return render_template("result.html")


@app.route("/video/", methods=['POST'])
def video():
    # get username + id
    username = session.get("username")
    statement_id = session.get("statement_id")
    uuid = session.get("uuid")
    files = request.files
    if 'file' not in files:
        flash('No file part, redo')
        return redirect(url_for('statement', id=statement_id))
    file = files.get('file')
    # upload to own user video folder
    folder = os.path.join(app.config['VIDEO_FOLDER'], username + "_" + str(uuid))
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"statement_{statement_id}.webm")
    file.save(filename)
    new_statement_id = int(statement_id) + 1
    if new_statement_id > len(statements):
        return redirect(url_for('index'))
    return redirect(url_for('statement', id=new_statement_id))


if __name__ == "__main__":
    app.run(debug=True) #debug needed for dev (autom. refresh after changes)
