from flask import Flask, redirect, render_template, url_for, request, flash
from flask import session
import uuid
from flask_cors import CORS, cross_origin
import os

app = Flask(__name__)
cors = CORS(app)
video_folder = f"{app.root_path}/input"
os.makedirs(video_folder, exist_ok=True)
app.config['VIDEO_FOLDER'] = video_folder
app.secret_key = "c4206643c7794893a9ecbc258c89f9c1"

statements = [
    ["Sex ist für mich nebensächlich."],
    ["Ich finde klassische Geschlechterrollen bei Mann und Frau wichtig.", "Der Mann sollte schon meistens die Initiative übernehmen."],
    ["Ich möchte auch viel Zeit mit Freunden ohne meinen Partner verbringen."],
    ["Eifersucht und ein bisschen Kontrolle ist ein Ausdruck von Liebe."],
    ["Paare die Kinder bekommen, haben keinen Spaß mehr im Leben."],
    ["Mein:e Partner:in sollte eine erfolgreiche Karriere haben."],
    ["Religion hat für mich einen sehr hohen Stellenwert."],
    ["Sport spielt in meinem Leben eine wesentliche Rolle."],
    ["Ich würde gerne irgendwann im Ausland leben."],
    ["Ich bin sehr sozial und gehe gerne auf Parties."],
    ["Ich bin verwurzelt in meiner Heimat und Familie ist mir wichtig. "],
    ["Ich mag Gewohnheit und Stabilität und stehe neuen Dingen eher skeptisch gegenüber."],
    ["Kunst, Kultur und Musik bedeuten mir viel."],
    ["Jeder Bürger hat die Pflicht sich politisch einzubringen."],
    ["Ich liebe romantische Gesten."]
]

percentages = None
all_partners = None

@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        username = request.form['usernameInput']
        if username is None:
            return render_template("index.html")
        session["username"] = username
        session["uuid"] = uuid.uuid4()
        return redirect(url_for('start'))
    elif request.method == 'GET':
        return render_template("index.html")


@app.route("/start/", methods=['POST', 'GET'])
def start():
    print(session.get("username"))
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
        return render_template(f'statement.html', last=last, statement_id=id, statements=statement, name=session.get("username"))
    elif request.method == 'POST':
        return redirect(url_for('result'))

@app.route("/result/", methods=['POST', 'GET'])
def result():
    return render_template("result.html", partners=session["partners"])


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
    folder = os.path.join(app.config['VIDEO_FOLDER'], username + "_" + "_" + str(uuid)[:5])
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"statement_{statement_id}.webm")
    file.save(filename)
    new_statement_id = int(statement_id) + 1
    # after last statement redirect to waiting room 
    if new_statement_id > len(statements):
        return redirect(url_for('partner_selection'))
    return redirect(url_for('statement', id=new_statement_id))


@app.route("/partner_selection/", methods=['GET', 'POST'])
def partner_selection():
    global all_partners, percentages
    username = session.get("username")
    
    if request.method == 'GET':
        session["partners"] = {}
        session["partner_id"] = 0
        partner_name = all_partners[0]
        return render_template("partner.html", partner_name=partner_name, name=username)
    elif request.method == 'POST':
        current_partner_id = session.get("partner_id")
        if request.form.get("yes"):
            session["partners"][all_partners[current_partner_id]] = percentages[current_partner_id]
        
        new_partner_id = int(current_partner_id) + 1
        if new_partner_id > len(all_partners) - 1:
            return redirect(url_for('result'))
        session["partner_id"] = new_partner_id
        return render_template("partner.html", partner_name=all_partners[new_partner_id], name=username)


@app.route("/waiting_room/")
def waiting_room():
    if percentages is not None:
        return redirect(url_for('partner_selection'))
    return render_template("waitRoom2.html")


@app.route("/upload_results/", methods=['POST'])
def upload_results():
    global percentages, all_partners
    json = request.json
    # percentages = list(json.values())
    percentages = list(map(lambda x: int(x.strip("%")), json.values())) 
    print(percentages)
    all_partners = list(json.keys())
    return "OK"


if __name__ == "__main__":
    app.run(debug=True) #debug needed for dev (autom. refresh after changes)
