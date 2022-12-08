from flask import Flask, redirect, render_template, url_for, request
app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        username = request.form['usernameInput']
        return redirect(url_for('start', name=username))
    elif request.method == 'GET':
        return render_template("index.html")
        
@app.route("/start/<name>", methods=['POST', 'GET'])
def start(name):
    if request.method == 'GET':
        username=name
        return render_template('waitRoom.html', test=username)
    elif request.method == 'POST':
        username=name
        return redirect(url_for('statement'))

@app.route("/statement/", methods=['POST', 'GET'])
def statement():
    if request.method == 'GET':
        return render_template('statement1.html')
    elif request.method == 'POST':
        return redirect(url_for('result'))

@app.route("/result/", methods=['POST', 'GET'])
def result():
    return render_template("result.html")

if __name__ == "__main__":
    app.run(debug=True) #debug needed for dev (autom. refresh after changes)
