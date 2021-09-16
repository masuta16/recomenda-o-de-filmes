from flask import Flask, request, render_template, jsonify
from app import app, model

#@app.route('/')
#def index():
#    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
    userId = request.form['text']
    movies = model.recommended_json(int(userId))
    neighbors = model.get_K_neighbors(int(userId), 3)
    return render_template('results.html', movies=movies, neighbors=neighbors,
    colnames = ['Identificador', 'Idade', 'Genero'])
