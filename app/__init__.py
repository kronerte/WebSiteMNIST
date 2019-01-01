#! /usr/bin/python
# -*- coding:utf-8 -*-
#FLASK_APP=main.py flask run
from flask import Flask, render_template, request
from PIL import Image
from app.TransImage import do
app = Flask(__name__)




@app.route('/', methods=['GET'])
def accueil():
    return render_template("accueil.html",image="first.png")

@app.route('/', methods=['POST'])
def transform():
    # On récupère les données
    image = Image.open(request.files["image"])
    name = request.form["name"]
    method = request.form["Transformation"]
    # Traitement
    image, name =do(method, image, name)
    # Enregistrement et affichage
    image.save(f'app/static/{name}.png')
    return render_template("transform.html",image=name)

if __name__ == '__main__':
    app.run(debug=True)
