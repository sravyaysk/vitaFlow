from flask import Flask, render_template

app = Flask(__name__, static_folder='static', static_url_path='/static')


@app.route('/test')
def index():
    return "Hello, World!"


@app.route('/')
def home_page():
    return render_template('index.html')
