from flask import Flask, render_template, request, Response

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload(): 
    print(request.form["content"])
    return Response("OK");

if __name__  == "__main__":
    app.run(debug=True)