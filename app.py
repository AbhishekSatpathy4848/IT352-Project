import base64, binascii
from flask import Flask, render_template, request, Response

app = Flask(__name__)


def decode_base_64(image_data):
    try:
        image = base64.b64decode(image_data.split(",")[1], validate=True)
        return image
    except binascii.Error as e:
        print(e)

def save_image(image, image_name):
    with open(image_name, "wb") as f:
        f.write(image)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload(): 
    image = decode_base_64(request.json["image_data"]);
    save_image(image, "uploaded_image.png");
    return Response(status=200);

if __name__  == "__main__":
    app.run(debug=True)

