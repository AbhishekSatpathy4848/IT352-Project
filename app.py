import base64, binascii
from flask import Flask, render_template, request, Response
import json;

app = Flask(__name__)


def decode_base_64(image_data):
    try:
        image = base64.b64decode(image_data, validate=True)
        return image
    except binascii.Error as e:
        print(e)

def save_image(image, image_name):
    with open(image_name, "wb") as f:
        f.write(image)

def encode_base_64(image):
    return base64.b64encode(image)

def read_image(image_name):
    with open(image_name, "rb") as f:
        image = f.read()
    return image

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload(): 
    image = decode_base_64(request.json["image_data"]);
    save_image(image, "uploaded_image.png");
    # image = get_image();
    return Response(status=200);

# def get_image():
#     base64_string = ""
#     with open("uploaded_image.png", "rb") as f:
#         base64_string = base64.b64encode(f.read())
#     print(base64_string)
#     return base64_string

if __name__  == "__main__":
    app.run(host="0.0.0.0", port=5000)

