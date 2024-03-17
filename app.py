import base64, binascii
from flask import Flask, render_template, request, Response, jsonify
import json;
from visual_cryptography.color_hvc import evcs_decrypt, evcs_encrypt

app = Flask(__name__)

vc_scheme = (3, 4)
resolution = (128, 128)
cover_imgs = ["Lena.png", "Baboon.png", "Barbara.bmp", "House.bmp"]

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
    data = generate_image_shares("uploaded_image.png");
    return jsonify(data);

def generate_image_shares(input_image_path):
    evcs_encrypt(vc_scheme, resolution, input_image_path, "visual_cryptography/output_files", "visual_cryptography/cover_imgs", cover_imgs)
    data = {"image_shares" : []}
    for i in range(vc_scheme[1]):
        image_string = ""
        with open(f"visual_cryptography/output_files/shares_{i}.png", "rb") as f:
            image_string = base64.b64encode(f.read()).decode('utf-8')
            image_string = "data:image/png;base64," + image_string
            data["image_shares"].append(image_string)
  
    return data;

@app.route("/decode", methods=["GET"])
def decode_image_shares(image_shares):
    evcs_decrypt("visual_cryptography/output_files", "final_image.png", vc_scheme, resolution)
    image_string = ""
    with open(f"visual_cryptography/final_image.png", "rb") as f:
            image_string = base64.b64encode(f.read()).decode('utf-8')
            image_string = "data:image/png;base64," + image_string
            image_string = image_string
    return jsonify({"decoded_image", image_string})

@app.route("/verify", methods=["POST"])
def verify():
    image_shares = request.json["image_shares"]
    image_shares = decode_image_shares(image_shares)
    image_shares = [encode_base_64(image_share) for image_share in image_shares]
    return Response(json.dumps({"image_shares": image_shares}), status=200, mimetype="application/json");

if __name__  == "__main__":
    app.run(debug=True)

