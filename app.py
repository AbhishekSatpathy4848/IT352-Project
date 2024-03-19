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
def decode_image_shares():
    evcs_decrypt("visual_cryptography/output_files", "visual_cryptography/final_image.png", vc_scheme, resolution)
    image_string = ""
    data = {}
    with open(f"visual_cryptography/final_image.png", "rb") as f:
            image_string = base64.b64encode(f.read()).decode('utf-8')
            image_string = "data:image/png;base64," + image_string
            image_string = image_string
    data["decoded_image"] = image_string
    return jsonify(data);


def verify_image_function():
        return True

@app.route("/verify", methods=["POST"])
def verify():
    image = decode_base_64(request.json["image_data"]);
    save_image(image, "uploaded_image_to_verify.png");
    uploaded_image = ""

    with open(f"uploaded_image_to_verify.png", "rb") as f:
        uploaded_image = base64.b64encode(f.read()).decode('utf-8')
        uploaded_image = "data:image/png;base64," + uploaded_image
    
    decoded_image = ""
    with open(f"visual_cryptography/final_image.png", "rb") as f:
        decoded_image = base64.b64encode(f.read()).decode('utf-8')
        decoded_image = "data:image/png;base64," + decoded_image

    result = verify_image_function()
    return jsonify({"verified": result, "uploaded_image": uploaded_image, "decoded_image": decoded_image});

if __name__  == "__main__":
    app.run(debug=True)

