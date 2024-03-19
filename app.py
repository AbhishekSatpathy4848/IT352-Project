import base64, binascii
import random
from flask import Flask, render_template, request, Response, jsonify
import json;
from visual_cryptography.color_hvc import evcs_decrypt, evcs_encrypt
from transformers import AutoFeatureExtractor, AutoModel
import torch
import torchvision
from torchvision import transforms

app = Flask(__name__)

vc_scheme = (3, 4)
resolution = (128, 128)
cover_imgs = ["Lena.png", "Baboon.png", "Barbara.bmp", "House.bmp"]


model_ckpt = "nateraw/vit-base-beans"
extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
hidden_dim = model.config.hidden_size

from datasets import Dataset

torch_dataset = torchvision.datasets.ImageFolder(root="Photos", transform=transforms.Compose([transforms.Resize((128, 128))]))

def gen():
    for image, label in torch_dataset:
        yield {"image": image, "labels": label}

dataset = Dataset.from_generator(gen)
# dataset = load_dataset("nuwandaa/ffhq128")

labels = dataset["labels"]
label2id, id2label = dict(), dict()

for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label
    
num_samples = 100
seed = 42
candidate_subset = dataset.shuffle(seed=seed).select(range(num_samples))

import torchvision.transforms as T


transformation_chain = T.Compose(
    [
        T.Resize(int((256 / 224) * extractor.size["height"])),
        T.CenterCrop(extractor.size["height"]),
        T.ToTensor(),
        T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
    ]
)

import torch


def extract_embeddings(model: torch.nn.Module):
    """Utility to compute embeddings."""
    device = model.device

    def pp(batch):
        images = batch["image"]
        image_batch_transformed = torch.stack(
            [transformation_chain(image) for image in images]
        )
        new_batch = {"pixel_values": image_batch_transformed.to(device)}
        with torch.no_grad():
            embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()
        return {"embeddings": embeddings}

    return pp


# batch_size = 24
device = "cpu"
model = model.to(device)
extract_fn = extract_embeddings(model.to(device))
candidate_subset_emb = candidate_subset.map(extract_fn, batched=True, batch_size=24)

from tqdm.auto import tqdm

candidate_ids = []

for id in tqdm(range(len(candidate_subset_emb))):
    label = candidate_subset_emb[id]["labels"]

    entry = str(id) + "_" + str(label)

    candidate_ids.append(entry)

import numpy as np

all_candidate_embeddings = np.array(candidate_subset_emb["embeddings"])
all_candidate_embeddings = torch.from_numpy(all_candidate_embeddings)


def compute_scores(emb_one, emb_two):
    """Computes cosine similarity between two vectors."""
    scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)
    return scores.numpy().tolist()


def find_similarity(img_1, img_2):
    img_1 = transformation_chain(img_1).unsqueeze(0)
    img_2 = transformation_chain(img_2).unsqueeze(0)

    new_batch = {"pixel_values": img_1.to(device)}
    with torch.no_grad():
        emb_1 = model(**new_batch).last_hidden_state[:, 0].cpu()

    new_batch = {"pixel_values": img_2.to(device)}
    with torch.no_grad():
        emb_2 = model(**new_batch).last_hidden_state[:, 0].cpu()

    return compute_scores(emb_1, emb_2)

def is_similar(img_1_file, img_2_file):
    from PIL import Image
    img_transform = T.Compose([T.Resize((128, 128))])

    img_1 = Image.open(img_1_file)
    img_2 = Image.open(img_2_file)

    img_1 = img_transform(img_1)
    img_2 = img_transform(img_2)

    # make sure both have only 3 channels
    img_1 = img_1.convert("RGB")
    img_2 = img_2.convert("RGB")

    similarity = find_similarity(img_1, img_2)

    print(similarity[0])
    
    if similarity[0] > 0.7:
        return (True, similarity[0])
    else:
        return (False, similarity[0])
    

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
    return is_similar("uploaded_image_to_verify.png", "uploaded_image.png") 

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
    return jsonify({"verified": result[0], "similarity_score": result[1], "uploaded_image": uploaded_image, "decoded_image": decoded_image});

if __name__  == "__main__":
    app.run(debug=True)

