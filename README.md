# Extended visual cryptography for facial data

Team Members:
- Srinivasa R
- Vignaraj Pai
- Abhishek Satpathy

## Introduction

Here we implement the extended visual cryptography with error diffusion. The extended visual cryptography scheme involves using cover images to make the shares of the secret image look less like noise.

## Visual Cryptography

Visual cryptography is a cryptographic technique which allows visual information (pictures, text, etc.) to be encrypted in such a way that decryption can be performed by the human visual system, without the aid of computers.

## Error Diffusion

Error diffusion is a technique used in digital halftoning. It is a type of halftoning in which the quantization error is distributed to neighboring pixels that have not yet been processed.

## Extended Visual Cryptography with Error Diffusion

The extended visual cryptography with error diffusion scheme involves using cover images to make the shares of the secret image look less like noise. The shares are then printed on transparencies and stacked on top of each other to reveal the secret image.

## Facial data

A pre-trained ResNet model is trained on both the decrypted lossy images and the lossless original images using contrastive learning to ensure that the facial recognition system can recognize faces from the decrypted data.
The model was finetuned using [this](https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset) facial recognition dataset by encrypting and augmenting the dataset with the recovered images.

## Deployed tool

We have used HTML, CSS, and JS to render the frontend of the website, and Flask to write the backend.

The following API routes have been defined in the backend:

### Route: /
- **Request Type:** GET
- **Response:** Renders the index.html template.
- **Description:** Renders the homepage of the website.

### Route: /upload
- **Request Type:** POST
- **Request:**
  ```
  {
    “image_data”:  <base64_encoded_image>
  }
  ```
- **Response:**
  ```
  {
    “image_shares”: 
    [
      <base64_encoded_share_0>,
      <base64_encoded_share_1>,
      <base64_encoded_share_3>,
      ...
    ]
  }
  ```
- **Description:** Accepts a POST request containing base64-encoded original image data, generates all 4 image shares using extended visual cryptography, and returns all base64-encoded image shares in a JSON response.

### Route: /decode
- **Request Type:** GET
- **Response:**
  ```
  {
    “reconstructed_image_data”:  <base64_encoded_image>
  }
  ```
- **Description:** Combines the image shares to return the reconstructed original image.

### Route: /verify
- **Request Type:** GET
- **Response:**
  ```
  {
    “result”: true/false
  }
  ```
- **Description:** Verifies if the reconstructed image matches the original uploaded image.


## Experimental data

### Average time taken for encryption and decryption

<b>(2, 2) Visual Cryptography</b>:

- Time for Encryption: 2.0392 seconds
- Time for Decryption: 0.0029 seconds
- Time taken per share: 1.0196 seconds

<b>(3, 4) Visual Cryptography</b>:

- Time for Encryption: 5.9154 seconds
- Time for Decryption: 0.0068 seconds
- Time taken per share: 1.4789 seconds