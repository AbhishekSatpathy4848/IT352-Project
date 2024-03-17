let imageDataUrl;

document.addEventListener('DOMContentLoaded', function() {
    // document.getElementById('loading_message').style.display = "none";
})

document.getElementById("fileInput").onchange = function() {
    var previewImg = document.getElementById('previewImg');
    var file = document.querySelector('input[type=file]').files[0];
    var reader = new FileReader();

    reader.onloadend = function() {
        previewImg.src = reader.result;
        imageDataUrl = reader.result;
    }

    if (file) {
        reader.readAsDataURL(file);
    } else {
        previewImg.src = "";
    }
};

document.getElementById("uploadForm").onsubmit = async function(e) {
    e.preventDefault();

    document.getElementById("loading_message").style.display = "block";
    let response = await post_image_to_endpoint(imageDataUrl, "/upload");
    document.getElementById("loading_message").style.display = "none";
    response = await response.json();
    img_sources = response["image_shares"];
    for (let i = 0; i < img_sources.length; i++){
        let img = document.createElement("img");
        img.src = img_sources[i];
        img.className = "img-share";
        document.getElementById("image-shares").appendChild(img);
    }
    //add a decode button as well
    let decodeButton = document.createElement("button");
    decodeButton.className = "decode-button";
    decodeButton.className = "glass";
    decodeButton.innerText = "Decode";
    document.getElementById("main").appendChild(decodeButton)
    document.getElementsByClassName("decode-button")[0].onclick = async function(){
        let response = await fetch("/decode");
        response = await response.json();
        let decodedMessage = response["decoded_image"];
        console.log(decodedMessage)
        // document.getElementById("decoded-message").innerText = decodedMessage;
    }
};

async function post_image_to_endpoint(imageDataUrl, endpoint){
    let base64Image = imageDataUrl.split(',').pop();
    let options = {
        method: 'POST',
        body: JSON.stringify({"image_data": base64Image}),
        headers: {
            "Content-type": "application/json; charset=UTF-8"
        }
    }
    let response = await fetch(endpoint, options);
    return response;
}
