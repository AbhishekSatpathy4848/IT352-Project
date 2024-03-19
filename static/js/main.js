let imageDataUrl;

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

document.addEventListener('DOMContentLoaded', function() {
    particlesJS.load('particles-js', "static/assets/particles-js-config.json", function() {
    console.log('callback - particles.js config loaded');
  });
});

document.getElementById("uploadForm").onsubmit = async function(e) {
    e.preventDefault();

    if(!imageDataUrl){
        alert("Please upload an image");
        return;
    }

    if(document.getElementById('submitButton').innerHTML == "Verify"){
        let response = await post_image_to_endpoint(imageDataUrl, "/verify");
        response = await response.json();
        console.log(response);
        let verified = response["verified"];
        let similarity_score = response["similarity_score"];
        let uploaded_image = response["uploaded_image"];
        let decoded_image = response["decoded_image"];
        let uploaded_image_element = document.createElement("img");
        uploaded_image_element.src = uploaded_image;
        uploaded_image_element.className = "img-share";
        let decoded_image_element = document.createElement("img");
        decoded_image_element.src = decoded_image;
        decoded_image_element.className = "img-share";
        let div = document.createElement("div");
        div.className = "row-flex";
        let h2 = document.createElement("h2");
        h2.innerText = "Uploaded Image";
        h2.className = "generated-title";
        let div_1 = document.createElement("div");
        div_1.appendChild(h2);
        div_1.appendChild(uploaded_image_element);
        h2 = document.createElement("h2");
        h2.innerText = "Decoded Image";
        h2.className = "generated-title";
        let div_2 = document.createElement("div");
        div_2.appendChild(h2);
        div_2.appendChild(decoded_image_element);

        div_1.style.display = "flex";
        div_1.style.flexDirection = "column";
        div_2.style.display = "flex";
        div_2.style.flexDirection = "column";
        div_1.style.alignItems = "center";
        div_2.style.alignItems = "center";
        div.appendChild(div_1);
        div.appendChild(div_2);
        div.style.width = "100%";
        document.getElementById("main").appendChild(div);


        if(verified){
            alert(`Verified with Similarity Score ${similarity_score}`);
        } else {
            alert(`Not verified with Similarity Score ${similarity_score}`);
        }
        return;
    }
    document.getElementById("loading_message").style.display = "block";
    let response = await post_image_to_endpoint(imageDataUrl, "/upload");
    document.getElementById("loading_message").style.display = "none";
    let h2 = document.createElement("h2");
    h2.innerText = "Generated Image Shares";
    h2.className = "generated-title";
    document.getElementById("main").appendChild(h2);
    response = await response.json();
    img_sources = response["image_shares"];
    let images_shares_div = document.createElement("div");
    images_shares_div.id = "image-shares";
    document.getElementById("main").appendChild(images_shares_div);
    for (let i = 0; i < img_sources.length; i++){
        let img = document.createElement("img");
        img.src = img_sources[i];
        img.className = "img-share";
        document.getElementById("image-shares").appendChild(img);
    }
    let decodeButton = document.createElement("button");
    decodeButton.className = "decode-button";
    decodeButton.className = "glass";
    decodeButton.innerText = "Decode";
    decodeButton.onclick = async function(){
        let response = await fetch("/decode");
        response = await response.json();

        let decodedImage = response["decoded_image"];
        let img = document.createElement("img");

        img.src = decodedImage;
        img.className = "img-share";

        let h2 = document.createElement("h2");
        h2.innerText = "Decoded Image";
        h2.className = "generated-title";
        document.getElementById("main").appendChild(h2);
        document.getElementById("main").appendChild(img);
        decodeButton.style.display = "none";
        let submitButton = document.getElementById('submitButton');
        submitButton.innerHTML = "Verify";
    }
    
    document.getElementById("main").appendChild(decodeButton);
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
