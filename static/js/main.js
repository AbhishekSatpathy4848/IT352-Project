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

    console.log(response);
};

async function post_image_to_endpoint(image_data, endpoint){
    let options = {
        method: 'POST',
        body: JSON.stringify({"image_data": imageDataUrl}),
        headers: {
            "Content-type": "application/json; charset=UTF-8"
        }
    }
    let response = await fetch("/upload", options);
    return response;
}
