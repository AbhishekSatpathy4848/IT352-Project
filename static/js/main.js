let imageDataUrl;

document.getElementById("fileInput").onchange = function() {
    var previewImg = document.getElementById('previewImg');
    var file = document.querySelector('input[type=file]').files[0];
    var reader = new FileReader();

    reader.onloadend = function() {
        previewImg.src = reader.result;
    }

    if (file) {
        reader.readAsDataURL(file);
    } else {
        previewImg.src = "";
    }
};

document.getElementById("uploadForm").onsubmit = async function(e) {
    e.preventDefault();
    console.log(imageDataUrl);
    // let fileInput = document.getElementById("fileInput");
    // file = fileInput.files[0];
    // console.log(file.readAsDataURL);
    // let options = {
    //     method: 'POST',
    //     body: file,
    //     headers: {
    //         "Content-type": "application/json; charset=UTF-8"
    //     }
    // }

    // await fetch("/upload", options);

    // alert("Image uploaded successfully!");
};