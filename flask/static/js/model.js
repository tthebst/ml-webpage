

var loadFile = function (event) {
    //change choose file name to image name
    var name = document.getElementById('inputGroupFile04').files.item(0).name;
    document.getElementById("choosefile_label").innerHTML = name;

    //display image in img with id output
    var image = document.getElementById('output');
    image.src = URL.createObjectURL(event.target.files[0]);
};


function image_predict(result, spinner, fetch_url) {

    //Get input image
    let photo = document.getElementById("inputGroupFile04").files[0];


    //check if there is a legit photo

    if (document.getElementById("inputGroupFile04").files.length == 0) {
        confirm("You need to add an image!")
        return 0
    }
    let formData = new FormData();
    formData.append("photo", photo);


    //remove predict button and add loader symbol
    document.getElementById(result).innerHTML = `<div class="spinner-border text-dark" id=${spinner} role="status"><span class="sr-only"> Loading...</span ></div >`;

    fetch(fetch_url, {
        method: "POST", body: formData
    }).then((response) => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response
    }).then(function (a) {
        document.getElementById(spinner).classList.add('invisible')
        return a.json(); // call the json method on the response to get JSON
    }).then(function (json) {
        res = document.getElementById(result)
        var listDiv = res;
        res.removeChild(res.firstChild)
        var ul = document.createElement('ul');
        ul.classList.add("list-unstyled");

        var li = document.createElement('li');
        li.innerHTML = '<h5><u>Prediction</u></h5>'
        ul.appendChild(li);
        for (var i = 0; i < json.length; ++i) {
            var li = document.createElement('li');
            pred = json[i][0][0].toUpperCase() + json[i][0].slice(1)
            li.innerHTML = new String(pred.bold() + ": " + json[i][1]);
            //li.classList.add("list-group-item");
            li.classList.add("border-0")   // Use innerHTML to set the text
            ul.appendChild(li);


        }
        li = document.createElement('li');
        li.innerHTML = `<i class="fas fa-redo fa-1x text-right" style="padding-top: 0.3em"></i>`;
        li.firstChild.addEventListener("click", function () { image_predict(result, spinner, fetch_url); }, false);
        ul.appendChild(li);
        listDiv.appendChild(ul);
    });
}


function image_detect(result, spinner, fetch_url) {

    //Get input image
    let photo = document.getElementById("inputGroupFile04").files[0];


    //check if there is a legit photo

    if (document.getElementById("inputGroupFile04").files.length == 0) {
        confirm("You need to add an image!");
        return 0;
    }
    let formData = new FormData();
    formData.append("photo", photo);


    //remove predict button and add loader symbol
    document.getElementById(result).innerHTML = `<div class="spinner-border text-dark" id=${spinner} role="status"><span class="sr-only"> Loading...</span ></div >`;
    console.log("fwetch");
    fetch(fetch_url, {
        method: "POST", body: formData
    }).then((response) => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response
    }).then(function (a) {
        document.getElementById(spinner).classList.add('invisible');
        return a; // call the json method on the response to get JSON
    }).then(function (json) {
        res = document.getElementById(result);
        var listDiv = res;
        res.removeChild(res.firstChild);


        console.log(json);
        imgElem.setAttribute('src', "data:image/png;base64," + json);



        li = document.createElement('li');
        li.innerHTML = `<i class="fas fa-redo fa-1x text-right" style="padding-top: 0.3em"></i>`;
        li.firstChild.addEventListener("click", function () { image_predict(result, spinner, fetch_url); }, false);
        ul.appendChild(li);
        listDiv.appendChild(ul);
    });
}