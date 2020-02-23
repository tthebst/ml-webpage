

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
        return a.text(); // call the json method on the response to get JSON
    }).then(function (json) {
        res = document.getElementById(result);
        var listDiv = res;
        res.removeChild(res.firstChild);
        res = document.getElementById(result);

        //add predicted image
        d = document.createElement('div');
        d.classList.add("img-hover");
        var img = new Image();
        img.src = "data:image/jpg;base64," + json.substring(3, json.length - 7);
        img.classList.add("img-fluid");
        d.appendChild(img);
        res.appendChild(d);

        //add reload button
        li = document.createElement('p');
        li.innerHTML = `<i class="fas fa-redo fa-1x text-right" style="padding-top: 0.3em"></i>`;
        li.firstChild.addEventListener("click", function () { image_detect(result, spinner, fetch_url); }, false);
        res.appendChild(li);
    });
}

function image_generate(result, spinner, fetch_url) {



    //remove predict button and add loader symbol
    document.getElementById(result).innerHTML = `<div class="spinner-border text-dark" id=${spinner} role="status"><span class="sr-only"> Loading...</span ></div >`;
    console.log("fwetch");
    console.log(document.getElementById("biggan_options").value);
    fetch(fetch_url, {
        method: "POST", body: JSON.stringify({ a: document.getElementById("biggan_options").value })
    }).then((response) => {
        if (!response.ok) {
            d = document.createElement('h4');
            d.innerHTML = "Something went wrong... Roboters still asleep";
            d.classList.add("text-danger");
            res = document.getElementById(result);
            res.appendChild(d);

        }
        return response
    }).then(function (a) {
        document.getElementById(spinner).classList.add('invisible');
        return a.text(); // call the json method on the response to get JSON
    }).then(function (json) {
        //txt = response.text()
        res = document.getElementById(result);
        var listDiv = res;
        res.removeChild(res.firstChild);
        res = document.getElementById(result);

        //add predicted image

        d = document.createElement('div');
        d.classList.add("img-hover");
        var img = new Image();
        console.log(json)
        img.src = "data:image/png;base64," + json.substring(3, json.length - 3);
        console.log(img.src)
        img.classList.add("img-fluid");
        d.appendChild(img);
        res.appendChild(d);


        //add reload button
        li = document.createElement('p');
        console.log("hh");
        li.innerHTML = `<i class="fas fa-redo fa-1x text-right" style="padding-top: 0.3em"></i>`;
        li.firstChild.addEventListener("click", function () { image_generate(result, spinner, fetch_url); }, false);
        document.getElementById(result).appendChild(li);
    });
}

function image_generate_pgan(result, spinner, fetch_url) {



    //remove predict button and add loader symbol
    document.getElementById(result).innerHTML = `<div class="spinner-border text-dark" id=${spinner} role="status"><span class="sr-only"> Loading...</span ></div >`;
    console.log("fwetch");
    console.log(document.getElementById("biggan_options").value);
    fetch(fetch_url, {
        method: "GET"
    }).then((response) => {
        if (!response.ok) {
            d = document.createElement('h4');
            d.innerHTML = "Something went wrong... Roboters still asleep";
            d.classList.add("text-danger");
            res = document.getElementById(result);
            res.appendChild(d);

        }
        return response
    }).then(function (a) {
        document.getElementById(spinner).classList.add('invisible');
        return a.text(); // call the json method on the response to get JSON
    }).then(function (json) {
        //txt = response.text()
        res = document.getElementById(result);
        var listDiv = res;
        res.removeChild(res.firstChild);
        res = document.getElementById(result);

        //add predicted image

        d = document.createElement('div');
        d.classList.add("img-hover");
        var img = new Image();
        console.log(json)
        img.src = "data:image/png;base64," + json.substring(3, json.length - 3);
        console.log(img.src)
        img.classList.add("img-fluid");
        d.appendChild(img);
        res.appendChild(d);


        //add reload button
        li = document.createElement('p');
        console.log("hh");
        li.innerHTML = `<i class="fas fa-redo fa-1x text-right" style="padding-top: 0.3em"></i>`;
        li.firstChild.addEventListener("click", function () { image_generate_pgan(result, spinner, fetch_url); }, false);
        document.getElementById(result).appendChild(li);
    });
}



function language_predict(result, spinner, fetch_url) {



    //remove predict button and add loader symbol
    document.getElementById(result).innerHTML = `<div class="container-fluid text-center"><div class="spinner-border text-dark spinners" id=${spinner} role="status"><span class="sr-only"> Loading...</span ></div></div>`;
    console.log("fwetch");
    console.log(document.getElementById("to_translate").value);
    fetch(fetch_url, {
        method: "POST", body: JSON.stringify({ a: document.getElementById("to_translate").value })
    }).then((response) => {
        if (!response.ok) {
            d = document.createElement('h4');
            d.innerHTML = "Something went wrong... Roboters still asleep";
            d.classList.add("text-danger");
            res = document.getElementById(result);
            res.appendChild(d);

        }
        return response
    }).then(function (a) {
        document.getElementById(spinner).classList.add('invisible');
        return a.text(); // call the json method on the response to get JSON
    }).then(function (json) {
        //txt = response.text()
        res = document.getElementById(result);
        var listDiv = res;
        res.removeChild(res.firstChild);
        res = document.getElementById(result);

        //add predicted image
        console.log(json)
        d = document.createElement('div');
        d.classList.add("container-fluid");
        d.innerHTML = "<b>Translation:</b> <br>" + json;
        res.appendChild(d);


        //add reload button
        li = document.createElement('div');
        li.classList.add("container-fluid");
        console.log("hh");
        li.innerHTML = `<i class="fas fa-redo fa-1x text-right float-left" style="padding-top: 0.3em"></i>`;
        li.firstChild.addEventListener("click", function () { language_predict(result, spinner, fetch_url); }, false);
        document.getElementById(result).appendChild(li);
    });
}

function deepspeech_transcribe(result, spinner, fetch_url) {



    //remove predict button and add loader symbol
    document.getElementById(result).innerHTML = `<div class="container-fluid text-center"><div class="spinner-border text-dark spinners" id=${spinner} role="status"><span class="sr-only"> Loading...</span ></div></div>`;
    console.log("fwetch");
    console.log(document.getElementById("to_translate").value);


    //create FormData to send audio

    var fd = new FormData();
    fd.append('data', audio);

    console.log(audio)

    console.log(fd);
    fetch(fetch_url, {
        method: "POST", body: fd
    }).then((response) => {
        if (!response.ok) {
            d = document.createElement('h4');
            d.innerHTML = "Something went wrong... Roboters still asleep";
            d.classList.add("text-danger");
            res = document.getElementById(result);
            res.appendChild(d);

        }
        return response
    }).then(function (a) {
        document.getElementById(spinner).classList.add('invisible');
        return a.text(); // call the json method on the response to get JSON
    }).then(function (json) {
        //txt = response.text()
        res = document.getElementById(result);
        var listDiv = res;
        res.removeChild(res.firstChild);
        res = document.getElementById(result);



        //add predicted image


        console.log(json);
        j = JSON.parse(json);
        d = document.createElement('div');
        d.classList.add("container-fluid");
        d.innerHTML = "<b>Translation:</b> <br>" + j.substring(3, j.length - 3);
        res.appendChild(d);


        //add reload button
        li = document.createElement('div');
        li.classList.add("container-fluid");
        console.log("hh");
        li.innerHTML = `<i class="fas fa-redo fa-1x text-right float-left" style="padding-top: 0.3em"></i>`;
        li.firstChild.addEventListener("click", function () { deepspeech_transcribe(result, spinner, fetch_url); }, false);
        document.getElementById(result).appendChild(li);
    });
}