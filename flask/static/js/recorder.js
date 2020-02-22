var audioChunks;

startRecord.onclick = e => {
    startRecord.disabled = true;
    stopRecord.disabled = false;
    // This will prompt for permission if not allowed earlier
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(function (stream) {
            audioChunks = [];
            var options = {
                mimeType: 'audio/wav'
            };



            rec = new MediaRecorder(stream, options);
            console.log(rec);
            rec.ondataavailable = e => {
                audioChunks.push(e.data);
                if (rec.state == "inactive") {
                    let blob = new Blob(audioChunks, { type: 'audio/wav' });
                    recordedAudio.src = URL.createObjectURL(blob);
                    recordedAudio.controls = true;
                    recordedAudio.autoplay = true;
                }
            }
            rec.start();
        })
        .catch(e => console.log(e));
}
stopRecord.onclick = e => {
    startRecord.disabled = false;
    stopRecord.disabled = true;
    rec.stop();
    console.log(audioChunks)
}



