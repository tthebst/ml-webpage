
var audio;
startRecord.onclick = e => {
    startRecord.disabled = true;
    stopRecord.disabled = false;
    // This will prompt for permission if not allowed earlier
    var constraint = { audio: true, video: false };
    navigator.mediaDevices.getUserMedia(constraint)
        .then(function (stream) {
            const options = { mimeType: 'audio/mpeg' };

            audioContext = new AudioContext;
            input = audioContext.createMediaStreamSource(stream);
            /* Create the Recorder object and configure to record mono sound (1 channel) Recording 2 channels will double the file size */
            rec = new Recorder(input, {
                numChannels: 1
            })
            //start the recording process 
            rec.record()
        })
        .catch(e => console.log(e));
}
stopRecord.onclick = e => {
    startRecord.disabled = false;
    stopRecord.disabled = true;
    rec.stop();
    rec.exportWAV(save_blob);
    console.log(audio);
}

function save_blob(blob) {
    console.log("saving blob");
    console.log(blob);
    audio = blob
    recordedAudio.src = URL.createObjectURL(blob);
    recordedAudio.controls = true;
    recordedAudio.autoplay = true;


}



