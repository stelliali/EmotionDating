const startBtn = document.querySelector("#startRecording");
const exportBtn = document.querySelector("#exportVideo");
const videoEl = document.querySelector("video");
const nextItem = document.getElementById('next-item');

let videoData = [];
let cameraStream = null;
let mediaRecorder = null;

window.addEventListener('load', () => {
    const constraints = {
        // audio: true,
        video: {
            width: 300,
            height: 300
        }
    };
    navigator.mediaDevices.getUserMedia(constraints).then(stream => {
        cameraStream = stream;
        videoEl.controls = false;
        videoEl.srcObject = cameraStream;
        videoEl.play();
    }).catch(info => {
        alert('error' + info);
    });

/*    var str = window.location.search;
    if (str.indexOf(name) != -1) {
        var username = str.slice(6);
        if (username.length != 0) {
            document.getElementById('userName').innerHTML = "Username: " + username;
        } else {
            alert("No value found");
        }
    };*/
});

startBtn.addEventListener('click', () => {
    statement = document.getElementById('statement_block');
    statement.style.display = "flex";
    startBtn.style.display = "none";
    mediaRecorder = new MediaRecorder(cameraStream, {
        mimeType: 'video/webm'
    });
    mediaRecorder.start();
    console.log("recording started");
    mediaRecorder.addEventListener('dataavailable', ev => {
        videoData.push(ev.data);
    });
    mediaRecorder.addEventListener('stop', () => {
        videoData = new Blob(videoData);
    });

    var t = 9;
    var t_1 = 3;
    setInterval(() => {
        if (t < 0) {
            console.log("recording stopped");
            mediaRecorder.stop();
            nextItem.style.display = "flex";
            return;
        }
        document.getElementById('show').innerHTML = "Recording: " + t + "s";
        t--;
    }, 1000);

});



// stopBtn.addEventListener('click', () => {
//     mediaRecorder.stop();
// });

// playBtn.addEventListener('click', () => {
//     if (videoData === null) return false;
//     videoEl.srcObject = null;
//     videoEl.src = URL.createObjectURL(videoData);
//     videoEl.play();
//     videoEl.controls = true;
//     cameraStream = null;
// });

exportBtn.addEventListener('click', () => {
    if (videoData === null) return false;
    //videoFile = URL.createObjectURL(videoData);
    var data = new FormData()
    data.append('file', videoData, 'file')

    fetch('http://localhost:5000/video/', {
        method: 'POST',
        body: data

    }).then((response)=>{
        if(response.redirected){
            window.location.href = response.url;
        }
    }).catch(function(e){
        console.log(e)
    })
});

