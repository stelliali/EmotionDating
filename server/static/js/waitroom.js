const openBtn = document.querySelector("#openCamera");
const start = document.querySelector("#start");
const videoEl = document.querySelector("video");
const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');

function onResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(
        results.image, 0, 0, canvasElement.width, canvasElement.height);
    if (results.multiFaceLandmarks) {
        for (const landmarks of results.multiFaceLandmarks) {
            drawConnectors(canvasCtx, landmarks, FACEMESH_TESSELATION, {
                color: '#C0C0C070',
                lineWidth: 1
            });
            drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_EYE, {
                color: '#FF3030'
            });
            drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_EYEBROW, {
                color: '#FF3030'
            });
            drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_IRIS, {
                color: '#FF3030'
            });
            drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_EYE, {
                color: '#30FF30'
            });
            drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_EYEBROW, {
                color: '#30FF30'
            });
            drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_IRIS, {
                color: '#30FF30'
            });
            drawConnectors(canvasCtx, landmarks, FACEMESH_FACE_OVAL, {
                color: '#E0E0E0'
            });
            drawConnectors(canvasCtx, landmarks, FACEMESH_LIPS, {
                color: '#E0E0E0'
            });
        }
    }
    canvasCtx.restore();
}

const faceMesh = new FaceMesh({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
    }
});
faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});
faceMesh.onResults(onResults);

const camera = new Camera(videoElement, {
    onFrame: async() => {
        await faceMesh.send({
            image: videoElement
        });
    },
    width: 300,
    height: 300
});
camera.start();

// openBtn.addEventListener('click', () => {
//     const constraints = {
//         audio: true,
//         video: {
//         width: 300,
//         height: 300
//         }
//     };
//     navigator.mediaDevices.getUserMedia(constraints).then(stream => {
//         cameraStream = stream;
//         videoEl.controls = false;
//         videoEl.srcObject = cameraStream;
//         videoEl.play();
//     }).catch(info => {
//         alert('error' + info);
//     });
// });

window.addEventListener('load', () => {
    var str = window.location.search;
    console.log(str)
    if (str.indexOf(name) != -1) {
        // var pos_start = str.indexOf(name) + name.length + 1;
        // var pos_end = str.indexOf("=", pos_start);
        var username = str.slice(6)
        console.log(username)
        if (username.length != 0) {
            document.getElementById('userName').innerHTML = "Username: " + username;
        } else {
            alert("No value found");
        }
    }
})

/**
start.addEventListener('click', () => {
    {
        var str = window.location.search;
        if (str.indexOf(name) != -1) {
            var username = str.slice(6)
            console.log(username)
        }
        location = "statement1.html?name=" + username;
    }
})
*/