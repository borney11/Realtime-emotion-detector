const video = document.getElementById("video");
const resultText = document.getElementById("result");

// Access webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    });

function sendFrame() {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    canvas.toBlob(async (blob) => {
        let formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        let res = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        let data = await res.json();
        resultText.innerHTML =
            `Emotion: ${data.emotion.toUpperCase()} ( ${(data.confidence * 100).toFixed(1)}% )`;
    }, "image/jpeg");
}

// Send frame every 500 ms
setInterval(sendFrame, 500);
