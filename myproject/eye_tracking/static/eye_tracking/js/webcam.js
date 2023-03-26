const videoFeed = document.getElementById('video_feed');
const constraints = {
    video: true
};

navigator.mediaDevices.getUserMedia(constraints)
    .then(stream => {
        videoFeed.srcObject = stream;
    })
    .catch(error => {
        console.error(error);
    });
