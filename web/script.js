const clips = [
    "../examples/thumbs_up.mp4",
    "../examples/peace_sign.mp4",
    "../examples/teeth.mp4",
    "../examples/eating_burger.mp4",
];

const video = document.getElementById("clip");
const button = document.getElementById("play-btn");

function playRandomClip() {
    const randomIndex = Math.floor(Math.random() * clips.length);
    video.src = clips[randomIndex];
    video.play();
    console.log("Playing clip number", randomIndex);
}

button.addEventListener("click", playRandomClip);
function showScreen(id) {
    document.getElementById("entrance-screen").classList.add("hidden");
    document.getElementById("lobby-screen").classList.add("hidden");
    document.getElementById("game-screen").classList.add("hidden");

    document.getElementById(id).classList.remove("hidden");
}
function startGame() {
    showScreen("game-screen");
}

document.getElementById("start-btn").addEventListener("click", startGame);
