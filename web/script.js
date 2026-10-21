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

function createLobby() {
    const name = document.getElementById("username-input").value.trim();

    if (name === "") {
        alert("Type your name first!");
        return;
    }

    const code = Math.random().toString(36).slice(2, 6).toUpperCase();
    document.getElementById("lobby-code").textContent = code;
    showScreen("lobby-screen");
}
document.getElementById("create-btn").addEventListener("click", createLobby);

function joinLobby(){
    const name = document.getElementById("username-input").value.trim();


    if(name==="")
    {
        alert("Type your name!");
        return;
    }

    const code = document.getElementById("code-input").value.trim();
    if(code === "")
    {
        alert("Type the lobby code that you want to join!");
        return;
    }

    document.getElementById("lobby-code").textContent = code;

    showScreen("lobby-screen");


}

document.getElementById("join-btn").addEventListener("click", joinLobby);
