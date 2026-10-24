const socket = io();

const clips = [
    { file: "../examples/thumbs_up.mp4", answer: "thumbs up" },
    { file: "../examples/peace_sign.mp4", answer: "peace sign" },
    { file: "../examples/teeth.mp4", answer: "teeth" },
    { file: "../examples/eating_burger.mp4", answer: "eating burger" }
];


const video = document.getElementById("clip");
const button = document.getElementById("play-btn");

let timeLeft = 30;
let timerId = null;
let currentClip = null;
let currentTeam = 0;
let team1Score = 0;
let team2Score = 0;

function startTimer() {
    clearInterval(timerId);
    timeLeft = 30;
    document.getElementById("timer").textContent = "Time: " + timeLeft;

    timerId = setInterval(tick, 1000);
}

function tick() {
    timeLeft -= 1;
    document.getElementById("timer").textContent = "Time: " + timeLeft;

    if (timeLeft === 0) {
        clearInterval(timerId);
        document.getElementById("timer").textContent = "Time's up!";
    }
}

function playRandomClip() {
    const randomIndex = Math.floor(Math.random() * clips.length);
    currentClip = clips[randomIndex];

    if (currentTeam === 1) {
        currentTeam = 2;
    } else {
        currentTeam = 1;
    }
    document.getElementById("turn").textContent = "Team " + currentTeam + "'s turn";

    video.src = currentClip.file;
    document.getElementById("hint").textContent = makeBlanks(currentClip.answer);
    document.getElementById("result").textContent = "";
    video.play();
    startTimer();
    console.log("Playing clip number", randomIndex);
}

button.addEventListener("click", playRandomClip);

function checkGuess() {
    if (currentClip === null) {
        return;
    }

    const guess = document.getElementById("guess-input").value.trim().toUpperCase();

    if (guess === currentClip.answer.toUpperCase()) {
        document.getElementById("result").textContent = "Correct!";

        if (currentTeam === 1) {
            team1Score += 1;
            document.getElementById("score1").textContent = "Score: " + team1Score;
        } else {
            team2Score += 1;
            document.getElementById("score2").textContent = "Score: " + team2Score;
        }

        currentClip = null;
    } else {
        document.getElementById("result").textContent = "Wrong, try again!";
    }
}

document.getElementById("guess-btn").addEventListener("click", checkGuess);

document.getElementById("guess-input").addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
        checkGuess();
    }
});


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

    socket.emit("createLobby", name);

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

    const code = document.getElementById("code-input").value.trim().toUpperCase();
    if(code === "")
    {
        alert("Type the lobby code that you want to join!");
        return;
    }

    document.getElementById("lobby-code").textContent = code;

    showScreen("lobby-screen");


}


document.getElementById("join-btn").addEventListener("click", joinLobby);

function makeBlanks(answer) {
    let blanks = "";
    for (const letter of answer) {
        if (letter === " ") {
            blanks += "   ";
        } else {
            blanks += "_ ";
        }
    }
    return blanks;
}


