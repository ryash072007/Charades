const express = require("express");
const http = require("http");
const path = require("path");
const { Server } = require("socket.io");

const app = express();
const server = http.createServer(app);
const io = new Server(server);

app.use(express.static(path.join(__dirname, "../web")));
app.use("/examples", express.static(path.join(__dirname, "../examples")));

io.on("connection", function (socket) {
    console.log("A player connected:", socket.id);
});

server.listen(3000, function () {
    console.log("Server running on http://localhost:3000");
});
