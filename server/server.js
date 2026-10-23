const { Server } = require("socket.io");

const io = new Server(3000, {
    cors: { origin: "*" }
});

io.on("connection", function (socket) {
    console.log("A player connected:", socket.id);
});

console.log("Server running on port 3000");
