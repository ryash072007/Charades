# How the AI Charades website was built

My step-by-step notes, written to look back on. Each step matches a git commit
on the `feature/website` branch (run `git log --oneline` to see them).

**The 3 languages of a web page:**

| File | Language | Job | Think of it as |
|---|---|---|---|
| `index.html` | HTML | *What* is on the page (buttons, text, video) | The skeleton |
| `style.css` | CSS | *How it looks* (colours, sizes, layout) | The clothes |
| `script.js` | JavaScript | *What it does* (clicks, timers, logic) | The brain |

The browser reads `index.html` first. That file pulls in the other two:
`<link rel="stylesheet" href="style.css">` in the `<head>`, and
`<script src="script.js"></script>` at the very end of `<body>`. The script tag
goes last so that every element already exists when the JavaScript runs.

---

## Part 1: The page itself (HTML + CSS)

### Step 1: A first web page (`added web page`)
- Made `web/index.html`, opened it with **Live Server** (a VS Code extension
  that shows the page in Chrome and reloads it each time I save).
- HTML is made of **tags**: `<h1>Title</h1>` has an opening tag, the content, and a closing tag.
- `<!DOCTYPE html>` at the top tells the browser "this is modern HTML".
- `<head>` holds info *about* the page (title, CSS link). `<body>` holds what you *see*.

### Step 2: Show the 4 example clips (`showed all 4 example clips`)
- `<video src="../examples/thumbs_up.mp4" controls></video>` shows a video.
  `controls` adds the play/pause bar.
- `../` means "go up one folder". `web/` and `examples/` sit next to each other,
  so from `web/` we go up and then into `examples/`.

### Step 3: Styling with CSS (`styled page with css`)
A CSS rule = **selector** + `{ property: value; }`.

```css
body { background-color: #c4622d; }   /* tag selector: every <body> */
.card { border-radius: 12px; }         /* class selector (dot): class="card" */
#clip { max-width: 350px; }            /* id selector (hash): id="clip" */
```
- **class** = reusable label, many elements can share it (`class="card"`).
- **id** = unique name, only one element (`id="clip"`).
- Colours like `#c4622d` are hex codes: red, green, blue, two digits each.
- `.grid` used `display: grid` + `grid-template-columns: repeat(4, 1fr)` to put 4 clips
  in a row (`1fr` = "one equal share of the space").

---

## Part 2: Making it do things (JavaScript)

### Step 4: Play a random clip (`play a random clip with JS`)
Key ideas:

```js
const video = document.getElementById("clip");    // grab an HTML element by its id
button.addEventListener("click", playRandomClip); // "when clicked, run this function"
```
- `const` = a variable you won't reassign. `let` = one you will. (Python has no keywords
  for this, you just write `x = 5`.)
- `document` = the whole web page, as an object JS can poke at.
- **Event listener** = "when X happens, call function Y". Note: we pass
  `playRandomClip` *without* `()`. We hand over the function itself, we don't call it now.
  (Same as passing a function as an argument in Python.)
- Random pick:
  ```js
  Math.floor(Math.random() * clips.length)  // like Python: random.randrange(len(clips))
  ```
  `Math.random()` gives 0 to 0.999…, times 4 gives 0 to 3.999…, and `floor` chops it down to 0, 1, 2 or 3.

### Step 5: Three screens (`split page into entrance lobb and game screen`)
The page is ONE HTML file with three `<div>`s: entrance, lobby, game. Only one shows at a time.

```css
.hidden { display: none; }
```
```js
function showScreen(id) {
    // hide all three…
    document.getElementById("entrance-screen").classList.add("hidden");
    document.getElementById("lobby-screen").classList.add("hidden");
    document.getElementById("game-screen").classList.add("hidden");
    // …then un-hide the one we want
    document.getElementById(id).classList.remove("hidden");
}
```
- `classList.add/remove` adds or removes a class on an element, so CSS rules switch on and off.
- An element can have several classes: `class="screen card hidden"`.

### Step 6: Create / join a lobby (`create and join a lobby`, `fixed the create and join lobby`)
**Fake for now**: no other players can really join yet (that needs a server, see Part 4).

```js
const name = document.getElementById("username-input").value.trim();
if (name === "") {
    alert("Type your name first!");
    return;          // stop the function early
}
```
- `.value` = the text typed in an `<input>`.
- `.trim()` = remove spaces at the start/end (Python: `.strip()`).
- `===` = "equal" in JS. Always use `===`, not `==` (`==` does weird type conversions).
- A random 4-letter code:
  `Math.random().toString(36).slice(2, 6).toUpperCase()`. That writes a random number
  in base 36 (digits + letters) like `"0.k3x9f…"`, keeps characters 2–5, and uppercases them.
  (`.slice(2, 6)` is Python's `[2:6]`.)
- `.textContent = code` puts text inside an element on the page.

### Step 7: Game layout with flexbox (`made layout of the game screen…`)
```css
.game-layout { display: flex; gap: 50px; }   /* children sit side by side in a row */
.team  { flex: 1; }   /* each team column gets 1 share */
.stage { flex: 2; }   /* the middle gets 2 shares, so it's twice as wide */
```
Result: `[ Team 1 | stickman clip | Team 2 ]`, widths 1 : 2 : 1.

### Step 8: Clips as data + blank-letter hints (`indexing changes…`, `show blank letters…`)
Clips became an **array of objects**:
```js
const clips = [
    { file: "../examples/thumbs_up.mp4", answer: "thumbs up" },
    ...
];
```
- Array `[ ]` = Python list. Object `{ key: value }` = like a Python dict.
- Read a field with a dot: `clip.answer` (Python dict would be `clip["answer"]`).

```js
function makeBlanks(answer) {
    let blanks = "";
    for (const letter of answer) {      // Python: for letter in answer:
        if (letter === " ") blanks += "   ";
        else                blanks += "_ ";
    }
    return blanks;                      // "thumbs up" becomes "_ _ _ _ _ _    _ _ "
}
```
- CSS `white-space: pre` on `#hint` stops the browser from squashing several spaces
  into one (HTML normally does that), so the gap between words stays visible.

### Step 9: 30-second timer (`added 30 second timer`)
```js
timerId = setInterval(tick, 1000);   // call tick() every 1000 ms (= 1 second)
clearInterval(timerId);              // stop it
```
- `setInterval` returns an id, and you need that id to stop the timer later.
- `startTimer()` first calls `clearInterval` so playing a new clip doesn't start a
  *second* timer running on top of the old one (it would count down twice as fast!).
- `tick()` subtracts 1 and stops at 0, showing "Time's up!".

### Step 10: Guess box (`added guess box`)
- HTML: `<input id="guess-input">`, a Guess `<button>`, and `<p id="result">` for "Correct!/Wrong".
- `playRandomClip` saves the chosen clip in `currentClip`, so `checkGuess` knows the answer.
- Compare fairly:
  ```js
  guess.trim().toUpperCase() === currentClip.answer.toUpperCase()
  ```
  so `" thumbs UP "` still counts as correct.
- CSS `text-transform: uppercase` *shows* what you type in capitals, and
  `::placeholder { text-transform: none; }` keeps the grey hint text ("Your guess") normal.
  (This only changes how it looks. The real value is still whatever was typed.)

### Step 11: Enter key submits (`enter key to submit the guesses`)
```js
input.addEventListener("keydown", function (event) {
    if (event.key === "Enter") checkGuess();
});
```
- The browser passes an **event object** describing what happened. `event.key` = which key.
- `function (event) { … }` is an **anonymous function** (like a Python `lambda`, but it can be many lines).

### Step 12: Teams, turns and scores (`added team scores and turns`)
- `currentTeam` flips 1 → 2 → 1 each time a new clip plays.
- A correct guess adds 1 to that team's score and updates `#score1` / `#score2`.
- Then `currentClip = null` so spamming the button can't score the same clip twice.
  (`null` ≈ Python's `None`.) `checkGuess` returns early if `currentClip === null`.
- **Known gap:** guesses are still accepted after "Time's up!".

---

## Part 3: Git routine (how I save progress)
Commit only when a change adds **one working thing**.
```
git status                 # what changed?
git add web/               # stage the files (or a specific file)
git commit -m "short message saying what now works"
```
- I'm on branch `feature/website`. Yash owns `master` and the Python code, so I don't touch those.

---

## Part 4: The server (backend) — `set up socket.io`

### Why a server at all?
Right now every browser runs its own separate copy of the game. My browser can't see
your browser. For real multiplayer, everyone needs to talk to **one shared program**
that sits in the middle and passes messages along: the **server**.

```
 Player A's browser ─┐
                     ├──►  SERVER (one program, always running)
 Player B's browser ─┘     knows the lobbies, scores, answers
```

### Words to know
- **Server**: a program that runs non-stop and waits for others to connect to it.
  (The browsers are the **clients**.)
- **Port**: a numbered "door" on a computer. One computer can run many servers, each
  on its own door. Ours uses door **3000**. `localhost` = "this same computer".
- **Request**: like a **letter**. The browser asks once ("give me index.html"),
  gets one reply, and that's the end of it. The server can't send anything else later on its own.
- **Socket**: like a **phone call that stays open**. Either side can talk at any moment.
  Games need this: when a team scores, the server must tell *everyone* instantly.
- **Node.js**: lets JavaScript run *outside* the browser, on a computer/server
  (like running `python file.py`). Run it with `node server.js`.
- **npm**: Node's package installer (like `pip`). `npm install socket.io` downloaded
  Socket.IO into `node_modules/` and wrote it down in `package.json` (≈ `requirements.txt`).
  `package-lock.json` records exact versions. I don't edit those by hand.
- **`.gitignore`** with `node_modules/`: that folder is huge and can be rebuilt with
  `npm install`, so git skips it.

### `server/server.js` line by line
```js
const { Server } = require("socket.io");
```
Load the Socket.IO library and take out its `Server` class.
Python equivalent: `from socket_io import Server`.

```js
const io = new Server(3000, {
    cors: { origin: "*" }
});
```
Create a Socket.IO server listening on port 3000 (`new` makes an object from a class,
like `Server(3000)` in Python).
**cors**: browsers block a page from one address (Live Server, e.g. `localhost:5500`)
from talking to a different address (`localhost:3000`) unless the server says it's
allowed. `origin: "*"` = "allow anyone". (Planned to remove this once the server
serves the page itself, because then both use the same address.)

```js
io.on("connection", function (socket) {
    console.log("A player connected:", socket.id);
});
```
"Whenever a browser connects, run this function." It's the same idea as
`addEventListener`, but on the server. `socket` is that one player's open
phone line, and `socket.id` is a random unique name for it. `console.log` = Python `print`
(it prints in the terminal running `node`, not in Chrome).

```js
console.log("Server running on port 3000");
```
Just a message so I know it started.

**Run it:** `cd server` then `node server.js` (or `npm start`). Stop with `Ctrl + C`.
**Right now nothing connects to it yet**, since the page has no client code.

---

## Coming next (the plan)
1. Install **Express** so the server also sends out the web page itself
   (`web/`, plus `examples/` at `/examples`). Use `const PORT = process.env.PORT || 3000`
   (hosting sites choose the port for you). Then open `localhost:3000` instead of Live Server.
2. Connect the page to the server (Socket.IO client), so "A player connected" appears.
3. Real lobbies with Socket.IO **rooms**. The server keeps the answers and checks
   guesses, because anything in `script.js` can be read by anyone with F12.
4. Deploy to **Render** or **Railway** (they keep a server running. GitHub Pages /
   Netlify / Vercel only host plain files).

Later: team colours, cream body text, small style tidy-ups.

---

## Quick JS ↔ Python cheat sheet
| JavaScript | Python |
|---|---|
| `let x = 5;` / `const x = 5;` | `x = 5` |
| `[1, 2, 3]` array | list |
| `{ a: 1 }` object, `obj.a` | dict, `d["a"]` |
| `null` | `None` |
| `===` | `==` |
| `console.log(x)` | `print(x)` |
| `s.trim()` | `s.strip()` |
| `s.toUpperCase()` | `s.upper()` |
| `s.slice(2, 6)` | `s[2:6]` |
| `for (const x of list)` | `for x in list:` |
| `function f(a) { … }` | `def f(a): …` |
| `require("lib")` | `import lib` |
| `npm install` | `pip install` |
