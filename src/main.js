import "./style.css";
import { Hands } from "@mediapipe/hands";

const KEYPOINT_MODEL =
  `${import.meta.env.BASE_URL}models/keypoint_classifier.tflite`;

const POINT_HISTORY_MODEL =
  `${import.meta.env.BASE_URL}models/point_history_classifier.tflite`;


const KEYPOINT_LABELS = [
    "1", "2", "3", "4", "5",
    "6", "7", "8", "9", "0",
    "A", "B", "C", "D", "E", "F",
    "G", "H", "I", "J", "K", "L",
    "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X",
    "Y", "Z"
];

const POINT_HISTORY_LABELS = [
  "Stop",
  "Clockwise",
  "Counter Clockwise",
  "Move"
];

const STABLE_THRESHOLD = 35;
const WORDS_PER_BATCH = 2;
const WORD_BATCH_DELAY_MS = 3000;
const cameraCard = document.querySelector("#cameraCard");
const outputCard = document.querySelector("#outputCard"); 

const video = document.querySelector("#video");
const overlay = document.querySelector("#overlay");
const ctx = overlay.getContext("2d");

const statusEl = document.querySelector("#status");
const cameraMessage = document.querySelector("#cameraMessage");
const detectedLetterEl = document.querySelector("#detectedLetter");
const stabilityEl = document.querySelector("#stability");
const fpsEl = document.querySelector("#fps");
const wordOutput = document.querySelector("#wordOutput");

const signModeBtn = document.querySelector("#signModeBtn");
const speechModeBtn = document.querySelector("#speechModeBtn");
const signOutput = document.querySelector("#signOutput");
const speechOutput = document.querySelector("#speechOutput");

const startCameraBtn = document.querySelector("#startCameraBtn");
const stopCameraBtn = document.querySelector("#stopCameraBtn");
const backspaceBtn = document.querySelector("#backspaceBtn");
const clearBtn = document.querySelector("#clearBtn");
const speakBtn = document.querySelector("#speakBtn");

const listenBtn = document.querySelector("#listenBtn");
const stopListenBtn = document.querySelector("#stopListenBtn");
const speechTextEl = document.querySelector("#speechText");
const speechImages = document.querySelector("#speechImages");

let keypointModel = null;
let pointHistoryModel = null;
let hands = null;
let stream = null;
let animationFrame = null;
const processingCanvas = document.createElement("canvas");
const processingCtx = processingCanvas.getContext("2d");

let translatorMode = 0; // 0 = sign -> speech, 1 = speech -> sign
let signToSpeechWord = "";
let stableLetter = null;
let stableCounter = 0;
let waitingForNewSign = false;

let pointHistory = [];
const HISTORY_LENGTH = 16;

let lastFrameTime = performance.now();
let frameCounter = 0;
let displayedFps = 0;

let inferenceBusy = false;
let lastResults = null;
let backspaceLocked = false;

let currentWordBatch = 0;
let lastSpeechText = "";
let lastBatchTime = 0;

let recognition = null;
let recognitionSupported = false;

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.style.borderColor = isError ? "#ef4444" : "#334155";
  statusEl.style.color = isError ? "#fca5a5" : "#cbd5e1";
}

function updateWordOutput() {
  wordOutput.textContent = signToSpeechWord || "—";
}

function resetStability() {
  stableLetter = null;
  stableCounter = 0;
  stabilityEl.textContent = `0 / ${STABLE_THRESHOLD}`;
}

function clearSignText() {
  signToSpeechWord = "";
  resetStability();
  updateWordOutput();
}

function backspaceSignText() {
  if (backspaceLocked) return;
  if (signToSpeechWord.length === 0) return;

  backspaceLocked = true;

  signToSpeechWord = signToSpeechWord.substring(
    0,
    signToSpeechWord.length - 1
  );

  updateWordOutput();

  stableLetter = null;
  stableCounter = 0;

  stabilityEl.textContent = `0 / ${STABLE_THRESHOLD}`;

  // Prevent another deletion until the key is released.
  setTimeout(() => {
    backspaceLocked = false;
  }, 100);
}

function speakText(text) {
  const clean = text.trim();
  if (!clean) return;

  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(clean);
  utterance.lang = "en-US";
  utterance.rate = 1;
  utterance.pitch = 1;

  window.speechSynthesis.speak(utterance);
}

function setMode(mode) {
  translatorMode = mode;

  const signMode = mode === 0;
  signModeBtn.classList.toggle("active", signMode);
  speechModeBtn.classList.toggle("active", !signMode);

  signOutput.hidden = !signMode;
  speechOutput.hidden = signMode;

  resetStability();

  if (signMode) {
    stopSpeechRecognition();
  } else {
    startSpeechRecognition();
  }
}

function preprocessLandmarks(landmarks, width, height) {
    // Step 1: Convert MediaPipe's normalized coordinates
    // to the same integer pixel coordinates used by Python.
    const landmarkList = landmarks.map((landmark) => [
        Math.min(Math.floor(landmark.x * width), width - 1),
        Math.min(Math.floor(landmark.y * height), height - 1)
    ]);

    // Step 2: Use landmark 0 (wrist) as the origin.
    const baseX = landmarkList[0][0];
    const baseY = landmarkList[0][1];

    const relativeLandmarks = landmarkList.map(([x, y]) => [
        x - baseX,
        y - baseY
    ]);

    // Step 3: Flatten [ [x,y], [x,y], ... ]
    // into [x,y,x,y,...]
    const flattened = relativeLandmarks.flat();

    // Step 4: Find the single largest absolute value.
    const maxValue = Math.max(
        ...flattened.map((value) => Math.abs(value))
    );

    // Prevent division by zero.
    if (maxValue === 0) {
        return flattened.map(() => 0);
    }

    // Step 5: Normalize exactly like Python.
    return flattened.map((value) => value / maxValue);
}

function preprocessPointHistory(history, imageWidth, imageHeight) {
  if (history.length === 0) return [];

  const baseX = history[0][0];
  const baseY = history[0][1];

  const flat = [];

  for (const [x, y] of history) {
    flat.push((x - baseX) / imageWidth);
    flat.push((y - baseY) / imageHeight);
  }

  return flat;
}

function argmax(values) {
  let bestIndex = 0;
  let bestValue = values[0] ?? -Infinity;

  for (let i = 1; i < values.length; i++) {
    if (values[i] > bestValue) {
      bestValue = values[i];
      bestIndex = i;
    }
  }

  return bestIndex;
}

function tensorDataToArray(output) {
  if (Array.isArray(output)) {
    return output[0].dataSync();
  }
  return output.dataSync();
}

async function predictKeypoint(input42) {
  const inputTensor = window.tf.tensor2d(
    [input42],
    [1, 42],
    "float32"
  );

  let output = null;

  try {
    output = keypointModel.predict(inputTensor);

    const data = output.dataSync();

    return argmax(data);
  } finally {
    inputTensor.dispose();

    if (output && typeof output.dispose === "function") {
      output.dispose();
    }
  }
}

async function predictPointHistory(input32) {
  const inputTensor = window.tf.tensor2d(
    [input32],
    [1, 32],
    "float32"
  );

  let output = null;

  try {
    output = pointHistoryModel.predict(inputTensor);

    const data = output.dataSync();

    return argmax(data);
  } finally {
    inputTensor.dispose();

    if (output && typeof output.dispose === "function") {
      output.dispose();
    }
  }
}

function addStableLetter(letter) {
  if (!letter) return;

  if (letter === stableLetter) {
    stableCounter += 1;
  } else {
    stableLetter = letter;
    stableCounter = 0;
  }

  stabilityEl.textContent =
    `${Math.min(stableCounter, STABLE_THRESHOLD)} / ${STABLE_THRESHOLD}`;

  // Add the letter every time the stability threshold is reached.
  if (stableCounter >= STABLE_THRESHOLD) {
    signToSpeechWord += letter;

    updateWordOutput();

    // Reset the counter so the same held sign
    // can be detected again.
    stableCounter = 0;

    stabilityEl.textContent = `0 / ${STABLE_THRESHOLD}`;
  }
}

function drawResults(results) {
  const width = overlay.width;
  const height = overlay.height;

  ctx.clearRect(0, 0, width, height);

  if (!results?.multiHandLandmarks?.length) {
    detectedLetterEl.textContent = "—";
    stabilityEl.textContent = `0 / ${STABLE_THRESHOLD}`;
    return;
  }

  const landmarks = results.multiHandLandmarks[0];

  // Mirror the x coordinate to match the mirrored camera display.
  const points = landmarks.map((p) => ({
    x: (1 - p.x) * width,
    y: p.y * height
  }));

  // Connections corresponding to the 21-point MediaPipe hand skeleton.
  const connections = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],
    [0,17]
  ];

  ctx.lineWidth = 3;
  ctx.strokeStyle = "#ffffff";

  for (const [a, b] of connections) {
    ctx.beginPath();
    ctx.moveTo(points[a].x, points[a].y);
    ctx.lineTo(points[b].x, points[b].y);
    ctx.stroke();
  }

  for (let i = 0; i < points.length; i++) {
    ctx.beginPath();
    ctx.arc(points[i].x, points[i].y, [4,8,12,16,20].includes(i) ? 6 : 4, 0, Math.PI * 2);
    ctx.fillStyle = "#111827";
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  const minX = Math.min(...points.map((p) => p.x));
  const minY = Math.min(...points.map((p) => p.y));
  const maxX = Math.max(...points.map((p) => p.x));
  const maxY = Math.max(...points.map((p) => p.y));

  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1;
  ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);
}

async function processResults(results) {
  lastResults = results;

  if (!results?.multiHandLandmarks?.length || translatorMode !== 0) {
    return;
  }

  if (inferenceBusy) return;
  inferenceBusy = true;

  try {
    const landmarks = results.multiHandLandmarks[0];

    const imageWidth = video.videoWidth;
    const imageHeight = video.videoHeight;

    // The Python version converts MediaPipe normalized coordinates into
    // integer pixel coordinates before preprocessing.
    const preprocessed = preprocessLandmarks(
    landmarks,
    imageWidth,
    imageHeight
    );  

    console.log("Input length:", preprocessed.length);
console.log("Input:", preprocessed);
console.log(
  "Contains NaN:",
  preprocessed.some((v) => Number.isNaN(v))
);
console.log(
  "Contains Infinity:",
  preprocessed.some((v) => !Number.isFinite(v))
);

    if (preprocessed.length !== 42) {
      throw new Error(`Expected 42 keypoint values, got ${preprocessed.length}`);
    }

    const handSignId = await predictKeypoint(preprocessed);
    const currentLetter = KEYPOINT_LABELS[handSignId] ?? "";

    detectedLetterEl.textContent = currentLetter || "—";
    addStableLetter(currentLetter);

    // Pixel coordinates are still needed for point history.
    const landmarkList = landmarks.map((landmark) => [
    Math.min(Math.trunc(landmark.x * imageWidth), imageWidth - 1),
    Math.min(Math.trunc(landmark.y * imageHeight), imageHeight - 1)
    ]);

    // The Python project tracks landmark 8 only when the classifier
    // predicts class ID 2 ("3" according to the provided label file).
    if (handSignId === 2) {
      pointHistory.push(landmarkList[8]);
    } else {
      pointHistory.push([0, 0]);
    }

    if (pointHistory.length > HISTORY_LENGTH) {
      pointHistory.shift();
    }

    if (pointHistory.length === HISTORY_LENGTH && pointHistoryModel) {
      const historyInput = preprocessPointHistory(
        pointHistory,
        imageWidth,
        imageHeight
      );

      if (historyInput.length === 32) {
        await predictPointHistory(historyInput);
      }
    }
  } catch (error) {
    console.error("Inference error:", error);
    setStatus(`Inference error: ${error.message}`, true);
  } finally {
    inferenceBusy = false;
  }
}

async function initModels() {
  setStatus("Loading TensorFlow…");

  if (!window.tf) {
    throw new Error("TensorFlow.js failed to load.");
  }

  if (!window.tflite) {
    throw new Error("TensorFlow Lite failed to load.");
  }

  console.log("TensorFlow:", window.tf);
  console.log("TensorFlow version:", window.tf.version.tfjs);
  console.log("TFLite:", window.tflite);

  await window.tf.setBackend("cpu");
  await window.tf.ready();

  console.log("Backend:", window.tf.getBackend());

  setStatus("Loading keypoint model…");

  keypointModel = await window.tflite.loadTFLiteModel(
    KEYPOINT_MODEL
  );

  console.log("Keypoint model loaded:", keypointModel);

  setStatus("Loading point-history model…");

  pointHistoryModel = await window.tflite.loadTFLiteModel(
    POINT_HISTORY_MODEL
  );

  console.log("Point-history model loaded:", pointHistoryModel);

  setStatus("Loading MediaPipe Hands…");

  hands = new Hands({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`
  });

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.5
  });

  hands.onResults((results) => {
    drawResults(results);
    processResults(results);
  });

  setStatus("Ready");
}

async function startCamera() {
  if (!hands) {
    setStatus("Models are still loading…", true);
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 960 },
        height: { ideal: 540 },
        facingMode: "user"
      },
      audio: false
    });

    video.srcObject = stream;
    await video.play();

    overlay.width = video.videoWidth || 960;
    overlay.height = video.videoHeight || 540;

    cameraMessage.style.display = "none";
    startCameraBtn.disabled = true;
    stopCameraBtn.disabled = false;

    setStatus("Camera running");

    processVideo();
  } catch (error) {
    console.error(error);
    setStatus(`Camera error: ${error.message}`, true);
    cameraMessage.textContent =
      "Camera access was blocked. Allow camera permission and try again.";
  }
}

function stopCamera() {
  if (animationFrame) {
    cancelAnimationFrame(animationFrame);
    animationFrame = null;
  }

  if (stream) {
    for (const track of stream.getTracks()) {
      track.stop();
    }
    stream = null;
  }

  video.srcObject = null;
  startCameraBtn.disabled = false;
  stopCameraBtn.disabled = true;
  cameraMessage.style.display = "grid";

  ctx.clearRect(0, 0, overlay.width, overlay.height);

  resetStability();
  detectedLetterEl.textContent = "—";
  setStatus("Camera stopped");
}

async function processVideo() {
  if (!stream) return;

  const now = performance.now();
  frameCounter++;

  if (now - lastFrameTime >= 1000) {
    displayedFps = frameCounter;
    frameCounter = 0;
    lastFrameTime = now;
    fpsEl.textContent = displayedFps;
  }

  if (video.readyState >= 2) {
  const width = video.videoWidth;
  const height = video.videoHeight;

  processingCanvas.width = width;
  processingCanvas.height = height;

  // Mirror the frame exactly like the Python version:
  // image = cv.flip(image, 1)
  processingCtx.save();
  processingCtx.translate(width, 0);
  processingCtx.scale(-1, 1);

  processingCtx.drawImage(
    video,
    0,
    0,
    width,
    height
  );

  processingCtx.restore();

  await hands.send({
    image: processingCanvas
  });
}

  animationFrame = requestAnimationFrame(processVideo);
}

function updateSpeechImages(text) {
  speechImages.innerHTML = "";

  const words = text.toUpperCase().trim().split(/\s+/).filter(Boolean);

  if (!words.length) return;

  const start = currentWordBatch * WORDS_PER_BATCH;
  const batch = words.slice(start, start + WORDS_PER_BATCH);

  for (const word of batch) {
    const group = document.createElement("div");
    group.className = "word-group";

    for (const character of word) {
      if (!/^[A-Z]$/.test(character)) continue;

      const card = document.createElement("div");
      card.className = "letter-card";

      const img = document.createElement("img");
      img.src = `${import.meta.env.BASE_URL}asl_images/${character}.jpg`;
      img.alt = `ASL letter ${character}`;

      img.onerror = () => {
        img.style.display = "none";
      };

      const label = document.createElement("span");
      label.textContent = character;

      card.append(img, label);
      group.appendChild(card);
    }

    speechImages.appendChild(group);
  }
}

function handleSpeechText(text) {
  const normalized = text.trim().toUpperCase();

  if (normalized !== lastSpeechText) {
    currentWordBatch = 0;
    lastBatchTime = performance.now();
    lastSpeechText = normalized;
  }

  speechTextEl.textContent = normalized || "—";
  updateSpeechImages(normalized);
}

function startSpeechRecognition() {
  if (!recognitionSupported || !recognition) return;

  try {
    recognition.start();
  } catch {
    // Calling start while already running throws in some browsers.
  }
}

function stopSpeechRecognition() {
  if (!recognition) return;

  try {
    recognition.stop();
  } catch {
    // Ignore when already stopped.
  }

  listenBtn.disabled = false;
  stopListenBtn.disabled = true;
}

function setupSpeechRecognition() {
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    listenBtn.disabled = true;
    stopListenBtn.disabled = true;
    speechTextEl.textContent =
      "Speech recognition is not supported by this browser.";
    return;
  }

  recognitionSupported = true;

  recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = false;
  recognition.lang = "en-US";

  recognition.onstart = () => {
    listenBtn.disabled = true;
    stopListenBtn.disabled = false;
    setStatus("Listening…");
  };

  recognition.onresult = (event) => {
    let latest = "";

    for (let i = event.resultIndex; i < event.results.length; i++) {
      if (event.results[i].isFinal) {
        latest += event.results[i][0].transcript + " ";
      }
    }

    if (latest.trim()) {
  handleSpeechText(latest.trim());
}
  };

  recognition.onerror = (event) => {
    console.warn("Speech recognition:", event.error);
    setStatus(`Speech recognition: ${event.error}`, true);
  };

  recognition.onend = () => {
    listenBtn.disabled = false;
    stopListenBtn.disabled = true;

    if (translatorMode === 1) {
      setStatus("Speech recognition stopped");
    }
  };
}

setInterval(() => {
  if (translatorMode !== 1) return;

  const text = speechTextEl.textContent;
  if (!text || text === "—") return;

  const words = text.trim().split(/\s+/).filter(Boolean);

  if (
    words.length > (currentWordBatch + 1) * WORDS_PER_BATCH &&
    performance.now() - lastBatchTime >= WORD_BATCH_DELAY_MS
  ) {
    currentWordBatch += 1;
    lastBatchTime = performance.now();
    updateSpeechImages(text);
  }
}, 100);

startCameraBtn.addEventListener("click", startCamera);
stopCameraBtn.addEventListener("click", stopCamera);

signModeBtn.addEventListener("click", () => setMode(0));
speechModeBtn.addEventListener("click", () => setMode(1));

backspaceBtn.addEventListener("click", backspaceSignText);
clearBtn.addEventListener("click", clearSignText);
speakBtn.addEventListener("click", () => speakText(signToSpeechWord));

listenBtn.addEventListener("click", startSpeechRecognition);
stopListenBtn.addEventListener("click", stopSpeechRecognition);

document.addEventListener("keydown", (event) => {
  // Backspace → delete last detected letter
  if (event.key === "Backspace") {
  event.preventDefault();

  if (!event.repeat) {
    backspaceSignText();
  }
}

  // C → clear the entire sign-to-speech output
  if (event.key.toLowerCase() === "c") {
    // Don't trigger if the user is typing into an input field
    if (
      event.target.tagName !== "INPUT" &&
      event.target.tagName !== "TEXTAREA"
    ) {
      event.preventDefault();
      clearSignText();
    }
  }

  document.addEventListener("keydown", (event) => {
  // 1 → Sign to Speech
  if (event.key === "1") {
    setMode(0);
  }

  // 2 → Speech to Sign
  if (event.key === "2") {
    setMode(1);
  }

  // Backspace → delete last detected letter
  if (event.key === "Backspace") {
    event.preventDefault();
    backspaceSignText();
  }

  // C → clear sign-to-speech output
  if (
    event.key.toLowerCase() === "c" &&
    event.target.tagName !== "INPUT" &&
    event.target.tagName !== "TEXTAREA"
  ) {
    event.preventDefault();
    clearSignText();
  }

  // Enter → speak detected word
  if (event.key === "Enter") {
    event.preventDefault();
    speakText(signToSpeechWord);
  }
});

  // Enter → speak the current detected word
  if (event.key === "Enter") {
    event.preventDefault();
    speakText(signToSpeechWord);
  }
});

window.addEventListener("beforeunload", () => {
  stopCamera();
  stopSpeechRecognition();
  window.speechSynthesis.cancel();
});

async function main() {
  setupSpeechRecognition();

  try {
    await initModels();
  } catch (error) {
    console.error(error);
    setStatus(`Startup error: ${error.message}`, true);
    cameraMessage.textContent =
      "Could not load the models. Check the model files in public/models.";
  }
}

main();
