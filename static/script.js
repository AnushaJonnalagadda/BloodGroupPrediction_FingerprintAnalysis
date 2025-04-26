// Global variables
const fileInput = document.getElementById('file-upload');
const fileNameDisplay = document.getElementById('file-name');
const predictButton = document.getElementById('predict-btn');
const predictionResult = document.getElementById('prediction-result');
const bloodDropAnimation = document.getElementById('blood-drop');
const logoutButton = document.getElementById('logout-btn');

// Initialize Speech Synthesis
const synth = window.speechSynthesis;

// Handle file selection
fileInput.addEventListener('change', function () {
    const file = fileInput.files[0];
    if (file) {
        fileNameDisplay.textContent = file.name;
    } else {
        fileNameDisplay.textContent = 'No file chosen';
    }
});

// Function to speak text
function speak(text) {
    if (synth.speaking) {
        console.error('Speech synthesis is already in progress');
        return;
    }

    const utterance = new SpeechSynthesisUtterance(text);
    synth.speak(utterance);
}

// Predict Blood Group Function with Voice Output
function predictBloodGroup() {
    const file = fileInput.files[0];

    if (!file) {
        alert("Please upload a fingerprint image.");
        return;
    }

    // Start animation
    bloodDropAnimation.style.animation = 'none'; // Reset animation
    bloodDropAnimation.offsetHeight; // Trigger reflow to restart the animation
    bloodDropAnimation.style.animation = 'dropAnimation 2s infinite'; // Restart animation

    // Simulate prediction (Replace with actual API call for prediction)
    setTimeout(() => {
        // Simulate prediction response
        const bloodGroups = ['A-', 'A+', 'AB-', 'AB+', 'B-', 'B+', 'O-', 'O+'];
        const randomIndex = Math.floor(Math.random() * bloodGroups.length);
        const predictedBloodGroup = bloodGroups[randomIndex];

        // Display result
        predictionResult.textContent = `✅ The predicted blood group is: ${predictedBloodGroup}`;
        
        // Voice Feedback
        speak(`The predicted blood group is: ${predictedBloodGroup}`);
    }, 3000); // Simulate loading time (3 seconds)
}

// Speech Recognition for Voice Command to Upload File
function startVoiceRecognition() {
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';

    recognition.onstart = function () {
        console.log('Voice recognition started. Speak now.');
    };

    recognition.onresult = function (event) {
        const transcript = event.results[0][0].transcript;
        console.log('Recognized text:', transcript);

        // If user says "upload" or similar, trigger file input
        if (transcript.toLowerCase().includes('upload') || transcript.toLowerCase().includes('choose file')) {
            fileInput.click();
            speak("Please upload a fingerprint image.");
        }
        // If user says "predict", trigger the prediction
        else if (transcript.toLowerCase().includes('predict') || transcript.toLowerCase().includes('blood group')) {
            predictBloodGroup();
        }
    };

    recognition.onerror = function (event) {
        console.error('Speech recognition error', event);
        speak("Sorry, I didn't catch that. Please try again.");
    };

    recognition.onend = function () {
        console.log('Voice recognition ended.');
    };

    recognition.start();
}

// Logout function (for demo)
function logout() {
    window.location.href = 'login.html'; // Redirect to login page (adjust URL as needed)
}

// Additional helper functions (if needed for features such as error handling)
function displayError(message) {
    alert(message);
}

// Voice Assistant Button (Start Recognition)
document.getElementById('voice-assistant-btn').addEventListener('click', startVoiceRecognition);
