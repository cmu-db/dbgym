// Constant for the total timer duration in seconds
// Remember to update welcome.html to match this value
const TIMER_DURATION = 120;

let timerInterval;
let hasPlayedSound = false;
let clockSound = null;
const timesUpSound = new Audio('../sounds/times-up.mp3');

// Add CSS for flashing animation
const style = document.createElement('style');
style.textContent = `
    @keyframes flash {
        0% { color: white; }
        50% { color: red; }
        100% { color: white; }
    }
    .flashing {
        animation: flash 1s infinite;
    }
`;
document.head.appendChild(style);

function startClockSound() {
    if (!clockSound) {
        clockSound = new Audio('../sounds/clock.mp3');
        clockSound.loop = true;
        clockSound.play().catch(error => console.error("Error playing clock sound:", error));
    }
}

function stopClockSound() {
    if (clockSound) {
        clockSound.pause();
        clockSound.currentTime = 0;
        clockSound = null;
    }
}

function startTimer() {
    const startTime = Date.now();
    localStorage.setItem('startTime', startTime);
}

function loadPausedTimer() {
    const startTime = localStorage.getItem('startTime');
    setText(startTime);
}

function continueTimer() {
    const startTime = localStorage.getItem('startTime');
    setText(startTime);
    timerInterval = setInterval(() => {
        setText(startTime);
    }, 10);
}

function clearTimer() {
    localStorage.removeItem('startTime');
    setText(null);
}

function setText(startTime) {
    if (!startTime) {
        document.getElementById('timerDisplay').innerText = `${TIMER_DURATION}.0s Remaining`;
        document.getElementById('timerDisplay').classList.remove('flashing');
        stopClockSound();
    } else {
        const elapsedTime = (Date.now() - startTime) / 1000; // in seconds
        const remainingTime = Math.max(0, TIMER_DURATION - elapsedTime);
        const timerDisplay = document.getElementById('timerDisplay');
        timerDisplay.innerText = `${remainingTime.toFixed(1)}s Remaining`;

        // Add flashing effect and clock sound when 10 seconds or less remain
        if (remainingTime <= 10 && remainingTime > 0) {
            timerDisplay.classList.add('flashing');
            startClockSound();
        } else {
            timerDisplay.classList.remove('flashing');
            stopClockSound();
        }

        // Show the popup and play sound when time is up
        if (remainingTime <= 0 && !hasPlayedSound) {
            document.getElementById('timesUpPopup').style.display = 'flex';
            timesUpSound.play().catch(error => console.error("Error playing sound:", error));
            hasPlayedSound = true;
            stopClockSound();
        }
    }
}

// Call loadTimer on page load
document.addEventListener('DOMContentLoaded', function() {
    if (window.location.pathname == '/pages/welcome.html') {
        clearTimer();
    } else if (window.location.pathname == '/pages/results.html') {
        loadPausedTimer();
    } else {
        continueTimer();
    }
});
