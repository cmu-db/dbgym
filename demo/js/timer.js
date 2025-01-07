// Constant for the total timer duration in seconds
// Remember to update welcome.html to match this value
const TIMER_DURATION = 10000; // DEBUG: make it 60

let timerInterval;

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
    } else {
        const elapsedTime = (Date.now() - startTime) / 1000; // in seconds
        const remainingTime = Math.max(0, TIMER_DURATION - elapsedTime);
        document.getElementById('timerDisplay').innerText = `${remainingTime.toFixed(1)}s Remaining`;

        // Show the popup when time is up
        if (remainingTime <= 0) {
            document.getElementById('timesUpPopup').style.display = 'flex';
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
