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
        document.getElementById('timerDisplay').innerText = `60.0s Remaining`;
    } else {
        const elapsedTime = (Date.now() - startTime) / 1000; // in seconds
        const remainingTime = Math.max(0, 60 - elapsedTime);
        document.getElementById('timerDisplay').innerText = `${remainingTime.toFixed(1)}s Remaining`;
    }
}

// Call loadTimer on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log(window.location.pathname);

    if (window.location.pathname == '/pages/welcome.html') {
        clearTimer();
    } else if (window.location.pathname == '/pages/leaderboard.html') {
        loadPausedTimer();
    } else {
        continueTimer();
    }
});
