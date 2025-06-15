function showDistractionPopup() {
    const overlay = document.getElementById('distractionPopup');
    const popup = document.getElementById('distractionPopupContent');
    overlay.style.display = 'block';

    // Get viewport size
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    // Get popup size (after display:block)
    popup.style.left = '0px';
    popup.style.top = '0px';
    const rect = popup.getBoundingClientRect();
    const pw = rect.width;
    const ph = rect.height;

    // Add buffer zone (50px from edges)
    const buffer = 50;
    const maxLeft = Math.max(buffer, vw - pw - buffer);
    const maxTop = Math.max(buffer, vh - ph - buffer);
    const left = Math.random() * (maxLeft - buffer) + buffer;
    const top = Math.random() * (maxTop - buffer) + buffer;

    popup.style.left = left + 'px';
    popup.style.top = top + 'px';
}

function closeDistractionPopup() {
    document.getElementById('distractionPopup').style.display = 'none';
}

// Start showing popups every second until time runs out
window.addEventListener('DOMContentLoaded', function() {
    const emailInterval = setInterval(() => {
        const timerDisplay = document.getElementById('timerDisplay');
        const timeText = timerDisplay.innerText;
        const remainingTime = parseFloat(timeText);
        
        if (remainingTime <= 0) {
            clearInterval(emailInterval);
        } else {
            showDistractionPopup();
        }
    }, 1000);
});