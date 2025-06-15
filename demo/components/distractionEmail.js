function showDistractionPopup() {
    // Generate unique IDs for this popup instance
    const uniqueId = 'distraction_' + Date.now();
    const overlayId = uniqueId + '_overlay';
    const contentId = uniqueId + '_content';

    // Load the distraction popup component
    fetch('../components/distractionEmail.html')
        .then(response => response.text())
        .then(data => {
            // Replace the static IDs with unique ones
            data = data.replace('id="distractionPopup"', `id="${overlayId}"`);
            data = data.replace('id="distractionPopupContent"', `id="${contentId}"`);

            

            // Insert the popup before the times up popup container if it exists
            const timesUpContainer = document.getElementById('timesUpPopupContainer');
            if (timesUpContainer) {
                timesUpContainer.insertAdjacentHTML('beforebegin', data);
            } else {
                document.body.innerHTML += data;
            }

            // Show the distraction popup (now that it's in the DOM)
            const overlay = document.getElementById(overlayId);
            const popup = document.getElementById(contentId);
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
        })
        .catch(error => console.error("Error loading distraction popup:", error));
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