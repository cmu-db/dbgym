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

    // Calculate random position (ensure popup stays within viewport)
    const maxLeft = Math.max(0, vw - pw);
    const maxTop = Math.max(0, vh - ph);
    const left = Math.random() * maxLeft;
    const top = Math.random() * maxTop;

    popup.style.left = left + 'px';
    popup.style.top = top + 'px';
}

function closeDistractionPopup() {
    document.getElementById('distractionPopup').style.display = 'none';
}

// Start showing popups every 5 seconds
window.addEventListener('DOMContentLoaded', function() {
    setInterval(() => {
        showDistractionPopup();
    }, 1000);
});