// Global array of distraction emails
let distractionEmails = [
    {
        subject: "HELP! Database broken!",
        from: "intern@company.com",
        body: "I think I broke the database... all the queries are slow. Should I turn it off and on again?",
        signature: "- The Intern",
        sound: new Audio("../sounds/youve-got-mail.mp3")
    },
    {
        subject: "URGENT: Database Crisis",
        from: "cto@company.com",
        body: "Fix the database now or you're fired. Customers are leaving.",
        signature: "- Your CTO",
        sound: new Audio("../sounds/futuristic-ding.mp3")
    },
    {
        subject: "We need to talk about Alex",
        from: "spouse@personal.com",
        body: "I've been seeing a lot of messages from someone named Alex. Who is this person? Are you hiding something from me?",
        signature: "- Your Spouse",
        sound: new Audio("../sounds/four-bells.mp3")
    },
    {
        subject: "Your Replacement is Being Hired",
        from: "cto@company.com",
        body: "I've already posted your job on LinkedIn. Better hurry up with that database fix if you want to keep it.",
        signature: "- Your CTO",
        sound: new Audio("../sounds/email.mp3")
    },
    {
        subject: "Taking Over Database Fix",
        from: "coworker@company.com",
        body: "Hey, I've started working on the database fix. The CTO says I'll get your job if I solve it first. Just FYI 😉",
        signature: "- Your 'Friend' at Work",
        sound: new Audio("../sounds/boing.mp3")
    },
    {
        subject: "Intern's Gone, You're Next",
        from: "cto@company.com",
        body: "Just fired the intern for breaking the database. You're the only one left to blame. Clock is ticking.",
        signature: "- Your CTO",
        sound: new Audio("../sounds/tune.mp3")
    }
];

// Initialize the email index from localStorage or default to 0
let currentEmailIndex = parseInt(localStorage.getItem('currentEmailIndex')) || 0;

// Get active popups from localStorage or initialize empty array
let activePopups = JSON.parse(localStorage.getItem('activePopups')) || [];

// Get shown thresholds from localStorage or initialize empty array
let shownThresholds = JSON.parse(localStorage.getItem('shownThresholds')) || [];

// Shared function to create and display a popup
function createPopup(popupInfo, position = null) {
    // Play notification sound for new emails (not for restored ones)
    if (!position) {
        popupInfo.email.sound.play().catch(error => console.error("Error playing sound:", error));
    }

    return fetch('../components/distractionEmail.html')
        .then(response => response.text())
        .then(data => {
            // Replace the static IDs with unique ones
            data = data.replace('id="distractionPopup"', `id="${popupInfo.overlayId}"`);
            data = data.replace('id="distractionPopupContent"', `id="${popupInfo.contentId}"`);

            // Replace the email content
            data = data.replace('{{subject}}', popupInfo.email.subject);
            data = data.replace('{{from}}', popupInfo.email.from);
            data = data.replace('{{body}}', popupInfo.email.body);
            data = data.replace('{{signature}}', popupInfo.email.signature);

            // Insert the popup
            const timesUpContainer = document.getElementById('timesUpPopupContainer');
            if (timesUpContainer) {
                timesUpContainer.insertAdjacentHTML('beforebegin', data);
            } else {
                document.body.innerHTML += data;
            }

            // Show the popup
            const overlay = document.getElementById(popupInfo.overlayId);
            const popup = document.getElementById(popupInfo.contentId);
            overlay.style.display = 'block';

            // Position the popup
            if (position) {
                popup.style.left = position.left + 'px';
                popup.style.top = position.top + 'px';
            } else {
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
                
                // Update position in popupInfo
                popupInfo.position = { left, top };
                localStorage.setItem('activePopups', JSON.stringify(activePopups));
            }

            // Add close handler
            const closeBtn = overlay.querySelector('.popup-close');
            closeBtn.onclick = function() {
                overlay.style.display = 'none';
                activePopups = activePopups.filter(p => p.id !== popupInfo.id);
                localStorage.setItem('activePopups', JSON.stringify(activePopups));
            };
        })
        .catch(error => console.error("Error creating popup:", error));
}

function showDistractionPopup() {
    // Get the next email in sequence
    const email = distractionEmails[currentEmailIndex];
    currentEmailIndex = (currentEmailIndex + 1) % distractionEmails.length;
    localStorage.setItem('currentEmailIndex', currentEmailIndex.toString());

    // Generate unique IDs for this popup instance
    const uniqueId = 'distraction_' + currentEmailIndex;
    const overlayId = uniqueId + '_overlay';
    const contentId = uniqueId + '_content';

    // Store popup info in activePopups
    const popupInfo = {
        id: uniqueId,
        overlayId: overlayId,
        contentId: contentId,
        email: email,
        position: null
    };
    activePopups.push(popupInfo);
    localStorage.setItem('activePopups', JSON.stringify(activePopups));

    // Create and show the popup
    createPopup(popupInfo);
}

// Function to restore active popups
function restoreActivePopups() {
    activePopups.forEach(popupInfo => {
        createPopup(popupInfo, popupInfo.position);
    });
}

// Start showing popups based on remaining time
window.addEventListener('DOMContentLoaded', function() {
    // Restore any existing popups first
    restoreActivePopups();
    
    // Check timer every second
    const checkInterval = setInterval(() => {
        const timerDisplay = document.getElementById('timerDisplay');
        const timeText = timerDisplay.innerText;
        const remainingTime = parseFloat(timeText);
        
        if (remainingTime <= 0) {
            clearInterval(checkInterval);
            return;
        }

        // Show popups at specific time thresholds (85s, 70s, etc.)
        const thresholds = [110, 90, 70, 50, 30, 10];
        // const thresholds = [118, 116, 114, 112, 110, 108];
        thresholds.forEach(threshold => {
            if (remainingTime <= threshold && !shownThresholds.includes(threshold)) {
                showDistractionPopup();
                shownThresholds.push(threshold);
                localStorage.setItem('shownThresholds', JSON.stringify(shownThresholds));
            }
        });
    }, 500);
});