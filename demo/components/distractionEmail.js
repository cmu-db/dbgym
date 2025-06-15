// Global array of distraction emails
let distractionEmails = [
    {
        subject: "HELP! Database broken!",
        from: "intern@company.com",
        body: "I think I broke the database... all the queries are slow. Should I turn it off and on again?",
        signature: "- The Intern"
    },
    {
        subject: "URGENT: Database Crisis",
        from: "cto@company.com",
        body: "Fix the database now or you're fired. Customers are leaving.",
        signature: "- Your CTO"
    },
    {
        subject: "We need to talk about Alex",
        from: "spouse@personal.com",
        body: "I've been seeing a lot of messages from someone named Alex. Who is this person? Are you hiding something from me?",
        signature: "- Your Spouse"
    },
    {
        subject: "Your Replacement is Being Hired",
        from: "cto@company.com",
        body: "I've already posted your job on LinkedIn. Better hurry up with that database fix if you want to keep it.",
        signature: "- Your CTO"
    },
    {
        subject: "Taking Over Database Fix",
        from: "coworker@company.com",
        body: "Hey, I've started working on the database fix. The CTO says I'll get your job if I solve it first. Just FYI 😉",
        signature: "- Your 'Friend' at Work"
    },
    {
        subject: "Intern's Gone, You're Next",
        from: "cto@company.com",
        body: "Just fired the intern for breaking the database. You're the only one left to blame. Clock is ticking.",
        signature: "- Your CTO"
    }
];

// Initialize the email index
let currentEmailIndex = 0;

function showDistractionPopup() {
    // Get the next email in sequence
    const email = distractionEmails[currentEmailIndex];
    currentEmailIndex = (currentEmailIndex + 1) % distractionEmails.length;

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

            // Replace the email content
            data = data.replace('{{subject}}', email.subject);
            data = data.replace('{{from}}', email.from);
            data = data.replace('{{body}}', email.body);
            data = data.replace('{{signature}}', email.signature);

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
    // Add initial 10 second delay
    setTimeout(() => {
        const emailInterval = setInterval(() => {
            const timerDisplay = document.getElementById('timerDisplay');
            const timeText = timerDisplay.innerText;
            const remainingTime = parseFloat(timeText);
            
            if (remainingTime <= 0) {
                clearInterval(emailInterval);
            } else {
                showDistractionPopup();
            }
        }, 20000);
    }, 10000);
});