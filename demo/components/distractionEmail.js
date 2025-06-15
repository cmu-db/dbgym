function showDistractionPopup() {
    // List of possible distraction emails
    const distractionEmails = [
        {
            subject: "Urgent: Meeting Tomorrow",
            from: "boss@company.com",
            body: "Hi there, I need to confirm your attendance for tomorrow's important strategy meeting at 10 AM. Please respond ASAP.",
            signature: "- Your Boss"
        },
        {
            subject: "Your Package Delivery",
            from: "delivery@amazon.com",
            body: "Your package #12345 has been delivered to your front porch. Please confirm receipt.",
            signature: "- Amazon Delivery"
        },
        {
            subject: "Password Reset Request",
            from: "security@company.com",
            body: "We noticed unusual activity on your account. Please reset your password immediately.",
            signature: "- Security Team"
        },
        {
            subject: "New Message from LinkedIn",
            from: "notifications@linkedin.com",
            body: "You have 3 new connection requests and 2 unread messages waiting for you.",
            signature: "- LinkedIn Team"
        },
        {
            subject: "Your Subscription is Expiring",
            from: "support@netflix.com",
            body: "Your premium subscription will expire in 3 days. Click here to renew now.",
            signature: "- Netflix Support"
        }
    ];

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

            // Randomly select an email from the list
            const randomEmail = distractionEmails[Math.floor(Math.random() * distractionEmails.length)];
            data = data.replace('{{subject}}', randomEmail.subject);
            data = data.replace('{{from}}', randomEmail.from);
            data = data.replace('{{body}}', randomEmail.body);
            data = data.replace('{{signature}}', randomEmail.signature);

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