// Function to load the navbar
function loadNavbar() {
    fetch('../components/navbar.html')
        .then(response => response.text())
        .then(data => {
            document.getElementById('navbar-placeholder').innerHTML = data;
        });
}

// Load the navbar when the page loads
document.addEventListener('DOMContentLoaded', loadNavbar);
