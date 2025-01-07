function initializePopup() {
    fetch('/components/genned_query1a.html')
        .then(response => response.text())
        .then(data => {
            document.getElementById('query1View').innerHTML = data;
        })
        .catch(error => console.error('Error loading HTML:', error));

    fetch('/components/genned_query2a.html')
        .then(response => response.text())
        .then(data => {
            document.getElementById('query2View').innerHTML = data;
        })
        .catch(error => console.error('Error loading HTML:', error));

    fetch('/components/genned_query4a.html')
        .then(response => response.text())
        .then(data => {
            document.getElementById('query3View').innerHTML = data;
        })
        .catch(error => console.error('Error loading HTML:', error));
}

function initializeAccordion() {
    const headers = document.getElementsByClassName("accordion-header");
    for (let i = 0; i < headers.length; i++) {
        headers[i].addEventListener("click", function() {
            this.classList.toggle("active");
            const panel = this.nextElementSibling;
            if (panel.style.display === "block") {
                panel.style.display = "none";
            } else {
                panel.style.display = "block";
            }
        });
    }
}