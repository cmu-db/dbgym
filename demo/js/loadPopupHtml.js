fetch('/components/genned_query1a.html')
    .then(response => response.text())
    .then(data => {
        document.getElementById('query1View').innerHTML = data;
    })
    .catch(error => console.error('Error loading HTML:', error));

fetch('/components/genned_query11a.html')
    .then(response => response.text())
    .then(data => {
        document.getElementById('query2View').innerHTML = data;
    })
    .catch(error => console.error('Error loading HTML:', error));