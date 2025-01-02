fetch('/components/genned_query1a.html')
    .then(response => response.text())
    .then(data => {
        document.getElementById('query1View').innerHTML = data;
    })
    .catch(error => console.error('Error loading HTML:', error));

fetch('/components/genned_query3b.html')
    .then(response => response.text())
    .then(data => {
        document.getElementById('query2View').innerHTML = data;
    })
    .catch(error => console.error('Error loading HTML:', error));

fetch('/components/genned_query6a.html')
    .then(response => response.text())
    .then(data => {
        document.getElementById('query3View').innerHTML = data;
    })
    .catch(error => console.error('Error loading HTML:', error));