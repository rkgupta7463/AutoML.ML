function validateForm() {
    const fileInput = document.getElementById('csvFile');
    const selectedFile = fileInput.files[0];
    const problemType = document.getElementById('problemType').value;

    if (!selectedFile) {
        alert('Please select a CSV file.');
        return false;
    }

    if (selectedFile.type !== 'application/vnd.ms-excel') {
        alert('Please select a valid CSV file.');
        return false;
    }

    if (problemType === 'Open this select menu') {
        alert('Please select a problem type.');
        return false;
    }

    return true;
}

function showInputForm() {
    document.getElementById('dataset-form').style.display = 'none';
    document.getElementById('input-form').style.display = 'block';
}