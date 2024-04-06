function handleDragOver(event) {
    event.preventDefault();
    document.getElementById('dropArea').classList.add('active-drag');
}

function handleDragLeave(event) {
    event.preventDefault();
    document.getElementById('dropArea').classList.remove('active-drag');
}

function handleDrop(event) {
    event.preventDefault();
    document.getElementById('dropArea').classList.remove('active-drag');

    const files = event.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];

        if (file.name.toLowerCase().endsWith('.txt')) {
            readTextFileContent(file);
        } else if (file.name.toLowerCase().endsWith('.pdf')) {
            readPdfFileContent(file);
        } else if (file.name.toLowerCase().endsWith('.docx')) {
            readDocxFileContent(file);
        } else {
            $('#invalidFileModal').modal('show');
        }
    }
}

function readTextFileContent(file) {
    const reader = new FileReader();

    reader.onload = function (e) {
        const textContent = e.target.result;
        document.getElementById('textInput').value = textContent;
        $('#summarizeButton').prop('disabled', false);
    };

    reader.readAsText(file);
}

async function readPdfFileContent(file) {
    const arrayBuffer = await file.arrayBuffer();
    const pdfData = new Uint8Array(arrayBuffer);

    const pdfjsLib = window['pdfjs-dist/build/pdf'];

    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.10.377/pdf.worker.min.js';

    const pdfDoc = await pdfjsLib.getDocument({
        data: pdfData
    }).promise;

    let textContent = '';

    for (let i = 0; i < pdfDoc.numPages; i++) {
        const page = await pdfDoc.getPage(i + 1);
        const textContentData = await page.getTextContent();

        textContentData.items.forEach(item => {
            textContent += item.str + ' ';
        });
    }

    document.getElementById('textInput').value = textContent;
}

async function readDocxFileContent(file) {
    const arrayBuffer = await file.arrayBuffer();
    const mammothResult = await mammoth.extractRawText({
        arrayBuffer
    });

    const textContent = mammothResult.value;
    document.getElementById('textInput').value = textContent;
}

document.getElementById('file').addEventListener('change', function (event) {
    const files = event.target.files;

    if (files.length > 0) {
        const file = files[0];

        if (file.name.toLowerCase().endsWith('.txt')) {
            readTextFileContent(file);
        } else if (file.name.toLowerCase().endsWith('.pdf')) {
            readPdfFileContent(file);
        } else if (file.name.toLowerCase().endsWith('.docx')) {
            readDocxFileContent(file);
        } else {
            $('#invalidFileModal').modal('show');
        }
    }
});

async function summarizeText() {
    var inputText = document.getElementById('textInput').value.trim();

    if (inputText === '') {
        $('#errorModal').modal('show');
        document.getElementById("summarizedText").value = '';
    } else {
        try {
            var response = await fetch('/get_summarized_data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: 'input_text=' + encodeURIComponent(inputText),
            });

            if (response.ok) {
                var data = await response.json();
                document.getElementById('summarizedText').value = data.summarized_text;
            } else {
                console.error('Error:', response.statusText);
            }
        } catch (error) {
            console.error('Error:', error);
        }
    }
}


document.addEventListener("DOMContentLoaded", function () {
    var downloadButton = document.getElementById("download");
    var summarizedTextArea = document.getElementById("summarizedText");

    downloadButton.addEventListener("click", function () {
        var summarizedText = summarizedTextArea.value;
        if (summarizedText === '') {
            $('#errorSummarizedTextModal').modal('show');
        } else {
            var docDefinition = {
                content: [{
                    text: summarizedText,
                    fontSize: 12
                }]
            };
            pdfMake.createPdf(docDefinition).download("summarized_text.pdf");
        }
    });
});