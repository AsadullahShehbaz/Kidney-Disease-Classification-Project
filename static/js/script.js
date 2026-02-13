// ============================================
// DOM ELEMENTS
// ============================================

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const removeBtn = document.getElementById('removeBtn');
const predictBtn = document.getElementById('predictBtn');
const loader = document.getElementById('loader');
const resultSection = document.getElementById('resultSection');
const resultPrediction = document.getElementById('resultPrediction');
const resultIcon = document.getElementById('resultIcon');
const newPredictionBtn = document.getElementById('newPredictionBtn');


// ============================================
// FILE UPLOAD HANDLING
// ============================================

// Click to upload
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.background = '#e0f2fe';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.background = '#f8fafc';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.background = '#f8fafc';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// File input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});


// ============================================
// FILE HANDLING FUNCTION
// ============================================

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('⚠️ Please upload an image file (JPG, JPEG, PNG)');
        return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        alert('⚠️ File size should be less than 10MB');
        return;
    }
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        previewSection.style.display = 'block';
        resultSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}


// ============================================
// REMOVE IMAGE BUTTON
// ============================================

removeBtn.addEventListener('click', (e) => {
    e.stopPropagation(); // Prevent triggering parent click
    fileInput.value = '';
    previewSection.style.display = 'none';
    resultSection.style.display = 'none';
    uploadArea.style.background = '#f8fafc';
});


// ============================================
// PREDICT BUTTON - SEND TO API
// ============================================

predictBtn.addEventListener('click', async () => {
    // Get the file
    const file = fileInput.files[0];
    if (!file) {
        alert('⚠️ Please select an image first');
        return;
    }
    
    // Create FormData (to send file via POST request)
    const formData = new FormData();
    formData.append('file', file);
    
    // Show loader, hide preview and result
    previewSection.style.display = 'none';
    loader.style.display = 'block';
    resultSection.style.display = 'none';
    
    try {
        // Send POST request to /predict endpoint
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        // Parse JSON response
        const data = await response.json();
        
        // Hide loader
        loader.style.display = 'none';
        
        // Check if prediction was successful
        if (data.status === 'success') {
            displayResult(data.prediction);
        } else {
            alert('❌ Error: ' + (data.message || 'Prediction failed'));
        }
        
    } catch (error) {
        // Handle network errors
        loader.style.display = 'none';
        alert('❌ Network error. Please check your connection and try again.');
        console.error('Error:', error);
    }
});


// ============================================
// DISPLAY RESULT FUNCTION
// ============================================

function displayResult(prediction) {
    // Show result section
    resultSection.style.display = 'block';
    
    // Normalize prediction text
    const predictionLower = prediction.toLowerCase();
    
    // Update result based on prediction
    if (predictionLower.includes('normal')) {
        resultPrediction.textContent = '✅ Normal Kidney';
        resultPrediction.className = 'result-prediction normal';
        resultIcon.className = 'result-icon success';
        resultIcon.innerHTML = '<i class="fas fa-check-circle"></i>';
    } else if (predictionLower.includes('tumor')) {
        resultPrediction.textContent = '⚠️ Tumor Detected';
        resultPrediction.className = 'result-prediction tumor';
        resultIcon.className = 'result-icon danger';
        resultIcon.innerHTML = '<i class="fas fa-exclamation-circle"></i>';
    } else {
        // Fallback for unknown predictions
        resultPrediction.textContent = prediction;
        resultPrediction.className = 'result-prediction';
        resultIcon.className = 'result-icon';
        resultIcon.innerHTML = '<i class="fas fa-question-circle"></i>';
    }
    
    // Scroll to result
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
}


// ============================================
// NEW PREDICTION BUTTON
// ============================================

newPredictionBtn.addEventListener('click', () => {
    // Reset everything
    fileInput.value = '';
    previewSection.style.display = 'none';
    resultSection.style.display = 'none';
    loader.style.display = 'none';
    uploadArea.style.background = '#f8fafc';
    
    // Scroll to upload area
    uploadArea.scrollIntoView({ behavior: 'smooth', block: 'center' });
});


// ============================================
// CONSOLE MESSAGE
// ============================================

console.log('%c🏥 Kidney Disease Classifier', 'color: #2563eb; font-size: 20px; font-weight: bold;');
console.log('%cPowered by VGG16 CNN & FastAPI', 'color: #10b981; font-size: 14px;');