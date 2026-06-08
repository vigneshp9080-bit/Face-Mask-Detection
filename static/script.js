document.addEventListener('DOMContentLoaded', () => {
    // Mode Switch
    const modeUploadBtn = document.getElementById('mode-upload');
    const modeCameraBtn = document.getElementById('mode-camera');
    const sectionTitle = document.getElementById('section-title');
    const dropArea = document.getElementById('drop-area');
    const cameraArea = document.getElementById('camera-area');
    
    // Camera Elements
    const webcam = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const startCameraBtn = document.getElementById('start-camera-btn');
    const stopCameraBtn = document.getElementById('stop-camera-btn');
    const cameraStatus = document.getElementById('camera-status');
    
    // Upload Elements
    const fileInput = document.getElementById('file-input');
    const uploadPrompt = document.getElementById('upload-prompt');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const resetBtn = document.getElementById('reset-btn');
    const predictBtn = document.getElementById('predict-btn');
    
    // Result Elements
    const analysisLayout = document.getElementById('analysis-layout');
    const resultContainer = document.getElementById('result-container');
    const loader = document.getElementById('loader');
    const errorMessage = document.getElementById('error-message');
    const predictionResult = document.getElementById('prediction-result');
    const resultBadge = document.getElementById('result-badge');
    const predLabel = document.getElementById('pred-label');
    const predConfidence = document.getElementById('pred-confidence');
    const confidenceFill = document.getElementById('confidence-fill');

    let currentFile = null;
    let currentMode = 'upload'; // 'upload' or 'camera'
    let stream = null;
    let detectionInterval = null;

    // --- Mode Switching ---
    modeUploadBtn.addEventListener('click', () => {
        if (currentMode === 'upload') return;
        currentMode = 'upload';
        
        modeUploadBtn.classList.add('active');
        modeCameraBtn.classList.remove('active');
        sectionTitle.textContent = 'Image Analysis';
        
        dropArea.classList.remove('hidden');
        cameraArea.classList.add('hidden');
        
        // Restore predict button state
        predictBtn.classList.remove('hidden');
        
        stopCamera();
        resetResults();
    });

    modeCameraBtn.addEventListener('click', () => {
        if (currentMode === 'camera') return;
        currentMode = 'camera';
        
        modeCameraBtn.classList.add('active');
        modeUploadBtn.classList.remove('active');
        sectionTitle.textContent = 'Real-Time Detection';
        
        dropArea.classList.add('hidden');
        cameraArea.classList.remove('hidden');
        
        // Hide predict button since camera is real-time
        predictBtn.classList.add('hidden');
        
        resetUpload();
        resetResults();
    });

    // --- Camera Logic ---
    startCameraBtn.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
            webcam.srcObject = stream;
            
            startCameraBtn.classList.add('hidden');
            stopCameraBtn.classList.remove('hidden');
            cameraStatus.textContent = 'Camera is active. Analyzing...';
            cameraStatus.style.color = 'var(--status-with-mask)'; // Greenish text
            
            resetResults();
            resultContainer.classList.remove('hidden');
            analysisLayout.classList.add('has-results');
            predictionResult.classList.remove('hidden'); 
            
            predLabel.textContent = 'Scanning...';
            predConfidence.textContent = '0%';
            confidenceFill.style.width = '0%';
            resultBadge.className = 'result-badge';
            
            // Start detection loop
            startDetectionLoop();
            
        } catch (err) {
            console.error('Error accessing webcam:', err);
            cameraStatus.textContent = 'Error: Please allow camera access.';
            cameraStatus.style.color = 'var(--status-without-mask)';
            showError('Could not access the camera. Please ensure permissions are granted.');
        }
    });

    stopCameraBtn.addEventListener('click', stopCamera);

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            webcam.srcObject = null;
            stream = null;
        }
        
        stopDetectionLoop();
        
        startCameraBtn.classList.remove('hidden');
        stopCameraBtn.classList.add('hidden');
        cameraStatus.textContent = 'Camera is off';
        cameraStatus.style.color = 'var(--text-muted)';
        
        if (currentMode === 'camera') {
            resetResults();
        }
    }

    function startDetectionLoop() {
        if (detectionInterval) clearInterval(detectionInterval);
        
        detectionInterval = setInterval(async () => {
            if (!stream || !webcam.videoWidth) return;
            
            const ctx = canvas.getContext('2d');
            canvas.width = webcam.videoWidth;
            canvas.height = webcam.videoHeight;
            
            ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);
            const base64Image = canvas.toDataURL('image/jpeg', 0.8);
            
            try {
                const response = await fetch('/predict_base64', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: base64Image })
                });

                if (response.ok) {
                    const data = await response.json();
                    if (currentMode === 'camera') { 
                        displayResult(data);
                    }
                }
            } catch (error) {
                console.error("Frame prediction error:", error);
            }
        }, 500); // 2 FPS processing
    }

    // --- Upload Logic ---
    // Handle drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            if (currentMode === 'upload') dropArea.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            if (currentMode === 'upload') dropArea.classList.remove('dragover');
        }, false);
    });

    dropArea.addEventListener('drop', handleDrop, false);
    
    uploadPrompt.addEventListener('click', () => {
        if (currentMode === 'upload') fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0] && currentMode === 'upload') {
            handleFile(this.files[0]);
        }
    });

    function handleDrop(e) {
        if (currentMode !== 'upload') return;
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files[0]) {
            handleFile(files[0]);
        }
    }

    function handleFile(file) {
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        if (!validTypes.includes(file.type)) {
            showError('Invalid file type. Please upload a JPG, JPEG, or PNG image.');
            return;
        }

        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadPrompt.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            predictBtn.disabled = false;
            resetResults();
        };
        reader.readAsDataURL(file);
    }

    resetBtn.addEventListener('click', (e) => {
        e.stopPropagation(); 
        resetUpload();
    });

    function resetUpload() {
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        uploadPrompt.classList.remove('hidden');
        predictBtn.disabled = true;
        resetResults();
    }

    // --- Prediction Request for Upload ---
    predictBtn.addEventListener('click', async () => {
        if (!currentFile || currentMode !== 'upload') return;

        resultContainer.classList.remove('hidden');
        analysisLayout.classList.add('has-results');
        loader.classList.remove('hidden');
        predictionResult.classList.add('hidden');
        hideError();
        predictBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Something went wrong during prediction.');
            }

            displayResult(data);

        } catch (error) {
            showError(error.message);
        } finally {
            loader.classList.add('hidden');
            predictBtn.disabled = false;
        }
    });

    // --- Common UI Update Functions ---
    function displayResult(data) {
        // Only show loader briefly for upload mode, for camera it stays hidden after first frame
        if (currentMode === 'upload') {
            loader.classList.add('hidden');
        }
        resultContainer.classList.remove('hidden');
        analysisLayout.classList.add('has-results');
        predictionResult.classList.remove('hidden');
        hideError();
        
        const formattedLabel = data.label.replace(/_/g, ' ');
        predLabel.textContent = formattedLabel;
        
        const confidenceValue = parseFloat(data.confidence);
        predConfidence.textContent = `${confidenceValue.toFixed(2)}%`;
        
        // Trigger reflow for animation
        confidenceFill.style.width = '0%';
        // Need a tiny timeout for css transition to trigger
        setTimeout(() => {
            confidenceFill.style.width = `${confidenceValue}%`;
        }, 10);

        resultBadge.className = 'result-badge';
        
        const exactLabel = data.label.toLowerCase();
        
        if (exactLabel === 'with_mask') {
            resultBadge.classList.add('status-with_mask');
        } else if (exactLabel === 'without_mask') {
            resultBadge.classList.add('status-without_mask');
        } else if (exactLabel === 'mask_weared_incorrect') {
            resultBadge.classList.add('status-mask_weared_incorrect');
        } else {
            if (exactLabel.includes('incorrect')) {
                 resultBadge.classList.add('status-mask_weared_incorrect');
            } else if (exactLabel.includes('without')) {
                 resultBadge.classList.add('status-without_mask');
            } else if (exactLabel.includes('with')) {
                 resultBadge.classList.add('status-with_mask');
            } else {
                 resultBadge.classList.add('status-with_mask'); 
            }
        }
    }

    function showError(message) {
        resultContainer.classList.remove('hidden');
        analysisLayout.classList.add('has-results');
        loader.classList.add('hidden');
        predictionResult.classList.add('hidden');
        errorMessage.textContent = message;
        errorMessage.classList.remove('hidden');
    }

    function hideError() {
        errorMessage.classList.add('hidden');
        errorMessage.textContent = '';
    }

    function resetResults() {
        resultContainer.classList.add('hidden');
        analysisLayout.classList.remove('has-results');
        loader.classList.add('hidden');
        predictionResult.classList.add('hidden');
        hideError();
    }
});
