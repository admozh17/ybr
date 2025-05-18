/**
 * Progress Tracker for Video Analysis
 * Handles polling and UI updates for the pipeline visualization
 */
class ProgressTracker {
    /**
     * Create a new progress tracker
     * @param {string} resultId - The ID of the result to track
     */
    constructor(resultId) {
        this.resultId = resultId;
        this.pollInterval = 2000; // 2 seconds
        this.pollTimer = null;
        this.progressBar = null;
        this.statusMessage = null;
        this.pipelineStages = null;
        this.resultsSection = null;
        this.finishedCallback = null;
        this.errorCallback = null;
        
        // Stage descriptions
        this.stageDescriptions = [
            "Downloading and extracting metadata",
            "Analyzing audio and speech",
            "Processing video frames and visual content",
            "Combining data and finalizing results"
        ];
    }
    
    /**
     * Initialize the progress tracker with UI elements
     * @param {HTMLElement} progressBar - The progress bar element
     * @param {HTMLElement} statusMessage - The status message element
     * @param {HTMLElement} pipelineStages - The pipeline stages container
     * @param {HTMLElement} resultsSection - The results section element
     */
    init(progressBar, statusMessage, pipelineStages, resultsSection) {
        this.progressBar = progressBar;
        this.statusMessage = statusMessage;
        this.pipelineStages = pipelineStages;
        this.resultsSection = resultsSection;
        
        // Initialize pipeline stages
        this.initPipelineStages();
        
        // Start polling
        this.startPolling();
        
        return this;
    }
    
    /**
     * Set a callback function to be called when processing is finished
     * @param {Function} callback - The callback function
     */
    setFinishedCallback(callback) {
        this.finishedCallback = callback;
        return this;
    }
    
    /**
     * Set a callback function to be called when an error occurs
     * @param {Function} callback - The callback function
     */
    setErrorCallback(callback) {
        this.errorCallback = callback;
        return this;
    }
    
    /**
     * Initialize the pipeline stages visualization
     */
    initPipelineStages() {
        // Clear existing content
        this.pipelineStages.innerHTML = '';
        
        // Create stages
        for (let i = 0; i < this.stageDescriptions.length; i++) {
            const stageElement = document.createElement('div');
            stageElement.className = 'pipeline-stage';
            stageElement.id = `stage-${i}`;
            
            stageElement.innerHTML = `
                <div class="stage-indicator">
                    <div class="stage-number">${i + 1}</div>
                    <div class="stage-line"></div>
                </div>
                <div class="stage-details">
                    <h5>${this.stageDescriptions[i]}</h5>
                    <div class="progress mb-2" style="height: 5px;">
                        <div class="progress-bar" role="progressbar" style="width: 0%"></div>
                    </div>
                    <div class="stage-results d-none"></div>
                </div>
            `;
            
            this.pipelineStages.appendChild(stageElement);
        }
        
        // Activate first stage
        this.updateStageStatus(0, 'active');
    }
    
    /**
     * Start polling for status updates
     */
    startPolling() {
        this.poll();
    }
    
    /**
     * Stop polling
     */
    stopPolling() {
        if (this.pollTimer) {
            clearTimeout(this.pollTimer);
            this.pollTimer = null;
        }
    }
    
    /**
     * Poll the server for status updates
     */
    poll() {
        fetch(`/analysis_status/${this.resultId}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Server returned ${response.status}: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                this.handleStatusUpdate(data);
                
                // Schedule next poll if still processing
                if (data.status === 'processing') {
                    this.pollTimer = setTimeout(() => this.poll(), this.pollInterval);
                }
            })
            .catch(error => {
                console.error('Error polling for status:', error);
                
                // Try again after a longer delay
                this.pollTimer = setTimeout(() => this.poll(), this.pollInterval * 2);
            });
    }
    
    /**
     * Handle status update from the server
     * @param {Object} data - The status data
     */
    handleStatusUpdate(data) {
        // Update progress bar
        if (this.progressBar && data.progress !== undefined) {
            this.progressBar.style.width = `${data.progress}%`;
            this.progressBar.setAttribute('aria-valuenow', data.progress);
        }
        
        // Update status message
        if (this.statusMessage) {
            const stageDesc = this.stageDescriptions[data.current_stage] || 'Processing';
            this.statusMessage.textContent = `${stageDesc} (${Math.round(data.progress)}%)`;
        }
        
        // Update pipeline stages
        this.updatePipelineStages(data);
        
        // Handle completion or error
        if (data.status === 'completed') {
            this.handleCompletion(data);
        } else if (data.status === 'error') {
            this.handleError(data.error || 'An unknown error occurred');
        }
    }
    
    /**
     * Update the pipeline stages based on current status
     * @param {Object} data - The status data
     */
    updatePipelineStages(data) {
        const currentStage = data.current_stage || 0;
        
        // Update stage statuses
        for (let i = 0; i < this.stageDescriptions.length; i++) {
            if (i < currentStage) {
                // Previous stages are completed
                this.updateStageStatus(i, 'completed', 100);
                
                // Show stage results if available
                this.showStageResults(i, data);
            } else if (i === currentStage) {
                // Current stage is active
                this.updateStageStatus(i, 'active', data.progress);
                
                // Show partial results if available
                this.showStageResults(i, data);
            } else {
                // Future stages are pending
                this.updateStageStatus(i, '', 0);
                
                // Hide any results
                const resultsElement = document.querySelector(`#stage-${i} .stage-results`);
                if (resultsElement) {
                    resultsElement.classList.add('d-none');
                    resultsElement.innerHTML = '';
                }
            }
        }
    }
    
    /**
     * Update the status of a pipeline stage
     * @param {number} stageIndex - The index of the stage
     * @param {string} status - The status ('active', 'completed', or '')
     * @param {number} progress - The progress percentage
     */
    updateStageStatus(stageIndex, status, progress = 0) {
        const stageElement = document.getElementById(`stage-${stageIndex}`);
        if (!stageElement) return;
        
        // Update stage class
        stageElement.className = 'pipeline-stage';
        if (status) {
            stageElement.classList.add(status);
        }
        
        // Update progress bar if present
        const progressBar = stageElement.querySelector('.progress-bar');
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
        }
    }
    
    /**
     * Show stage results if available
     * @param {number} stageIndex - The index of the stage
     * @param {Object} data - The status data
     */
    showStageResults(stageIndex, data) {
        const resultsElement = document.querySelector(`#stage-${stageIndex} .stage-results`);
        if (!resultsElement) return;
        
        // Check if we have stage-specific data
        if (!data.stage_data) return;
        
        let hasResults = false;
        resultsElement.innerHTML = '';
        
        // Stage 0: Caption
        if (stageIndex === 0 && data.stage_data.caption_text) {
            hasResults = true;
            const captionElement = document.createElement('div');
            captionElement.className = 'partial-result';
            captionElement.innerHTML = `
                <h5>Caption:</h5>
                <p class="text-muted small">${data.stage_data.caption_text}</p>
            `;
            resultsElement.appendChild(captionElement);
        }
        
        // Stage 1: Speech
        if (stageIndex === 1 && data.stage_data.speech_text) {
            hasResults = true;
            const speechElement = document.createElement('div');
            speechElement.className = 'partial-result';
            speechElement.innerHTML = `
                <h5>Speech Transcription:</h5>
                <p class="text-muted small">${data.stage_data.speech_text}</p>
            `;
            resultsElement.appendChild(speechElement);
            
            // Add frame count if available
            if (data.stage_data.num_frames) {
                const framesElement = document.createElement('div');
                framesElement.className = 'partial-result';
                framesElement.innerHTML = `
                    <h5>Extracted Frames:</h5>
                    <p class="text-muted small">${data.stage_data.num_frames} frames</p>
                `;
                resultsElement.appendChild(framesElement);
            }
        }
        
        // Stage 2: OCR & Visual
        if (stageIndex === 2) {
            // OCR text
            if (data.stage_data.frame_text) {
                hasResults = true;
                const ocrElement = document.createElement('div');
                ocrElement.className = 'partial-result';
                ocrElement.innerHTML = `
                    <h5>Text from Video Frames:</h5>
                    <p class="text-muted small">${data.stage_data.frame_text}</p>
                `;
                resultsElement.appendChild(ocrElement);
            }
            
            // Visual results
            if (data.stage_data.visual_results && data.stage_data.visual_results.length > 0) {
                hasResults = true;
                const visualElement = document.createElement('div');
                visualElement.className = 'partial-result';
                
                const result = data.stage_data.visual_results[0].analysis || {};
                let visualHTML = '<h5>Visual Analysis:</h5>';
                
                if (result.place_type) {
                    visualHTML += `<p class="text-muted small"><strong>Place Type:</strong> ${result.place_type}</p>`;
                }
                
                if (result.scene) {
                    visualHTML += `<p class="text-muted small"><strong>Scene:</strong> ${result.scene}</p>`;
                }
                
                if (result.objects && result.objects.length > 0) {
                    visualHTML += `<p class="text-muted small"><strong>Objects:</strong> ${result.objects.join(', ')}</p>`;
                }
                
                if (result.food_items && result.food_items.length > 0) {
                    const foodItems = result.food_items.map(item => item.name).join(', ');
                    visualHTML += `<p class="text-muted small"><strong>Food Items:</strong> ${foodItems}</p>`;
                }
                
                visualElement.innerHTML = visualHTML;
                resultsElement.appendChild(visualElement);
            }
        }
        
        // Show or hide results container
        if (hasResults) {
            resultsElement.classList.remove('d-none');
        } else {
            resultsElement.classList.add('d-none');
        }
    }
    
    /**
     * Handle completion of processing
     * @param {Object} data - The status data
     */
    handleCompletion(data) {
        // Stop polling
        this.stopPolling();
        
        // Update UI
        if (this.progressBar) {
            this.progressBar.style.width = '100%';
            this.progressBar.setAttribute('aria-valuenow', 100);
            this.progressBar.classList.remove('progress-bar-animated', 'progress-bar-striped');
            this.progressBar.classList.add('bg-success');
        }
        
        if (this.statusMessage) {
            this.statusMessage.textContent = 'Analysis completed successfully!';
        }
        
        // Mark all stages as completed
        for (let i = 0; i < this.stageDescriptions.length; i++) {
            this.updateStageStatus(i, 'completed', 100);
        }
        
        // Call finished callback if defined
        if (this.finishedCallback) {
            this.finishedCallback(data);
        }
    }
    
    /**
     * Handle error during processing
     * @param {string} errorMessage - The error message
     */
    handleError(errorMessage) {
        // Stop polling
        this.stopPolling();
        
        // Call error callback if defined
        if (this.errorCallback) {
            this.errorCallback(errorMessage);
        }
    }
}