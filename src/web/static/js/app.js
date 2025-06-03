// DekDataset Generator - JavaScript Application
class DatasetGenerator {
    constructor() {
        this.selectedTaskId = null;
        this.tasks = [];
        this.currentAbortController = null; // Add abort controller for cancellation
        this.isGenerating = false; // Track generation state
        this.init();
    }

    init() {
        console.log('Initializing DatasetGenerator...');
        this.bindEvents();
        
        // Load tasks first, then other components
        this.loadTasks().then(() => {
            console.log('Tasks loaded, loading other components...');
            this.loadQualityConfig();
            this.loadAvailableModels();
        }).catch(error => {
            console.error('Error during initialization:', error);
            this.showAlert('Error initializing application: ' + error.message, 'danger');
        });
    }

    bindEvents() {
        // Task selection
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('select-task')) {
                this.selectTask(e.target.closest('.task-item').dataset.taskId);
            }
            
            if (e.target.classList.contains('delete-task')) {
                e.preventDefault();
                e.stopPropagation();
                this.deleteTask(e.target.closest('.task-item').dataset.taskId);
            }
        });

        // Model provider selection
        document.addEventListener('change', (e) => {
            if (e.target.name === 'modelProvider') {
                this.switchModelProvider(e.target.value);
            }
        });

        // Generation buttons
        const testBtn = document.getElementById('testGeneration');
        const generateBtn = document.getElementById('generateDataset');
        const stopBtn = document.getElementById('stopGeneration'); // Add stop button
        
        if (testBtn) {
            testBtn.addEventListener('click', () => this.testGeneration());
        }
        
        if (generateBtn) {
            generateBtn.addEventListener('click', () => this.generateDataset());
        }

        // Add stop button event listener
        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopGeneration());
        }

        // Modal buttons
        const saveTaskBtn = document.getElementById('saveTask');
        const saveQualityBtn = document.getElementById('saveQuality');
        
        if (saveTaskBtn) {
            saveTaskBtn.addEventListener('click', () => this.createTask());
        }
        
        if (saveQualityBtn) {
            saveQualityBtn.addEventListener('click', () => this.updateQualitySettings());
        }

        // Download buttons
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('download-btn')) {
                this.downloadDataset(e.target.dataset.format);
            }
        });

        // Slider update
        const similaritySlider = document.getElementById('similarityThreshold');
        if (similaritySlider) {
            similaritySlider.addEventListener('input', (e) => {
                const thresholdValue = document.getElementById('thresholdValue');
                if (thresholdValue) {
                    thresholdValue.textContent = e.target.value;
                }
            });
        }

        // Refresh button
        const refreshBtn = document.getElementById('refreshTasks');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.loadTasks());
        }
    }

    switchModelProvider(provider) {
        const deepseekSection = document.getElementById('deepseekModelSection');
        const ollamaSection = document.getElementById('ollamaModelSection');
        
        if (provider === 'deepseek') {
            deepseekSection.style.display = 'block';
            ollamaSection.style.display = 'none';
            deepseekSection.classList.add('active');
            ollamaSection.classList.remove('active');
        } else if (provider === 'ollama') {
            deepseekSection.style.display = 'none';
            ollamaSection.style.display = 'block';
            ollamaSection.classList.add('active');
            deepseekSection.classList.remove('active');
        }
    }

    getSelectedModel() {
        const providerRadio = document.querySelector('input[name="modelProvider"]:checked');
        if (!providerRadio) return null;

        const provider = providerRadio.value;
        
        if (provider === 'deepseek') {
            const deepseekSelect = document.getElementById('deepseekModelSelect');
            return deepseekSelect ? deepseekSelect.value : null;
        } else if (provider === 'ollama') {
            const ollamaSelect = document.getElementById('ollamaModelSelect');
            return ollamaSelect ? ollamaSelect.value : null;
        }
        
        return null;
    }

    async loadTasks() {
        try {
            this.showAlert('Loading tasks...', 'info');
            const response = await fetch('/api/tasks');
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log('API Response:', data);
            
            this.tasks = data.tasks || data || [];
            this.updateTaskList(this.tasks);
            this.showAlert(`Loaded ${this.tasks.length} tasks successfully`, 'success');
        } catch (error) {
            console.error('Error loading tasks:', error);
            this.showAlert('Error loading tasks: ' + error.message, 'danger');
            this.updateTaskList([]);
        }
    }

    updateTaskList(tasks) {
        const taskListContainer = document.getElementById('taskList');
        if (!taskListContainer) {
            console.error('Task list container not found');
            return;
        }

        if (!tasks || tasks.length === 0) {
            taskListContainer.innerHTML = `
                <div class="alert alert-info">
                    <h5>No tasks found</h5>
                    <p>Create your first task to get started!</p>
                    <button class="btn btn-primary" data-bs-toggle="modal" data-bs-target="#createTaskModal">
                        Create Task
                    </button>
                </div>
            `;
            return;
        }

        taskListContainer.innerHTML = tasks.map(task => `
            <div class="task-item card mb-2" data-task-id="${task.id}">
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-start">
                        <div class="flex-grow-1">
                            <h6 class="card-title mb-1">${this.escapeHtml(task.id)}</h6>
                            <p class="card-text text-muted small mb-2">
                                ${this.escapeHtml(task.description || 'No description')}
                            </p>
                            <div class="d-flex gap-2">
                                <span class="badge bg-secondary">${task.type || 'custom'}</span>
                                ${task.created_at ? `<small class="text-muted">Created: ${new Date(task.created_at).toLocaleDateString()}</small>` : ''}
                            </div>
                        </div>
                        <div class="btn-group-vertical btn-group-sm">
                            <button class="btn btn-outline-primary btn-sm select-task">
                                Select
                            </button>
                            <button class="btn btn-outline-danger btn-sm delete-task">
                                Delete
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    }

    selectTask(taskId) {
        this.selectedTaskId = taskId;
        
        // Update UI to show selected task
        document.querySelectorAll('.task-item').forEach(item => {
            item.classList.remove('selected');
        });
        
        const selectedItem = document.querySelector(`[data-task-id="${taskId}"]`);
        if (selectedItem) {
            selectedItem.classList.add('selected');
        }
        
        // Update selected task display
        const selectedTaskInput = document.getElementById('selectedTask');
        if (selectedTaskInput) {
            selectedTaskInput.value = taskId;
        }
        
        // Enable generation buttons only if not currently generating
        if (!this.isGenerating) {
            this.updateGenerationUI(false);
        }
        
        this.showAlert(`Selected task: ${taskId}`, 'success');
    }

    async testGeneration() {
        if (!this.selectedTaskId) {
            this.showAlert('Please select a task first', 'warning');
            return;
        }
        
        const model = this.getSelectedModel() || 'deepseek-chat';
        if (!model) {
            this.showAlert('Please select a model first', 'warning');
            return;
        }

        // Set generation state
        this.isGenerating = true;
        this.updateGenerationUI(true);
        this.currentAbortController = new AbortController();
        
        this.showProgress('Testing generation...', 30);
        
        try {
            const response = await fetch('/api/test-generation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: this.selectedTaskId, model }),
                signal: this.currentAbortController.signal
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            this.hideProgress();

            if (result.success) {
                const output = result.sample || result.output || 'Generation completed successfully';
                this.showAlert('Test generation completed successfully!', 'success');
                
                // Show result in a modal or alert
                this.displayGenerationResult('Test Result', output);
            } else {
                this.showAlert('Test generation failed: ' + (result.error || 'Unknown error'), 'danger');
            }
        } catch (error) {
            this.hideProgress();
            
            if (error.name === 'AbortError') {
                this.showAlert('Test generation was cancelled by user', 'warning');
            } else {
                console.error('Error during test generation:', error);
                this.showAlert('Error during test generation: ' + error.message, 'danger');
            }
        } finally {
            // Reset generation state
            this.isGenerating = false;
            this.updateGenerationUI(false);
            this.currentAbortController = null;
        }
    }

    async generateDataset() {
        if (!this.selectedTaskId) {
            this.showAlert('Please select a task first', 'warning');
            return;
        }
        
        const model = this.getSelectedModel() || 'deepseek-chat';
        if (!model) {
            this.showAlert('Please select a model first', 'warning');
            return;
        }
        
        const entryCount = parseInt(document.getElementById('entryCount')?.value || '10');

        // Set generation state
        this.isGenerating = true;
        this.updateGenerationUI(true);
        this.currentAbortController = new AbortController();
        
        this.showProgress('Generating dataset...', 10);
        
        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    task_id: this.selectedTaskId, 
                    model: model,
                    count: entryCount // Fix: use 'count' instead of 'entry_count'
                }),
                signal: this.currentAbortController.signal
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            this.hideProgress();

            if (result.entries && result.entries.length > 0) {
                this.showAlert(`Dataset generation completed! Generated ${result.entries.length} entries.`, 'success');
                this.displayDatasetResult(result);
            } else {
                this.showAlert('Dataset generation completed but no entries were generated.', 'warning');
            }
        } catch (error) {
            this.hideProgress();
            
            if (error.name === 'AbortError') {
                this.showAlert('Dataset generation was cancelled by user', 'warning');
            } else {
                console.error('Error during dataset generation:', error);
                this.showAlert('Error during dataset generation: ' + error.message, 'danger');
            }
        } finally {
            // Reset generation state
            this.isGenerating = false;
            this.updateGenerationUI(false);
            this.currentAbortController = null;
        }
    }

    stopGeneration() {
        if (this.currentAbortController && this.isGenerating) {
            this.currentAbortController.abort();
            this.showAlert('Stopping generation...', 'info');
        }
    }

    updateGenerationUI(isGenerating) {
        const testBtn = document.getElementById('testGeneration');
        const generateBtn = document.getElementById('generateDataset');
        const stopBtn = document.getElementById('stopGeneration');
        const entryCountInput = document.getElementById('entryCount');
        const modelProviderRadios = document.querySelectorAll('input[name="modelProvider"]');
        const modelSelects = document.querySelectorAll('#deepseekModelSelect, #ollamaModelSelect');

        if (isGenerating) {
            // Disable generation buttons and inputs
            if (testBtn) {
                testBtn.disabled = true;
                testBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Testing...';
            }
            if (generateBtn) {
                generateBtn.disabled = true;
                generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
            }
            if (stopBtn) {
                stopBtn.style.display = 'inline-flex';
                stopBtn.disabled = false;
            }
            if (entryCountInput) entryCountInput.disabled = true;
            
            modelProviderRadios.forEach(radio => radio.disabled = true);
            modelSelects.forEach(select => select.disabled = true);
        } else {
            // Re-enable generation buttons and inputs
            if (testBtn) {
                testBtn.disabled = !this.selectedTaskId;
                testBtn.innerHTML = '<i class="fas fa-flask"></i> Test Generation';
            }
            if (generateBtn) {
                generateBtn.disabled = !this.selectedTaskId;
                generateBtn.innerHTML = '<i class="fas fa-rocket"></i> Generate Dataset';
            }
            if (stopBtn) {
                stopBtn.style.display = 'none';
            }
            if (entryCountInput) entryCountInput.disabled = false;
            
            modelProviderRadios.forEach(radio => radio.disabled = false);
            modelSelects.forEach(select => select.disabled = false);
        }
    }

    showProgress(text, percent) {
        const progressSection = document.getElementById('progressSection');
        if (!progressSection) return;

        const progressBar = progressSection.querySelector('.progress-bar');
        const progressText = document.getElementById('progressText');

        if (progressText) progressText.innerHTML = `<i class="fas fa-cog fa-spin me-2"></i>${text}`;
        if (progressBar) {
            progressBar.style.width = `${percent}%`;
            progressBar.setAttribute('aria-valuenow', Math.min(percent, 100));
        }

        progressSection.style.display = 'block';
    }

    hideProgress() {
        const progressSection = document.getElementById('progressSection');
        if (progressSection) {
            progressSection.style.display = 'none';
        }
        
        // Reset UI state
        this.updateGenerationUI(false);
    }

    showAlert(message, type = 'info') {
        const alertContainer = document.getElementById('alertContainer');
        if (!alertContainer) return;

        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        alertContainer.appendChild(alertDiv);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Additional methods would go here (createTask, deleteTask, loadQualityConfig, etc.)
    async createTask() {
        const taskId = document.getElementById('taskId')?.value?.trim();
        const taskDescription = document.getElementById('taskDescription')?.value?.trim();
        const taskType = document.getElementById('taskType')?.value || 'custom';
        const systemPrompt = document.getElementById('systemPrompt')?.value?.trim();
        const userTemplate = document.getElementById('userTemplate')?.value?.trim();

        if (!taskId) {
            this.showAlert('Task ID is required', 'warning');
            return;
        }

        if (!taskDescription && !systemPrompt && !userTemplate) {
            this.showAlert('At least one of description, system prompt, or user template is required', 'warning');
            return;
        }

        try {
            const response = await fetch('/api/tasks', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    id: taskId,
                    description: taskDescription,
                    type: taskType,
                    system_prompt: systemPrompt,
                    user_template: userTemplate
                })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            
            // Close the modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('createTaskModal'));
            if (modal) modal.hide();

            // Clear the form
            document.getElementById('createTaskForm')?.reset();

            this.showAlert(`Task "${taskId}" created successfully!`, 'success');
            
            // Reload tasks to show the new one
            await this.loadTasks();

        } catch (error) {
            console.error('Error creating task:', error);
            this.showAlert('Error creating task: ' + error.message, 'danger');
        }
    }

    async deleteTask(taskId) {
        if (!taskId) {
            this.showAlert('Invalid task ID', 'warning');
            return;
        }

        // Show confirmation dialog
        const confirmed = confirm(`Are you sure you want to delete task "${taskId}"? This action cannot be undone.`);
        if (!confirmed) return;

        try {
            const response = await fetch(`/api/tasks/${encodeURIComponent(taskId)}`, {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
            }

            // If this was the selected task, clear selection
            if (this.selectedTaskId === taskId) {
                this.selectedTaskId = null;
                const testBtn = document.getElementById('testGeneration');
                const generateBtn = document.getElementById('generateDataset');
                if (testBtn) testBtn.disabled = true;
                if (generateBtn) generateBtn.disabled = true;
            }

            this.showAlert(`Task "${taskId}" deleted successfully!`, 'success');
            
            // Reload tasks to reflect the deletion
            await this.loadTasks();

        } catch (error) {
            console.error('Error deleting task:', error);
            this.showAlert('Error deleting task: ' + error.message, 'danger');
        }
    }

    async loadQualityConfig() {
        try {
            const response = await fetch('/api/quality-config');
            
            if (response.ok) {
                const config = await response.json();
                
                // Update UI elements with loaded config
                const similaritySlider = document.getElementById('similarityThreshold');
                const thresholdValue = document.getElementById('thresholdValue');
                const enableFilter = document.getElementById('enableQualityFilter');
                const minLength = document.getElementById('minResponseLength');
                const maxLength = document.getElementById('maxResponseLength');
                const enableDuplicateDetection = document.getElementById('enableDuplicateDetection');

                if (similaritySlider && config.similarity_threshold !== undefined) {
                    similaritySlider.value = config.similarity_threshold;
                    if (thresholdValue) thresholdValue.textContent = config.similarity_threshold;
                }
                
                if (enableFilter && config.enable_quality_filter !== undefined) {
                    enableFilter.checked = config.enable_quality_filter;
                }
                
                if (minLength && config.min_response_length !== undefined) {
                    minLength.value = config.min_response_length;
                }
                
                if (maxLength && config.max_response_length !== undefined) {
                    maxLength.value = config.max_response_length;
                }
                
                if (enableDuplicateDetection && config.enable_duplicate_detection !== undefined) {
                    enableDuplicateDetection.checked = config.enable_duplicate_detection;
                }

                console.log('Quality config loaded:', config);
            } else {
                console.warn('Could not load quality config, using defaults');
            }
        } catch (error) {
            console.error('Error loading quality config:', error);
            // Don't show error alert for config loading as it's not critical
        }
    }

    async updateQualitySettings() {
        const similarityThreshold = parseFloat(document.getElementById('similarityThreshold')?.value || '0.8');
        const enableFilter = document.getElementById('enableQualityFilter')?.checked || false;
        const minLength = parseInt(document.getElementById('minResponseLength')?.value || '10');
        const maxLength = parseInt(document.getElementById('maxResponseLength')?.value || '2000');
        const enableDuplicateDetection = document.getElementById('enableDuplicateDetection')?.checked || true;

        // Validate inputs
        if (minLength < 0 || maxLength < 0 || minLength > maxLength) {
            this.showAlert('Invalid length settings. Min length must be less than max length and both must be positive.', 'warning');
            return;
        }

        if (similarityThreshold < 0 || similarityThreshold > 1) {
            this.showAlert('Similarity threshold must be between 0 and 1.', 'warning');
            return;
        }

        try {
            const response = await fetch('/api/quality-config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    similarity_threshold: similarityThreshold,
                    enable_quality_filter: enableFilter,
                    min_response_length: minLength,
                    max_response_length: maxLength,
                    enable_duplicate_detection: enableDuplicateDetection
                })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
            }

            // Close the modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('qualityModal'));
            if (modal) modal.hide();

            this.showAlert('Quality settings updated successfully!', 'success');

        } catch (error) {
            console.error('Error updating quality settings:', error);
            this.showAlert('Error updating quality settings: ' + error.message, 'danger');
        }
    }

    async loadAvailableModels() {
        try {
            // Load DeepSeek models (always available)
            const deepseekSelect = document.getElementById('deepseekModelSelect');
            if (deepseekSelect) {
                deepseekSelect.innerHTML = `
                    <option value="deepseek-chat">DeepSeek-V3-0324</option>
                    <option value="deepseek-reasoner">DeepSeek-R1-0528</option>
                `;
            }

            // Load Ollama models
            try {
                const response = await fetch('/api/models/ollama');
                if (response.ok) {
                    const data = await response.json();
                    const ollamaSelect = document.getElementById('ollamaModelSelect');
                    
                    if (ollamaSelect) {
                        if (data.models && data.models.length > 0) {
                            ollamaSelect.innerHTML = data.models.map(model => 
                                `<option value="ollama:${model}">${model}</option>`
                            ).join('');
                            
                            // Enable Ollama provider if models are available
                            const ollamaRadio = document.querySelector('input[name="modelProvider"][value="ollama"]');
                            if (ollamaRadio) {
                                ollamaRadio.disabled = false;
                                const ollamaLabel = ollamaRadio.closest('label');
                                if (ollamaLabel) {
                                    ollamaLabel.classList.remove('disabled');
                                }
                            }
                        } else {
                            ollamaSelect.innerHTML = '<option value="">No Ollama models available</option>';
                            
                            // Disable Ollama provider if no models
                            const ollamaRadio = document.querySelector('input[name="modelProvider"][value="ollama"]');
                            if (ollamaRadio) {
                                ollamaRadio.disabled = true;
                                const ollamaLabel = ollamaRadio.closest('label');
                                if (ollamaLabel) {
                                    ollamaLabel.classList.add('disabled');
                                }
                            }
                        }
                    }
                }
            } catch (error) {
                console.warn('Could not load Ollama models:', error);
                const ollamaSelect = document.getElementById('ollamaModelSelect');
                if (ollamaSelect) {
                    ollamaSelect.innerHTML = '<option value="">Ollama server not available</option>';
                }
            }

            console.log('Available models loaded');
        } catch (error) {
            console.error('Error loading available models:', error);
            // Don't show error alert as this is not critical for basic functionality
        }
    }

    displayGenerationResult(title, output) {
        // Create a modal to display generation results
        const modalHtml = `
            <div class="modal fade" id="generationResultModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">${this.escapeHtml(title)}</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <pre class="bg-light p-3 rounded" style="max-height: 400px; overflow-y: auto;">${this.escapeHtml(JSON.stringify(output, null, 2))}</pre>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Remove existing modal if any
        const existingModal = document.getElementById('generationResultModal');
        if (existingModal) {
            existingModal.remove();
        }

        // Add modal to document
        document.body.insertAdjacentHTML('beforeend', modalHtml);

        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('generationResultModal'));
        modal.show();

        // Clean up modal when hidden
        document.getElementById('generationResultModal').addEventListener('hidden.bs.modal', function() {
            this.remove();
        });
    }

    displayDatasetResult(result) {
        // Show dataset generation results and enable download
        const downloadSection = document.getElementById('downloadSection');
        if (!downloadSection) return;

        downloadSection.style.display = 'block';
        
        // Update result summary
        const resultSummary = document.getElementById('resultSummary');
        if (resultSummary) {
            resultSummary.innerHTML = `
                <div class="alert alert-success">
                    <h5><i class="fas fa-check-circle"></i> Dataset Generated Successfully!</h5>
                    <p><strong>Task:</strong> ${this.escapeHtml(result.task_id)}</p>
                    <p><strong>Entries:</strong> ${result.count}</p>
                    <p><strong>Generated:</strong> ${new Date(result.generated_at).toLocaleString()}</p>
                    ${result.quality_report ? `<p><strong>Quality Score:</strong> ${(result.quality_report.quality_score * 100).toFixed(1)}%</p>` : ''}
                </div>
            `;
        }

        // Enable download buttons
        const downloadBtns = document.querySelectorAll('.download-btn');
        downloadBtns.forEach(btn => {
            btn.disabled = false;
            btn.dataset.taskId = result.task_id;
        });

        // Show preview table
        this.displayPreviewTable(result.entries);

        // Show a sample of the generated data
        if (result.entries && result.entries.length > 0) {
            const sampleEntry = result.entries[0];
            const sampleDisplay = document.getElementById('sampleDisplay');
            if (sampleDisplay) {
                sampleDisplay.innerHTML = `
                    <h6>Sample Entry (JSON):</h6>
                    <pre class="bg-light p-2 rounded" style="max-height: 200px; overflow-y: auto;">${this.escapeHtml(JSON.stringify(sampleEntry, null, 2))}</pre>
                `;
            }
        }
    }

    displayPreviewTable(entries) {
        const previewContainer = document.getElementById('previewTableContainer');
        if (!previewContainer || !entries || entries.length === 0) return;

        // Extract all unique fields from content across all entries
        const allFields = new Set(['id']); // Always include ID
        entries.forEach(entry => {
            if (entry.content && typeof entry.content === 'object') {
                Object.keys(entry.content).forEach(field => allFields.add(field));
            }
        });

        const fields = Array.from(allFields);
        
        // Create table HTML
        const tableHtml = `
            <div class="card mt-3">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <h5><i class="fas fa-table"></i> Dataset Preview</h5>
                    <div class="btn-group btn-group-sm">
                        <button class="btn btn-outline-primary" onclick="datasetGenerator.toggleTableView('compact')">
                            <i class="fas fa-compress"></i> Compact
                        </button>
                        <button class="btn btn-outline-primary" onclick="datasetGenerator.toggleTableView('full')">
                            <i class="fas fa-expand"></i> Full
                        </button>
                        <button class="btn btn-outline-secondary" onclick="datasetGenerator.exportTableToCSV()">
                            <i class="fas fa-download"></i> Export CSV
                        </button>
                    </div>
                </div>
                <div class="card-body p-0">
                    <div class="table-responsive" style="max-height: 400px; overflow-y: auto;">
                        <table class="table table-striped table-hover mb-0" id="previewTable">
                            <thead class="table-dark sticky-top">
                                <tr>
                                    ${fields.map(field => `<th class="text-nowrap">${this.escapeHtml(field)}</th>`).join('')}
                                    <th class="text-nowrap">Source</th>
                                    <th class="text-nowrap">Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${entries.slice(0, 50).map((entry, index) => `
                                    <tr data-entry-index="${index}">
                                        <td class="text-nowrap">${this.escapeHtml(entry.id || `entry-${index + 1}`)}</td>
                                        ${fields.slice(1).map(field => {
                                            const value = entry.content && entry.content[field] ? entry.content[field] : '';
                                            const displayValue = typeof value === 'string' ? value : JSON.stringify(value);
                                            const truncated = displayValue.length > 100 ? displayValue.substring(0, 100) + '...' : displayValue;
                                            return `<td class="entry-content" data-field="${field}" data-full-value="${this.escapeHtml(displayValue)}" title="${this.escapeHtml(displayValue)}">${this.escapeHtml(truncated)}</td>`;
                                        }).join('')}
                                        <td class="text-nowrap">
                                            <small class="text-muted">${this.escapeHtml(entry.metadata?.source || 'Unknown')}</small>
                                        </td>
                                        <td class="text-nowrap">
                                            <button class="btn btn-sm btn-outline-info me-1" onclick="datasetGenerator.viewEntryDetails(${index})" title="View Details">
                                                <i class="fas fa-eye"></i>
                                            </button>
                                            <button class="btn btn-sm btn-outline-secondary" onclick="datasetGenerator.copyEntryToClipboard(${index})" title="Copy JSON">
                                                <i class="fas fa-copy"></i>
                                            </button>
                                        </td>
                                    </tr>
                                `).join('')}
                                ${entries.length > 50 ? `
                                    <tr>
                                        <td colspan="${fields.length + 2}" class="text-center text-muted py-3">
                                            <i class="fas fa-info-circle"></i> Showing first 50 entries out of ${entries.length} total entries.
                                            <button class="btn btn-link btn-sm" onclick="datasetGenerator.showAllEntries()">Show All</button>
                                        </td>
                                    </tr>
                                ` : ''}
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="card-footer text-muted">
                    <small>
                        <i class="fas fa-info-circle"></i> Total: ${entries.length} entries | 
                        Fields: ${fields.length} | 
                        Click on entries to view full content
                    </small>
                </div>
            </div>
        `;

        previewContainer.innerHTML = tableHtml;
        
        // Store entries for later use
        this.currentEntries = entries;
        this.currentFields = fields;

        // Add click handlers for expandable content
        this.addTableInteractivity();
    }

    addTableInteractivity() {
        // Add click to expand content
        document.querySelectorAll('.entry-content').forEach(cell => {
            cell.addEventListener('click', function() {
                const fullValue = this.getAttribute('data-full-value');
                const field = this.getAttribute('data-field');
                
                if (fullValue && fullValue.length > 100) {
                    const modal = document.createElement('div');
                    modal.className = 'modal fade';
                    modal.innerHTML = `
                        <div class="modal-dialog modal-lg">
                            <div class="modal-content">
                                <div class="modal-header">
                                    <h5 class="modal-title">Field: ${field}</h5>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                                </div>
                                <div class="modal-body">
                                    <pre class="bg-light p-3 rounded" style="white-space: pre-wrap; word-break: break-word;">${fullValue}</pre>
                                </div>
                                <div class="modal-footer">
                                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                                    <button type="button" class="btn btn-primary" onclick="navigator.clipboard.writeText('${fullValue.replace(/'/g, "\\'")}')">Copy</button>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    document.body.appendChild(modal);
                    const bsModal = new bootstrap.Modal(modal);
                    bsModal.show();
                    
                    modal.addEventListener('hidden.bs.modal', () => {
                        modal.remove();
                    });
                }
            });
        });
    }

    toggleTableView(viewType) {
        const table = document.getElementById('previewTable');
        if (!table) return;

        if (viewType === 'compact') {
            table.classList.add('table-sm');
            document.querySelectorAll('.entry-content').forEach(cell => {
                const fullValue = cell.getAttribute('data-full-value');
                if (fullValue && fullValue.length > 50) {
                    cell.textContent = fullValue.substring(0, 50) + '...';
                }
            });
        } else {
            table.classList.remove('table-sm');
            document.querySelectorAll('.entry-content').forEach(cell => {
                const fullValue = cell.getAttribute('data-full-value');
                if (fullValue && fullValue.length > 100) {
                    cell.textContent = fullValue.substring(0, 100) + '...';
                } else if (fullValue) {
                    cell.textContent = fullValue;
                }
            });
        }
    }

    viewEntryDetails(index) {
        if (!this.currentEntries || !this.currentEntries[index]) return;

        const entry = this.currentEntries[index];
        const modalHtml = `
            <div class="modal fade" id="entryDetailsModal" tabindex="-1">
                <div class="modal-dialog modal-xl">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Entry Details: ${this.escapeHtml(entry.id || `Entry ${index + 1}`)}</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h6>Content</h6>
                                    <pre class="bg-light p-3 rounded" style="max-height: 300px; overflow-y: auto;">${this.escapeHtml(JSON.stringify(entry.content, null, 2))}</pre>
                                </div>
                                <div class="col-md-6">
                                    <h6>Metadata</h6>
                                    <pre class="bg-light p-3 rounded" style="max-height: 300px; overflow-y: auto;">${this.escapeHtml(JSON.stringify(entry.metadata, null, 2))}</pre>
                                </div>
                            </div>
                            <div class="mt-3">
                                <h6>Full Entry (JSON)</h6>
                                <pre class="bg-dark text-light p-3 rounded" style="max-height: 200px; overflow-y: auto;">${this.escapeHtml(JSON.stringify(entry, null, 2))}</pre>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                            <button type="button" class="btn btn-primary" onclick="datasetGenerator.copyEntryToClipboard(${index})">
                                <i class="fas fa-copy"></i> Copy JSON
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Remove existing modal
        const existingModal = document.getElementById('entryDetailsModal');
        if (existingModal) existingModal.remove();

        // Add and show modal
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        const modal = new bootstrap.Modal(document.getElementById('entryDetailsModal'));
        modal.show();
    }

    copyEntryToClipboard(index) {
        if (!this.currentEntries || !this.currentEntries[index]) return;

        const entry = this.currentEntries[index];
        const jsonString = JSON.stringify(entry, null, 2);
        
        navigator.clipboard.writeText(jsonString).then(() => {
            this.showAlert('Entry JSON copied to clipboard!', 'success');
        }).catch(err => {
            console.error('Failed to copy to clipboard:', err);
            this.showAlert('Failed to copy to clipboard', 'warning');
        });
    }

    exportTableToCSV() {
        if (!this.currentEntries || !this.currentFields) return;

        // Create CSV content
        const headers = [...this.currentFields, 'source', 'generated_at'];
        const csvRows = [headers.join(',')];

        this.currentEntries.forEach(entry => {
            const row = [];
            
            // Add ID
            row.push(this.escapeCSV(entry.id || ''));
            
            // Add content fields
            this.currentFields.slice(1).forEach(field => {
                const value = entry.content && entry.content[field] ? entry.content[field] : '';
                const cellValue = typeof value === 'string' ? value : JSON.stringify(value);
                row.push(this.escapeCSV(cellValue));
            });
            
            // Add metadata
            row.push(this.escapeCSV(entry.metadata?.source || ''));
            row.push(this.escapeCSV(entry.metadata?.generated_at || ''));
            
            csvRows.push(row.join(','));
        });

        // Download CSV
        const csvContent = csvRows.join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        
        link.setAttribute('href', url);
        link.setAttribute('download', `dataset_preview_${new Date().getTime()}.csv`);
        link.style.visibility = 'hidden';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        this.showAlert('CSV exported successfully!', 'success');
    }

    escapeCSV(value) {
        if (value === null || value === undefined) return '';
        const stringValue = String(value);
        if (stringValue.includes(',') || stringValue.includes('"') || stringValue.includes('\n')) {
            return `"${stringValue.replace(/"/g, '""')}"`;
        }
        return stringValue;
    }

    showAllEntries() {
        if (!this.currentEntries) return;
        
        // Re-render table with all entries
        this.displayPreviewTable(this.currentEntries);
    }

    // ...existing code...
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.datasetGenerator = new DatasetGenerator();
});
