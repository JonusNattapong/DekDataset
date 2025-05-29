class DatasetGenerator {
    constructor() {
        this.selectedTaskId = null;
        this.tasks = [];
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadTasks();
        this.loadQualityConfig();
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

        // Generation buttons
        const testBtn = document.getElementById('testGeneration');
        const generateBtn = document.getElementById('generateDataset');
        
        if (testBtn) {
            testBtn.addEventListener('click', () => this.testGeneration());
        }
        
        if (generateBtn) {
            generateBtn.addEventListener('click', () => this.generateDataset());
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

    async loadTasks() {
        try {
            this.showAlert('Loading tasks...', 'info');
            const response = await fetch('/api/tasks');
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log('API Response:', data); // Debug log
            
            // Handle both old and new API response formats
            this.tasks = data.tasks || data || [];
            this.updateTaskList(this.tasks);
            
            this.showAlert(`Loaded ${this.tasks.length} tasks successfully`, 'success');
        } catch (error) {
            console.error('Error loading tasks:', error);
            this.showAlert('Error loading tasks: ' + error.message, 'danger');
            this.updateTaskList([]); // Show empty state
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
        
        // Update selected task display
        const selectedTaskElement = document.getElementById('selectedTask');
        if (selectedTaskElement) {
            selectedTaskElement.value = taskId;
        }
        
        // Update UI - remove previous selection
        document.querySelectorAll('.task-item').forEach(item => {
            item.classList.remove('border-primary', 'bg-light');
        });
        
        // Add selection styling
        const selectedElement = document.querySelector(`[data-task-id="${taskId}"]`);
        if (selectedElement) {
            selectedElement.classList.add('border-primary', 'bg-light');
        }

        // Enable generation buttons
        const testBtn = document.getElementById('testGeneration');
        const generateBtn = document.getElementById('generateDataset');
        
        if (testBtn) testBtn.disabled = false;
        if (generateBtn) generateBtn.disabled = false;

        this.showAlert(`Selected task: ${taskId}`, 'info');
    }

    async createTask() {
        const taskData = {
            task_id: document.getElementById('taskId')?.value?.trim(),
            task_type: document.getElementById('taskType')?.value || 'custom',
            description: document.getElementById('taskDescription')?.value?.trim(),
            prompt_template: document.getElementById('promptTemplate')?.value?.trim()
        };

        // Validation
        if (!taskData.task_id) {
            this.showAlert('Task ID is required', 'warning');
            return;
        }

        if (!taskData.description) {
            this.showAlert('Task description is required', 'warning');
            return;
        }

        if (!taskData.prompt_template) {
            this.showAlert('Prompt template is required', 'warning');
            return;
        }

        try {
            const response = await fetch('/api/tasks', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(taskData)
            });

            const result = await response.json();

            if (response.ok) {
                this.showAlert('Task created successfully!', 'success');
                
                // Close modal
                const modal = bootstrap.Modal.getInstance(document.getElementById('createTaskModal'));
                if (modal) modal.hide();
                
                // Clear form
                document.getElementById('createTaskForm')?.reset();
                
                // Reload tasks
                await this.loadTasks();
            } else {
                this.showAlert('Error: ' + (result.error || 'Unknown error'), 'danger');
            }
        } catch (error) {
            console.error('Error creating task:', error);
            this.showAlert('Error creating task: ' + error.message, 'danger');
        }
    }

    async deleteTask(taskId) {
        if (!confirm(`Are you sure you want to delete task "${taskId}"?`)) {
            return;
        }

        try {
            const response = await fetch(`/api/tasks/${taskId}`, {
                method: 'DELETE'
            });

            const result = await response.json();

            if (response.ok) {
                this.showAlert('Task deleted successfully!', 'success');
                
                // Clear selection if deleted task was selected
                if (this.selectedTaskId === taskId) {
                    this.selectedTaskId = null;
                    const selectedTaskElement = document.getElementById('selectedTask');
                    if (selectedTaskElement) selectedTaskElement.value = '';
                }
                
                await this.loadTasks();
            } else {
                this.showAlert('Error: ' + (result.error || 'Unknown error'), 'danger');
            }
        } catch (error) {
            console.error('Error deleting task:', error);
            this.showAlert('Error deleting task: ' + error.message, 'danger');
        }
    }

    async testGeneration() {
        if (!this.selectedTaskId) {
            this.showAlert('Please select a task first', 'warning');
            return;
        }

        this.showProgress('Testing generation...', 30);

        try {
            const response = await fetch('/api/test-generation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: this.selectedTaskId })
            });

            const result = await response.json();
            
            if (response.ok) {
                this.showProgress('Test completed!', 100);
                this.displayResults(result);
                this.showAlert('Test generation completed successfully!', 'success');
            } else {
                this.showAlert('Test failed: ' + (result.error || 'Unknown error'), 'danger');
            }
        } catch (error) {
            console.error('Test generation error:', error);
            this.showAlert('Error: ' + error.message, 'danger');
        } finally {
            setTimeout(() => this.hideProgress(), 2000);
        }
    }

    async generateDataset() {
        if (!this.selectedTaskId) {
            this.showAlert('Please select a task first', 'warning');
            return;
        }

        const countElement = document.getElementById('entryCount');
        const count = parseInt(countElement?.value || '10');

        if (count <= 0 || count > 1000) {
            this.showAlert('Entry count must be between 1 and 1000', 'warning');
            return;
        }

        this.showProgress('Generating dataset...', 0);

        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    task_id: this.selectedTaskId,
                    count: count
                })
            });

            // Simulate progress
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += Math.random() * 15;
                if (progress > 90) progress = 90;
                this.showProgress(`Generating dataset... ${Math.round(progress)}%`, progress);
            }, 1000);

            const result = await response.json();
            clearInterval(progressInterval);

            if (response.ok) {
                this.showProgress('Generation completed!', 100);
                this.displayResults(result);
                this.showAlert(`Generated ${result.count || count} entries successfully!`, 'success');
            } else {
                this.showAlert('Generation failed: ' + (result.error || 'Unknown error'), 'danger');
            }
        } catch (error) {
            console.error('Generation error:', error);
            this.showAlert('Error: ' + error.message, 'danger');
        } finally {
            setTimeout(() => this.hideProgress(), 2000);
        }
    }

    displayResults(result) {
        const resultsSection = document.getElementById('resultsSection');
        const qualityReport = document.getElementById('qualityReport');
        const datasetPreview = document.getElementById('datasetPreview');

        if (!resultsSection) return;

        const entries = result.entries || result.test_entries || [];
        
        // Show quality report
        if (qualityReport) {
            const report = result.quality_report || {};
            qualityReport.innerHTML = `
                <div class="card">
                    <div class="card-header">
                        <h6 class="mb-0">Quality Report</h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <strong>Total entries:</strong> ${entries.length}<br>
                                <strong>Generated at:</strong> ${result.generated_at ? new Date(result.generated_at).toLocaleString() : 'N/A'}
                            </div>
                            <div class="col-md-6">
                                ${Object.entries(report).map(([key, value]) => 
                                    `<strong>${key}:</strong> ${value}<br>`).join('')}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Show preview
        if (datasetPreview) {
            if (entries.length === 0) {
                datasetPreview.innerHTML = '<div class="alert alert-warning">No entries generated</div>';
            } else {
                datasetPreview.innerHTML = `
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0">Dataset Preview</h6>
                        </div>
                        <div class="card-body">
                            <div class="table-responsive">
                                <table class="table table-sm table-striped">
                                    <thead>
                                        <tr>
                                            <th>#</th>
                                            <th>Content</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${entries.slice(0, 5).map((entry, idx) => `
                                            <tr>
                                                <td>${idx + 1}</td>
                                                <td>
                                                    <pre class="mb-0" style="max-height: 100px; overflow-y: auto;">${this.escapeHtml(JSON.stringify(entry, null, 2))}</pre>
                                                </td>
                                            </tr>
                                        `).join('')}
                                    </tbody>
                                </table>
                                ${entries.length > 5 ? `<small class="text-muted">Showing 5 of ${entries.length} entries</small>` : ''}
                            </div>
                        </div>
                    </div>
                `;
            }
        }

        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    async loadQualityConfig() {
        try {
            const response = await fetch('/api/quality-config');
            if (response.ok) {
                const data = await response.json();
                this.updateQualityConfigUI(data.config);
            }
        } catch (error) {
            console.warn('Could not load quality config:', error);
        }
    }

    updateQualityConfigUI(config) {
        if (!config) return;

        const elements = {
            'minLength': config.min_length,
            'maxLength': config.max_length,
            'similarityThreshold': config.similarity_threshold
        };

        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element && value !== undefined) {
                element.value = value;
            }
        });

        // Update threshold display
        const thresholdValue = document.getElementById('thresholdValue');
        if (thresholdValue && config.similarity_threshold) {
            thresholdValue.textContent = config.similarity_threshold;
        }
    }

    async updateQualitySettings() {
        const config = {
            min_length: parseInt(document.getElementById('minLength')?.value || '10'),
            max_length: parseInt(document.getElementById('maxLength')?.value || '1000'),
            similarity_threshold: parseFloat(document.getElementById('similarityThreshold')?.value || '0.8')
        };

        try {
            const response = await fetch('/api/quality-config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });

            const result = await response.json();

            if (response.ok) {
                this.showAlert('Quality settings updated successfully!', 'success');
                
                // Close modal if exists
                const modal = bootstrap.Modal.getInstance(document.getElementById('qualityModal'));
                if (modal) modal.hide();
            } else {
                this.showAlert('Error: ' + (result.error || 'Unknown error'), 'danger');
            }
        } catch (error) {
            console.error('Error updating quality settings:', error);
            this.showAlert('Error updating quality settings: ' + error.message, 'danger');
        }
    }

    showProgress(text, percent) {
        const progressSection = document.getElementById('progressSection');
        if (!progressSection) return;

        const progressBar = progressSection.querySelector('.progress-bar');
        const progressText = document.getElementById('progressText');

        progressSection.style.display = 'block';
        if (progressBar) {
            progressBar.style.width = Math.min(percent, 100) + '%';
            progressBar.setAttribute('aria-valuenow', Math.min(percent, 100));
        }
        if (progressText) {
            progressText.textContent = text;
        }
    }

    hideProgress() {
        const progressSection = document.getElementById('progressSection');
        if (progressSection) {
            progressSection.style.display = 'none';
        }
    }

    showAlert(message, type = 'info') {
        // Remove existing alerts of the same type
        document.querySelectorAll(`.alert-${type}`).forEach(alert => {
            if (alert.textContent.includes(message.substring(0, 20))) {
                alert.remove();
            }
        });

        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
        alertDiv.innerHTML = `
            ${this.escapeHtml(message)}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;

        document.body.appendChild(alertDiv);

        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    async downloadDataset(format) {
        if (!this.selectedTaskId) {
            this.showAlert('No task selected for download', 'warning');
            return;
        }

        try {
            // Check if dataset exists
            const response = await fetch(`/api/download/${format}/${this.selectedTaskId}`);
            
            if (response.ok) {
                // Create download link
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `dataset_${this.selectedTaskId}.${format}`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                
                this.showAlert(`Dataset downloaded as ${format.toUpperCase()}`, 'success');
            } else {
                const error = await response.json();
                this.showAlert('Download failed: ' + (error.error || 'Unknown error'), 'danger');
            }
        } catch (error) {
            console.error('Download error:', error);
            this.showAlert('Download error: ' + error.message, 'danger');
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Debug helper
    getStatus() {
        return {
            selectedTaskId: this.selectedTaskId,
            tasksCount: this.tasks.length,
            tasks: this.tasks
        };
    }
}

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    window.datasetGenerator = new DatasetGenerator();
    
    // Debug helper
    window.debugDatasetGenerator = () => {
        console.log('Dataset Generator Status:', window.datasetGenerator.getStatus());
    };
});

// Add global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    if (window.datasetGenerator) {
        window.datasetGenerator.showAlert('An unexpected error occurred', 'danger');
    }
});