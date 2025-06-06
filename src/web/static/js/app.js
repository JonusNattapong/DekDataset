// DekDataset Web Application - Main JavaScript
(function () {
  "use strict";

  // Global application state
  window.datasetGenerator = {
    currentTask: null,
    isGenerating: false,
    models: {
      deepseek: [],
      ollama: [],
    },

    // Initialize the application
    init: function () {
      console.log("🚀 DekDataset Web App initializing...");
      this.setupEventListeners();
      this.loadTasks();
      this.loadModels();
      this.loadQualitySettings();
      this.checkApiStatus();
    },

    // Setup event listeners
    setupEventListeners: function () {
      // Task management
      const taskDropdown = document.getElementById("taskDropdown");
      const refreshTasks = document.getElementById("refreshTasks");
      const testGeneration = document.getElementById("testGeneration");
      const generateDataset = document.getElementById("generateDataset");
      const stopGeneration = document.getElementById("stopGeneration");

      if (taskDropdown) {
        taskDropdown.addEventListener("change", (e) =>
          this.selectTask(e.target.value)
        );
      }

      if (refreshTasks) {
        refreshTasks.addEventListener("click", () => this.loadTasks());
      }

      if (testGeneration) {
        testGeneration.addEventListener("click", () => this.testGeneration());
      }

      if (generateDataset) {
        generateDataset.addEventListener("click", () => this.generateDataset());
      }

      if (stopGeneration) {
        stopGeneration.addEventListener("click", () => this.stopGeneration());
      }

      // Model provider switching
      const modelProviderInputs = document.querySelectorAll(
        'input[name="modelProvider"]'
      );
      modelProviderInputs.forEach((input) => {
        input.addEventListener("change", (e) =>
          this.switchModelProvider(e.target.value)
        );
      });

      // Download buttons
      document.querySelectorAll(".download-btn").forEach((btn) => {
        btn.addEventListener("click", (e) =>
          this.downloadDataset(e.target.dataset.format)
        );
      });

      // Modal button listeners - setup immediately
      this.setupModalEventListeners();

      // RAG Management
      this.setupRagEventListeners();
      this.loadRagStatus();

      // PDF Drop Zone (Mistral Document AI)
      const pdfDropZone = document.getElementById("pdfDropZone");
      const pdfFileInput = document.getElementById("pdfFileInput");
      const uploadPdfBtn = document.getElementById("uploadPdfBtn");

      if (pdfDropZone && pdfFileInput) {
        // Click to open file dialog
        pdfDropZone.addEventListener("click", (e) => {
          if (e.target.tagName !== "INPUT") {
            pdfFileInput.value = "";
            pdfFileInput.click();
          }
        });

        // Drag & drop support
        ["dragenter", "dragover"].forEach((eventName) => {
          pdfDropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            pdfDropZone.classList.add("dragover");
          });
        });
        ["dragleave", "drop"].forEach((eventName) => {
          pdfDropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            pdfDropZone.classList.remove("dragover");
          });
        });
        pdfDropZone.addEventListener("drop", (e) => {
          e.preventDefault();
          e.stopPropagation();
          pdfDropZone.classList.remove("dragover");
          if (
            e.dataTransfer &&
            e.dataTransfer.files &&
            e.dataTransfer.files.length > 0
          ) {
            pdfFileInput.files = e.dataTransfer.files;
            this.handlePdfFile(pdfFileInput.files[0]);
          }
        });

        // File input change
        pdfFileInput.addEventListener("change", (e) => {
          if (pdfFileInput.files && pdfFileInput.files.length > 0) {
            this.handlePdfFile(pdfFileInput.files[0]);
          }
        });
      } // Upload PDF button
      if (uploadPdfBtn) {
        uploadPdfBtn.addEventListener("click", (e) => {
          e.preventDefault();
          if (pdfFileInput.files && pdfFileInput.files.length > 0) {
            this.handlePdfFile(pdfFileInput.files[0], true);
          } else {
            notifications.warning("Please select or drag a PDF file first.");
          }
        });
      }

      // Dataset creation checkbox toggle
      const createDatasetCheckbox = document.getElementById(
        "createDatasetFromPdf"
      );
      const datasetOptionsPanel = document.getElementById(
        "datasetOptionsPanel"
      );
      if (createDatasetCheckbox && datasetOptionsPanel) {
        createDatasetCheckbox.addEventListener("change", (e) => {
          datasetOptionsPanel.style.display = e.target.checked
            ? "block"
            : "none";
        });
      }
    },

    // Setup Modal Event Listeners (works with custom modal system)
    setupModalEventListeners: function () {
      // API Configuration listeners
      document.addEventListener("click", (e) => {
        // Handle Test API button click
        if (
          e.target &&
          (e.target.id === "testApiBtn" || e.target.closest("#testApiBtn"))
        ) {
          e.preventDefault();
          e.stopPropagation();
          console.log("Test API button clicked");
          this.testApiKey();
        }

        // Handle Save API Key button click
        if (
          e.target &&
          (e.target.id === "saveApiKey" || e.target.closest("#saveApiKey"))
        ) {
          e.preventDefault();
          e.stopPropagation();
          console.log("Save API Key button clicked");
          this.saveApiKey();
        }

        // Handle Create Task button click
        if (
          e.target &&
          (e.target.id === "saveTask" || e.target.closest("#saveTask"))
        ) {
          e.preventDefault();
          e.stopPropagation();
          console.log("Save Task button clicked");
          this.createTask();
        }

        // Handle Save Quality Settings button click
        if (
          e.target &&
          (e.target.id === "saveQuality" || e.target.closest("#saveQuality"))
        ) {
          e.preventDefault();
          e.stopPropagation();
          console.log("Save Quality button clicked");
          this.saveQualitySettings();
        }

        // Handle JSON editing buttons
        if (
          e.target &&
          (e.target.id === "updateTaskJsonl" ||
            e.target.closest("#updateTaskJsonl"))
        ) {
          e.preventDefault();
          e.stopPropagation();
          this.updateTaskFromJson();
        }

        if (
          e.target &&
          (e.target.id === "formatJsonBtn" ||
            e.target.closest("#formatJsonBtn"))
        ) {
          e.preventDefault();
          e.stopPropagation();
          this.formatJson();
        }

        if (
          e.target &&
          (e.target.id === "validateJsonBtn" ||
            e.target.closest("#validateJsonBtn"))
        ) {
          e.preventDefault();
          e.stopPropagation();
          this.validateJson();
        }

        if (
          e.target &&
          (e.target.id === "copyJsonBtn" || e.target.closest("#copyJsonBtn"))
        ) {
          e.preventDefault();
          e.stopPropagation();
          this.copyJson();
        }
      });

      // Quality settings range slider listener
      document.addEventListener("input", (e) => {
        if (e.target && e.target.id === "similarityThreshold") {
          const thresholdValue = document.getElementById("thresholdValue");
          if (thresholdValue) {
            thresholdValue.textContent = e.target.value;
          }
        }
      });

      console.log("Modal event listeners setup complete");
    },

    // Enhanced API key input finding with debugging
    findApiKeyInput: function () {
      // Multiple strategies to find the API key input
      const selectors = [
        ".custom-modal-body #apiKey", // Custom modal body (modal ที่แสดงจริง)
        ".custom-modal-overlay #apiKey", // Anywhere in custom modal
      ];

      for (const selector of selectors) {
        const input = document.querySelector(selector);
        if (input) {
          console.log(`Found API key input using selector: ${selector}`);
          console.log("Input element:", input);
          console.log("Input value length:", input.value.length);
          console.log(
            "Input value preview:",
            input.value.substring(0, 10) + "..."
          );
          return input;
        }
      }

      console.error("API key input not found with any selector!");
      console.log(
        "Available inputs in document:",
        document.querySelectorAll("input")
      );
      return null;
    },

    // Enhanced API status finding with debugging
    findApiStatus: function () {
      const selectors = [
        "#apiStatus",
        ".custom-modal-body #apiStatus",
        ".custom-modal-overlay #apiStatus",
        ".alert", // Fallback to any alert in modal
      ];

      for (const selector of selectors) {
        const status = document.querySelector(selector);
        if (
          status &&
          (selector === ".alert" ? status.closest(".custom-modal-body") : true)
        ) {
          console.log(`Found API status using selector: ${selector}`);
          return status;
        }
      }

      console.error("API status element not found!");
      return null;
    },

    // Test API Key with enhanced error handling and UI feedback
    testApiKey: async function () {
      console.log("Starting API key test...");

      // Find API key input with enhanced detection
      const apiKeyInput = this.findApiKeyInput();
      if (!apiKeyInput) {
        notifications.error("ไม่พบช่องใส่ API key กรุณาลองใหม่");
        return;
      }

      const apiKey = apiKeyInput.value.trim();
      console.log("API key length:", apiKey.length);

      if (!apiKey) {
        this.updateApiStatus("warning", "กรุณาใส่ API key ก่อน");
        notifications.warning("กรุณาใส่ API key ก่อน");
        // Focus on the input field
        apiKeyInput.focus();
        return;
      }

      if (!apiKey.startsWith("sk-")) {
        this.updateApiStatus(
          "danger",
          "รูปแบบ API key ไม่ถูกต้อง (ต้องขึ้นต้นด้วย sk-)"
        );
        notifications.error("รูปแบบ API key ไม่ถูกต้อง");
        return;
      }

      // Find and update test button
      let testApiBtn = document.querySelector("#testApiBtn");
      if (!testApiBtn) {
        testApiBtn = document.querySelector(".custom-modal-footer #testApiBtn");
      }

      try {
        // Update button state
        if (testApiBtn) {
          testApiBtn.disabled = true;
          testApiBtn.innerHTML =
            '<i class="fas fa-spinner fa-spin"></i> กำลังทดสอบ...';
        }

        this.updateApiStatus(
          "info",
          '<i class="fas fa-spinner fa-spin"></i> กำลังทดสอบ API key...'
        );
        notifications.info("กำลังทดสอบ API key...", 0);

        console.log("Sending test request to API...");

        const response = await fetch("/api/test-api-key", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify({ api_key: apiKey }),
        });

        console.log("Response status:", response.status);

        const data = await response.json();
        console.log("Response data:", data);

        notifications.clear();

        if (response.ok && data.valid) {
          this.updateApiStatus(
            "success",
            `✅ API key ใช้งานได้! รุ่น: ${
              data.model_accessible || "deepseek-chat"
            }`
          );
          notifications.success("✅ API key ทดสอบสำเร็จ!");
          console.log("API key test successful");
        } else {
          const errorMsg = data.message || "API key ไม่ถูกต้อง";
          this.updateApiStatus("danger", `❌ ${errorMsg}`);
          notifications.error(`❌ ${errorMsg}`);
          console.error("API key test failed:", errorMsg);
        }
      } catch (error) {
        console.error("API test error:", error);
        notifications.clear();
        this.updateApiStatus("danger", `❌ เกิดข้อผิดพลาด: ${error.message}`);
        notifications.error(`เกิดข้อผิดพลาดในการทดสอบ: ${error.message}`);
      } finally {
        // Restore button state
        if (testApiBtn) {
          testApiBtn.disabled = false;
          testApiBtn.innerHTML = '<i class="fas fa-flask"></i> Test API';
        }
      }
    },

    // Save API Key with enhanced input detection
    saveApiKey: async function () {
      console.log("Starting API key save...");

      // Find API key input with enhanced detection
      const apiKeyInput = this.findApiKeyInput();
      if (!apiKeyInput) {
        notifications.error("ไม่พบช่องใส่ API key");
        return;
      }

      const apiKey = apiKeyInput.value.trim();
      console.log("API key for save - length:", apiKey.length);

      if (!apiKey) {
        notifications.warning("กรุณาใส่ API key ก่อน");
        apiKeyInput.focus();
        return;
      }

      // Find save button
      let saveApiKey = document.querySelector("#saveApiKey");
      if (!saveApiKey) {
        saveApiKey = document.querySelector(".custom-modal-footer #saveApiKey");
      }

      try {
        if (saveApiKey) {
          saveApiKey.disabled = true;
          saveApiKey.innerHTML =
            '<i class="fas fa-spinner fa-spin"></i> กำลังบันทึก...';
        }

        const response = await fetch("/api/config/api-key", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ api_key: apiKey }),
        });

        const data = await response.json();

        if (response.ok && data.valid) {
          notifications.success("✅ บันทึก API key สำเร็จ!");
          this.updateApiStatus("success", "✅ API key ถูกตั้งค่าและใช้งานได้");

          // Close modal
          this.closeModalBySelector(".custom-modal-overlay");

          // Refresh generation buttons
          this.updateGenerationButtons();
          this.loadModels(); // Reload models now that API is configured
        } else {
          throw new Error(data.message || "การบันทึก API key ล้มเหลว");
        }
      } catch (error) {
        console.error("Error saving API key:", error);
        notifications.error("ไม่สามารถบันทึก API key ได้: " + error.message);
        this.updateApiStatus("danger", "❌ " + error.message);
      } finally {
        if (saveApiKey) {
          saveApiKey.disabled = false;
          saveApiKey.innerHTML = '<i class="fas fa-save"></i> Save & Set';
        }
      }
    },

    // Update API Status display with better element finding
    updateApiStatus: function (type, message) {
      const apiStatus = this.findApiStatus();

      if (!apiStatus) {
        console.warn("API status element not found, using notifications only");
        return;
      }

      const iconMap = {
        success: "fas fa-check-circle",
        danger: "fas fa-exclamation-circle",
        warning: "fas fa-exclamation-triangle",
        info: "fas fa-info-circle",
      };

      apiStatus.className = `alert alert-${type}`;
      apiStatus.innerHTML = message.includes("<i class=")
        ? message
        : `<i class="${iconMap[type] || "fas fa-info-circle"}"></i> ${message}`;
    },

    // Helper to close modal by selector
    closeModalBySelector: function (selector) {
      const modal = document.querySelector(selector);
      if (modal && modal.classList.contains("custom-modal-overlay")) {
        modal.classList.remove("show");
        setTimeout(() => {
          if (modal.parentNode) {
            modal.parentNode.removeChild(modal);
          }
        }, 300);
      }
    },

    // Load available tasks
    loadTasks: async function () {
      try {
        const response = await fetch("/api/tasks");
        const data = await response.json();

        if (response.ok) {
          this.updateTaskList(data.tasks);
          this.updateTaskDropdown(data.tasks);
        } else {
          throw new Error(data.detail || "Failed to load tasks");
        }
      } catch (error) {
        console.error("Error loading tasks:", error);
        notifications.error("Failed to load tasks: " + error.message);
      }
    },

    // Update task list display
    updateTaskList: function (tasks) {
      const taskList = document.getElementById("taskList");
      if (!taskList) return;

      if (tasks.length === 0) {
        taskList.innerHTML =
          '<div class="text-muted text-center py-4">No tasks available. Create your first task!</div>';
        return;
      }

      let html = "";
      tasks.forEach((task) => {
        html += `
        <div class="task-card shadow rounded-4 p-4 mb-3 bg-white border position-relative" data-task-id="${
          task.id
        }">
            <div class="mb-2">
                <h5 class="fw-semibold mb-1 text-primary">${
                  task.name || task.id
                }</h5>
                <small class="text-muted d-block mb-3">${
                  task.description || "No description available"
                }</small>
                <div class="d-flex gap-2">
                    <button class="btn btn-sm btn-outline-success" title="Select Task"
                        onclick="datasetGenerator.selectTask('${task.id}')">
                        <i class="fas fa-check"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-warning" title="Edit Task"
                        onclick="datasetGenerator.editTask('${task.id}')">
                        <i class="fas fa-pen"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-danger" title="Delete Task"
                        onclick="datasetGenerator.deleteTask('${task.id}')">
                        <i class="fas fa-trash-alt"></i>
                    </button>
                </div>
            </div>
        </div>
    `;
      });

      taskList.innerHTML = html;
    },

    // Update task dropdown
    updateTaskDropdown: function (tasks) {
      const dropdown = document.getElementById("taskDropdown");
      if (!dropdown) return;

      dropdown.innerHTML = '<option value="">-- Select Task --</option>';
      tasks.forEach((task) => {
        dropdown.innerHTML += `<option value="${task.id}">${
          task.name || task.id
        }</option>`;
      });
    },

    // Load available models
    loadModels: async function () {
      try {
        const response = await fetch("/api/models/all");
        if (!response.ok) {
          throw new Error("Not Found");
        }
        const data = await response.json();
        console.log("Models loaded successfully:", data);
        
        // Store models data
        this.models.deepseek = data.deepseek?.models || [];
        this.models.ollama = data.ollama?.models || [];
        
        this.updateModelSelects(data);
      } catch (error) {
        console.error("Error loading models:", error);
        
        // Provide fallback model data
        const fallbackData = {
          deepseek: {
            available: false,
            models: [
              {
                id: "deepseek-chat",
                name: "DeepSeek Chat",
                description: "Advanced reasoning model (API key required)"
              }
            ]
          },
          ollama: {
            available: false,
            models: []
          }
        };
        
        this.models.deepseek = fallbackData.deepseek.models;
        this.models.ollama = fallbackData.ollama.models;
        
        this.updateModelSelects(fallbackData);
      }
    },

    // Update model select elements
    updateModelSelects: function (data) {
      const deepseekSelect = document.getElementById("deepseekModelSelect");
      const ollamaSelect = document.getElementById("ollamaModelSelect");

      // Ensure data structure exists
      if (!data) {
        console.error("No model data provided");
        return;
      }

      if (deepseekSelect) {
        deepseekSelect.innerHTML = "";
        
        // Check if DeepSeek data exists and has models
        if (data.deepseek && data.deepseek.models && Array.isArray(data.deepseek.models)) {
          if (data.deepseek.available && data.deepseek.models.length > 0) {
            data.deepseek.models.forEach((model) => {
              const option = document.createElement("option");
              option.value = model.id;
              option.textContent = model.name;
              option.title = model.description || "";
              deepseekSelect.appendChild(option);
            });
          } else {
            const option = document.createElement("option");
            option.value = "";
            option.textContent = data.deepseek.available 
              ? "No DeepSeek models available" 
              : "DeepSeek API key required";
            option.disabled = true;
            deepseekSelect.appendChild(option);
          }
        } else {
          console.warn("Invalid DeepSeek model data structure");
          const option = document.createElement("option");
          option.value = "";
          option.textContent = "Error loading DeepSeek models";
          option.disabled = true;
          deepseekSelect.appendChild(option);
        }
      }

      if (ollamaSelect) {
        ollamaSelect.innerHTML = "";
        
        // Check if Ollama data exists and has models
        if (data.ollama && data.ollama.models && Array.isArray(data.ollama.models)) {
          if (data.ollama.available && data.ollama.models.length > 0) {
            data.ollama.models.forEach((model) => {
              const option = document.createElement("option");
              option.value = model.id;
              option.textContent = model.name;
              option.title = model.description || "";
              ollamaSelect.appendChild(option);
            });
          } else {
            const option = document.createElement("option");
            option.value = "";
            option.textContent = data.ollama.available 
              ? "No Ollama models found" 
              : "Ollama server not available";
            option.disabled = true;
            ollamaSelect.appendChild(option);
          }
        } else {
          console.warn("Invalid Ollama model data structure");
          const option = document.createElement("option");
          option.value = "";
          option.textContent = "Ollama not available";
          option.disabled = true;
          ollamaSelect.appendChild(option);
        }
      }

      // Hide loading indicator
      const loadingIndicator = document.getElementById("modelLoadingIndicator");
      if (loadingIndicator) {
        loadingIndicator.style.display = "none";
      }

      this.updateGenerationButtons();
    },

    // Switch model provider
    switchModelProvider: function (provider) {
      const deepseekSection = document.getElementById("deepseekModelSection");
      const ollamaSection = document.getElementById("ollamaModelSection");

      if (provider === "deepseek") {
        deepseekSection.style.display = "block";
        ollamaSection.style.display = "none";
        deepseekSection.classList.add("active");
        ollamaSection.classList.remove("active");
      } else if (provider === "ollama") {
        deepseekSection.style.display = "none";
        ollamaSection.style.display = "block";
        ollamaSection.classList.add("active");
        deepseekSection.classList.remove("active");
      }

      this.updateGenerationButtons();
    },

    // Select a task
    selectTask: function (taskId) {
      if (!taskId) {
        this.currentTask = null;
        this.updateSelectedTask(null);
        this.updateGenerationButtons();
        return;
      }

      fetch(`/api/tasks/${taskId}`)
        .then((response) => response.json())
        .then((task) => {
          if (task) {
            this.currentTask = task;
            this.updateSelectedTask(task);
            this.updateGenerationButtons();

            // Update dropdown selection
            const dropdown = document.getElementById("taskDropdown");
            if (dropdown) dropdown.value = taskId;

            // Update task list selection
            document.querySelectorAll(".task-item").forEach((item) => {
              item.classList.remove("selected");
            });
            const selectedItem = document.querySelector(
              `.task-item[data-task-id="${taskId}"]`
            );
            if (selectedItem) selectedItem.classList.add("selected");

            notifications.success(`Selected task: ${task.name || task.id}`);
          }
        })
        .catch((error) => {
          console.error("Error selecting task:", error);
          notifications.error("Failed to select task: " + error.message);
        });
    },

    // Update selected task display
    updateSelectedTask: function (task) {
      const selectedTaskInput = document.getElementById("selectedTask");
      if (selectedTaskInput) {
        selectedTaskInput.value = task ? task.name || task.id : "";
      }
    },

    // Update generation buttons state
    updateGenerationButtons: function () {
      const testBtn = document.getElementById("testGeneration");
      const generateBtn = document.getElementById("generateDataset");

      const hasTask = this.currentTask !== null;
      const hasModel = this.getSelectedModel() !== null;
      const canGenerate = hasTask && hasModel && !this.isGenerating;

      if (testBtn) testBtn.disabled = !canGenerate;
      if (generateBtn) generateBtn.disabled = !canGenerate;
    },

    // Get selected model
    getSelectedModel: function () {
      const provider = document.querySelector(
        'input[name="modelProvider"]:checked'
      )?.value;

      if (provider === "deepseek") {
        const select = document.getElementById("deepseekModelSelect");
        return select?.value || null;
      } else if (provider === "ollama") {
        const select = document.getElementById("ollamaModelSelect");
        return select?.value || null;
      }

      return null;
    },

    // Test generation
    testGeneration: async function () {
      if (!this.currentTask) {
        notifications.warning("Please select a task first");
        return;
      }

      const model = this.getSelectedModel();
      if (!model) {
        notifications.warning("Please select a model first");
        return;
      }

      try {
        this.showProgress("Testing generation...", 0);

        const response = await fetch("/api/test-generation", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            task_id: this.currentTask.id,
            model: model,
          }),
        });

        const data = await response.json();

        if (response.ok) {
          this.hideProgress();
          this.displayTestResults(data);
          notifications.success("Test generation completed successfully!");
        } else {
          throw new Error(data.detail || "Test generation failed");
        }
      } catch (error) {
        this.hideProgress();
        console.error("Test generation error:", error);
        notifications.error("Test generation failed: " + error.message);
      }
    },

    // Generate dataset
    generateDataset: async function () {
      if (!this.currentTask) {
        notifications.warning("Please select a task first");
        return;
      }

      const model = this.getSelectedModel();
      if (!model) {
        notifications.warning("Please select a model first");
        return;
      }

      const entryCount = parseInt(
        document.getElementById("entryCount")?.value || "10"
      );
      if (entryCount <= 0 || entryCount > 1000) {
        notifications.warning("Entry count must be between 1 and 1000");
        return;
      }

      try {
        this.isGenerating = true;
        this.updateGenerationButtons();
        this.showProgress("Generating dataset...", 0);

        const response = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            task_id: this.currentTask.id,
            model: model,
            count: entryCount,
          }),
        });

        const data = await response.json();

        if (response.ok) {
          this.hideProgress();
          this.displayResults(data);
          notifications.success(
            `Successfully generated ${data.entries.length} entries!`
          );
        } else {
          throw new Error(data.detail || "Dataset generation failed");
        }
      } catch (error) {
        this.hideProgress();
        console.error("Generation error:", error);
        notifications.error("Dataset generation failed: " + error.message);
      } finally {
        this.isGenerating = false;
        this.updateGenerationButtons();
      }
    },

    // Stop generation (placeholder)
    stopGeneration: function () {
      notifications.info("Stop generation functionality not implemented yet");
    },

    // Show progress
    showProgress: function (message, progress) {
      const progressSection = document.getElementById("progressSection");
      const progressBar = progressSection?.querySelector(".progress-bar");
      const progressText = document.getElementById("progressText");
      const stopBtn = document.getElementById("stopGeneration");

      if (progressSection) progressSection.style.display = "block";
      if (progressBar) progressBar.style.width = `${progress}%`;
      if (progressText)
        progressText.innerHTML = `<i class="fas fa-cog fa-spin me-2"></i>${message}`;
      if (stopBtn) stopBtn.style.display = "inline-flex";
    },

    // Hide progress
    hideProgress: function () {
      const progressSection = document.getElementById("progressSection");
      const stopBtn = document.getElementById("stopGeneration");

      if (progressSection) progressSection.style.display = "none";
      if (stopBtn) stopBtn.style.display = "none";
    },

    // Display test results
    displayTestResults: function (data) {
      const resultsSection = document.getElementById("resultsSection");
      if (!resultsSection) return;

      let html = `
                <h4><i class="fas fa-flask"></i> Test Results</h4>
                <div class="alert alert-info">
                    <strong>Test completed!</strong> Generated ${data.test_entries.length} sample entries.
                </div>
            `;

      if (data.test_entries.length > 0) {
        html += "<h5>Sample Entries:</h5>";
        data.test_entries.slice(0, 3).forEach((entry, index) => {
          html += `
                        <div class="card mb-2">
                            <div class="card-body">
                                <h6>Entry ${index + 1}</h6>
                                <pre class="bg-light p-2 rounded"><code>${JSON.stringify(
                                  entry.content,
                                  null,
                                  2
                                )}</code></pre>
                            </div>
                        </div>
                    `;
        });
      }

      resultsSection.innerHTML = html;
      resultsSection.style.display = "block";
    },

    // Display generation results
    displayResults: function (data) {
      const downloadSection = document.getElementById("downloadSection");
      const resultSummary = document.getElementById("resultSummary");
      const previewTableContainer = document.getElementById(
        "previewTableContainer"
      );

      if (!downloadSection) return;

      // Update summary
      if (resultSummary) {
        const quality = data.quality_report;
        resultSummary.innerHTML = `
                    <div class="alert alert-success">
                        <h5><i class="fas fa-check-circle"></i> Dataset Generated Successfully!</h5>
                        <div class="row mt-3">
                            <div class="col-md-3">
                                <strong>Entries Generated:</strong><br>
                                <span class="h4 text-primary">${
                                  data.entries.length
                                }</span>
                            </div>
                            <div class="col-md-3">
                                <strong>Quality Score:</strong><br>
                                <span class="h4 text-success">${(
                                  quality.quality_score * 100
                                ).toFixed(1)}%</span>
                            </div>
                            <div class="col-md-3">
                                <strong>Average Length:</strong><br>
                                <span class="h4 text-info">${Math.round(
                                  quality.average_length
                                )} chars</span>
                            </div>
                            <div class="col-md-3">
                                <strong>Duplicates Removed:</strong><br>
                                <span class="h4 text-warning">${
                                  quality.duplicates_removed
                                }</span>
                            </div>
                        </div>
                    </div>
                `;
      }

      // Update preview table
      if (previewTableContainer && data.entries.length > 0) {
        let tableHTML = `
                    <h5><i class="fas fa-table"></i> Dataset Preview (First 10 entries)</h5>
                    <div class="table-responsive">
                        <table class="table table-striped table-hover">
                            <thead class="table-dark">
                                <tr>
                                    <th>ID</th>
                                    <th>Content Preview</th>
                                    <th>Metadata</th>
                                </tr>
                            </thead>
                            <tbody>
                `;

        data.entries.slice(0, 10).forEach((entry) => {
          const contentPreview = Object.keys(entry.content)
            .map(
              (key) =>
                `<strong>${key}:</strong> ${String(
                  entry.content[key]
                ).substring(0, 100)}...`
            )
            .join("<br>");

          const metadataPreview = Object.entries(entry.metadata)
            .slice(0, 3)
            .map(([k, v]) => `${k}: ${v}`)
            .join("<br>");

          tableHTML += `
                        <tr>
                            <td><code>${entry.id}</code></td>
                            <td>${contentPreview}</td>
                            <td class="small text-muted">${metadataPreview}</td>
                        </tr>
                    `;
        });

        tableHTML += `
                            </tbody>
                        </table>
                    </div>
                `;

        previewTableContainer.innerHTML = tableHTML;
      }

      // Enable download buttons
      document.querySelectorAll(".download-btn").forEach((btn) => {
        btn.disabled = false;
      });

      downloadSection.style.display = "block";
    },

    // Download dataset
    downloadDataset: function (format) {
      if (!this.currentTask) {
        notifications.warning("No dataset to download");
        return;
      }

      const url = `/api/download/${format}/${this.currentTask.id}`;
      const link = document.createElement("a");
      link.href = url;
      link.style.display = "none";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      notifications.success(
        `Downloading dataset in ${format.toUpperCase()} format...`
      );
    },

    // Create new task
    createTask: async function () {
      let taskId = document.getElementById("taskId")?.value.trim();
      let taskType = document.getElementById("taskType")?.value;
      let description = document
        .getElementById("taskDescription")
        ?.value.trim();
      let systemPrompt = document.getElementById("systemPrompt")?.value.trim();
      let userTemplate = document.getElementById("userTemplate")?.value.trim();

      // Try to find in custom modal if not found
      if (!taskId) {
        taskId = document
          .querySelector(".custom-modal-body #taskId")
          ?.value.trim();
        taskType = document.querySelector(
          ".custom-modal-body #taskType"
        )?.value;
        description = document
          .querySelector(".custom-modal-body #taskDescription")
          ?.value.trim();
        systemPrompt = document
          .querySelector(".custom-modal-body #systemPrompt")
          ?.value.trim();
        userTemplate = document
          .querySelector(".custom-modal-body #userTemplate")
          ?.value.trim();
      }

      if (!taskId) {
        notifications.warning("Task ID is required");
        return;
      }

      try {
        const response = await fetch("/api/tasks", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            id: taskId,
            type: taskType,
            description: description,
            system_prompt: systemPrompt,
            user_template: userTemplate,
          }),
        });

        const data = await response.json();

        if (response.ok) {
          notifications.success("Task created successfully!");
          this.loadTasks();

          // Close modal
          this.closeModalBySelector(".custom-modal-overlay");

          // Clear form
          const form = document.getElementById("createTaskForm");
          if (form) form.reset();
        } else {
          throw new Error(data.detail || "Failed to create task");
        }
      } catch (error) {
        console.error("Error creating task:", error);
        notifications.error("Failed to create task: " + error.message);
      }
    },

    // Save Quality Settings
    saveQualitySettings: async function () {
      let enableQualityFilter = document.getElementById(
        "enableQualityFilter"
      )?.checked;
      let minResponseLength = parseInt(
        document.getElementById("minResponseLength")?.value || "10"
      );
      let maxResponseLength = parseInt(
        document.getElementById("maxResponseLength")?.value || "2000"
      );
      let similarityThreshold = parseFloat(
        document.getElementById("similarityThreshold")?.value || "0.8"
      );
      let enableDuplicateDetection = document.getElementById(
        "enableDuplicateDetection"
      )?.checked;

      // Try custom modal if not found
      if (enableQualityFilter === undefined) {
        enableQualityFilter = document.querySelector(
          ".custom-modal-body #enableQualityFilter"
        )?.checked;
        minResponseLength = parseInt(
          document.querySelector(".custom-modal-body #minResponseLength")
            ?.value || "10"
        );
        maxResponseLength = parseInt(
          document.querySelector(".custom-modal-body #maxResponseLength")
            ?.value || "2000"
        );
        similarityThreshold = parseFloat(
          document.querySelector(".custom-modal-body #similarityThreshold")
            ?.value || "0.8"
        );
        enableDuplicateDetection = document.querySelector(
          ".custom-modal-body #enableDuplicateDetection"
        )?.checked;
      }

      try {
        const response = await fetch("/api/quality-config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            min_length: minResponseLength,
            max_length: maxResponseLength,
            similarity_threshold: similarityThreshold,
            required_fields: [],
            custom_validators: [],
          }),
        });

        const data = await response.json();

        if (response.ok) {
          notifications.success("Quality settings saved successfully!");
          this.closeModalBySelector(".custom-modal-overlay");
        } else {
          throw new Error(data.detail || "Failed to save quality settings");
        }
      } catch (error) {
        console.error("Error saving quality settings:", error);
        notifications.error(
          "Failed to save quality settings: " + error.message
        );
      }
    },

    // Load quality settings
    loadQualitySettings: async function () {
      try {
        const response = await fetch("/api/quality-config");
        const data = await response.json();

        if (response.ok && data.config) {
          const config = data.config;

          const minLength = document.getElementById("minResponseLength");
          const maxLength = document.getElementById("maxResponseLength");
          const threshold = document.getElementById("similarityThreshold");
          const thresholdValue = document.getElementById("thresholdValue");

          if (minLength) minLength.value = config.min_length || 10;
          if (maxLength) maxLength.value = config.max_length || 2000;
          if (threshold) threshold.value = config.similarity_threshold || 0.8;
          if (thresholdValue)
            thresholdValue.textContent = config.similarity_threshold || 0.8;
        }
      } catch (error) {
        console.error("Error loading quality settings:", error);
      }
    },

    // Check API status
    checkApiStatus: async function () {
      try {
        const response = await fetch("/api/status");
        const data = await response.json();

        if (response.ok) {
          const apiStatus = document.getElementById("apiStatus");
          if (apiStatus) {
            if (data.deepseek_api_configured) {
              this.updateApiStatus("success", "✅ DeepSeek API พร้อมใช้งาน");
            } else {
              this.updateApiStatus(
                "warning",
                "⚠ ยังไม่ได้ตั้งค่า DeepSeek API key"
              );
            }
          }
        }
      } catch (error) {
        console.error("Error checking API status:", error);
      }
    },

    // Edit task
    editTask: async function (taskId) {
      try {
        const response = await fetch(`/api/tasks/${taskId}`);
        const task = await response.json();

        if (response.ok) {
          this.populateEditForm(task);

          // Show edit modal using custom modal system
          const modalElement = document.getElementById("editTaskModal");
          if (modalElement && window.CustomModal) {
            const customModal = new window.CustomModal(modalElement);
            customModal.show();
          }
        } else {
          throw new Error(task.detail || "Failed to load task");
        }
      } catch (error) {
        console.error("Error loading task for edit:", error);
        notifications.error("Failed to load task: " + error.message);
      }
    },

    // Populate edit form
    populateEditForm: function (taskData) {
      const jsonlTextarea = document.getElementById("editTaskJsonl");
      if (jsonlTextarea) {
        jsonlTextarea.value = JSON.stringify(taskData, null, 2);
      }
    },

    // Format JSON
    formatJson: function () {
      let textarea = document.getElementById("editTaskJsonl");
      if (!textarea) {
        textarea = document.querySelector(".custom-modal-body #editTaskJsonl");
      }

      if (!textarea) return;

      try {
        const parsed = JSON.parse(textarea.value);
        textarea.value = JSON.stringify(parsed, null, 2);
        notifications.success("JSON formatted successfully!");
      } catch (error) {
        notifications.error("Invalid JSON format: " + error.message);
      }
    },

    // Validate JSON
    validateJson: function () {
      let textarea = document.getElementById("editTaskJsonl");
      if (!textarea) {
        textarea = document.querySelector(".custom-modal-body #editTaskJsonl");
      }

      if (!textarea) return;

      try {
        const parsed = JSON.parse(textarea.value);
        notifications.success("JSON is valid!");
        return true;
      } catch (error) {
        notifications.error("Invalid JSON format: " + error.message);
        return false;
      }
    },

    // Copy JSON
    copyJson: function () {
      let textarea = document.getElementById("editTaskJsonl");
      if (!textarea) {
        textarea = document.querySelector(".custom-modal-body #editTaskJsonl");
      }

      if (!textarea) return;

      try {
        textarea.select();
        document.execCommand('copy');
        notifications.success("JSON copied to clipboard!");
      } catch (error) {
        notifications.error("Failed to copy JSON: " + error.message);
      }
    },
    // Update task from JSON
    updateTaskFromJson: async function () {
      let jsonTextarea = document.getElementById("editTaskJsonl");
      if (!jsonTextarea) {
        jsonTextarea = document.querySelector(
          ".custom-modal-body #editTaskJsonl"
        );
      }

      if (!jsonTextarea) return;

      try {
        const taskData = JSON.parse(jsonTextarea.value);
        const taskId = taskData.id;

        if (!taskId) {
          notifications.warning("Task ID is required in JSON data");
          return;
        }

        const response = await fetch(`/api/tasks/${taskId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(taskData),
        });

        const result = await response.json();

        if (response.ok) {
          notifications.success("Task updated successfully!");
          this.loadTasks();
          this.closeModalBySelector(".custom-modal-overlay");
        } else {
          throw new Error(result.detail || "Failed to update task");
        }
      } catch (error) {
        if (error instanceof SyntaxError) {
          notifications.error("Invalid JSON format. Please check your syntax.");
        } else {
          console.error("Error updating task:", error);
          notifications.error("Failed to update task: " + error.message);
        }
      }
    },

    // Delete task
    deleteTask: async function (taskId) {
      if (
        !confirm(
          "Are you sure you want to delete this task? This action cannot be undone."
        )
      ) {
        return;
      }

      try {
        const response = await fetch(`/api/tasks/${taskId}`, {
          method: "DELETE",
        });

        const result = await response.json();

        if (response.ok) {
          notifications.success("Task deleted successfully!");
          this.loadTasks();

          // Clear selection if deleted task was selected
          if (this.currentTask && this.currentTask.id === taskId) {
            this.currentTask = null;
            this.updateSelectedTask(null);
            this.updateGenerationButtons();
          }
        } else {
          throw new Error(result.detail || "Failed to delete task");
        }
      } catch (error) {
        console.error("Error deleting task:", error);
        notifications.error("Failed to delete task: " + error.message);
      }
    },

    // Handle PDF file selection or drop
    handlePdfFile: function (file, triggerUpload) {
      if (!file) {
        notifications.warning("No file selected.");
        return;
      }
      if (
        !file.name.toLowerCase().endsWith(".pdf") &&
        !file.type.startsWith("image/")
      ) {
        notifications.error("Only PDF or image files are supported.");
        return;
      }
      // Show file name and enable upload button if present
      const pdfFileLabel = document.getElementById("pdfFileLabel");
      if (pdfFileLabel) {
        pdfFileLabel.textContent = file.name;
      }
      if (triggerUpload) {
        this.uploadPdfFile(file);
      }
    },
    uploadPdfFile: async function (file) {
      if (!file) {
        notifications.warning("No file selected for upload.");
        return;
      }

      this.showProgress("Uploading and processing file...", 10);

      try {
        const formData = new FormData();
        formData.append("file", file);

        // Add Mistral API key if provided
        const apiKeyInput = document.getElementById("mistralApiKey");
        if (apiKeyInput && apiKeyInput.value) {
          formData.append("mistral_api_key", apiKeyInput.value.trim());
        }

        // Add dataset creation options
        const createDatasetCheckbox = document.getElementById(
          "createDatasetFromPdf"
        );
        const datasetTypeSelect = document.getElementById("pdfDatasetType");

        if (createDatasetCheckbox && createDatasetCheckbox.checked) {
          formData.append("create_dataset", "true");
          if (datasetTypeSelect && datasetTypeSelect.value) {
            formData.append("dataset_type", datasetTypeSelect.value);
          }
        }

        // Add OCR options
        const bboxAnnotation = document.getElementById("bboxAnnotation");
        const docAnnotation = document.getElementById("docAnnotation");
        const ocrPages = document.getElementById("ocrPages");

        if (bboxAnnotation && bboxAnnotation.checked) {
          formData.append("bbox_annotation", "true");
        }
        if (docAnnotation && docAnnotation.checked) {
          formData.append("doc_annotation", "true");
        }
        if (ocrPages && ocrPages.value.trim()) {
          formData.append("pages", ocrPages.value.trim());
        }

        const response = await fetch("/api/pdf-upload", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          const err = await response.json().catch(() => ({}));
          throw new Error(err.detail || "Failed to process file.");
        }

        const data = await response.json();
        this.hideProgress();

        notifications.success("File processed successfully!");
        this.displayPdfResults(data);
      } catch (error) {
        this.hideProgress();
        notifications.error("Error: " + (error.message || error));
      }
    },

    displayPdfResults: function (data) {
      console.log("PDF Results:", data);

      // Show results in the upload status area
      const statusDiv = document.getElementById("pdfUploadStatus");
      const previewDiv = document.getElementById("pdfEntriesPreview");

      if (statusDiv) {
        let statusHtml = `
                    <div class="alert alert-success mt-3">
                        <h5><i class="fas fa-check-circle"></i> Processing Complete</h5>
                        <p><strong>Text Extracted:</strong> ${
                          data.extracted_text ? data.extracted_text.length : 0
                        } characters</p>
                        ${
                          data.language_detected
                            ? `<p><strong>Language:</strong> ${data.language_detected}</p>`
                            : ""
                        }
                        ${
                          data.confidence_score
                            ? `<p><strong>Confidence:</strong> ${(
                                data.confidence_score * 100
                              ).toFixed(1)}%</p>`
                            : ""
                        }
                `;

        if (data.dataset_created) {
          statusHtml += `
                        <p><strong>Dataset Created:</strong> ${data.dataset_info.entries_count} entries</p>
                        <p><strong>Dataset Type:</strong> ${data.dataset_info.dataset_type}</p>
                        <p><strong>Task ID:</strong> <code>${data.task_id}</code></p>
                    `;
        }

        statusHtml += "</div>";
        statusDiv.innerHTML = statusHtml;
      }

      // Show dataset preview if available
      if (data.dataset_preview && previewDiv) {
        let previewHtml = '<div class="mt-3"><h6>Dataset Preview:</h6>';

        if (data.dataset_preview.length > 0) {
          previewHtml += '<div class="card"><div class="card-body">';

          data.dataset_preview.forEach((entry, index) => {
            previewHtml += `<div class="border-bottom pb-2 mb-2">`;
            previewHtml += `<small class="text-muted">Entry ${
              index + 1
            }:</small><br>`;

            // Display based on dataset type
            if (entry.instruction && entry.response) {
              previewHtml += `<strong>Instruction:</strong> ${this.truncateText(
                entry.instruction,
                150
              )}<br>`;
              previewHtml += `<strong>Response:</strong> ${this.truncateText(
                entry.response,
                150
              )}`;
            } else if (entry.question && entry.answer) {
              previewHtml += `<strong>Question:</strong> ${this.truncateText(
                entry.question,
                150
              )}<br>`;
              previewHtml += `<strong>Answer:</strong> ${this.truncateText(
                entry.answer,
                150
              )}`;
            } else if (entry.text && entry.label) {
              previewHtml += `<strong>Text:</strong> ${this.truncateText(
                entry.text,
                150
              )}<br>`;
              previewHtml += `<strong>Label:</strong> ${entry.label}`;
            } else if (entry.source && entry.target) {
              previewHtml += `<strong>Source:</strong> ${this.truncateText(
                entry.source,
                150
              )}<br>`;
              previewHtml += `<strong>Target:</strong> ${this.truncateText(
                entry.target,
                150
              )}`;
            } else if (entry.text && entry.summary) {
              previewHtml += `<strong>Text:</strong> ${this.truncateText(
                entry.text,
                150
              )}<br>`;
              previewHtml += `<strong>Summary:</strong> ${this.truncateText(
                entry.summary,
                150
              )}`;
            } else {
              // Fallback for unknown format
              previewHtml += `<pre class="small">${JSON.stringify(
                entry,
                null,
                2
              )}</pre>`;
            }

            previewHtml += "</div>";
          });

          previewHtml += "</div></div>";

          // Add download button if dataset was created
          if (data.dataset_created && data.task_id) {
            previewHtml += `
                            <div class="mt-3">
                                <button class="btn btn-success" onclick="datasetGenerator.downloadPdfDataset('${data.task_id}')">
                                    <i class="fas fa-download"></i> Download Dataset
                                </button>
                            </div>
                        `;
          }
        } else {
          previewHtml +=
            '<p class="text-muted">No dataset entries to preview.</p>';
        }

        previewHtml += "</div>";
        previewDiv.innerHTML = previewHtml;
      }

      // Show raw extracted text if no dataset was created
      if (!data.dataset_created && data.extracted_text && previewDiv) {
        previewDiv.innerHTML = `
                    <div class="mt-3">
                        <h6>Extracted Text:</h6>
                        <div class="card">
                            <div class="card-body">
                                <pre class="small" style="max-height: 300px; overflow-y: auto;">${this.escapeHtml(
                                  data.extracted_text
                                )}</pre>
                            </div>
                        </div>
                    </div>
                `;
      }
    },

    downloadPdfDataset: function (taskId) {
      if (!taskId) {
        notifications.error("No task ID provided for download.");
        return;
      }

      // Trigger download
      const downloadUrl = `/api/download/${taskId}?format=json`;
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = `pdf-dataset-${taskId}.jsonl`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      notifications.success("Dataset download started.");
    },

    truncateText: function (text, maxLength) {
      if (!text) return "";
      if (text.length <= maxLength) return this.escapeHtml(text);
      return this.escapeHtml(text.substring(0, maxLength)) + "...";
    },

    escapeHtml: function (text) {
      if (!text) return "";
      const div = document.createElement("div");
      div.textContent = text;
      return div.innerHTML;
    },

    // RAG Management Functions
    setupRagEventListeners: function () {
      const manageRagBtn = document.getElementById("manageRagBtn");
      const refreshRagStatus = document.getElementById("refreshRagStatus");
      const clearRagBtn = document.getElementById("clearRagBtn");
      const ragSearchBtn = document.getElementById("ragSearchBtn");
      const ragSearchQuery = document.getElementById("ragSearchQuery");

      if (manageRagBtn) {
        manageRagBtn.addEventListener("click", () =>
          this.openRagManagementModal()
        );
      }

      if (refreshRagStatus) {
        refreshRagStatus.addEventListener("click", () => this.loadRagStatus());
      }

      if (clearRagBtn) {
        clearRagBtn.addEventListener("click", () => this.clearRagDocuments());
      }

      if (ragSearchBtn) {
        ragSearchBtn.addEventListener("click", () => this.searchRagDocuments());
      }

      if (ragSearchQuery) {
        ragSearchQuery.addEventListener("keypress", (e) => {
          if (e.key === "Enter") {
            this.searchRagDocuments();
          }
        });
      }
    },

    loadRagStatus: async function () {
      try {
        const response = await fetch("/api/rag/status");
        const data = await response.json();

        // Update main RAG panel
        const ragStatusIcon = document.getElementById("ragStatusIcon");
        const ragStatusMessage = document.getElementById("ragStatusMessage");
        const ragDocumentInfo = document.getElementById("ragDocumentInfo");

        if (ragStatusIcon && ragStatusMessage && ragDocumentInfo) {
          if (data.status === "active") {
            ragStatusIcon.className = "fas fa-circle text-success me-1";
            ragStatusMessage.textContent = "RAG system active";
            ragDocumentInfo.textContent = `${data.document_count} documents loaded`;
          } else {
            ragStatusIcon.className = "fas fa-circle text-secondary me-1";
            ragStatusMessage.textContent = "RAG system empty";
            ragDocumentInfo.textContent = "No PDF documents loaded";
          }
        }

        // Update modal status
        const ragModalStatus = document.getElementById("ragModalStatus");
        const ragModalDocCount = document.getElementById("ragModalDocCount");
        const ragModalIndexed = document.getElementById("ragModalIndexed");
        const ragDocumentsList = document.getElementById("ragDocumentsList");

        if (ragModalStatus) {
          ragModalStatus.className =
            data.status === "active"
              ? "badge bg-success"
              : "badge bg-secondary";
          ragModalStatus.textContent =
            data.status === "active" ? "Active" : "Empty";
        }

        if (ragModalDocCount) {
          ragModalDocCount.textContent = data.document_count;
        }

        if (ragModalIndexed) {
          ragModalIndexed.className = data.indexed
            ? "badge bg-success"
            : "badge bg-secondary";
          ragModalIndexed.textContent = data.indexed ? "Yes" : "No";
        }

        // Update documents list
        if (ragDocumentsList) {
          if (data.documents && data.documents.length > 0) {
            let documentsHtml = "";
            data.documents.forEach((doc) => {
              documentsHtml += `
                                <div class="border rounded p-2 mb-2">
                                    <div class="d-flex justify-content-between align-items-start">
                                        <div>
                                            <h6 class="mb-1">${
                                              doc.filename
                                            }</h6>
                                            <small class="text-muted">
                                                ${
                                                  doc.chunks
                                                } chunks • Uploaded: ${new Date(
                doc.upload_time
              ).toLocaleString()}
                                            </small>
                                        </div>
                                        <span class="badge bg-primary">${
                                          doc.id.split("_")[0]
                                        }</span>
                                    </div>
                                </div>
                            `;
            });
            ragDocumentsList.innerHTML = documentsHtml;
          } else {
            ragDocumentsList.innerHTML =
              '<p class="text-muted">No documents loaded</p>';
          }
        }
      } catch (error) {
        console.error("Error loading RAG status:", error);
        window.notifications.error("Failed to load RAG status");
      }
    },

    openRagManagementModal: function () {
      // Refresh status when opening modal
      this.loadRagStatus();

      // Use custom modal system if available, otherwise fallback to Bootstrap
      if (window.customModals && window.customModals.ragManagementModal) {
        window.customModals.ragManagementModal.show();
      } else {
        // Create custom modal for RAG management
        const modalElement = document.getElementById("ragManagementModal");
        if (modalElement) {
          const customModal = new CustomModal(modalElement);
          window.customModals = window.customModals || {};
          window.customModals.ragManagementModal = customModal;
          customModal.show();
        }
      }
    },

    searchRagDocuments: async function () {
      const queryInput = document.getElementById("ragSearchQuery");
      const resultsDiv = document.getElementById("ragSearchResults");

      if (!queryInput || !resultsDiv) return;

      const query = queryInput.value.trim();
      if (!query) {
        window.notifications.warning("Please enter a search query");
        return;
      }

      try {
        resultsDiv.innerHTML =
          '<div class="text-center"><i class="fas fa-spinner fa-spin"></i> Searching...</div>';

        const response = await fetch("/api/rag/search", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ query: query }),
        });

        const data = await response.json();

        if (data.results && data.results.length > 0) {
          let resultsHtml = `<h6 class="mb-3">Found ${data.result_count} results for "${query}"</h6>`;

          data.results.forEach((result, index) => {
            resultsHtml += `
                            <div class="border rounded p-3 mb-2">
                                <div class="d-flex justify-content-between align-items-start mb-2">
                                    <h6 class="mb-0">${result.filename}</h6>
                                    <span class="badge bg-primary">${(
                                      result.score * 100
                                    ).toFixed(1)}%</span>
                                </div>
                                <small class="text-muted">Chunk ${
                                  result.chunk_index + 1
                                }</small>
                                <p class="mb-0 mt-2">${result.text_preview}</p>
                            </div>
                        `;
          });

          resultsDiv.innerHTML = resultsHtml;
        } else {
          resultsDiv.innerHTML = '<p class="text-muted">No results found</p>';
        }
      } catch (error) {
        console.error("Error searching RAG documents:", error);
        resultsDiv.innerHTML =
          '<div class="alert alert-danger">Search failed</div>';
        window.notifications.error("Failed to search documents");
      }
    },

    clearRagDocuments: async function () {
      if (
        !confirm(
          "Are you sure you want to clear all RAG documents? This action cannot be undone."
        )
      ) {
        return;
      }

      try {
        const response = await fetch("/api/rag/clear", {
          method: "DELETE",
        });

        const data = await response.json();

        if (response.ok) {
          window.notifications.success(data.message);
          this.loadRagStatus(); // Refresh status

          // Clear search results
          const resultsDiv = document.getElementById("ragSearchResults");
          if (resultsDiv) {
            resultsDiv.innerHTML = "";
          }
        } else {
          throw new Error(data.detail || "Failed to clear documents");
        }
      } catch (error) {
        console.error("Error clearing RAG documents:", error);
        window.notifications.error("Failed to clear RAG documents");
      }
    },
  };
})();

// Enhanced Notification system with better display
window.notifications = {
  container: null,
  notifications: [],

  init: function () {
    this.container = document.getElementById("alertContainer");
    if (!this.container) {
      this.container = document.createElement("div");
      this.container.id = "alertContainer";
      document.body.appendChild(this.container);
    }
  },

  show: function (message, type = "info", duration = 5000) {
    if (!this.container) this.init();
    const icons = {
      success: '<i class="fas fa-check-circle notification-icon"></i>',
      error: '<i class="fas fa-times-circle notification-icon"></i>',
      warning:
        '<i class="fas fa-exclamation-triangle notification-icon"></i>',
      info: '<i class="fas fa-info-circle notification-icon"></i>',
    };
    const notif = document.createElement("div");
    notif.className = `notification notification-${type}`;
    notif.innerHTML = `
                ${icons[type] || icons.info}
                <span style="flex:1;">${message}</span>
                <button class="notification-close" aria-label="Close">&times;</button>
            `;
    // Close button
    notif.querySelector(".notification-close").onclick = () =>
      this._remove(notif);
    // Auto-remove after duration
    setTimeout(() => this._remove(notif), duration);
    this.container.appendChild(notif);
    // Animate exit on click (for accessibility)
    notif.addEventListener("click", (e) => {
      if (!e.target.classList.contains("notification-close"))
        this._remove(notif);
    });
  },

  _remove: function (notif) {
    if (!notif) return;
    notif.classList.add("notification-exit");
    setTimeout(() => {
      if (notif.parentNode) notif.parentNode.removeChild(notif);
    }, 350);
  },

  clear: function () {
    if (!this.container) return;
    this.container.innerHTML = "";
  },

  success: function (message, duration = 5000) {
    this.show(message, "success", duration);
  },
  error: function (message, duration = 8000) {
    this.show(message, "error", duration);
  },
  warning: function (message, duration = 6000) {
    this.show(message, "warning", duration);
  },
  info: function (message, duration = 5000) {
    this.show(message, "info", duration);
  },
}; // Initialize when DOM is loaded
document.addEventListener("DOMContentLoaded", function () {
  console.log("DOM loaded, initializing dataset generator...");
  window.datasetGenerator.init();
  window.notifications.init();

  // Initialize RAG event listeners
  initializeRagEventListeners();

  // Load initial RAG status
  if (window.ragManager) {
    window.ragManager.loadRagStatus();
  }

  // Add a test notification to verify system is working
  setTimeout(() => {
    console.log("DekDataset Web App fully loaded! 🚀");
  }, 1000);
});

// RAG Event Listeners
function initializeRagEventListeners() {
  // RAG Management Modal button
  const manageRagBtn = document.getElementById("manageRagBtn");
  if (manageRagBtn) {
    manageRagBtn.addEventListener("click", function () {
      if (window.ragManager) {
        window.ragManager.openRagManagementModal();
      }
    });
  }

  // RAG Search button
  const ragSearchBtn = document.getElementById("ragSearchBtn");
  if (ragSearchBtn) {
    ragSearchBtn.addEventListener("click", function () {
      if (window.ragManager) {
        window.ragManager.searchRagDocuments();
      }
    });
  }

  // RAG Search input (Enter key)
  const ragSearchQuery = document.getElementById("ragSearchQuery");
  if (ragSearchQuery) {
    ragSearchQuery.addEventListener("keypress", function (e) {
      if (e.key === "Enter") {
        e.preventDefault();
        if (window.ragManager) {
          window.ragManager.searchRagDocuments();
        }
      }
    });
  }

  // RAG Clear button
  const ragClearBtn = document.getElementById("ragClearBtn");
  if (ragClearBtn) {
    ragClearBtn.addEventListener("click", function () {
      if (window.ragManager) {
        window.ragManager.clearRagDocuments();
      }
    });
  }

  // RAG Status refresh button (if exists)
  const ragRefreshBtn = document.getElementById("ragRefreshBtn");
  if (ragRefreshBtn) {
    ragRefreshBtn.addEventListener("click", function () {
      if (window.ragManager) {
        window.ragManager.loadRagStatus();
      }
    });
  }
}
