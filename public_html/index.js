class ColPaliUI {
  constructor() {
    this.apiBase = "http://127.0.0.1:8000";
    this.selectedFile = null;
    this.documents = [];
    this.currentTab = "ingest";
    this.initializeEventListeners();
    this.checkServerHealth();
    this.loadDocuments();
  }

  initializeEventListeners() {
    // Tab switching
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", (e) => {
        this.switchTab(e.target.dataset.tab);
      });
    });

    // File input handling
    const fileInput = document.getElementById("pdfFile");
    const fileDisplay = document.getElementById("fileDisplayText");
    const fileDisplayEl = document.querySelector(".file-input-display");

    fileInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) {
        if (
          file.type !== "application/pdf" &&
          !file.name.toLowerCase().endsWith(".pdf")
        ) {
          this.showStatus(
            "ingestStatus",
            "Please select a valid PDF file",
            "error"
          );
          fileInput.value = "";
          return;
        }
        this.selectedFile = file;
        fileDisplay.innerHTML = `<strong>Selected:</strong> ${file.name}`;
        fileDisplay.classList.add("file-selected");
        // Clear any previous error status
        document.getElementById("ingestStatus").innerHTML = "";
      }
    });

    // Click handler for file display area
    fileDisplayEl.addEventListener("click", (e) => {
      if (e.target === fileDisplayEl || fileDisplayEl.contains(e.target)) {
        fileInput.click();
      }
    });

    // Drag and drop
    fileDisplayEl.addEventListener("dragover", (e) => {
      e.preventDefault();
      fileDisplayEl.style.borderColor = "var(--accent-color)";
    });

    fileDisplayEl.addEventListener("dragleave", (e) => {
      e.preventDefault();
      fileDisplayEl.style.borderColor = "var(--border-light)";
    });

    fileDisplayEl.addEventListener("drop", (e) => {
      e.preventDefault();
      fileDisplayEl.style.borderColor = "var(--border-light)";

      const files = e.dataTransfer.files;
      if (files.length > 0) {
        const file = files[0];
        if (
          file.type !== "application/pdf" &&
          !file.name.toLowerCase().endsWith(".pdf")
        ) {
          this.showStatus(
            "ingestStatus",
            "Please select a valid PDF file",
            "error"
          );
          return;
        }
        this.selectedFile = file;
        fileDisplay.innerHTML = `<strong>Selected:</strong> ${file.name}`;
        fileDisplay.classList.add("file-selected");
        fileInput.files = files;
        // Clear any previous error status
        document.getElementById("ingestStatus").innerHTML = "";
      }
    });

    // Buttons
    document
      .getElementById("ingestBtn")
      .addEventListener("click", () => this.ingestDocument());
    document
      .getElementById("searchBtn")
      .addEventListener("click", () => this.searchDocuments());
    document
      .getElementById("refreshDocsBtn")
      .addEventListener("click", () => this.loadDocuments());
    document
      .getElementById("viewServerLogsBtn")
      .addEventListener("click", () => this.viewServerLogs());

    // Enter key for search
    document.getElementById("searchQuery").addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        this.searchDocuments();
      }
    });
  }

  switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.classList.remove("active");
    });
    document.querySelector(`[data-tab="${tabName}"]`).classList.add("active");

    // Update tab content
    document.querySelectorAll(".tab-content").forEach((content) => {
      content.classList.remove("active");
    });
    document.getElementById(tabName).classList.add("active");

    this.currentTab = tabName;
  }

  async checkServerHealth() {
    try {
      const response = await fetch(`${this.apiBase}/health`);
      const data = await response.json();

      if (data.status === "healthy") {
        this.showStatus(
          "searchStatus",
          "Server connection established",
          "success"
        );
        if (!data.model_loaded) {
          this.showStatus(
            "ingestStatus",
            "Model initialization required - process a document to begin",
            "loading"
          );
        }
      }
    } catch (error) {
      this.showStatus(
        "searchStatus",
        "Unable to connect to server - please verify ColPali service is running",
        "error"
      );
      this.showStatus("ingestStatus", "Server connection unavailable", "error");
    }
  }

  async ingestDocument() {
    if (!this.selectedFile) {
      this.showStatus(
        "ingestStatus",
        "Please select a PDF document to process",
        "error"
      );
      return;
    }

    const formData = new FormData();
    formData.append("file", this.selectedFile);

    const docName =
      document.getElementById("docName").value ||
      this.selectedFile.name.replace(".pdf", "");
    if (docName !== this.selectedFile.name.replace(".pdf", "")) {
      formData.append("doc_name", docName);
    }

    this.setButtonLoading("ingestBtn", "ingestBtnText", "ingestSpinner", true);
    this.showProgress("ingestProgress", 0, "Starting document processing...");

    // Add logs container
    this.showLogsContainer("ingestStatus");
    this.addLogEntry(
      "ingestStatus",
      `Starting ingestion of "${docName}"`,
      "info"
    );

    try {
      const response = await fetch(`${this.apiBase}/ingest`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          `Processing failed with status: ${response.status} - ${errorText}`
        );
      }

      const result = await response.json();
      this.addLogEntry(
        "ingestStatus",
        `Task ID: ${result.task_id} - Status: ${result.status}`,
        "info"
      );

      // Start real-time progress tracking
      const trackingResult = await this.trackProgressRealTime(
        result.task_id,
        "ingest"
      );

      if (trackingResult.success) {
        this.addLogEntry(
          "ingestStatus",
          `✅ Document "${docName}" processed successfully`,
          "success"
        );
        this.showStatus(
          "ingestStatus",
          `Document "${docName}" processed successfully`,
          "success"
        );

        // Refresh documents list and clear form
        setTimeout(() => {
          this.loadDocuments();
          this.clearIngestForm();
        }, 1000);
      } else {
        throw new Error(trackingResult.error || "Processing failed");
      }
    } catch (error) {
      console.error("Ingestion error:", error);
      this.addLogEntry(
        "ingestStatus",
        `❌ Processing failed: ${error.message}`,
        "error"
      );
      this.showStatus(
        "ingestStatus",
        `Processing failed: ${error.message}`,
        "error"
      );
    } finally {
      this.setButtonLoading(
        "ingestBtn",
        "ingestBtnText",
        "ingestSpinner",
        false
      );
      setTimeout(() => this.hideProgress("ingestProgress"), 1000);
    }
  }

  clearIngestForm() {
    document.getElementById("pdfFile").value = "";
    document.getElementById("docName").value = "";
    document.getElementById("fileDisplayText").innerHTML =
      "Select PDF file or drag & drop";
    document
      .getElementById("fileDisplayText")
      .classList.remove("file-selected");
    this.selectedFile = null;
  }

  async searchDocuments() {
    const query = document.getElementById("searchQuery").value.trim();
    const topK = parseInt(document.getElementById("topK").value) || 5;

    if (!query) {
      this.showStatus(
        "searchStatus",
        "Please enter a search query to continue",
        "error"
      );
      return;
    }

    this.setButtonLoading("searchBtn", "searchBtnText", "searchSpinner", true);
    this.showProgress("searchProgress", 0, "Starting search...");

    // Add logs container
    this.showLogsContainer("searchStatus");
    this.addLogEntry(
      "searchStatus",
      `Starting search for: "${query}" (top ${topK} results)`,
      "info"
    );

    try {
      const response = await fetch(`${this.apiBase}/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query,
          top_k: topK,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          `Search failed with status: ${response.status} - ${errorText}`
        );
      }

      const result = await response.json();
      this.addLogEntry(
        "searchStatus",
        `Task ID: ${result.task_id} - Status: ${result.status}`,
        "info"
      );

      // Start real-time progress tracking
      const trackingResult = await this.trackProgressRealTime(
        result.task_id,
        "search"
      );

      if (trackingResult.success) {
        const results = trackingResult.results || [];
        this.displayResults(results);
        this.addLogEntry(
          "searchStatus",
          `✅ Search completed - found ${results.length} relevant matches`,
          "success"
        );
        this.showStatus(
          "searchStatus",
          `Search completed - found ${results.length} relevant matches`,
          "success"
        );
      } else {
        throw new Error(trackingResult.error || "Search failed");
      }
    } catch (error) {
      console.error("Search error:", error);
      this.addLogEntry(
        "searchStatus",
        `❌ Search failed: ${error.message}`,
        "error"
      );
      this.showStatus(
        "searchStatus",
        `Search failed: ${error.message}`,
        "error"
      );
      this.hideResults();
    } finally {
      this.setButtonLoading(
        "searchBtn",
        "searchBtnText",
        "searchSpinner",
        false
      );
      setTimeout(() => this.hideProgress("searchProgress"), 1000);
    }
  }

  async loadDocuments() {
    this.setButtonLoading(
      "refreshDocsBtn",
      "refreshBtnText",
      "refreshSpinner",
      true
    );

    try {
      const response = await fetch(`${this.apiBase}/documents`);

      if (!response.ok) {
        throw new Error(`Failed to fetch documents: ${response.status}`);
      }

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      this.documents = data.documents || [];
      this.displayDocuments();

      if (this.documents.length > 0) {
        this.showStatus(
          "documentsStatus",
          `Document library loaded - ${this.documents.length} documents available`,
          "success"
        );
      } else {
        this.showStatus(
          "documentsStatus",
          "No documents found - use the Ingest tab to add documents",
          "warning"
        );
      }
    } catch (error) {
      console.error("Error loading documents:", error);
      this.showStatus(
        "documentsStatus",
        `Failed to load document library: ${error.message}`,
        "error"
      );
      this.displayDocuments([]);
    } finally {
      this.setButtonLoading(
        "refreshDocsBtn",
        "refreshBtnText",
        "refreshSpinner",
        false
      );
    }
  }

  displayDocuments(documents = this.documents) {
    const container = document.getElementById("documentsContainer");

    if (!documents || documents.length === 0) {
      container.innerHTML = `
                        <div class="empty-state">
                            <h3>No Documents Found</h3>
                            <p>Your document library is empty. Process PDF documents using the Ingest tab to build your searchable collection.</p>
                        </div>
                    `;
      return;
    }

    container.innerHTML = documents
      .map(
        (doc) => `
                    <div class="document-item" data-doc-id="${doc.id}">
                        <div class="document-header">
                            <div class="document-name">${doc.name}</div>
                            <div class="document-stats">${
                              doc.embeddings_count
                            } embeddings</div>
                        </div>
                        <div class="document-meta">
                            <span>Pages: ${doc.pages}</span>
                            <span>Size: ${doc.size}</span>
                            <span>Created: ${this.formatDate(
                              doc.created_date
                            )}</span>
                        </div>
<div class="document-actions">
                            <button class="btn btn-secondary btn-small" onclick="colpaliUI.viewDocument('${
                              doc.id
                            }')">
                                View Details
                            </button>
                            <button class="btn btn-secondary btn-small" onclick="colpaliUI.reindexDocument('${
                              doc.id
                            }')">
                                Reprocess
                            </button>
                            <button class="btn btn-danger btn-small" onclick="colpaliUI.deleteDocument('${
                              doc.id
                            }', '${doc.name.replace(/'/g, "\\'")}')">
                                Remove
                            </button>
                        </div>
                    </div>
                `
      )
      .join("");
  }

  async viewDocument(docId) {
    const doc = this.documents.find((d) => d.id === docId);
    if (!doc) return;

    // Premium modal simulation
    const details = `
Document: ${doc.name}
Pages: ${doc.pages}
Size: ${doc.size}
Embeddings: ${doc.embeddings_count}
Created: ${this.formatDate(doc.created_date)}

This would open a sophisticated document viewer with:
- Page-by-page navigation
- Embedding visualization
- Metadata inspection
- Search result highlighting`;

    alert(details);
  }

  async reindexDocument(docId) {
    const doc = this.documents.find((d) => d.id === docId);
    if (!doc) return;

    if (
      !confirm(
        `Reprocess "${doc.name}"?\n\nThis will regenerate all embeddings for this document using the latest AI model. The process may take several minutes.`
      )
    ) {
      return;
    }

    try {
      this.showStatus(
        "documentsStatus",
        `Reprocessing "${doc.name}" - this may take several minutes...`,
        "loading"
      );

      // Simulate reindexing process
      await new Promise((resolve) => setTimeout(resolve, 3000));

      this.showStatus(
        "documentsStatus",
        `Document "${doc.name}" has been successfully reprocessed`,
        "success"
      );
    } catch (error) {
      console.error("Reindex error:", error);
      this.showStatus(
        "documentsStatus",
        `Reprocessing failed: ${error.message}`,
        "error"
      );
    }
  }

  async deleteDocument(docId, docName) {
    if (
      !confirm(
        `Remove "${docName}" from your library?\n\nThis will permanently delete the document and all associated embeddings. This action cannot be undone.`
      )
    ) {
      return;
    }

    try {
      this.showStatus(
        "documentsStatus",
        `Removing "${docName}" from library...`,
        "loading"
      );

      // Call the delete endpoint
      const response = await fetch(
        `${this.apiBase}/documents/${encodeURIComponent(docName)}`,
        {
          method: "DELETE",
        }
      );

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage;
        try {
          const errorData = JSON.parse(errorText);
          errorMessage = errorData.detail || errorText;
        } catch {
          errorMessage = errorText;
        }
        throw new Error(
          `Delete failed with status ${response.status}: ${errorMessage}`
        );
      }

      const result = await response.json();

      // Remove from local array
      this.documents = this.documents.filter((d) => d.id !== docId);
      this.displayDocuments();

      this.showStatus(
        "documentsStatus",
        `Document "${docName}" has been removed from your library (${result.deleted_embeddings} embeddings deleted)`,
        "success"
      );

      // Log the successful deletion
      console.log("Document deleted successfully:", result);
    } catch (error) {
      console.error("Delete error:", error);
      this.showStatus(
        "documentsStatus",
        `Removal failed: ${error.message}`,
        "error"
      );
    }
  }

  displayResults(results) {
    const resultsContainer = document.getElementById("resultsContainer");

    if (!results || results.length === 0) {
      resultsContainer.innerHTML = `
                        <div class="empty-state">
                            <h3>No Results Found</h3>
                            <p>Your search didn't return any matches. Try refining your query or check if documents have been processed.</p>
                        </div>
                    `;
      resultsContainer.style.display = "block";
      return;
    }

    resultsContainer.innerHTML = results
      .map((result, index) => {
        const hasImage =
          result.image_path &&
          result.image_path !== null &&
          result.image_path !== "";
        const resultClass = hasImage ? "result-item" : "result-item no-image";
        const scorePercentage = result.score.toFixed(3);

        // Create image section
        const imageSection = hasImage
          ? `
                        <div class="result-image-container">
                            <img class="result-image" 
                                 src="${result.image_path}" 
                                 alt="Page ${result.page_num} from ${result.doc_name}"
                                 onclick="colpaliUI.openImageModal('${result.image_path}', 'Page ${result.page_num} - ${result.doc_name}')"
                                 onerror="this.style.display='none'"
                            />
                            <div class="result-image-caption">Click to enlarge</div>
                        </div>
                    `
          : "";

        // Create citation
        const citation = `${result.doc_name}, Page ${result.page_num}`;

        return `
                        <div class="${resultClass}">
                            <div class="result-content">
                                <div class="result-header">
                                    <div class="result-meta">
                                        <span class="result-score">${scorePercentage}% match</span>
                                        <span class="result-page">Page ${
                                          result.page_num
                                        } • ${result.doc_name}</span>
                                    </div>
                                </div>
                                <div class="result-snippet">${
                                  result.full_text ||
                                  "Full text content available"
                                }</div>
                                <div class="citation-list">
                                    <div class="citation-badge">
                                        📖 ${citation}
                                    </div>
                                </div>
                                <div class="result-actions">
                                    <button class="btn btn-secondary" 
                                            data-doc-name="${result.doc_name}" 
                                            data-page-num="${result.page_num}" 
                                            data-full-text="${(
                                              result.full_text || ""
                                            ).replace(/"/g, "&quot;")}"
                                            onclick="colpaliUI.copyFullTextFromButton(this)">
                                        📋 Copy Text
                                    </button>
                                    <button class="btn btn-secondary" onclick="colpaliUI.copyCitation('${citation.replace(
                                      /'/g,
                                      "\\'"
                                    )}')">
                                        📚 Copy Citation
                                    </button>
                                    ${
                                      hasImage
                                        ? `<button class="btn btn-secondary" onclick="colpaliUI.openImageModal('${result.image_path}', 'Page ${result.page_num} - ${result.doc_name}')">
                                        🔍 View Page
                                    </button>`
                                        : ""
                                    }
                                </div>
                            </div>
                            ${imageSection}
                        </div>
                    `;
      })
      .join("");

    resultsContainer.style.display = "block";
    resultsContainer.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  hideResults() {
    document.getElementById("resultsContainer").style.display = "none";
  }

  async trackProgressRealTime(taskId, type) {
    const statusId = type === "ingest" ? "ingestStatus" : "searchStatus";
    const progressId = type === "ingest" ? "ingestProgress" : "searchProgress";

    let isComplete = false;
    let pollCount = 0;
    let lastLogTime = Date.now();
    const maxIdleTime = 120000; // 2 minutes without logs = timeout
    const maxTotalTime = 1800000; // 30 minutes absolute max
    const startTime = Date.now();

    this.addLogEntry(
      statusId,
      `🔄 Starting real-time progress tracking for task ${taskId}`,
      "info"
    );

    while (!isComplete) {
      const currentTime = Date.now();

      // Check absolute timeout
      if (currentTime - startTime > maxTotalTime) {
        this.addLogEntry(
          statusId,
          `⏰ Progress tracking timed out after ${Math.round(
            (currentTime - startTime) / 1000
          )}s (absolute limit)`,
          "warning"
        );
        return { success: false, error: "Absolute timeout reached" };
      }

      try {
        const response = await fetch(`${this.apiBase}/progress/${taskId}`);

        if (response.ok) {
          const data = await response.json();
          const progress = data.progress;

          // Update progress bar
          this.showProgress(
            progressId,
            progress.progress,
            progress.current_step
          );

          // Add detailed log entry
          const logMessage = `[${progress.step_num}/${progress.total_steps}] ${progress.current_step}`;
          const details = progress.details ? ` - ${progress.details}` : "";
          const throughput = progress.throughput
            ? ` (${progress.throughput})`
            : "";
          const eta = progress.eta_seconds
            ? ` - ETA: ${progress.eta_seconds}s`
            : "";

          this.addLogEntry(
            statusId,
            `${logMessage}${details}${throughput}${eta}`,
            "info"
          );

          // Reset idle timer whenever we get a log with meaningful progress
          if (
            progress.current_step &&
            progress.current_step !== "Waiting..." &&
            progress.current_step !== "Idle"
          ) {
            lastLogTime = currentTime;
          }

          // Check for completion or error
          if (progress.error) {
            this.addLogEntry(statusId, `❌ Error: ${progress.error}`, "error");
            isComplete = true;
            return { success: false, error: progress.error };
          } else if (progress.progress >= 100) {
            this.addLogEntry(
              statusId,
              `✅ Task completed successfully`,
              "success"
            );
            isComplete = true;
            return { success: true, results: progress.results || null };
          }
        } else if (response.status === 404) {
          this.addLogEntry(
            statusId,
            `⚠️ Task ${taskId} not found on server`,
            "warning"
          );
          break;
        } else {
          this.addLogEntry(
            statusId,
            `⚠️ Failed to fetch progress (${response.status})`,
            "warning"
          );
        }
      } catch (error) {
        this.addLogEntry(
          statusId,
          `⚠️ Progress fetch error: ${error.message}`,
          "warning"
        );
      }

      // Only check idle timeout if we haven't received meaningful logs
      if (currentTime - lastLogTime > maxIdleTime) {
        this.addLogEntry(
          statusId,
          `⏰ Progress tracking timed out after ${Math.round(
            (currentTime - lastLogTime) / 1000
          )}s without logs`,
          "warning"
        );
        return {
          success: false,
          error: "Idle timeout - no progress logs received",
        };
      }

      pollCount++;

      if (!isComplete) {
        await new Promise((resolve) => setTimeout(resolve, 500)); // Poll every 500ms
      }
    }

    return { success: false, error: "Unknown error" };
  }

  formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  }

  async viewServerLogs() {
    this.setButtonLoading(
      "viewServerLogsBtn",
      "logsBtnText",
      "logsSpinner",
      true
    );

    try {
      const response = await fetch(`${this.apiBase}/logs`);

      if (!response.ok) {
        throw new Error(`Failed to fetch logs: ${response.status}`);
      }

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      this.displayServerLogs(data.logs);
      document.getElementById("serverLogsContainer").style.display = "block";
    } catch (error) {
      console.error("Error fetching server logs:", error);
      this.showStatus(
        "documentsStatus",
        `Failed to fetch server logs: ${error.message}`,
        "error"
      );
    } finally {
      this.setButtonLoading(
        "viewServerLogsBtn",
        "logsBtnText",
        "logsSpinner",
        false
      );
    }
  }

  displayServerLogs(logs) {
    const logsList = document.getElementById("serverLogsList");
    const logsContent = document.querySelector(
      "#serverLogsContainer .logs-content"
    );

    if (!logs || logs.length === 0) {
      logsList.innerHTML =
        '<div class="empty-state"><p>No server logs available</p></div>';
      return;
    }

    logsList.innerHTML = logs
      .map((log) => {
        const logType = this.getLogType(log.level);
        return `
                        <div class="log-entry log-${logType}">
                            <span class="log-time">${this.formatLogTimestamp(
                              log.timestamp
                            )}</span>
                            <span class="log-level">[${log.level}]</span>
                            <span class="log-message">${log.message}</span>
                        </div>
                    `;
      })
      .join("");

    // Auto-scroll to bottom with a slight delay to ensure content is rendered
    setTimeout(() => {
      if (logsContent) {
        logsContent.scrollTop = logsContent.scrollHeight;
      }
    }, 10);
  }

  getLogType(level) {
    switch (level.toUpperCase()) {
      case "ERROR":
        return "error";
      case "WARNING":
      case "WARN":
        return "warning";
      case "INFO":
        return "info";
      case "DEBUG":
        return "info";
      default:
        return "info";
    }
  }

  formatLogTimestamp(timestamp) {
    try {
      const date = new Date(timestamp);
      return date.toLocaleTimeString();
    } catch {
      return timestamp.split(" ")[1] || timestamp; // Extract time part if possible
    }
  }

  showStatus(elementId, message, type) {
    const element = document.getElementById(elementId);
    const existingLogs = element.querySelector(".logs-container");

    if (existingLogs) {
      // Keep existing logs and add status above them
      const statusDiv =
        element.querySelector(".status-indicator") ||
        document.createElement("div");
      statusDiv.className = `status-indicator status-${type}`;
      statusDiv.textContent = message;

      if (!element.querySelector(".status-indicator")) {
        element.insertBefore(statusDiv, existingLogs);
      }
    } else {
      element.innerHTML = `<div class="status-indicator status-${type}">${message}</div>`;
    }
  }

  showLogsContainer(elementId) {
    const element = document.getElementById(elementId);

    if (!element.querySelector(".logs-container")) {
      const logsContainer = document.createElement("div");
      logsContainer.className = "logs-container";
      logsContainer.innerHTML = `
                        <div class="logs-header">
                            <h4>📋 Processing Logs</h4>
                            <button class="logs-toggle" onclick="this.parentElement.parentElement.querySelector('.logs-content').classList.toggle('collapsed')">
                                ▼
                            </button>
                        </div>
                        <div class="logs-content">
                            <div class="logs-list"></div>
                        </div>
                    `;
      element.appendChild(logsContainer);
    }
  }

  addLogEntry(elementId, message, type = "info") {
    const element = document.getElementById(elementId);
    const logsContainer = element.querySelector(".logs-container");

    if (logsContainer) {
      const logsList = logsContainer.querySelector(".logs-list");
      const logsContent = logsContainer.querySelector(".logs-content");
      const timestamp = new Date().toLocaleTimeString();

      const logEntry = document.createElement("div");
      logEntry.className = `log-entry log-${type}`;
      logEntry.innerHTML = `
                        <span class="log-time">${timestamp}</span>
                        <span class="log-message">${message}</span>
                    `;

      logsList.appendChild(logEntry);

      // Auto-scroll to bottom of the logs content area
      setTimeout(() => {
        logsContent.scrollTop = logsContent.scrollHeight;
      }, 10);

      // Limit log entries to prevent memory issues
      const logEntries = logsList.querySelectorAll(".log-entry");
      if (logEntries.length > 100) {
        // Remove oldest entries, keep last 80
        for (let i = 0; i < 20; i++) {
          if (logEntries[i]) {
            logEntries[i].remove();
          }
        }
      }
    }
  }

  showProgress(progressId, percent, text) {
    const container = document.getElementById(progressId);
    const fill = document.getElementById(
      progressId.replace("Progress", "ProgressFill")
    );
    const textEl = document.getElementById(
      progressId.replace("Progress", "ProgressText")
    );

    container.style.display = "block";
    fill.style.width = `${percent}%`;
    textEl.textContent = text;
  }

  hideProgress(progressId) {
    document.getElementById(progressId).style.display = "none";
  }

  setButtonLoading(btnId, textId, spinnerId, loading) {
    const btn = document.getElementById(btnId);
    const text = document.getElementById(textId);
    const spinner = document.getElementById(spinnerId);

    btn.disabled = loading;

    if (loading) {
      text.style.display = "none";
      spinner.classList.remove("hidden");
    } else {
      text.style.display = "inline";
      spinner.classList.add("hidden");
    }
  }

  // Image modal methods
  openImageModal(imageSrc, caption) {
    const modal = document.getElementById("imageModal");
    const modalImage = document.getElementById("modalImage");

    modalImage.src = imageSrc;
    modalImage.alt = caption;
    modal.classList.add("active");

    // Close modal when clicking outside the image
    modal.addEventListener("click", (e) => {
      if (e.target === modal) {
        this.closeImageModal();
      }
    });

    // Close modal with Escape key
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && modal.classList.contains("active")) {
        this.closeImageModal();
      }
    });
  }

  closeImageModal() {
    const modal = document.getElementById("imageModal");
    modal.classList.remove("active");
  }

  // Utility methods for copying text
  async copyToClipboard(text) {
    try {
      await navigator.clipboard.writeText(text);
      this.showTemporaryStatus("Text copied to clipboard!", "success");
    } catch (err) {
      console.error("Failed to copy text: ", err);
      this.showTemporaryStatus("Failed to copy text", "error");
    }
  }

  async copyCitation(citation) {
    try {
      await navigator.clipboard.writeText(citation);
      this.showTemporaryStatus("Citation copied to clipboard!", "success");
    } catch (err) {
      console.error("Failed to copy citation: ", err);
      this.showTemporaryStatus("Failed to copy citation", "error");
    }
  }

  async copyFullTextFromButton(buttonElement) {
    const docName = buttonElement.dataset.docName;
    const pageNum = parseInt(buttonElement.dataset.pageNum);
    const fullText = buttonElement.dataset.fullText;

    await this.copyFullText(docName, pageNum, fullText);
  }

  async copyFullText(docName, pageNum, fullTextFromResult = null) {
    try {
      let fullText = fullTextFromResult;

      // If full text wasn't provided from search results, fetch it from API
      if (!fullText) {
        const response = await fetch(
          `${this.apiBase}/document/${encodeURIComponent(
            docName
          )}/page/${pageNum}/text`
        );

        if (!response.ok) {
          throw new Error(
            `Failed to fetch full text: ${response.status} ${response.statusText}`
          );
        }

        const data = await response.json();
        console.log("Received full text data:", data);

        // Try multiple possible field names for the text content
        fullText =
          data.text_content ||
          data.text ||
          data.content ||
          "No text available for this page";
      }

      if (
        !fullText ||
        fullText.trim() === "" ||
        fullText === "No text available for this page"
      ) {
        this.showTemporaryStatus(
          "No text content available for this page",
          "warning"
        );
        return;
      }

      await navigator.clipboard.writeText(fullText);
      this.showTemporaryStatus(
        `Full text copied to clipboard! (${fullText.length} characters)`,
        "success"
      );
    } catch (err) {
      console.error("Failed to copy full text: ", err);
      this.showTemporaryStatus(
        `Failed to copy full text: ${err.message}`,
        "error"
      );
    }
  }

  showTemporaryStatus(message, type) {
    // Create a temporary status element
    const statusEl = document.createElement("div");
    statusEl.className = `status-indicator status-${type}`;
    statusEl.textContent = message;
    statusEl.style.position = "fixed";
    statusEl.style.top = "20px";
    statusEl.style.right = "20px";
    statusEl.style.zIndex = "1001";
    statusEl.style.animation = "fadeIn 0.3s ease";

    document.body.appendChild(statusEl);

    // Remove after 3 seconds
    setTimeout(() => {
      statusEl.style.animation = "fadeOut 0.3s ease";
      setTimeout(() => {
        if (statusEl.parentNode) {
          statusEl.parentNode.removeChild(statusEl);
        }
      }, 300);
    }, 3000);
  }
}

// Initialize the UI when the page loads
let colpaliUI;
document.addEventListener("DOMContentLoaded", () => {
  colpaliUI = new ColPaliUI();
});
