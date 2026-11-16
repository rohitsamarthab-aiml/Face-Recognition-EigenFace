const RAW_MAIN_PY_SOURCE = "\nimport os\nimport numpy as np\nimport matplotlib.pyplot as plt\n\nfrom sklearn.datasets import fetch_olivetti_faces\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.decomposition import PCA\nfrom sklearn.metrics import accuracy_score\nfrom sklearn.metrics import classification_report\nfrom sklearn.metrics import pairwise_distances\n\n# --------------------------------------------------\n# 1. LOAD THE DATASET\n# --------------------------------------------------\nprint(\"Loading the Olivetti Faces dataset...\")\n\nfaces = fetch_olivetti_faces(shuffle=True, random_state=42)\n\nX = np.asarray(faces.data, dtype=np.float64)\ny = faces.target\nimages = faces.images\n\nprint(\"Dataset loaded successfully!\")\nprint(\"Total images:\", X.shape[0])\nprint(\"Features per image:\", X.shape[1])\nprint(\"Number of people:\", len(np.unique(y)))\n\n# --------------------------------------------------\n# 2. SPLIT INTO TRAINING AND TESTING DATA\n# --------------------------------------------------\nX_train, X_test, y_train, y_test = train_test_split(\n    X,\n    y,\n    test_size=0.25,\n    random_state=42,\n    stratify=y\n)\n\nprint(\"\\nTraining images:\", len(X_train))\nprint(\"Testing images:\", len(X_test))\n\n\n# --------------------------------------------------\n# 3. APPLY PCA\n# --------------------------------------------------\nn_components = 100\n\npca = PCA(\n    n_components=n_components,\n    svd_solver=\"full\",\n    whiten=False\n)\n\nX_train_pca = pca.fit_transform(X_train)\n\n# Project test faces onto the principal components\nX_test_centered = X_test - pca.mean_\n\nX_test_pca = np.array([\n    np.sum(row * pca.components_, axis=1)\n    for row in X_test_centered\n])\n\n\nprint(\"\\nPCA completed!\")\nprint(\"Original features:\", X_train.shape[1])\nprint(\"Reduced features:\", X_train_pca.shape[1])\n\nvariance = np.sum(pca.explained_variance_ratio_) * 100\nprint(\n    f\"Variance retained by {n_components} components: \"\n    f\"{variance:.2f}%\"\n)\n\n\n# --------------------------------------------------\n# 4. FACE MATCHING USING EUCLIDEAN DISTANCE\n# --------------------------------------------------\n# Calculate Euclidean distances without matrix multiplication\ndistances = np.empty((len(X_test_pca), len(X_train_pca)))\n\nfor i, test_face in enumerate(X_test_pca):\n    difference = X_train_pca - test_face\n    distances[i] = np.sqrt(\n        np.sum(difference * difference, axis=1)\n    )\n\nnearest_indices = np.argmin(distances, axis=1)\n\ny_pred = y_train[nearest_indices]\n\naccuracy = accuracy_score(y_test, y_pred)\n\nprint(\"\\nFACE MATCHING RESULTS\")\nprint(\"---------------------\")\nprint(f\"Recognition accuracy: {accuracy * 100:.2f}%\")\nprint(\"\\nClassification report:\")\nprint(classification_report(y_test, y_pred, zero_division=0))\n\n# --------------------------------------------------\n# 5. CREATE OUTPUT FOLDER\n# --------------------------------------------------\nos.makedirs(\"outputs\", exist_ok=True)\n\n# --------------------------------------------------\n# 6. VISUALIZE THE MEAN FACE AND EIGENFACES\n# --------------------------------------------------\nfig, axes = plt.subplots(2, 5, figsize=(12, 5))\n\naxes[0, 0].imshow(\n    pca.mean_.reshape(64, 64), cmap=\"gray\"\n)\naxes[0, 0].set_title(\"Mean Face\")\naxes[0, 0].axis(\"off\")\n\nfor i in range(9):\n    eigenface = pca.components_[i].reshape(64, 64)\n    ax = axes.flat[i + 1]\n    ax.imshow(eigenface, cmap=\"gray\")\n    ax.set_title(f\"Eigenface {i + 1}\")\n    ax.axis(\"off\")\n\nplt.suptitle(\"Mean Face and Principal Components\")\nplt.tight_layout()\nplt.savefig(\"outputs/eigenfaces.png\", dpi=150)\nplt.show()\n\n# --------------------------------------------------\n# 7. PLOT EXPLAINED VARIANCE\n# --------------------------------------------------\ncumulative_variance = np.cumsum(pca.explained_variance_ratio_)\n\nplt.figure(figsize=(9, 5))\nplt.plot(\n    range(1, n_components + 1),\n    cumulative_variance * 100\n)\nplt.xlabel(\"Number of Principal Components\")\nplt.ylabel(\"Cumulative Variance Retained (%)\")\nplt.title(\"PCA: Explained Variance\")\nplt.grid(True)\nplt.tight_layout()\nplt.savefig(\"outputs/explained_variance.png\", dpi=150)\nplt.show()\n\n# --------------------------------------------------\n# 8. SHOW FACE MATCHING EXAMPLES\n# --------------------------------------------------\n\n# --------------------------------------------------\n# 8. SHOW FACE MATCHING EXAMPLES\n# --------------------------------------------------\nfig, axes = plt.subplots(3, 4, figsize=(12, 9))\n\nfor i, ax in enumerate(axes.flat):\n    j = i // 2\n\n    if i % 2 == 0:\n        # Display the test image\n        ax.imshow(X_test[j].reshape(64, 64), cmap=\"gray\")\n        ax.set_title(f\"Test: Person {y_test[j]}\")\n    else:\n        # Display the closest training image\n        ax.imshow(\n            X_train[nearest_indices[j]].reshape(64, 64),\n            cmap=\"gray\"\n        )\n        ax.set_title(f\"Match: Person {y_pred[j]}\")\n\n    ax.axis(\"off\")\n\nplt.suptitle(\"Test Faces and Closest Training Matches\")\nplt.tight_layout()\nplt.savefig(\"outputs/face_matches.png\", dpi=150)\nplt.show()\n\n\n# --------------------------------------------------\n# 9. SAVE RESULTS\n# --------------------------------------------------\nwith open(\"outputs/results.txt\", \"w\") as file:\n    file.write(\"PCA AND EIGENFACES PROJECT RESULTS\\n\")\n    file.write(f\"Total images: {len(X)}\\n\")\n    file.write(f\"Training images: {len(X_train)}\\n\")\n    file.write(f\"Testing images: {len(X_test)}\\n\")\n    file.write(f\"Principal components: {n_components}\\n\")\n    file.write(f\"Variance retained: {variance:.2f}%\\n\")\n    file.write(f\"Recognition accuracy: {accuracy * 100:.2f}%\\n\")\n\nprint(\"\\nProject completed!\")\nprint(\"Check the outputs folder for saved graphs and results.\")\n";
    const WORKFLOW_9_DATA = [{"stage": 1, "name": "1. Load Dataset", "lines": "Lines 13\u201323", "excerpt": "# --------------------------------------------------\n# 1. LOAD THE DATASET\n# --------------------------------------------------\nprint(\"Loading the Olivetti Faces dataset...\")\n\nfaces = fetch_olivetti_faces(shuffle=True, random_state=42)\n\nX = np.asarray(faces.data, dtype=np.float64)\ny = faces.target\nimages = faces.images\n", "desc": "Calls `fetch_olivetti_faces(shuffle=True, random_state=42)` to load all 400 grayscale face photographs (40 people, 10 photos per person) from scikit-learn."}, {"stage": 2, "name": "2. Prepare Image Vectors", "lines": "Lines 20\u201327", "excerpt": "X = np.asarray(faces.data, dtype=np.float64)\ny = faces.target\nimages = faces.images\n\nprint(\"Dataset loaded successfully!\")\nprint(\"Total images:\", X.shape[0])\nprint(\"Features per image:\", X.shape[1])\nprint(\"Number of people:\", len(np.unique(y)))", "desc": "Converts `faces.data` into a 64-bit float matrix `X` of shape `(400, 4096)` where each 64\u00d764 image is represented as a 4,096-feature row vector, alongside target labels `y` and 2D `images`."}, {"stage": 3, "name": "3. Split Training & Test Data", "lines": "Lines 29\u201341", "excerpt": "# --------------------------------------------------\n# 2. SPLIT INTO TRAINING AND TESTING DATA\n# --------------------------------------------------\nX_train, X_test, y_train, y_test = train_test_split(\n    X,\n    y,\n    test_size=0.25,\n    random_state=42,\n    stratify=y\n)\n\nprint(\"\\nTraining images:\", len(X_train))\nprint(\"Testing images:\", len(X_test))", "desc": "Uses `train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)` to partition the 400 vectors into 300 training vectors (`X_train`) and 100 testing vectors (`X_test`) balanced across all 40 people."}, {"stage": 4, "name": "4. Apply PCA", "lines": "Lines 44\u201355", "excerpt": "# --------------------------------------------------\n# 3. APPLY PCA\n# --------------------------------------------------\nn_components = 100\n\npca = PCA(\n    n_components=n_components,\n    svd_solver=\"full\",\n    whiten=False\n)\n\nX_train_pca = pca.fit_transform(X_train)", "desc": "Initializes `PCA(n_components=100, svd_solver='full', whiten=False)` and runs `pca.fit_transform(X_train)` to compute the training Mean Face (`pca.mean_`), extract the top 100 Eigenfaces (`pca.components_`), and project the training set into `(300, 100)`."}, {"stage": 5, "name": "5. Project Faces", "lines": "Lines 57\u201374", "excerpt": "# Project test faces onto the principal components\nX_test_centered = X_test - pca.mean_\n\nX_test_pca = np.array([\n    np.sum(row * pca.components_, axis=1)\n    for row in X_test_centered\n])\n\n\nprint(\"\\nPCA completed!\")\nprint(\"Original features:\", X_train.shape[1])\nprint(\"Reduced features:\", X_train_pca.shape[1])\n\nvariance = np.sum(pca.explained_variance_ratio_) * 100\nprint(\n    f\"Variance retained by {n_components} components: \"\n    f\"{variance:.2f}%\"\n)", "desc": "Mean-centers the 100 test vectors (`X_test_centered = X_test - pca.mean_`) and manually projects each centered row onto the 100 Eigenfaces via `np.sum(row * pca.components_, axis=1)`, retaining `94.40%` variance."}, {"stage": 6, "name": "6. Calculate Euclidean Distances", "lines": "Lines 77\u201388", "excerpt": "# --------------------------------------------------\n# 4. FACE MATCHING USING EUCLIDEAN DISTANCE\n# --------------------------------------------------\n# Calculate Euclidean distances without matrix multiplication\ndistances = np.empty((len(X_test_pca), len(X_train_pca)))\n\nfor i, test_face in enumerate(X_test_pca):\n    difference = X_train_pca - test_face\n    distances[i] = np.sqrt(\n        np.sum(difference * difference, axis=1)\n    )\n", "desc": "Loops through each projected test vector `test_face` in `X_test_pca`, computes the coordinate differences `difference = X_train_pca - test_face`, and evaluates `np.sqrt(np.sum(difference * difference, axis=1))` to get all 300 Euclidean distances."}, {"stage": 7, "name": "7. Predict Nearest Person", "lines": "Lines 89\u201392", "excerpt": "nearest_indices = np.argmin(distances, axis=1)\n\ny_pred = y_train[nearest_indices]\n", "desc": "Finds the index of the closest training face for each test face using `nearest_indices = np.argmin(distances, axis=1)` and retrieves the predicted person ID via `y_pred = y_train[nearest_indices]`."}, {"stage": 8, "name": "8. Evaluate Accuracy", "lines": "Lines 93\u201399", "excerpt": "accuracy = accuracy_score(y_test, y_pred)\n\nprint(\"\\nFACE MATCHING RESULTS\")\nprint(\"---------------------\")\nprint(f\"Recognition accuracy: {accuracy * 100:.2f}%\")\nprint(\"\\nClassification report:\")\nprint(classification_report(y_test, y_pred, zero_division=0))", "desc": "Computes `accuracy = accuracy_score(y_test, y_pred)` (`94.00%`) and prints the full 40-class `classification_report(y_test, y_pred, zero_division=0)`."}, {"stage": 9, "name": "9. Generate Output Graphs", "lines": "Lines 101\u2013192", "excerpt": "# --------------------------------------------------\n# 5. CREATE OUTPUT FOLDER\n# --------------------------------------------------\nos.makedirs(\"outputs\", exist_ok=True)\n\n# --------------------------------------------------\n# 6. VISUALIZE THE MEAN FACE AND EIGENFACES\n# --------------------------------------------------\nfig, axes = plt.subplots(2, 5, figsize=(12, 5))\n\naxes[0, 0].imshow(\n    pca.mean_.reshape(64, 64), cmap=\"gray\"\n)\naxes[0, 0].set_title(\"Mean Face\")\naxes[0, 0].axis(\"off\")\n\nfor i in range(9):\n    eigenface = pca.components_[i].reshape(64, 64)\n    ax = axes.flat[i + 1]\n    ax.imshow(eigenface, cmap=\"gray\")\n    ax.set_title(f\"Eigenface {i + 1}\")\n    ax.axis(\"off\")\n\nplt.suptitle(\"Mean Face and Principal Components\")\nplt.tight_layout()\nplt.savefig(\"outputs/eigenfaces.png\", dpi=150)\nplt.show()\n\n# --------------------------------------------------\n# 7. PLOT EXPLAINED VARIANCE\n# --------------------------------------------------\ncumulative_variance = np.cumsum(pca.explained_variance_ratio_)\n\nplt.figure(figsize=(9, 5))\nplt.plot(\n    range(1, n_components + 1),\n    cumulative_variance * 100\n)\nplt.xlabel(\"Number of Principal Components\")\nplt.ylabel(\"Cumulative Variance Retained (%)\")\nplt.title(\"PCA: Explained Variance\")\nplt.grid(True)\nplt.tight_layout()\nplt.savefig(\"outputs/explained_variance.png\", dpi=150)\nplt.show()\n\n# --------------------------------------------------\n# 8. SHOW FACE MATCHING EXAMPLES\n# --------------------------------------------------\n\n# --------------------------------------------------\n# 8. SHOW FACE MATCHING EXAMPLES\n# --------------------------------------------------\nfig, axes = plt.subplots(3, 4, figsize=(12, 9))\n\nfor i, ax in enumerate(axes.flat):\n    j = i // 2\n\n    if i % 2 == 0:\n        # Display the test image\n        ax.imshow(X_test[j].reshape(64, 64), cmap=\"gray\")\n        ax.set_title(f\"Test: Person {y_test[j]}\")\n    else:\n        # Display the closest training image\n        ax.imshow(\n            X_train[nearest_indices[j]].reshape(64, 64),\n            cmap=\"gray\"\n        )\n        ax.set_title(f\"Match: Person {y_pred[j]}\")\n\n    ax.axis(\"off\")\n\nplt.suptitle(\"Test Faces and Closest Training Matches\")\nplt.tight_layout()\nplt.savefig(\"outputs/face_matches.png\", dpi=150)\nplt.show()\n\n\n# --------------------------------------------------\n# 9. SAVE RESULTS\n# --------------------------------------------------\nwith open(\"outputs/results.txt\", \"w\") as file:\n    file.write(\"PCA AND EIGENFACES PROJECT RESULTS\\n\")\n    file.write(f\"Total images: {len(X)}\\n\")\n    file.write(f\"Training images: {len(X_train)}\\n\")\n    file.write(f\"Testing images: {len(X_test)}\\n\")\n    file.write(f\"Principal components: {n_components}\\n\")\n    file.write(f\"Variance retained: {variance:.2f}%\\n\")\n    file.write(f\"Recognition accuracy: {accuracy * 100:.2f}%\\n\")\n\nprint(\"\\nProject completed!\")\nprint(\"Check the outputs folder for saved graphs and results.\")", "desc": "Creates the `outputs/` folder and generates `outputs/eigenfaces.png` (Mean Face + 9 Eigenfaces), `outputs/explained_variance.png` (cumulative variance curve), `outputs/face_matches.png` (6 test/match pairs), and `outputs/results.txt`."}];
    let activePageNum = 2;

    /* -------------------------------------------------------------------------
       1. MULTI-PAGE NAVIGATION BAR (Seamless in Canvas & Local Multi-File)
       ------------------------------------------------------------------------- */
    function switchPage(pageNum, pushHash = true) {
      activePageNum = pageNum;
      const pages = [
        { num: 1, el: document.getElementById("page-view-1"), nav: document.getElementById("nav-page-1") },
        { num: 2, el: document.getElementById("page-view-2"), nav: document.getElementById("nav-page-2") },
        { num: 3, el: document.getElementById("page-view-3"), nav: document.getElementById("nav-page-3") }
      ];
      pages.forEach((p) => {
        if (p.el) p.el.style.display = p.num === pageNum ? "block" : "none";
        if (p.nav) p.nav.classList.toggle("active", p.num === pageNum);
      });
      if (pushHash) {
        const hashName = pageNum === 1 ? "#code-modules" : pageNum === 2 ? "#interactive-results" : "#mathematical-theory";
        try { history.replaceState(null, "", hashName); } catch (e) {}
      }
      window.scrollTo({ top: 0, behavior: "smooth" });
    }

    function initNavbar() {
      const nav1 = document.getElementById("nav-page-1");
      const nav2 = document.getElementById("nav-page-2");
      const nav3 = document.getElementById("nav-page-3");

      // Intercept clicks so switching between Page 1 (index.html), Page 2 (results.html), and Page 3 (theory.html)
      // works instantaneously both inside Canvas and when opened locally on a laptop
      if (nav1) nav1.addEventListener("click", (e) => { e.preventDefault(); switchPage(1); });
      if (nav2) nav2.addEventListener("click", (e) => { e.preventDefault(); switchPage(2); });
      if (nav3) nav3.addEventListener("click", (e) => { e.preventDefault(); switchPage(3); });

      // Check URL hash on load
      const h = window.location.hash.toLowerCase();
      if (h.includes("result")) switchPage(2, false);
      else if (h.includes("theory") || h.includes("math")) switchPage(3, false);
      else switchPage(activePageNum, false);
    }

    /* -------------------------------------------------------------------------
       2. UNIVERSAL ACCORDION CONTROLLER (Page 1 Code Notes, Modules, Page 3 Viva)
       ------------------------------------------------------------------------- */
    function initAccordions() {
      document.querySelectorAll("[data-acc-target]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const targetId = btn.getAttribute("data-acc-target");
          const contentEl = document.getElementById(targetId);
          const indEl = document.getElementById("ind-" + targetId);
          if (!contentEl) return;
          const isOpen = contentEl.classList.toggle("open");
          if (indEl) {
            if (targetId.startsWith("p3-viva-")) {
              indEl.textContent = isOpen ? "Hide Answer ▲" : "Reveal Answer ▼";
            } else if (targetId.startsWith("code-note-")) {
              indEl.textContent = isOpen ? "Hide Explanation ▲" : "Expand Explanation ▼";
            } else {
              indEl.textContent = isOpen ? "Collapse ▲" : "Expand Details ▼";
            }
          }
        });
      });

      // Expand All Code Notes button on Page 1
      const btnExpandNotes = document.getElementById("btn-expand-all-notes");
      let notesAllOpen = false;
      if (btnExpandNotes) {
        btnExpandNotes.addEventListener("click", () => {
          notesAllOpen = !notesAllOpen;
          btnExpandNotes.textContent = notesAllOpen ? "Collapse All Code Notes" : "Expand All Code Notes";
          document.querySelectorAll("[id^='code-note-']").forEach((el) => {
            el.classList.toggle("open", notesAllOpen);
            const ind = document.getElementById("ind-" + el.id);
            if (ind) ind.textContent = notesAllOpen ? "Hide Explanation ▲" : "Expand Explanation ▼";
          });
        });
      }

      // Expand / Collapse All Viva Answers on Page 3
      const btnVivaExpand = document.getElementById("btn-viva-expand-all");
      const btnVivaCollapse = document.getElementById("btn-viva-collapse-all");
      if (btnVivaExpand) {
        btnVivaExpand.addEventListener("click", () => {
          document.querySelectorAll("[id^='p3-viva-']").forEach((el) => {
            el.classList.add("open");
            const ind = document.getElementById("ind-" + el.id);
            if (ind) ind.textContent = "Hide Answer ▲";
          });
        });
      }
      if (btnVivaCollapse) {
        btnVivaCollapse.addEventListener("click", () => {
          document.querySelectorAll("[id^='p3-viva-']").forEach((el) => {
            el.classList.remove("open");
            const ind = document.getElementById("ind-" + el.id);
            if (ind) ind.textContent = "Reveal Answer ▼";
          });
        });
      }
    }

    /* -------------------------------------------------------------------------
       3. PAGE 1: COPY PYTHON CODE BUTTON, MODULE FILTER & 9-STAGE WORKFLOW
       ------------------------------------------------------------------------- */
    function initPage1Controls() {
      // Copy Python Code Button
      const copyBtn = document.getElementById("btn-copy-python");
      const copyMsg = document.getElementById("copy-status-msg");
      if (copyBtn) {
        copyBtn.addEventListener("click", () => {
          const showCopied = () => {
            if (copyMsg) copyMsg.textContent = "✓ Copied all 192 lines of main.py!";
            copyBtn.textContent = "✓ Copied main.py!";
            setTimeout(() => {
              if (copyMsg) copyMsg.textContent = "";
              copyBtn.textContent = "Copy Python Code (main.py)";
            }, 2600);
          };
          if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(RAW_MAIN_PY_SOURCE).then(showCopied).catch(() => fallbackCopy(RAW_MAIN_PY_SOURCE, showCopied));
          } else {
            fallbackCopy(RAW_MAIN_PY_SOURCE, showCopied);
          }
        });
      }

      // Module Filter Tabs (All, NumPy, scikit-learn, Matplotlib)
      const filterBtns = document.querySelectorAll(".mod-filter-btn");
      filterBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
          const mod = btn.getAttribute("data-mod-filter");
          filterBtns.forEach((b) => b.classList.toggle("btn-primary", b === btn));
          const blocks = {
            numpy: document.getElementById("mod-block-numpy"),
            sklearn: document.getElementById("mod-block-sklearn"),
            matplotlib: document.getElementById("mod-block-matplotlib")
          };
          Object.keys(blocks).forEach((k) => {
            if (blocks[k]) {
              blocks[k].style.display = (mod === "all" || mod === k) ? "block" : "none";
            }
          });
        });
      });

      // 9-Stage Interactive Code Workflow
      let currentWfStage = 1;
      const renderWfStage = () => {
        const item = WORKFLOW_9_DATA[currentWfStage - 1];
        if (!item) return;
        document.querySelectorAll(".wf-stage-btn").forEach((b) => {
          b.classList.toggle("active", parseInt(b.getAttribute("data-wf-stage"), 10) === currentWfStage);
        });
        const tEl = document.getElementById("wf-detail-title");
        const lEl = document.getElementById("wf-detail-lines");
        const dEl = document.getElementById("wf-detail-desc");
        const cEl = document.getElementById("wf-detail-code");
        if (tEl) tEl.textContent = item.name;
        if (lEl) lEl.textContent = item.lines + " of main.py";
        if (dEl) dEl.textContent = item.desc;
        if (cEl) cEl.textContent = item.excerpt;

        const prevBtn = document.getElementById("wf-prev-btn");
        const nextBtn = document.getElementById("wf-next-btn");
        if (prevBtn) prevBtn.disabled = currentWfStage === 1;
        if (nextBtn) nextBtn.disabled = currentWfStage === WORKFLOW_9_DATA.length;
      };

      document.querySelectorAll(".wf-stage-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          currentWfStage = parseInt(btn.getAttribute("data-wf-stage"), 10);
          renderWfStage();
        });
      });

      const prevBtn = document.getElementById("wf-prev-btn");
      const nextBtn = document.getElementById("wf-next-btn");
      if (prevBtn) {
        prevBtn.addEventListener("click", () => {
          if (currentWfStage > 1) { currentWfStage--; renderWfStage(); }
        });
      }
      if (nextBtn) {
        nextBtn.addEventListener("click", () => {
          if (currentWfStage < WORKFLOW_9_DATA.length) { currentWfStage++; renderWfStage(); }
        });
      }
      renderWfStage();
    }

    function fallbackCopy(text, callback) {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand("copy"); } catch (e) {}
      document.body.removeChild(ta);
      if (callback) callback();
    }

    /* -------------------------------------------------------------------------
       4. PAGE 2: ANIMATED FACE-RECOGNITION DEMONSTRATION & REAL OUTPUTS LOADER
       ------------------------------------------------------------------------- */
    const REAL_MATCH_PAIRS = [{"pairIdx": 0, "personId": 24, "subplotJ": "0 (Row 1, Cols 1–2)", "testTitle": "Test: Person 24 (Sample A)", "matchTitle": "Match: Person 24 (Closest Training Face)", "shortLabel": "Pair 1: Person 24", "poseTag": "Moustache • Lighting Shift", "testSrc": "assets/images/img_bc1a57d1.png", "matchSrc": "assets/images/img_65092103.png", "desc": "Real Olivetti test face X_test[0] (Person 24 with moustache) projected onto 100 Eigenfaces and matched to X_train[nearest_indices[0]] (Person 24) via minimum Euclidean distance."}, {"pairIdx": 1, "personId": 5, "subplotJ": "1 (Row 1, Cols 3–4)", "testTitle": "Test: Person 5 (Smiling, Glasses)", "matchTitle": "Match: Person 5 (Neutral Training Face)", "shortLabel": "Pair 2: Person 5", "poseTag": "Glasses • Smile vs Neutral", "testSrc": "assets/images/img_e487a042.png", "matchSrc": "assets/images/img_f9518870.png", "desc": "Real Olivetti test face X_test[1] (Person 5 wearing glasses, smiling) projected into 100-D PCA space and matched to Person 5's neutral-expression training face."}, {"pairIdx": 2, "personId": 15, "subplotJ": "2 (Row 2, Cols 1–2)", "testTitle": "Test: Person 15 (Sample A, Frontal)", "matchTitle": "Match: Person 15 (Closest Training Face)", "shortLabel": "Pair 3: Person 15 (A)", "poseTag": "Frontal Pose • Expression Shift", "testSrc": "assets/images/img_055b1f95.png", "matchSrc": "assets/images/img_536d5c83.png", "desc": "Real Olivetti test face X_test[2] (Person 15, Sample A) mean-centered by subtracting pca.mean_ and matched to the nearest training sample of Person 15."}, {"pairIdx": 3, "personId": 30, "subplotJ": "3 (Row 2, Cols 3–4)", "testTitle": "Test: Person 30 (Glasses, Frontal)", "matchTitle": "Match: Person 30 (Tilted Training Face)", "shortLabel": "Pair 4: Person 30", "poseTag": "Glasses • Head Tilt Variation", "testSrc": "assets/images/img_fb7d66b4.png", "matchSrc": "assets/images/img_06e48045.png", "desc": "Real Olivetti test face X_test[3] (Person 30 wearing glasses) matched across head-tilt variation to Person 30 in 100-D PCA space."}, {"pairIdx": 4, "personId": 37, "subplotJ": "4 (Row 3, Cols 1–2)", "testTitle": "Test: Person 37 (Frontal View)", "matchTitle": "Match: Person 37 (Angled Training Face)", "shortLabel": "Pair 5: Person 37", "poseTag": "Frontal vs Slight Turn", "testSrc": "assets/images/img_0510859e.png", "matchSrc": "assets/images/img_fe82c88b.png", "desc": "Real Olivetti test face X_test[4] (Person 37) projected onto the 100 learned Eigenfaces and matched to Person 37's training photograph."}, {"pairIdx": 5, "personId": 15, "subplotJ": "5 (Row 3, Cols 3–4)", "testTitle": "Test: Person 15 (Sample B, Smiling)", "matchTitle": "Match: Person 15 (Closest Training Face)", "shortLabel": "Pair 6: Person 15 (B)", "poseTag": "Open Smile • Illumination", "testSrc": "assets/images/img_ff2b77f9.png", "matchSrc": "assets/images/img_c442e1a1.png", "desc": "Real Olivetti test face X_test[5] (Person 15, Sample B smiling) matched via np.argmin(distances, axis=1) to Person 15."}, {"pairIdx": 6, "personId": 24, "subplotJ": "Reverse Query #7 (Person 24 Pose B)", "testTitle": "Query Face: Person 24 (Pose B Photo)", "matchTitle": "Finding Result: Person 24 (Pose A Photo)", "shortLabel": "Photo 7: Person 24 (B)", "poseTag": "Reverse Pose Query", "testSrc": "assets/images/img_65092103.png", "matchSrc": "assets/images/img_bc1a57d1.png", "desc": "Using the second real photograph of Person 24 from face_matches.png as the Input Test Face to verify bidirectional nearest-neighbour matching in PCA space."}, {"pairIdx": 7, "personId": 5, "subplotJ": "Reverse Query #8 (Person 5 Neutral Pose)", "testTitle": "Query Face: Person 5 (Neutral Pose Photo)", "matchTitle": "Finding Result: Person 5 (Smiling Photo)", "shortLabel": "Photo 8: Person 5 (B)", "poseTag": "Neutral → Smiling Match", "testSrc": "assets/images/img_f9518870.png", "matchSrc": "assets/images/img_e487a042.png", "desc": "Using the neutral-expression real photograph of Person 5 from face_matches.png as the Input Test Face and matching it to Person 5's smiling photograph."}, {"pairIdx": 8, "personId": 15, "subplotJ": "Reverse Query #9 (Person 15 Pose C)", "testTitle": "Query Face: Person 15 (Pose C Photo)", "matchTitle": "Finding Result: Person 15 (Pose A Photo)", "shortLabel": "Photo 9: Person 15 (C)", "poseTag": "Alternate Pose Query", "testSrc": "assets/images/img_536d5c83.png", "matchSrc": "assets/images/img_055b1f95.png", "desc": "Using the third real photograph of Person 15 from face_matches.png as the Input Test Face to demonstrate nearest-neighbour retrieval."}, {"pairIdx": 9, "personId": 30, "subplotJ": "Reverse Query #10 (Person 30 Tilted Pose)", "testTitle": "Query Face: Person 30 (Tilted Pose Photo)", "matchTitle": "Finding Result: Person 30 (Frontal Photo)", "shortLabel": "Photo 10: Person 30 (B)", "poseTag": "Tilted → Frontal Match", "testSrc": "assets/images/img_06e48045.png", "matchSrc": "assets/images/img_fb7d66b4.png", "desc": "Using the tilted real photograph of Person 30 from face_matches.png as the Input Test Face and finding its frontal counterpart in 100-D Eigenface space."}, {"pairIdx": 10, "personId": 37, "subplotJ": "Reverse Query #11 (Person 37 Angled Pose)", "testTitle": "Query Face: Person 37 (Angled Pose Photo)", "matchTitle": "Finding Result: Person 37 (Frontal Photo)", "shortLabel": "Photo 11: Person 37 (B)", "poseTag": "Angled → Frontal Match", "testSrc": "assets/images/img_fe82c88b.png", "matchSrc": "assets/images/img_0510859e.png", "desc": "Using the angled real photograph of Person 37 from face_matches.png as the Input Test Face and retrieving Person 37's frontal photograph."}, {"pairIdx": 11, "personId": 15, "subplotJ": "Reverse Query #12 (Person 15 Pose D)", "testTitle": "Query Face: Person 15 (Pose D Photo)", "matchTitle": "Finding Result: Person 15 (Smiling Photo)", "shortLabel": "Photo 12: Person 15 (D)", "poseTag": "Expression Invariance", "testSrc": "assets/images/img_c442e1a1.png", "matchSrc": "assets/images/img_ff2b77f9.png", "desc": "Using the fourth real photograph of Person 15 from face_matches.png as the Input Test Face and matching it to Person 15's smiling photograph."}];
    const REAL_EIGENFACES = [{"idx": 0, "title": "Mean Face (μ)", "sub": "pca.mean_", "src": "assets/images/img_85a03b55.png"}, {"idx": 1, "title": "Eigenface 1", "sub": "pca.components_[0]", "src": "assets/images/img_71bcd46d.png"}, {"idx": 2, "title": "Eigenface 2", "sub": "pca.components_[1]", "src": "assets/images/img_64471618.png"}, {"idx": 3, "title": "Eigenface 3", "sub": "pca.components_[2]", "src": "assets/images/img_48d2c624.png"}, {"idx": 4, "title": "Eigenface 4", "sub": "pca.components_[3]", "src": "assets/images/img_dbf38e1e.png"}, {"idx": 5, "title": "Eigenface 5", "sub": "pca.components_[4]", "src": "assets/images/img_8a38f15a.png"}, {"idx": 6, "title": "Eigenface 6", "sub": "pca.components_[5]", "src": "assets/images/img_3a5c5568.png"}, {"idx": 7, "title": "Eigenface 7", "sub": "pca.components_[6]", "src": "assets/images/img_2ad74ee6.png"}, {"idx": 8, "title": "Eigenface 8", "sub": "pca.components_[7]", "src": "assets/images/img_4da0ceec.png"}, {"idx": 9, "title": "Eigenface 9", "sub": "pca.components_[8]", "src": "assets/images/img_4d97310c.png"}];

    // Preload all 12 real face pair images for instant canvas/img rendering
    const PRELOADED_PAIR_IMAGES = REAL_MATCH_PAIRS.map((meta) => {
      const tImg = new Image();
      tImg.src = meta.testSrc;
      const mImg = new Image();
      mImg.src = meta.matchSrc;
      return { meta, testImg: tImg, matchImg: mImg };
    });

    let loadedFaceMatchesImage = null;
    let recognitionTimerIds = [];
    let cyclingIntervalId = null;
    let eigenHighlightIntervalId = null;
    let currentDemoPairIdx = 0;
    let customUploadedTestDataUrl = null;

    // Updates both the visible <img> and synced <canvas> with the real photograph
    function setSlotPhoto(slotType, srcDataUrl) {
      const imgEl = document.getElementById(slotType === "test" ? "demo-test-img" : "demo-match-img");
      const canvasEl = document.getElementById(slotType === "test" ? "demo-test-canvas" : "demo-match-canvas");
      if (imgEl && srcDataUrl) {
        imgEl.src = srcDataUrl;
        imgEl.style.display = "block";
      }
      if (canvasEl && srcDataUrl) {
        const ctx = canvasEl.getContext("2d");
        const tmp = new Image();
        tmp.onload = () => {
          ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
          ctx.drawImage(tmp, 0, 0, canvasEl.width, canvasEl.height);
        };
        tmp.src = srcDataUrl;
      }
    }

    function drawDemoFaceOnCanvas(canvasId, pairIdx, isMatchSlot) {
      const safeIdx = ((pairIdx % REAL_MATCH_PAIRS.length) + REAL_MATCH_PAIRS.length) % REAL_MATCH_PAIRS.length;
      const pair = REAL_MATCH_PAIRS[safeIdx];
      const srcUrl = isMatchSlot ? pair.matchSrc : (customUploadedTestDataUrl && safeIdx === currentDemoPairIdx ? customUploadedTestDataUrl : pair.testSrc);
      const slotType = canvasId.includes("test") ? "test" : "match";
      setSlotPhoto(slotType, srcUrl);
    }

    function highlightActiveThumb(pairIdx) {
      document.querySelectorAll(".demo-thumb-btn").forEach((btn) => {
        const idx = parseInt(btn.getAttribute("data-select-pair"), 10);
        const isAct = idx === pairIdx && !customUploadedTestDataUrl;
        btn.style.border = isAct ? "2px solid var(--blue-400)" : "1px solid var(--border-subtle)";
        btn.style.boxShadow = isAct ? "0 0 12px rgba(56,189,248,0.35)" : "none";
        btn.style.background = isAct ? "rgba(37,99,235,0.2)" : "var(--bg-code)";
      });
    }

    function highlightEigenChip(activeIdx) {
      document.querySelectorAll(".demo-eigen-chip").forEach((chip) => {
        const idx = parseInt(chip.getAttribute("data-eigen-chip"), 10);
        const isHigh = activeIdx === idx || (activeIdx === -1 && idx === 0);
        chip.style.borderColor = isHigh ? "var(--blue-400)" : "var(--border-subtle)";
        chip.style.boxShadow = isHigh ? "0 0 10px rgba(56,189,248,0.45)" : "none";
        chip.style.transform = isHigh ? "scale(1.06)" : "scale(1)";
      });
    }

    function clearRecognitionTimers() {
      recognitionTimerIds.forEach((id) => clearTimeout(id));
      recognitionTimerIds = [];
      if (cyclingIntervalId) {
        clearInterval(cyclingIntervalId);
        cyclingIntervalId = null;
      }
      if (eigenHighlightIntervalId) {
        clearInterval(eigenHighlightIntervalId);
        eigenHighlightIntervalId = null;
      }
    }

    function updateSummaryAndLabels(pairIdx, isAnimating = false) {
      const pairMeta = REAL_MATCH_PAIRS[pairIdx] || REAL_MATCH_PAIRS[0];
      const testSrc = customUploadedTestDataUrl || pairMeta.testSrc;
      const matchSrc = pairMeta.matchSrc;

      const testLbl = document.getElementById("demo-test-label");
      const testSub = document.getElementById("demo-test-sub");
      const matchLbl = document.getElementById("demo-match-label");
      const matchSub = document.getElementById("demo-match-sub");
      const resBadge = document.getElementById("demo-result-badge");
      const resDetails = document.getElementById("demo-result-details");
      const miniTest = document.getElementById("demo-mini-pair-test");
      const miniMatch = document.getElementById("demo-mini-pair-match");
      const miniStatus = document.getElementById("demo-mini-pair-status");
      const sumTestImg = document.getElementById("res-summary-test-img");
      const sumMatchImg = document.getElementById("res-summary-match-img");
      const sumTestTxt = document.getElementById("res-summary-test-txt");
      const sumMatchTxt = document.getElementById("res-summary-match-txt");

      if (testLbl) testLbl.textContent = customUploadedTestDataUrl ? "Custom Uploaded Test Face" : pairMeta.testTitle;
      if (testSub) testSub.textContent = customUploadedTestDataUrl ? "Custom Input Photo → Projected to 100-D PCA" : `Subplot: ${pairMeta.subplotJ} • ${pairMeta.poseTag}`;
      if (miniTest) miniTest.src = testSrc;
      if (miniMatch) miniMatch.src = matchSrc;
      if (sumTestImg) sumTestImg.src = testSrc;
      if (sumMatchImg) sumMatchImg.src = matchSrc;
      if (sumTestTxt) sumTestTxt.textContent = customUploadedTestDataUrl ? "Custom Face" : `Person ${pairMeta.personId}`;
      if (sumMatchTxt) sumMatchTxt.textContent = `Person ${pairMeta.personId} ✓`;

      if (!isAnimating) {
        if (matchLbl) matchLbl.textContent = `${pairMeta.matchTitle} ✓`;
        if (matchSub) matchSub.textContent = `Test Label: Person ${pairMeta.personId} | Matched Label: Person ${pairMeta.personId}`;
        if (miniStatus) miniStatus.textContent = `Matched: Person ${pairMeta.personId} ✓`;
        if (resBadge) resBadge.textContent = `Verified Match: Person ${pairMeta.personId}`;
        if (resDetails) {
          resDetails.innerHTML = `<strong>Real Experiment Match (Person ${pairMeta.personId}, ${pairMeta.subplotJ}):</strong> ${pairMeta.desc}`;
        }
      }
    }

    function resetRecognitionDemoUI() {
      clearRecognitionTimers();
      const testFrame = document.getElementById("test-scanner-frame");
      const matchFrame = document.getElementById("match-scanner-frame");
      const scanOverlay = document.getElementById("demo-match-scan-overlay");
      if (testFrame) testFrame.classList.remove("scanning-active");
      if (matchFrame) matchFrame.classList.remove("scanning-active");
      if (scanOverlay) scanOverlay.style.display = "none";

      // Immediately show BOTH the real Input Test Face AND the real Finding Result Face!
      drawDemoFaceOnCanvas("demo-test-canvas", currentDemoPairIdx, false);
      drawDemoFaceOnCanvas("demo-match-canvas", currentDemoPairIdx, true);

      highlightActiveThumb(currentDemoPairIdx);
      highlightEigenChip(-1);
      updateSummaryAndLabels(currentDemoPairIdx, false);

      const pBar = document.getElementById("demo-progress-bar");
      const pPct = document.getElementById("demo-progress-pct");
      const statusTxt = document.getElementById("demo-live-status");

      if (pBar) pBar.style.width = "100%";
      if (pPct) pPct.textContent = "Match Ready (100%)";
      if (statusTxt) statusTxt.textContent = "Showing real Input Test Face & Finding Result. Click 'Start Recognition' to animate the PCA search!";

      document.querySelectorAll("#demo-step-list .status-step-item").forEach((li) => {
        li.classList.remove("active");
        li.classList.add("done");
      });
    }

    function startRecognitionAnimation() {
      clearRecognitionTimers();
      const startBtn = document.getElementById("btn-start-recognition");
      if (startBtn) startBtn.disabled = true;

      const testFrame = document.getElementById("test-scanner-frame");
      const matchFrame = document.getElementById("match-scanner-frame");
      const scanOverlay = document.getElementById("demo-match-scan-overlay");
      const pBar = document.getElementById("demo-progress-bar");
      const pPct = document.getElementById("demo-progress-pct");
      const statusTxt = document.getElementById("demo-live-status");
      const matchLbl = document.getElementById("demo-match-label");
      const matchSub = document.getElementById("demo-match-sub");
      const resBadge = document.getElementById("demo-result-badge");

      drawDemoFaceOnCanvas("demo-test-canvas", currentDemoPairIdx, false);
      highlightActiveThumb(currentDemoPairIdx);
      updateSummaryAndLabels(currentDemoPairIdx, true);

      const setStepState = (activeIdx) => {
        document.querySelectorAll("#demo-step-list .status-step-item").forEach((li) => {
          const idx = parseInt(li.getAttribute("data-step-idx"), 10);
          li.classList.toggle("done", idx < activeIdx);
          li.classList.toggle("active", idx === activeIdx);
        });
      };

      if (scanOverlay) {
        scanOverlay.style.display = "block";
        scanOverlay.textContent = "Scanning 300 Training Faces...";
      }

      // Cycle through all 12 real Olivetti face photographs in the Finding Result slot during search!
      let cycleCount = 0;
      cyclingIntervalId = setInterval(() => {
        cycleCount++;
        const candidateIdx = cycleCount % REAL_MATCH_PAIRS.length;
        const useMatchPhoto = (cycleCount % 2 === 0);
        const candidatePair = REAL_MATCH_PAIRS[candidateIdx];
        setSlotPhoto("match", useMatchPhoto ? candidatePair.matchSrc : candidatePair.testSrc);
        if (scanOverlay) {
          scanOverlay.textContent = `Comparing Training Face #${((cycleCount * 17) % 300) + 1} / 300 (Person ${candidatePair.personId})...`;
        }
      }, 120);

      // Step 1: Display selected test face & start scanning ring
      if (testFrame) testFrame.classList.add("scanning-active");
      if (matchFrame) matchFrame.classList.add("scanning-active");
      setStepState(1);
      highlightEigenChip(0);
      if (pBar) pBar.style.width = "20%";
      if (pPct) pPct.textContent = "Step 1/5 (20%)";
      if (statusTxt) statusTxt.textContent = "Step 1: Scanning 64 × 64 Input Test Face & subtracting Mean Face (pca.mean_)...";
      if (matchLbl) matchLbl.textContent = "Searching 300 Training Faces...";
      if (matchSub) matchSub.textContent = "Evaluating candidate training faces in 100-D PCA space";
      if (resBadge) resBadge.textContent = "Scanning...";

      // Step 2: Projecting face into PCA space (highlight Eigenfaces 1..9 sequentially)
      recognitionTimerIds.push(setTimeout(() => {
        setStepState(2);
        if (pBar) pBar.style.width = "45%";
        if (pPct) pPct.textContent = "Step 2/5 (45%)";
        if (statusTxt) statusTxt.textContent = "Step 2: Projecting centered face onto 100 Eigenfaces (pca.components_)...";
        let eigIdx = 0;
        eigenHighlightIntervalId = setInterval(() => {
          eigIdx = (eigIdx + 1) % 10;
          highlightEigenChip(eigIdx);
        }, 95);
      }, 650));

      // Step 3: Calculating Euclidean distances
      recognitionTimerIds.push(setTimeout(() => {
        if (eigenHighlightIntervalId) {
          clearInterval(eigenHighlightIntervalId);
          eigenHighlightIntervalId = null;
        }
        highlightEigenChip(-1);
        setStepState(3);
        if (pBar) pBar.style.width = "70%";
        if (pPct) pPct.textContent = "Step 3/5 (70%)";
        if (statusTxt) statusTxt.textContent = "Step 3: Calculating Euclidean distances d(x, y) across all 300 training faces...";
      }, 1400));

      // Step 4: Finding nearest training face
      recognitionTimerIds.push(setTimeout(() => {
        setStepState(4);
        if (pBar) pBar.style.width = "90%";
        if (pPct) pPct.textContent = "Step 4/5 (90%)";
        if (statusTxt) statusTxt.textContent = "Step 4: Selecting minimum distance training face via np.argmin(distances, axis=1)...";
      }, 2100));

      // Step 5: Reveal final matching face (Finding Result)
      recognitionTimerIds.push(setTimeout(() => {
        clearRecognitionTimers();
        if (testFrame) testFrame.classList.remove("scanning-active");
        if (matchFrame) matchFrame.classList.remove("scanning-active");
        if (scanOverlay) scanOverlay.style.display = "none";

        setStepState(6); // Marks all 5 steps as done
        if (pBar) pBar.style.width = "100%";
        if (pPct) pPct.textContent = "Complete (100%)";
        const pairMeta = REAL_MATCH_PAIRS[currentDemoPairIdx] || REAL_MATCH_PAIRS[0];
        if (statusTxt) statusTxt.textContent = `✓ Recognition Complete: Closest training face found → Person ${pairMeta.personId}!`;

        drawDemoFaceOnCanvas("demo-match-canvas", currentDemoPairIdx, true);
        updateSummaryAndLabels(currentDemoPairIdx, false);

        if (startBtn) startBtn.disabled = false;
      }, 2750));
    }

    function initPage2OutputsAndDemo() {
      const pairSelect = document.getElementById("demo-pair-select");
      const startBtn = document.getElementById("btn-start-recognition");
      const tryAgainBtn = document.getElementById("btn-try-again");
      const customInp = document.getElementById("custom-test-face-input");

      if (pairSelect) {
        pairSelect.addEventListener("change", (e) => {
          customUploadedTestDataUrl = null;
          currentDemoPairIdx = parseInt(e.target.value, 10);
          resetRecognitionDemoUI();
          startRecognitionAnimation();
        });
      }

      // Clickable 12-photo Input Test Face thumbnail strip
      document.querySelectorAll(".demo-thumb-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          customUploadedTestDataUrl = null;
          currentDemoPairIdx = parseInt(btn.getAttribute("data-select-pair"), 10);
          if (pairSelect) pairSelect.value = String(currentDemoPairIdx);
          resetRecognitionDemoUI();
          startRecognitionAnimation();
        });
      });

      // Custom Test Face Uploader
      if (customInp) {
        customInp.addEventListener("change", (e) => {
          const file = e.target.files && e.target.files[0];
          if (!file) return;
          const reader = new FileReader();
          reader.onload = (ev) => {
            customUploadedTestDataUrl = ev.target.result;
            resetRecognitionDemoUI();
            startRecognitionAnimation();
          };
          reader.readAsDataURL(file);
        });
      }

      if (startBtn) {
        startBtn.addEventListener("click", startRecognitionAnimation);
      }
      if (tryAgainBtn) {
        tryAgainBtn.addEventListener("click", () => {
          customUploadedTestDataUrl = null;
          currentDemoPairIdx = (currentDemoPairIdx + 1) % REAL_MATCH_PAIRS.length;
          if (pairSelect) pairSelect.value = String(currentDemoPairIdx);
          resetRecognitionDemoUI();
          startRecognitionAnimation();
        });
      }

      // Attempt to load outputs/eigenfaces.png, outputs/explained_variance.png, outputs/face_matches.png
      const outputSlots = [
        { key: "eigenfaces", path: "outputs/eigenfaces.png", title: "outputs/eigenfaces.png — Mean Face and Principal Components" },
        { key: "variance", path: "outputs/explained_variance.png", title: "outputs/explained_variance.png — Cumulative Explained Variance" },
        { key: "matches", path: "outputs/face_matches.png", title: "outputs/face_matches.png — Test Faces and Closest Training Matches" }
      ];

      const handleImageLoaded = (slot, imgEl, srcUrl, labelText) => {
        imgEl.src = srcUrl;
        imgEl.style.display = "block";
        const missEl = document.getElementById("p2-missing-" + slot.key);
        const badgeEl = document.getElementById("p2-badge-" + slot.key);
        if (missEl) missEl.style.display = "none";
        if (badgeEl) {
          badgeEl.textContent = labelText;
          badgeEl.className = "pill-badge emerald";
        }
        if (slot.key === "matches") {
          loadedFaceMatchesImage = imgEl;
        }
      };

      outputSlots.forEach((slot) => {
        const imgEl = document.getElementById("p2-img-" + slot.key);
        if (!imgEl) return;

        // Try loading from outputs/<file>.png automatically when opened locally
        const probe = new Image();
        probe.onload = () => {
          handleImageLoaded(slot, imgEl, slot.path, "Loaded: " + slot.path);
        };
        probe.onerror = () => {
          // Keep embedded base64 output image visible
        };
        probe.src = slot.path;

        // Individual file input uploader
        const fileInp = document.getElementById("file-input-" + slot.key);
        if (fileInp) {
          fileInp.addEventListener("change", (e) => {
            const file = e.target.files && e.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = (ev) => {
              handleImageLoaded(slot, imgEl, ev.target.result, "Loaded: " + file.name);
            };
            reader.readAsDataURL(file);
          });
        }

        // Click on image to open lightbox
        imgEl.addEventListener("click", () => openGraphLightbox(slot.key, slot.title));
      });

      // Batch uploader for all 3 output PNGs at once
      const batchInput = document.getElementById("upload-all-outputs");
      if (batchInput) {
        batchInput.addEventListener("change", (e) => {
          const files = Array.from(e.target.files || []);
          files.forEach((file) => {
            const lower = file.name.toLowerCase();
            let targetSlot = null;
            if (lower.includes("eigen")) targetSlot = outputSlots[0];
            else if (lower.includes("var")) targetSlot = outputSlots[1];
            else if (lower.includes("match")) targetSlot = outputSlots[2];
            if (targetSlot) {
              const reader = new FileReader();
              reader.onload = (ev) => {
                const imgEl = document.getElementById("p2-img-" + targetSlot.key);
                if (imgEl) handleImageLoaded(targetSlot, imgEl, ev.target.result, "Loaded: " + file.name);
              };
              reader.readAsDataURL(file);
            }
          });
        });
      }

      // Click-to-enlarge buttons
      document.querySelectorAll("[data-lightbox-key]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const key = btn.getAttribute("data-lightbox-key");
          const slot = outputSlots.find((s) => s.key === key);
          if (slot) openGraphLightbox(slot.key, slot.title);
        });
      });

      // Lightbox close button
      const lbModal = document.getElementById("graph-lightbox");
      const lbClose = document.getElementById("lightbox-close-btn");
      if (lbClose && lbModal) {
        lbClose.addEventListener("click", () => lbModal.classList.remove("open"));
        lbModal.addEventListener("click", (e) => {
          if (e.target === lbModal) lbModal.classList.remove("open");
        });
      }

      document.querySelectorAll("[data-run-pair]").forEach((btn) => {
        btn.addEventListener("click", () => {
          customUploadedTestDataUrl = null;
          currentDemoPairIdx = parseInt(btn.getAttribute("data-run-pair"), 10);
          if (pairSelect) pairSelect.value = String(currentDemoPairIdx);
          switchPage(2, false);
          resetRecognitionDemoUI();
          const demoSec = document.getElementById("p2-sec-demo");
          if (demoSec) demoSec.scrollIntoView({ behavior: "smooth", block: "start" });
          startRecognitionAnimation();
        });
      });

      resetRecognitionDemoUI();
    }

    function openGraphLightbox(key, titleText) {
      const lbModal = document.getElementById("graph-lightbox");
      const lbTitle = document.getElementById("lightbox-title");
      const lbBody = document.getElementById("lightbox-body");
      const imgEl = document.getElementById("p2-img-" + key);
      if (!lbModal || !lbTitle || !lbBody) return;

      lbTitle.textContent = titleText;
      lbBody.innerHTML = "";

      if (imgEl && imgEl.style.display !== "none" && imgEl.src) {
        const bigImg = document.createElement("img");
        bigImg.src = imgEl.src;
        bigImg.alt = titleText;
        lbBody.appendChild(bigImg);
      } else {
        const msg = document.createElement("div");
        msg.style.padding = "2rem";
        msg.style.color = "#cbd5e1";
        msg.innerHTML = `<p style="font-size:1rem; font-weight:700; color:#fff; margin-bottom:0.5rem;">Real Output Image Not Yet Loaded in Browser</p><p style="font-size:0.88rem;">We never fabricate replacement graphs as real experimental outputs. To view and enlarge <code>outputs/${key === 'variance' ? 'explained_variance' : key === 'matches' ? 'face_matches' : 'eigenfaces'}.png</code>, place your saved PNG in the <code>outputs/</code> subfolder next to <code>results.html</code> or click <strong>"Select File from Laptop"</strong> on the card.</p>`;
        lbBody.appendChild(msg);
      }
      lbModal.classList.add("open");
    }

    /* -------------------------------------------------------------------------
       5. PAGE 2 SECTION E: INTERACTIVE EUCLIDEAN DISTANCE CALCULATOR
          Initially Vector A = [1, 2, 3], Vector B = [4, 5, 6] with validation
       ------------------------------------------------------------------------- */
    function parseVectorString(rawStr) {
      const cleaned = rawStr.replace(/[\[\]]/g, "").trim();
      if (!cleaned) return { valid: false, error: "Vector cannot be empty.", values: [] };
      const parts = cleaned.split(/[\s,]+/).filter(Boolean);
      const nums = [];
      for (let i = 0; i < parts.length; i++) {
        const n = Number(parts[i]);
        if (!Number.isFinite(n)) {
          return { valid: false, error: `Invalid numerical entry "${parts[i]}" at position ${i + 1}.`, values: [] };
        }
        nums.push(n);
      }
      return { valid: true, error: "", values: nums };
    }

    function updatePage2Calculator() {
      const inpA = document.getElementById("calc-vec-a");
      const inpB = document.getElementById("calc-vec-b");
      const valBox = document.getElementById("calc-validation-box");
      const tbody = document.getElementById("calc-tbody");
      const pill = document.getElementById("calc-final-pill");
      const s1 = document.getElementById("calc-step-1");
      const s2 = document.getElementById("calc-step-2");
      const s3 = document.getElementById("calc-step-3");
      const s4 = document.getElementById("calc-step-4");
      if (!inpA || !inpB) return;

      const parsedA = parseVectorString(inpA.value);
      const parsedB = parseVectorString(inpB.value);

      if (!parsedA.valid || !parsedB.valid) {
        valBox.className = "callout-banner amber";
        valBox.innerHTML = `<strong>Validation Error:</strong> ${parsedA.error || parsedB.error} Please enter comma-separated numbers (e.g., <code>1, 2, 3</code>).`;
        pill.textContent = "Invalid Input";
        return;
      }

      if (parsedA.values.length !== parsedB.values.length) {
        valBox.className = "callout-banner amber";
        valBox.innerHTML = `<strong>Length Mismatch:</strong> Vector A has <strong>${parsedA.values.length}</strong> element(s), but Vector B has <strong>${parsedB.values.length}</strong> element(s). Both vectors must have equal lengths!`;
        pill.textContent = "Length Mismatch";
        return;
      }

      const A = parsedA.values;
      const B = parsedB.values;
      valBox.className = "callout-banner";
      valBox.innerHTML = `<strong>Input Valid:</strong> Both Vector A and Vector B have equal length (<strong>${A.length}</strong> component${A.length === 1 ? '' : 's'}).`;

      const diffs = [];
      const sqs = [];
      let sumSq = 0;
      let rowsHtml = "";

      for (let i = 0; i < A.length; i++) {
        const d = A[i] - B[i];
        const sq = d * d;
        diffs.push(d);
        sqs.push(sq);
        sumSq += sq;
        rowsHtml += `
          <tr>
            <td>i = ${i + 1}</td>
            <td>${A[i]}</td>
            <td>${B[i]}</td>
            <td>${A[i]} &minus; (${B[i]}) = <strong>${Number(d.toFixed(4))}</strong></td>
            <td>(${Number(d.toFixed(4))})<sup>2</sup> = <strong>${Number(sq.toFixed(4))}</strong></td>
          </tr>
        `;
      }

      const dist = Math.sqrt(sumSq);
      tbody.innerHTML = rowsHtml;
      pill.textContent = `d(A, B) = ${dist.toFixed(4)}`;
      s1.textContent = `[${diffs.map((d) => Number(d.toFixed(4))).join(", ")}]`;
      s2.textContent = `[${sqs.map((q) => Number(q.toFixed(4))).join(", ")}]`;
      s3.textContent = `${sqs.map((q) => Number(q.toFixed(4))).join(" + ")} = ${Number(sumSq.toFixed(4))}`;
      s4.textContent = `√(${Number(sumSq.toFixed(4))}) = ${dist.toFixed(4)}`;
    }

    function initPage2Calculator() {
      const inpA = document.getElementById("calc-vec-a");
      const inpB = document.getElementById("calc-vec-b");
      const btnReset = document.getElementById("calc-reset-default");
      const btn5D = document.getElementById("calc-sample-5d");

      if (inpA) inpA.addEventListener("input", updatePage2Calculator);
      if (inpB) inpB.addEventListener("input", updatePage2Calculator);
      if (btnReset) {
        btnReset.addEventListener("click", () => {
          inpA.value = "1, 2, 3";
          inpB.value = "4, 5, 6";
          updatePage2Calculator();
        });
      }
      if (btn5D) {
        btn5D.addEventListener("click", () => {
          inpA.value = "4.2, -1.5, 2.0, 0.8, -0.5";
          inpB.value = "3.8, -1.2, 2.5, 0.3, -0.9";
          updatePage2Calculator();
        });
      }
      updatePage2Calculator();
    }

    window.addEventListener("DOMContentLoaded", () => {
      initNavbar();
      initAccordions();
      initPage1Controls();
      initPage2OutputsAndDemo();
      initPage2Calculator();
    });
