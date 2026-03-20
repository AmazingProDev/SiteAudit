(function () {
    const modal = document.getElementById('ssv-validation-modal');
    const openButtons = Array.from(document.querySelectorAll('[data-ssv-open]'));
    const closeButtons = Array.from(document.querySelectorAll('[data-ssv-close]'));
    const dropzone = document.getElementById('ssv-upload-dropzone');
    const input = document.getElementById('ssv-upload-input');
    const progress = document.getElementById('ssv-progress-bar');
    const progressLabel = document.getElementById('ssv-progress-label');
    const errorBox = document.getElementById('ssv-error');
    const summaryShell = document.getElementById('ssv-summary');
    const summaryGrid = summaryShell ? summaryShell.querySelector('.ssv-summary-grid') : null;
    const previewGrid = summaryShell ? summaryShell.querySelector('.ssv-preview-grid') : null;
    const previewImage = document.getElementById('ssv-preview-image');
    const annotatedPreviewImage = document.getElementById('ssv-annotated-image');
    const verdict = document.getElementById('ssv-verdict');
    const verdictMeta = document.getElementById('ssv-verdict-meta');
    const colorList = document.getElementById('ssv-color-list');
    const metricsGrid = document.getElementById('ssv-metrics-grid');
    const siteCenter = document.getElementById('ssv-site-center');
    const selectionMeta = document.getElementById('ssv-selection-meta');
    const analysisList = document.getElementById('ssv-analysis-list');
    const uploadHint = document.getElementById('ssv-upload-hint');
    const colorsCard = colorList ? colorList.closest('.ssv-detail-card') : null;
    const metricsCard = metricsGrid ? metricsGrid.closest('.ssv-detail-card') : null;
    const siteCenterCard = siteCenter ? siteCenter.closest('.ssv-detail-card') : null;
    const selectionCard = selectionMeta ? selectionMeta.closest('.ssv-detail-card') : null;

    function getApiUrl() {
        if (window.location.protocol === 'file:') {
            return 'http://127.0.0.1:8000/api/ssv-validation';
        }
        return '/api/ssv-validation';
    }

    function promptForWorkbook() {
        if (!input) return;
        input.value = '';
        input.click();
    }

    function openModal(shouldPrompt = false) {
        if (!modal) return;
        modal.hidden = false;
        requestAnimationFrame(() => {
            modal.classList.add('visible');
            if (shouldPrompt) {
                promptForWorkbook();
            }
        });
    }

    function closeModal() {
        if (!modal) return;
        modal.classList.remove('visible');
        setTimeout(() => {
            modal.hidden = true;
        }, 180);
    }

    function resetState() {
        setProgress(0, 'Select an .xlsx SSV workbook to begin.');
        hideError();
        if (summaryShell) {
            summaryShell.hidden = true;
        }
        analysisList.hidden = true;
        if (summaryGrid) {
            summaryGrid.classList.add('compact');
        }
        if (previewGrid) {
            previewGrid.hidden = true;
        }
        if (verdictMeta) {
            verdictMeta.hidden = true;
        }
        if (colorsCard) {
            colorsCard.hidden = true;
        }
        if (metricsCard) {
            metricsCard.hidden = true;
        }
        if (siteCenterCard) {
            siteCenterCard.hidden = true;
        }
        if (selectionCard) {
            selectionCard.hidden = true;
        }
        if (previewImage) {
            previewImage.removeAttribute('src');
        }
        if (annotatedPreviewImage) {
            annotatedPreviewImage.removeAttribute('src');
        }
        if (colorList) {
            colorList.innerHTML = '';
        }
        if (metricsGrid) {
            metricsGrid.innerHTML = '';
        }
        if (siteCenter) {
            siteCenter.textContent = '--';
        }
        if (selectionMeta) {
            selectionMeta.textContent = '--';
        }
        analysisList.innerHTML = '';
        verdict.textContent = 'Waiting for analysis';
        verdict.className = 'ssv-verdict-card';
        uploadHint.textContent = 'Drop a single SSV Excel workbook here or click to choose one.';
        if (input) {
            input.value = '';
        }
    }

    function showError(message) {
        errorBox.hidden = false;
        errorBox.textContent = message;
    }

    function hideError() {
        errorBox.hidden = true;
        errorBox.textContent = '';
    }

    function setProgress(value, label) {
        const clamped = Math.max(0, Math.min(value, 100));
        progress.style.width = `${clamped}%`;
        progressLabel.textContent = label;
    }

    function renderColors(colors, target) {
        target.innerHTML = '';
        colors.forEach((entry) => {
            const item = document.createElement('div');
            item.className = 'ssv-color-pill';

            const swatch = document.createElement('span');
            swatch.className = 'ssv-color-swatch';
            swatch.style.background = entry.hex;

            const text = document.createElement('span');
            text.textContent = `${entry.name} (${entry.dominant_angle.toFixed(1)}°)`;

            item.appendChild(swatch);
            item.appendChild(text);
            target.appendChild(item);
        });
    }

    function createThroughputNotice() {
        const notice = document.createElement('div');
        notice.className = 'ssv-throughput-notice';
        notice.textContent = 'Throughput depends on: Bandwidth (10 vs 20 MHz)/ MIMO (2x2 vs 4x4)/ UE category/ Load (even in SSV sometimes not zero)/ SINR (critical!)';
        return notice;
    }

    function getAnalysisDisplayTitle(analysis) {
        const metrics = analysis.metrics || {};
        const metricGroup = metrics.metric_group || analysis.selection?.metricGroup;
        if (analysis.analysisKind === 'degradation' && metricGroup === 'coverage') {
            return 'Coverage';
        }
        if (analysis.analysisKind === 'degradation' && metricGroup === 'quality') {
            return 'Quality';
        }
        return analysis.label || analysis.selection?.captionCell || 'SSV map';
    }

    function getAnalysisDisplayVerdict(analysis) {
        if (analysis.analysisKind === 'degradation' || analysis.analysisKind === 'throughput_average') {
            const isFailure = Boolean(analysis.isFailure ?? analysis.cross);
            return isFailure ? 'NOK' : 'OK';
        }
        return analysis.verdict;
    }

    function getThroughputSummary(analyses) {
        return analyses.find((analysis) => analysis.analysisKind === 'throughput_average') || null;
    }

    function createThroughputDetails(analysis, throughputSummary) {
        const metrics = analysis.metrics || {};
        const metricGroup = metrics.metric_group || analysis.selection?.metricGroup;
        if (metricGroup !== 'throughput' || analysis.analysisKind !== 'degradation' || !throughputSummary) {
            return null;
        }

        const summaryMetrics = throughputSummary.metrics || {};
        const isDl = (analysis.label || '').toLowerCase().includes('dl');
        const averageValue = isDl ? summaryMetrics.dl_average_mbps : summaryMetrics.ul_average_mbps;
        const minimumValue = isDl ? summaryMetrics.dl_threshold_mbps : summaryMetrics.ul_threshold_mbps;
        const metricLabel = isDl ? 'Throughput DL' : 'Throughput UL';
        const status = (typeof averageValue === 'number' && typeof minimumValue === 'number' && averageValue >= minimumValue) ? 'OK' : 'NOK';

        const details = document.createElement('div');
        details.className = `ssv-analysis-info-body ${status === 'NOK' ? 'nok' : 'ok'}`;
        details.textContent = `${metricLabel} ${status} : Avrage Débit ${isDl ? 'DL' : 'UL'} ${typeof averageValue === 'number' ? `${averageValue.toFixed(2)} Mbps` : '--'} (minimum is ${typeof minimumValue === 'number' ? `${minimumValue.toFixed(0)}Mbps` : 'n/a'})`;
        return details;
    }

    function orderAnalyses(analyses) {
        const filtered = analyses.filter((analysis) => analysis.analysisKind !== 'throughput_average');
        const sheetOrder = new Map();

        filtered.forEach((analysis) => {
            const sheetName = analysis.selection?.sheetName || '';
            if (!sheetOrder.has(sheetName)) {
                sheetOrder.set(sheetName, sheetOrder.size);
            }
        });

        function categoryPriority(analysis) {
            const label = analysis.label || '';
            const metricGroup = (analysis.metrics || {}).metric_group || analysis.selection?.metricGroup;

            if (label === 'Débit DL') return 0;
            if (label === 'Débit UL') return 1;
            if (analysis.analysisKind === 'cross') return 2;
            if (metricGroup === 'coverage') return 3;
            if (metricGroup === 'quality') return 4;
            return 5;
        }

        return [...filtered].sort((left, right) => {
            const leftSheetOrder = sheetOrder.get(left.selection?.sheetName || '') ?? 99;
            const rightSheetOrder = sheetOrder.get(right.selection?.sheetName || '') ?? 99;
            if (leftSheetOrder !== rightSheetOrder) {
                return leftSheetOrder - rightSheetOrder;
            }

            const leftPriority = categoryPriority(left);
            const rightPriority = categoryPriority(right);
            if (leftPriority !== rightPriority) {
                return leftPriority - rightPriority;
            }

            return 0;
        });
    }

    function renderAnalysisCards(analyses) {
        analysisList.innerHTML = '';
        analysisList.hidden = false;
        const throughputSummary = getThroughputSummary(analyses);
        const orderedAnalyses = orderAnalyses(analyses);

        if (throughputSummary) {
            analysisList.appendChild(createThroughputNotice());
        }

        orderedAnalyses.forEach((analysis) => {
            const isFailure = Boolean(analysis.isFailure ?? analysis.cross);
            const card = document.createElement('section');
            card.className = 'ssv-analysis-card';

            const header = document.createElement('div');
            header.className = 'ssv-analysis-header';

            const title = document.createElement('div');
            title.className = 'ssv-analysis-title';
            title.textContent = getAnalysisDisplayTitle(analysis);

            const chip = document.createElement('div');
            chip.className = `ssv-analysis-chip ${isFailure ? 'cross' : 'clear'}`;
            chip.textContent = getAnalysisDisplayVerdict(analysis);

            header.appendChild(title);
            header.appendChild(chip);

            const note = document.createElement('div');
            note.className = 'ssv-analysis-note';
            note.textContent = analysis.selection?.sheetName || '--';
            card.appendChild(note);

            if (analysis.warnings && analysis.warnings.length) {
                const warningBox = document.createElement('div');
                warningBox.className = 'ssv-analysis-warning';
                warningBox.textContent = analysis.warnings.join(' | ');
                card.appendChild(warningBox);
            }

            const throughputDetails = createThroughputDetails(analysis, throughputSummary);
            if (throughputDetails) {
                card.appendChild(throughputDetails);
            }

            const grid = document.createElement('div');
            grid.className = 'ssv-analysis-grid';
            let hasVisual = false;

            if (analysis.previewImage) {
                const extractedFigure = document.createElement('figure');
                extractedFigure.className = 'ssv-analysis-image';
                const extractedCaption = document.createElement('figcaption');
                extractedCaption.textContent = 'Extracted image';
                const extractedImage = document.createElement('img');
                extractedImage.src = analysis.previewImage;
                extractedImage.alt = `${analysis.label || 'SSV'} extracted preview`;
                extractedFigure.appendChild(extractedCaption);
                extractedFigure.appendChild(extractedImage);
                grid.appendChild(extractedFigure);
                hasVisual = true;
            }

            if (analysis.annotatedPreview || analysis.previewImage) {
                const annotatedFigure = document.createElement('figure');
                annotatedFigure.className = 'ssv-analysis-image';
                const annotatedCaption = document.createElement('figcaption');
                annotatedCaption.textContent = analysis.previewImage ? 'Annotated analysis' : 'Summary';
                const annotatedImage = document.createElement('img');
                annotatedImage.src = analysis.annotatedPreview || analysis.previewImage;
                annotatedImage.alt = `${analysis.label || 'SSV'} annotated preview`;
                annotatedFigure.appendChild(annotatedCaption);
                annotatedFigure.appendChild(annotatedImage);
                grid.appendChild(annotatedFigure);
                hasVisual = true;
            }

            const localColorList = document.createElement('div');
            localColorList.className = 'ssv-color-list';
            if ((analysis.detected_colors || []).length) {
                renderColors(analysis.detected_colors || [], localColorList);
            }

            card.appendChild(header);
            if (hasVisual) {
                card.appendChild(grid);
            }
            if ((analysis.detected_colors || []).length) {
                card.appendChild(localColorList);
            }
            analysisList.appendChild(card);
        });
    }

    function renderResult(payload, filename) {
        const analyses = payload.analyses && payload.analyses.length ? payload.analyses : [payload];
        verdict.textContent = payload.verdict;
        verdict.className = `ssv-verdict-card ${payload.isFailure ? 'cross' : 'clear'}`;
        if (summaryShell) {
            summaryShell.hidden = false;
        }
        if (summaryGrid) {
            summaryGrid.classList.add('compact');
        }
        if (previewGrid) {
            previewGrid.hidden = true;
        }
        if (verdictMeta) {
            verdictMeta.hidden = true;
        }
        if (colorsCard) {
            colorsCard.hidden = true;
        }
        if (metricsCard) {
            metricsCard.hidden = true;
        }
        if (siteCenterCard) {
            siteCenterCard.hidden = true;
        }
        if (selectionCard) {
            selectionCard.hidden = true;
        }
        if (previewImage) {
            previewImage.removeAttribute('src');
        }
        if (annotatedPreviewImage) {
            annotatedPreviewImage.removeAttribute('src');
        }
        if (colorList) {
            colorList.innerHTML = '';
        }
        if (metricsGrid) {
            metricsGrid.innerHTML = '';
        }
        if (siteCenter) {
            siteCenter.textContent = '--';
        }
        if (selectionMeta) {
            selectionMeta.textContent = '--';
        }
        renderAnalysisCards(analyses);
    }

    function handleUpload(file) {
        if (!file) return;

        if (!file.name.toLowerCase().endsWith('.xlsx')) {
            showError('Invalid file format. Please upload an .xlsx workbook.');
            return;
        }

        hideError();
        if (summaryShell) {
            summaryShell.hidden = true;
        }
        analysisList.hidden = true;
        uploadHint.textContent = `Selected workbook: ${file.name}`;
        setProgress(8, 'Uploading workbook...');

        const formData = new FormData();
        formData.append('file', file);
        formData.append('debug', '1');

        const xhr = new XMLHttpRequest();
        xhr.open('POST', getApiUrl(), true);

        xhr.upload.onprogress = function (event) {
            if (!event.lengthComputable) return;
            const ratio = event.loaded / event.total;
            setProgress(10 + (ratio * 55), `Uploading workbook... ${Math.round(ratio * 100)}%`);
        };

        xhr.onloadstart = function () {
            setProgress(10, 'Uploading workbook...');
        };

        xhr.onreadystatechange = function () {
            if (xhr.readyState === XMLHttpRequest.HEADERS_RECEIVED) {
                setProgress(78, 'Workbook received. Extracting SSV map images...');
            }
        };

        xhr.onload = function () {
            let payload;
            try {
                payload = JSON.parse(xhr.responseText || '{}');
            } catch (error) {
                showError('The server returned an unreadable response.');
                setProgress(0, 'Upload failed.');
                return;
            }

            if (xhr.status >= 200 && xhr.status < 300 && payload.success) {
                setProgress(100, 'Analysis complete.');
                renderResult(payload, file.name);
                return;
            }

            const message = payload.error || 'SSV validation failed.';
            showError(message);
            setProgress(0, 'Validation failed.');
        };

        xhr.onerror = function () {
            showError('Unable to reach the SSV validation API. Start the local server with `python3 server.py` and try again.');
            setProgress(0, 'Connection failed.');
        };

        xhr.send(formData);
        setTimeout(() => {
            if (xhr.readyState !== XMLHttpRequest.DONE) {
                setProgress(86, 'Analyzing extracted maps and computing validation metrics...');
            }
        }, 450);
    }

    openButtons.forEach((button) => button.addEventListener('click', promptForWorkbook));
    closeButtons.forEach((button) => button.addEventListener('click', closeModal));

    if (dropzone && input) {
        dropzone.addEventListener('click', () => input.click());
        input.addEventListener('change', (event) => {
            const file = event.target.files && event.target.files[0];
            if (file) {
                openModal(false);
            }
            handleUpload(file);
        });

        ['dragenter', 'dragover'].forEach((eventName) => {
            dropzone.addEventListener(eventName, (event) => {
                event.preventDefault();
                dropzone.classList.add('dragging');
            });
        });

        ['dragleave', 'dragend'].forEach((eventName) => {
            dropzone.addEventListener(eventName, (event) => {
                event.preventDefault();
                dropzone.classList.remove('dragging');
            });
        });

        dropzone.addEventListener('drop', (event) => {
            event.preventDefault();
            dropzone.classList.remove('dragging');
            const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
            handleUpload(file);
        });
    }

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && modal && !modal.hidden) {
            closeModal();
        }
    });

    resetState();
})();
