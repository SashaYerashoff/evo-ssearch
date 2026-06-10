    const folderInput = document.getElementById('folderPath');
    const AUTH_ENABLED = {auth_enabled_json};
    const AUTH_CSRF_COOKIE = {auth_csrf_cookie_json};
    const authGate = document.getElementById('authGate');
    const authLoginForm = document.getElementById('authLoginForm');
    const authUsernameInput = document.getElementById('authUsername');
    const authPasswordInput = document.getElementById('authPassword');
    const authLoginBtn = document.getElementById('authLoginBtn');
    const authLoginStatus = document.getElementById('authLoginStatus');
    const authTokenBtn = document.getElementById('authTokenBtn');
    let authCurrentUser = null;
    const indexBtn = document.getElementById('indexBtn');
    const indexStatus = document.getElementById('indexStatus');
    const searchInput = document.getElementById('searchQuery');
    const searchBtn = document.getElementById('searchBtn');
    const imageUpload = document.getElementById('imageUpload');
    const imageUploadName = document.getElementById('imageUploadName');
    const imageQueryPanel = document.getElementById('imageQueryPanel');
    const queryImagePreview = document.getElementById('queryImagePreview');
    const queryImageThumb = document.getElementById('queryImageThumb');
    const imageSearchBtn = document.getElementById('imageSearchBtn');
    const archiveModeBtn = document.getElementById('archiveModeBtn');
    const videoModeBtn = document.getElementById('videoModeBtn');
    const archiveBox = document.getElementById('archiveBox');
    const videoBox = document.getElementById('videoBox');
    const videoPathInput = document.getElementById('videoPath');
    const videoModelInput = document.getElementById('videoModel');
    const videoFrameCount = document.getElementById('videoFrameCount');
    const videoSampleFpsInput = document.getElementById('videoSampleFps');
    const videoPromptInput = document.getElementById('videoPrompt');
    const saveVideoPromptInput = document.getElementById('saveVideoPrompt');
    const videoRunBtn = document.getElementById('videoRunBtn');
    const videoStatus = document.getElementById('videoStatus');
    const videoOutput = document.getElementById('videoOutput');
    const videoFrames = document.getElementById('videoFrames');
    const saveSummaryBtn = document.getElementById('saveSummaryBtn');
    const monitorModeBtn = document.getElementById('monitorModeBtn');
    const monitorBox = document.getElementById('monitorBox');
    const agentModeBtn = document.getElementById('agentModeBtn');
    const agentBox = document.getElementById('agentBox');
    const agentSkillList = document.getElementById('agentSkillList');
    const agentCreateSkillBtn = document.getElementById('agentCreateSkillBtn');
    const agentSkillModal = document.getElementById('agentSkillModal');
    const closeAgentSkillModalBtn = document.getElementById('closeAgentSkillModal');
    const agentSkillCancelBtn = document.getElementById('agentSkillCancelBtn');
    const agentSkillSaveBtn = document.getElementById('agentSkillSaveBtn');
    const agentSkillModalTitle = document.getElementById('agentSkillModalTitle');
    const agentSkillNameInput = document.getElementById('agentSkillNameInput');
    const agentSkillSlugInput = document.getElementById('agentSkillSlugInput');
    const agentSkillMeta = document.getElementById('agentSkillMeta');
    const agentSkillContentInput = document.getElementById('agentSkillContentInput');
    const headerStatusText = document.querySelector('.header-status-text');
    const luxriotChannelSelect = document.getElementById('luxriotChannelSelect');
    const luxriotRefreshChannelsBtn = document.getElementById('luxriotRefreshChannels');
    const luxriotBatchSizeSelect = document.getElementById('luxriotBatchSize');
    const luxriotBatchInfo = document.getElementById('luxriotBatchInfo');
    const luxriotStatusLabel = document.getElementById('luxriotStatus');
    const luxriotPreviewImg = document.getElementById('luxriotPreview');
    const luxriotOverlay = document.getElementById('luxriotOverlay');
    const luxriotToggleCaptureBtn = document.getElementById('luxriotToggleCapture');
    const luxriotFlushCaptureBtn = document.getElementById('luxriotFlushCapture');
    const luxriotPromptSettingsBtn = document.getElementById('luxriotPromptSettingsBtn');
    const luxriotPromptModal = document.getElementById('luxriotPromptModal');
    const closeLuxriotPromptModalBtn = document.getElementById('closeLuxriotPromptModal');
    const luxriotPromptCloseBtn = document.getElementById('luxriotPromptCloseBtn');
    const luxriotPromptApplyBtn = document.getElementById('luxriotPromptApplyBtn');
    const luxriotPromptModalInput = document.getElementById('luxriotPromptModalInput');
    const luxriotPromptModalMeta = document.getElementById('luxriotPromptModalMeta');
    const luxriotPromptTabButtons = Array.from(document.querySelectorAll('[data-luxriot-prompt-tab]'));
    const luxriotRefreshSummariesBtn = document.getElementById('luxriotRefreshSummaries');
    const luxriotSummaryChannelSelect = document.getElementById('luxriotSummaryChannelSelect');
    const luxriotSummaryRunSelect = document.getElementById('luxriotSummaryRunSelect');
    const luxriotSummaryRangeSelect = document.getElementById('luxriotSummaryRangeSelect');
    const luxriotSummaryLevelSelect = document.getElementById('luxriotSummaryLevelSelect');
    const luxriotSummaryCustomTime = document.getElementById('luxriotSummaryCustomTime');
    const luxriotSummaryFromInput = document.getElementById('luxriotSummaryFromInput');
    const luxriotSummaryToInput = document.getElementById('luxriotSummaryToInput');
    const luxriotSummaryApplyFiltersBtn = document.getElementById('luxriotSummaryApplyFiltersBtn');
    const luxriotSummaryBackBtn = document.getElementById('luxriotSummaryBackBtn');
    const luxriotSummaryMeta = document.getElementById('luxriotSummaryMeta');
    const luxriotSummaryFollowBtn = document.getElementById('luxriotSummaryFollowBtn');
    const luxriotSummaryPauseBtn = document.getElementById('luxriotSummaryPauseBtn');
    const luxriotSummaryViewBtn = document.getElementById('luxriotSummaryViewBtn');
    const luxriotSummaryCollapseAllBtn = document.getElementById('luxriotSummaryCollapseAllBtn');
    const luxriotSummaryJumpBtn = document.getElementById('luxriotSummaryJumpBtn');
    const luxriotSummaries = document.getElementById('luxriotSummaries');
    const luxriotStreams = document.getElementById('luxriotStreams');
    const luxriotRefreshStreamsBtn = document.getElementById('luxriotRefreshStreams');
    const luxriotStopAllVideoBtn = document.getElementById('luxriotStopAllVideo');
    const luxriotStopAllAnalyticsBtn = document.getElementById('luxriotStopAllAnalytics');
    const luxriotPromptInput = document.getElementById('luxriotPrompt');
    const luxriotLiveModelInput = document.getElementById('luxriotLiveModel');
    const luxriotSystemPromptInput = document.getElementById('luxriotSystemPrompt');
    const luxriotRollupPromptL1Input = document.getElementById('luxriotRollupPromptL1');
    const luxriotRollupPromptL2Input = document.getElementById('luxriotRollupPromptL2');
    const luxriotRollupPromptL3Input = document.getElementById('luxriotRollupPromptL3');
    const luxriotJsonAlertPromptInput = document.getElementById('luxriotJsonAlertPrompt');
    const luxriotBookmarkEnabledInput = document.getElementById('luxriotBookmarkEnabled');
    const luxriotBookmarkCooldownInput = document.getElementById('luxriotBookmarkCooldown');
    const probeChannelSelect = document.getElementById('probeChannelSelect');
    const probeTopKInput = document.getElementById('probeTopK');
    const probePosFloorInput = document.getElementById('probePosFloor');
    const probeMarginInput = document.getElementById('probeMargin');
    const probeNameInput = document.getElementById('probeName');
    const probeRunBtn = document.getElementById('probeRunBtn');
    const probeSaveBtn = document.getElementById('probeSaveBtn');
    const probeDeleteBtn = document.getElementById('probeDeleteBtn');
    const probeEditBtn = document.getElementById('probeEditBtn');
    const probeEditorModal = document.getElementById('probeEditorModal');
    const closeProbeEditorBtn = document.getElementById('closeProbeEditor');
    const probeEditorCloseBtn = document.getElementById('probeEditorCloseBtn');
    const probeResults = document.getElementById('probeResults');
    const probeStatus = document.getElementById('probeStatus');
    const probeBookmarkSeverityInput = document.getElementById('probeBookmarkSeverity');
    const probeBookmarkToggle = document.getElementById('probeBookmarkToggle');
    const probeBookmarkCooldownLocalInput = document.getElementById('probeBookmarkCooldownSecLocal');
    const probeBookmarkDedupeWindowLocalInput = document.getElementById('probeBookmarkDedupeWindowSecLocal');
    const probeFpsInput = document.getElementById('probeFps');
    const probeWindowSecInput = document.getElementById('probeWindowSec');
    const probeStreamToggleBtn = document.getElementById('probeStreamToggle');
    const probeCaptureStatus = document.getElementById('probeCaptureStatus');
    const probeHitsMeta = document.getElementById('probeHitsMeta');
    const probeCards = document.getElementById('probeCards');
    const probeNewBtn = document.getElementById('probeNewBtn');
    const probeReloadBtn = document.getElementById('probeReloadBtn');
    const probePreviewImg = document.getElementById('probePreviewImg');
    const probePreviewViewport = probePreviewImg ? probePreviewImg.closest('.monitor-stream-preview') : null;
    const probePreviewOverlay = document.getElementById('probePreviewOverlay');
    const probeRoiLayer = document.getElementById('probeRoiLayer');
    const probeRoiBox = document.getElementById('probeRoiBox');
    const probeRoiToggleBtn = document.getElementById('probeRoiToggle');
    const probeRoiClearBtn = document.getElementById('probeRoiClear');
    const probeSnapBtn = document.getElementById('probeSnapBtn');
    const probeRoiInfo = document.getElementById('probeRoiInfo');
    const probeSnapModal = document.getElementById('probeSnapModal');
    const closeProbeSnapBtn = document.getElementById('closeProbeSnap');
    const probeSnapCloseBtn = document.getElementById('probeSnapCloseBtn');
    const probeSnapExportBtn = document.getElementById('probeSnapExportBtn');
    const probeSnapUseBtn = document.getElementById('probeSnapUseBtn');
    const probeSnapActualSizeInput = document.getElementById('probeSnapActualSize');
    const probeSnapMeta = document.getElementById('probeSnapMeta');
    const probeSnapPreview = document.getElementById('probeSnapPreview');
    const probeSnapImg = document.getElementById('probeSnapImg');
    const probePairsContainer = document.getElementById('probePairs');
    const probePairRows = document.getElementById('probePairRows');
    const probeImageFile = document.getElementById('probeImageFile');
    const probeImageFileName = document.getElementById('probeImageFileName');
    const probeImageClearBtn = document.getElementById('probeImageClear');
    const probeImageClearRow = probeImageClearBtn ? probeImageClearBtn.closest('.image-probe-clear-row') : null;
    const probeImageEnableToggle = document.getElementById('probeImageEnableToggle');
    const probeImageStatus = document.getElementById('probeImageStatus');
    const probeImageThumb = document.getElementById('probeImageThumb');
    const probeImageOverlay = document.getElementById('probeImageOverlay');
    const probeImagePanel = document.querySelector('.image-probe-panel');
    const probeImagePosInput = document.getElementById('probeImagePos');
    const probeDetLeftBtn = document.getElementById('probeDetLeft');
    const probeDetRightBtn = document.getElementById('probeDetRight');
    const resultLimitSelect = document.getElementById('resultLimit');
    const sortBySelect = document.getElementById('sortBy');
    const searchScopeSelect = document.getElementById('searchScope');
    const showCommentedBtn = document.getElementById('showCommentedBtn');
    const resultsContainer = document.getElementById('results');
    const archiveInspectorBody = document.getElementById('archiveInspectorBody');
    const archiveInspectorEmpty = document.getElementById('archiveInspectorEmpty');
    const archiveChannelFilter = document.getElementById('archiveChannelFilter');
    const archiveProbeFilter = document.getElementById('archiveProbeFilter');
    const archiveTimeFilter = document.getElementById('archiveTimeFilter');
    const archiveDetectionsLimit = document.getElementById('archiveDetectionsLimit');
    const loadDetectionsBtn = document.getElementById('loadDetectionsBtn');
    const refreshDetectionsFiltersBtn = document.getElementById('refreshDetectionsFiltersBtn');
    const archiveDetectionsPrevBtn = document.getElementById('archiveDetectionsPrev');
    const archiveDetectionsNextBtn = document.getElementById('archiveDetectionsNext');
    const archiveDetectionsMeta = document.getElementById('archiveDetectionsMeta');
    const probeBufferInfo = document.getElementById('probeBufferInfo');
    const probeEnableToggle = document.getElementById('probeEnableToggle');
    const probeBenchBtn = document.getElementById('probeBenchBtn');
    const probeBenchOutput = document.getElementById('probeBenchOutput');
    const monitorProbeSummary = document.getElementById('monitorProbeSummary');
    const monitorSelectionStatus = document.getElementById('monitorSelectionStatus');
    const imageLightboxModal = document.getElementById('imageLightboxModal');
    const closeImageLightboxBtn = document.getElementById('closeImageLightbox');
    const imageLightboxImg = document.getElementById('imageLightboxImg');
    const imageLightboxMeta = document.getElementById('imageLightboxMeta');
    
    let currentFolder = '';
    let currentMode = 'archive';
    let videoTimerHandle = null;
    let videoRequestStarted = 0;
    let lastSummaryText = '';
    let lastSummaryTarget = null;
    let segmentContextByIndex = {};
    let archiveRenderedResults = [];
    let archiveRenderedCommented = false;
    let activeArchiveInspectorIndex = -1;
    let luxriotSummaryLogCache = [];
    const luxriotSummaryChannelCache = {};
    const luxriotSummarySeenKeys = {};
    let luxriotSummaryUnread = 0;
    let luxriotSummaryChannel = null;
    let luxriotSummaryRunFilter = 'latest';
    let luxriotSummaryRangePreset = '6h';
    let luxriotSummaryFromTs = null;
    let luxriotSummaryToTs = null;
    let luxriotSummaryLevel = 'L0';
    let luxriotSummaryRollupStack = [];
    let luxriotSummaryRollupRows = [];
    const luxriotSummaryRollupCache = {};
    let luxriotSummaryFollowLive = true;
    let luxriotSummaryAutoRefresh = true;
    let luxriotSummaryCompactMode = false;
    const luxriotSummaryCollapsedByChannel = {};
    const luxriotDefaults = {
        channelId: {luxriot_default_channel},
        snapshotInterval: {luxriot_snapshot_interval},
        snapshotMaxEdge: {luxriot_snapshot_max_edge},
        baseUrl: {luxriot_base_url_json},
        batchSize: {luxriot_batch_default}
    };
    let luxriotActiveChannel = luxriotDefaults.channelId;
    let luxriotPreviewTimer = null;
    let luxriotSummaryTimer = null;
    let luxriotSummaryRefreshInFlight = false;
    let luxriotSummaryRefreshQueued = null;
    let luxriotStreamsCache = [];
    const luxriotChannelNameById = {};
    const luxriotCaptureRunningByChannel = {};
    let luxriotPromptModalTab = 'stream';
    let luxriotInitialized = false;
    const probeHitsCacheByKey = {};
    const probeHitsOffsetByKey = {};
    const probeFramesByKey = {};
    const probeHitsUpdatedByKey = {};
    const probeWindowSecByKey = {};
    let probePairsState = [];
    let probeImageState = null;
    let probeRoiEnabled = false;
    let probeRoiNorm = null;
    let probeRoiDraftNorm = null;
    let probeRoiDrawState = null;
    let probeSnapState = null;
    let imageProbeEnabled = false;
    let probeList = [];
    let probeCatalog = [];
    let activeProbeId = null;
    const probeCaptureState = {};
    const probeChannelRuntime = {};
    const probeCaptureManualStop = {};
    let probeRunTimer = null;
    let probeRunInFlight = false;
    let probePreviewTimer = null;
    let probePreviewChannelId = null;
    let lastProbeRefresh = 0;
    let probeStatusTimer = null;
    let archiveDetectionsOffset = 0;
    let archiveDetectionsTotal = 0;
    let archiveDetectionsHasMore = false;
    const channelCaptureConfig = {};
    const channelFpsDesired = {};
    const ADMIN_TOKEN_STORAGE_KEY = 'evs_admin_token';
    const LUXRIOT_LIVE_MODEL_STORAGE_KEY = 'evs_luxriot_live_model';
    const VIDEO_MODEL_STORAGE_KEY = 'evs_video_model';
    let lmModelCatalog = {
        models: [],
        defaultModel: '',
        source: 'fallback',
        error: '',
    };
    let lmModelCatalogPromise = null;
    let agentSkillDraft = null;

    function normalizeModelId(value) {
        return String(value || '').trim();
    }

    function uniqueModelIds(...values) {
        const seen = new Set();
        const out = [];
        values.flat().forEach((value) => {
            const normalized = normalizeModelId(value);
            if (!normalized || seen.has(normalized)) return;
            seen.add(normalized);
            out.push(normalized);
        });
        return out;
    }

    function setModelSelectOptions(selectEl, selectedValue = '', fallbackValue = '') {
        if (!(selectEl instanceof HTMLSelectElement)) return;
        const selected = normalizeModelId(selectedValue);
        const fallback = normalizeModelId(fallbackValue || lmModelCatalog.defaultModel);
        const options = uniqueModelIds(lmModelCatalog.models || [], selected, fallback);
        const nextValue = selected || fallback || options[0] || '';
        if (!options.length) {
            selectEl.innerHTML = '<option value="">No models available</option>';
            selectEl.value = '';
            return;
        }
        selectEl.innerHTML = options
            .map((modelId) => `<option value="${escapeHtml(modelId)}">${escapeHtml(modelId)}</option>`)
            .join('');
        if (options.includes(nextValue)) {
            selectEl.value = nextValue;
        } else {
            selectEl.value = options[0];
        }
    }

    function syncStoredModelSelection(selectEl, storageKey) {
        if (!(selectEl instanceof HTMLSelectElement) || !storageKey) return;
        selectEl.addEventListener('change', () => {
            const value = normalizeModelId(selectEl.value);
            if (value) {
                localStorage.setItem(storageKey, value);
            } else {
                localStorage.removeItem(storageKey);
            }
        });
    }

    function applyLmModelCatalogToUi() {
        const defaultModel = normalizeModelId(lmModelCatalog.defaultModel);
        if (luxriotLiveModelInput) {
            const preferredLiveModel = normalizeModelId(luxriotLiveModelInput.value)
                || normalizeModelId(localStorage.getItem(LUXRIOT_LIVE_MODEL_STORAGE_KEY))
                || defaultModel;
            setModelSelectOptions(luxriotLiveModelInput, preferredLiveModel, defaultModel);
        }
        if (videoModelInput) {
            const preferredVideoModel = normalizeModelId(videoModelInput.value)
                || normalizeModelId(localStorage.getItem(VIDEO_MODEL_STORAGE_KEY))
                || defaultModel;
            setModelSelectOptions(videoModelInput, preferredVideoModel, defaultModel);
        }
    }

    async function loadLmModelCatalog(force = false) {
        if (lmModelCatalogPromise && !force) {
            return lmModelCatalogPromise;
        }
        lmModelCatalogPromise = (async () => {
            try {
                const url = force ? '/lm/models?force=1' : '/lm/models';
                const response = await fetch(url, { cache: 'no-store' });
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.error || 'Failed to load models');
                }
                lmModelCatalog = {
                    models: uniqueModelIds(data.models || []),
                    defaultModel: normalizeModelId(data.default_model),
                    source: String(data.source || 'fallback'),
                    error: normalizeModelId(data.error),
                };
            } catch (error) {
                lmModelCatalog = {
                    models: uniqueModelIds(
                        lmModelCatalog.models || [],
                        luxriotLiveModelInput ? luxriotLiveModelInput.value : '',
                        videoModelInput ? videoModelInput.value : '',
                    ),
                    defaultModel: normalizeModelId(lmModelCatalog.defaultModel || ''),
                    source: 'fallback',
                    error: error.message || String(error),
                };
            }
            applyLmModelCatalogToUi();
            return lmModelCatalog;
        })();
        return lmModelCatalogPromise;
    }

    syncStoredModelSelection(luxriotLiveModelInput, LUXRIOT_LIVE_MODEL_STORAGE_KEY);
    syncStoredModelSelection(videoModelInput, VIDEO_MODEL_STORAGE_KEY);
    void loadLmModelCatalog();

    function getAdminToken() {
        return (localStorage.getItem(ADMIN_TOKEN_STORAGE_KEY) || '').trim();
    }

    function saveAdminToken(token) {
        const clean = (token || '').trim();
        if (clean) {
            localStorage.setItem(ADMIN_TOKEN_STORAGE_KEY, clean);
        } else {
            localStorage.removeItem(ADMIN_TOKEN_STORAGE_KEY);
        }
    }

    (function seedAdminTokenFromQuery() {
        try {
            const url = new URL(window.location.href);
            const qp = (url.searchParams.get('admin_token') || '').trim();
            if (!qp) return;
            saveAdminToken(qp);
            url.searchParams.delete('admin_token');
            window.history.replaceState({}, '', url.toString());
        } catch (_) {
            // no-op
        }
    })();

    const rawFetch = window.fetch.bind(window);
    function cookieValue(name) {
        const prefix = `${encodeURIComponent(name)}=`;
        const item = document.cookie
            .split(';')
            .map((part) => part.trim())
            .find((part) => part.startsWith(prefix));
        return item ? decodeURIComponent(item.slice(prefix.length)) : '';
    }

    function setAuthGateVisible(visible, message = '') {
        if (!AUTH_ENABLED || !authGate) return;
        authGate.classList.toggle('is-hidden', !visible);
        document.body.classList.toggle('auth-required', visible);
        if (authLoginStatus) authLoginStatus.textContent = message;
        if (visible && authUsernameInput) authUsernameInput.focus();
    }

    window.fetch = (input, init = {}) => {
        const options = init ? { ...init } : {};
        const method = String(
            options.method || (input instanceof Request ? input.method : 'GET')
        ).toUpperCase();
        const headers = new Headers(options.headers || {});
        if (AUTH_ENABLED && !['GET', 'HEAD', 'OPTIONS'].includes(method)) {
            const csrfToken = cookieValue(AUTH_CSRF_COOKIE);
            if (csrfToken && !headers.has('X-CSRF-Token')) {
                headers.set('X-CSRF-Token', csrfToken);
            }
        }
        const token = getAdminToken();
        if (token) {
            if (!headers.has('X-Admin-Token') && !headers.has('Authorization')) {
                headers.set('X-Admin-Token', token);
            }
        }
        options.headers = headers;
        return rawFetch(input, options).then((response) => {
            const url = typeof input === 'string' ? input : input.url;
            if (
                AUTH_ENABLED
                && response.status === 401
                && !String(url).includes('/auth/login')
            ) {
                authCurrentUser = null;
                setAuthGateVisible(true, 'Session expired. Sign in again.');
            }
            return response;
        });
    };

    async function loadCurrentUser() {
        if (!AUTH_ENABLED) return;
        try {
            const response = await fetch('/auth/me', { cache: 'no-store' });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Sign in required');
            authCurrentUser = data.user || null;
            setAuthGateVisible(false);
            if (authTokenBtn && authCurrentUser) {
                authTokenBtn.title = `${authCurrentUser.displayName || authCurrentUser.username} · Sign out`;
                authTokenBtn.style.opacity = '1';
            }
        } catch (error) {
            setAuthGateVisible(true, error.message || 'Sign in required');
        }
    }

    if (AUTH_ENABLED && authLoginForm) {
        authLoginForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            if (authLoginBtn) authLoginBtn.disabled = true;
            if (authLoginStatus) authLoginStatus.textContent = '';
            try {
                const response = await fetch('/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        username: authUsernameInput ? authUsernameInput.value : '',
                        password: authPasswordInput ? authPasswordInput.value : '',
                    }),
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || 'Sign in failed');
                authCurrentUser = data.user || null;
                if (authPasswordInput) authPasswordInput.value = '';
                await loadCurrentUser();
            } catch (error) {
                setAuthGateVisible(true, error.message || 'Sign in failed');
            } finally {
                if (authLoginBtn) authLoginBtn.disabled = false;
            }
        });
        void loadCurrentUser();
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function sanitizeUrl(url) {
        const value = String(url || '').trim();
        if (!value) return '';
        if (/^(https?:\/\/|\/|\.\.?\/|\?)/i.test(value)) {
            return value;
        }
        if (/^data:image\//i.test(value)) {
            return value;
        }
        return '#';
    }

    function isPreviewableImageUrl(url) {
        const value = String(url || '').trim();
        if (!value || value === '#') return false;
        if (/^data:image\//i.test(value)) return true;
        if (/\/detections\/image\?/i.test(value)) return true;
        if (/^\/image\//i.test(value)) return true;
        if (/\/luxriot\/snapshot/i.test(value)) return true;
        return /\.(?:png|jpe?g|webp|gif|bmp|svg)(?:[?#].*)?$/i.test(value);
    }

    function renderPreviewableLink(label, safeUrl, title = '') {
        const titleAttr = title ? ` title="${escapeHtml(title)}"` : '';
        const previewAttrs = isPreviewableImageUrl(safeUrl)
            ? ` class="markdown-preview-link" data-preview-image="${escapeHtml(safeUrl)}"`
            : '';
        return `<a href="${escapeHtml(safeUrl)}" target="_blank" rel="noopener noreferrer"${titleAttr}${previewAttrs}>${escapeHtml(label)}</a>`;
    }

    function parseMarkdownTableRow(line) {
        const raw = String(line || '').trim();
        if (!raw.includes('|')) return [];
        const normalized = raw.replace(/^\|/, '').replace(/\|$/, '');
        return normalized.split('|').map((cell) => cell.trim());
    }

    function isMarkdownTableDivider(line) {
        const cells = parseMarkdownTableRow(line);
        if (cells.length < 2) return false;
        return cells.every((cell) => /^:?-{3,}:?$/.test(cell));
    }

    function renderMarkdownTable(tableLines) {
        if (!Array.isArray(tableLines) || tableLines.length < 2) return '';
        const headerCells = parseMarkdownTableRow(tableLines[0]);
        if (headerCells.length < 2 || !isMarkdownTableDivider(tableLines[1])) return '';

        const alignments = parseMarkdownTableRow(tableLines[1]).map((cell) => {
            const trimmed = String(cell || '').trim();
            if (trimmed.startsWith(':') && trimmed.endsWith(':')) return 'center';
            if (trimmed.endsWith(':')) return 'right';
            return 'left';
        });

        const renderCells = (cells, tagName) => {
            return cells.map((cell, idx) => {
                const align = alignments[idx] || 'left';
                return `<${tagName} style="text-align:${align}">${renderMarkdownInline(cell)}</${tagName}>`;
            }).join('');
        };

        const headerHtml = `<thead><tr>${renderCells(headerCells, 'th')}</tr></thead>`;
        const bodyRows = tableLines.slice(2).map((line) => {
            const cells = parseMarkdownTableRow(line);
            if (!cells.length) return '';
            while (cells.length < headerCells.length) {
                cells.push('');
            }
            return `<tr>${renderCells(cells.slice(0, headerCells.length), 'td')}</tr>`;
        }).filter(Boolean);

        return `<div class="markdown-table-wrap"><table>${headerHtml}<tbody>${bodyRows.join('')}</tbody></table></div>`;
    }

    function renderMarkdownInline(text) {
        const source = String(text || '');
        const tokens = [];
        const makeToken = (html) => `\x00MD${tokens.push(html) - 1}\x00`;
        const placeholder = source
            .replace(/`([^`\n]+)`/g, (_, codeText) => {
                return makeToken(`<code>${escapeHtml(codeText)}</code>`);
            })
            .replace(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+"([^"]*)")?\)/g, (_, altText, url, title) => {
                const safeUrl = sanitizeUrl(url);
                const titleAttr = title ? ` title="${escapeHtml(title)}"` : '';
                return makeToken(
                    `<img class="markdown-inline-image" src="${escapeHtml(safeUrl)}" alt="${escapeHtml(altText || '')}"${titleAttr} loading="lazy" data-preview-image="${escapeHtml(safeUrl)}" />`
                );
            })
            .replace(/\[([^\]]+)\]\(([^)\s]+)(?:\s+"([^"]*)")?\)/g, (_, label, url, title) => {
                const safeUrl = sanitizeUrl(url);
                return makeToken(renderPreviewableLink(label, safeUrl, title));
            })
            .replace(/(^|[\s(])((?:https?:\/\/|\/)[^\s<]+?)(?=([),.!?]?(?:\s|$)))/g, (_, prefix, url) => {
                const safeUrl = sanitizeUrl(url);
                return `${prefix}${makeToken(renderPreviewableLink(url, safeUrl))}`;
            });
        let out = escapeHtml(placeholder);
        out = out
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/__(.+?)__/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/_(.+?)_/g, '<em>$1</em>');
        out = out.replace(/\x00MD(\d+)\x00/g, (_, idx) => tokens[Number(idx)] || '');
        return out;
    }

    function renderMarkdown(text) {
        const source = String(text || '').replace(/\r\n?/g, '\n').trim();
        if (!source) return '';

        const lines = source.split('\n');
        const htmlParts = [];
        let paragraphLines = [];
        let ulItems = [];
        let olItems = [];
        let inCodeFence = false;
        let codeFenceLang = '';
        let codeFenceLines = [];

        const flushParagraph = () => {
            if (!paragraphLines.length) return;
            const body = paragraphLines.map((line) => renderMarkdownInline(line)).join('<br>');
            htmlParts.push(`<p>${body}</p>`);
            paragraphLines = [];
        };

        const flushLists = () => {
            if (ulItems.length) {
                htmlParts.push(`<ul>${ulItems.map((item) => `<li>${item}</li>`).join('')}</ul>`);
                ulItems = [];
            }
            if (olItems.length) {
                htmlParts.push(`<ol>${olItems.map((item) => `<li>${item}</li>`).join('')}</ol>`);
                olItems = [];
            }
        };

        const flushCodeFence = () => {
            if (!inCodeFence) return;
            const classAttr = codeFenceLang ? ` class="language-${escapeHtml(codeFenceLang)}"` : '';
            htmlParts.push(
                `<pre><code${classAttr}>${escapeHtml(codeFenceLines.join('\n'))}</code></pre>`
            );
            inCodeFence = false;
            codeFenceLang = '';
            codeFenceLines = [];
        };

        for (let i = 0; i < lines.length; i += 1) {
            const rawLine = lines[i];
            const line = String(rawLine || '');
            const trimmed = line.trim();
            const fenceMatch = trimmed.match(/^```\s*([\w-]+)?\s*$/);
            if (fenceMatch) {
                if (inCodeFence) {
                    flushCodeFence();
                } else {
                    flushParagraph();
                    flushLists();
                    inCodeFence = true;
                    codeFenceLang = String(fenceMatch[1] || '').trim();
                    codeFenceLines = [];
                }
                continue;
            }

            if (inCodeFence) {
                codeFenceLines.push(line);
                continue;
            }

            if (!trimmed) {
                flushParagraph();
                flushLists();
                continue;
            }

            const nextLine = String(lines[i + 1] || '').trim();
            const maybeTable = trimmed.includes('|') && nextLine && isMarkdownTableDivider(nextLine);
            if (maybeTable) {
                flushParagraph();
                flushLists();
                const tableLines = [line, lines[i + 1]];
                i += 2;
                while (i < lines.length) {
                    const candidate = String(lines[i] || '');
                    const candidateTrimmed = candidate.trim();
                    if (!candidateTrimmed || !candidateTrimmed.includes('|')) {
                        i -= 1;
                        break;
                    }
                    tableLines.push(candidate);
                    i += 1;
                }
                const tableHtml = renderMarkdownTable(tableLines);
                if (tableHtml) {
                    htmlParts.push(tableHtml);
                    continue;
                }
                i -= 1;
            }

            const headingMatch = trimmed.match(/^(#{1,6})\s+(.+)$/);
            if (headingMatch) {
                flushParagraph();
                flushLists();
                const level = Math.min(6, Math.max(1, headingMatch[1].length));
                htmlParts.push(`<h${level}>${renderMarkdownInline(headingMatch[2])}</h${level}>`);
                continue;
            }

            const quoteMatch = trimmed.match(/^>\s?(.*)$/);
            if (quoteMatch) {
                flushParagraph();
                flushLists();
                htmlParts.push(`<blockquote>${renderMarkdownInline(quoteMatch[1] || '')}</blockquote>`);
                continue;
            }

            const ulMatch = trimmed.match(/^[-*]\s+(.+)$/);
            if (ulMatch) {
                flushParagraph();
                if (olItems.length) {
                    flushLists();
                }
                ulItems.push(renderMarkdownInline(ulMatch[1]));
                continue;
            }

            const olMatch = trimmed.match(/^\d+\.\s+(.+)$/);
            if (olMatch) {
                flushParagraph();
                if (ulItems.length) {
                    flushLists();
                }
                olItems.push(renderMarkdownInline(olMatch[1]));
                continue;
            }

            paragraphLines.push(line);
        }

        flushParagraph();
        flushLists();
        flushCodeFence();
        return htmlParts.join('');
    }

    function splitSummaryAndJson(text) {
        const full = String(text || '').trim();
        if (!full) {
            return { main: '', json: '' };
        }

        const fenced = full.match(/```json\s*([\s\S]*?)```/i);
        if (fenced && fenced[1]) {
            const jsonBlock = String(fenced[1] || '').trim();
            const mainText = full.replace(fenced[0], '').trim();
            return { main: mainText, json: jsonBlock };
        }

        const marker = 'ALERTS_JSON:';
        const markerIndex = full.toUpperCase().indexOf(marker);
        if (markerIndex >= 0) {
            const mainText = full.slice(0, markerIndex).trim();
            const jsonBlock = full.slice(markerIndex + marker.length).trim();
            if (jsonBlock) {
                return { main: mainText, json: jsonBlock };
            }
        }

        const trailingStart = full.lastIndexOf('\n{');
        const startIndex = trailingStart >= 0 ? trailingStart + 1 : (full.startsWith('{') ? 0 : -1);
        if (startIndex >= 0) {
            const jsonCandidate = full.slice(startIndex).trim();
            const looksLikeAlerts = (jsonCandidate.includes('"alerts"') || jsonCandidate.includes("'alerts'"));
            if (looksLikeAlerts && jsonCandidate.startsWith('{') && jsonCandidate.endsWith('}')) {
                const mainText = full.slice(0, startIndex).trim();
                return { main: mainText, json: jsonCandidate };
            }
        }

        return { main: full, json: '' };
    }

    function formatDuration(seconds) {
        if (!Number.isFinite(seconds)) return 'n/a';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}m ${secs}s`;
    }

    const buttonBusyState = new WeakMap();

    function setButtonBusy(button, busy) {
        if (!(button instanceof HTMLButtonElement)) return;
        if (busy) {
            if (!buttonBusyState.has(button)) {
                buttonBusyState.set(button, Boolean(button.disabled));
            }
            button.disabled = true;
            button.classList.add('is-loading');
            button.setAttribute('aria-busy', 'true');
            return;
        }
        const wasDisabled = buttonBusyState.has(button) ? Boolean(buttonBusyState.get(button)) : false;
        buttonBusyState.delete(button);
        button.classList.remove('is-loading');
        button.removeAttribute('aria-busy');
        button.disabled = wasDisabled;
    }

    function startVideoTimer() {
        videoRequestStarted = performance.now();
        if (videoTimerHandle) clearInterval(videoTimerHandle);
        videoTimerHandle = setInterval(() => {
            const elapsed = (performance.now() - videoRequestStarted) / 1000;
            const base = videoStatus.dataset.base || '';
            videoStatus.textContent = `${base} · ${elapsed.toFixed(1)}s`;
        }, 200);
    }

    function stopVideoTimer(finalize = false) {
        const elapsed = videoRequestStarted ? (performance.now() - videoRequestStarted) / 1000 : 0;
        if (videoTimerHandle) {
            clearInterval(videoTimerHandle);
            videoTimerHandle = null;
        }
        if (finalize) {
            const base = videoStatus.dataset.base || '';
            videoStatus.textContent = `${base} · ${elapsed.toFixed(1)}s`;
        }
        videoRequestStarted = 0;
    }

    function setMode(mode) {
        currentMode = mode;
        archiveModeBtn.classList.toggle('active', mode === 'archive');
        videoModeBtn.classList.toggle('active', mode === 'video');
        monitorModeBtn.classList.toggle('active', mode === 'monitor');
        if (agentModeBtn) agentModeBtn.classList.toggle('active', mode === 'agent');
        if (headerStatusText) {
            const statusByMode = {
                archive: 'Archive Research Ready',
                video: 'Live Video Ops',
                monitor: 'Probe Monitoring',
                agent: 'Agent Session Active',
            };
            headerStatusText.textContent = statusByMode[mode] || 'Command Center Online';
        }
        if (archiveBox) {
            archiveBox.style.display = mode === 'archive' ? 'grid' : 'none';
        }
        videoBox.style.display = mode === 'video' ? 'grid' : 'none';
        monitorBox.style.display = mode === 'monitor' ? 'grid' : 'none';
        if (agentBox) agentBox.style.display = mode === 'agent' ? 'grid' : 'none';
        if (mode === 'video') {
            ensureLuxriotInit();
            startLuxriotPreview();
            refreshLuxriotSummaryView(getSelectedSummaryChannel(), true);
            refreshLuxriotStreams();
            startLuxriotSummaryPoll();
            syncProbeChannelSelect();
        } else if (mode === 'monitor') {
            ensureLuxriotInit();
            syncProbeChannelSelect();
            syncProbePreview(getSelectedProbeChannelId());
            refreshProbeStatus();
            loadProbeList();
            startProbeStatusPoll();
        } else {
            stopLuxriotPreview();
            stopLuxriotSummaryPoll();
            stopProbePreview();
            stopProbeRunLoop();
            stopProbeStatusPoll();
            refreshArchiveFilters().catch(() => {});
            if (probeEditorModal) {
                setProbeEditorModalVisibility(false);
            }
        }
    }

    function setProbeEditorModalVisibility(visible) {
        if (!probeEditorModal) return;
        probeEditorModal.style.display = visible ? 'block' : 'none';
        if (visible) {
            syncProbePreview(getSelectedProbeChannelId());
        } else {
            stopProbePreview();
            setProbeSnapModalVisibility(false);
        }
    }

    function setProbeSnapModalVisibility(visible) {
        if (!probeSnapModal) return;
        probeSnapModal.style.display = visible ? 'block' : 'none';
        if (!visible) {
            probeSnapState = null;
            if (probeSnapImg) probeSnapImg.src = '';
            if (probeSnapMeta) probeSnapMeta.textContent = 'No snapshot captured.';
            if (probeSnapPreview) {
                probeSnapPreview.classList.remove('actual-size');
            }
            if (probeSnapActualSizeInput) {
                probeSnapActualSizeInput.checked = false;
            }
        }
    }

    function setLuxriotStatus(text, isError = false) {
        if (!luxriotStatusLabel) return;
        luxriotStatusLabel.textContent = text;
        luxriotStatusLabel.classList.toggle('error', Boolean(isError));
        if (isError) {
            luxriotStatusLabel.title = text;
        } else {
            luxriotStatusLabel.removeAttribute('title');
        }
    }

    function updateLuxriotBatchInfo() {
        if (!luxriotBatchInfo) return;
        const intervalSec = Number(luxriotDefaults.snapshotInterval) || 1;
        const fps = intervalSec > 0 ? (1 / intervalSec) : 0;
        const fpsLabel = fps >= 1 ? fps.toFixed(1).replace(/[.]0$/, '') : fps.toFixed(2);
        luxriotBatchInfo.textContent = `~${fpsLabel} fps, ${luxriotDefaults.snapshotMaxEdge}px`;
    }

    function stopLuxriotPreview() {
        if (luxriotPreviewTimer) {
            clearInterval(luxriotPreviewTimer);
            luxriotPreviewTimer = null;
        }
    }

    function stopLuxriotSummaryPoll() {
        if (luxriotSummaryTimer) {
            clearInterval(luxriotSummaryTimer);
            luxriotSummaryTimer = null;
        }
    }

    function getSelectedLuxriotChannel() {
        const raw = luxriotChannelSelect ? luxriotChannelSelect.value : '';
        const parsed = parseInt(raw || luxriotActiveChannel, 10);
        if (Number.isFinite(parsed)) {
            luxriotActiveChannel = parsed;
            return parsed;
        }
        return luxriotDefaults.channelId;
    }

    function getSelectedSummaryChannel() {
        const raw = luxriotSummaryChannelSelect ? luxriotSummaryChannelSelect.value : '';
        const fallback = luxriotSummaryChannel ?? getSelectedLuxriotChannel();
        const parsed = parseInt(raw || String(fallback || ''), 10);
        if (Number.isFinite(parsed)) {
            luxriotSummaryChannel = parsed;
            return parsed;
        }
        return getSelectedLuxriotChannel();
    }

    function normalizeSummaryRun(value) {
        const text = String(value || '').trim();
        if (!text) return 'latest';
        const lowered = text.toLowerCase();
        if (lowered === 'latest' || lowered === 'live' || lowered === 'all') {
            return lowered;
        }
        return text;
    }

    function normalizeSummaryRangePreset(value) {
        const text = String(value || '').trim().toLowerCase();
        if (text === '6h' || text === '24h' || text === '3d' || text === '7d' || text === '30d' || text === 'all' || text === 'custom') {
            return text;
        }
        return '24h';
    }

    function getSummaryRangeBounds(rangePreset, nowSec = null) {
        const normalized = normalizeSummaryRangePreset(rangePreset);
        const now = Number.isFinite(nowSec) ? Number(nowSec) : Math.floor(Date.now() / 1000);
        const toTs = now;
        if (normalized === '6h') return { fromTs: toTs - 6 * 3600, toTs };
        if (normalized === '24h') return { fromTs: toTs - 24 * 3600, toTs };
        if (normalized === '3d') return { fromTs: toTs - 3 * 24 * 3600, toTs };
        if (normalized === '7d') return { fromTs: toTs - 7 * 24 * 3600, toTs };
        if (normalized === '30d') return { fromTs: toTs - 30 * 24 * 3600, toTs };
        return { fromTs: null, toTs: null };
    }

    function getSummaryRangeLabel() {
        const preset = normalizeSummaryRangePreset(luxriotSummaryRangePreset);
        if (preset === '6h') return '6h';
        if (preset === '24h') return '1d';
        if (preset === '3d') return '3d';
        if (preset === '7d') return '7d';
        if (preset === '30d') return '30d';
        if (preset === 'all') return 'all';
        if (Number.isFinite(luxriotSummaryFromTs) || Number.isFinite(luxriotSummaryToTs)) {
            return `custom ${formatRollupRange(luxriotSummaryFromTs, luxriotSummaryToTs)}`;
        }
        return 'custom';
    }

    function syncSummaryRangeUI() {
        const preset = normalizeSummaryRangePreset(luxriotSummaryRangePreset);
        if (luxriotSummaryRangeSelect) {
            luxriotSummaryRangeSelect.value = preset;
        }
        if (luxriotSummaryCustomTime) {
            luxriotSummaryCustomTime.classList.toggle('is-hidden', preset !== 'custom');
        }
    }

    function parseSummaryDatetimeInput(value) {
        const text = String(value || '').trim();
        if (!text) return null;
        const ms = Date.parse(text);
        if (!Number.isFinite(ms)) return null;
        return ms / 1000;
    }

    function formatSummaryDatetimeInput(ts) {
        const sec = Number(ts);
        if (!Number.isFinite(sec)) return '';
        const d = new Date(sec * 1000);
        const yyyy = d.getFullYear();
        const mm = String(d.getMonth() + 1).padStart(2, '0');
        const dd = String(d.getDate()).padStart(2, '0');
        const hh = String(d.getHours()).padStart(2, '0');
        const mi = String(d.getMinutes()).padStart(2, '0');
        return `${yyyy}-${mm}-${dd}T${hh}:${mi}`;
    }

    function readSummaryFiltersFromInputs() {
        const run = normalizeSummaryRun(luxriotSummaryRunSelect ? luxriotSummaryRunSelect.value : luxriotSummaryRunFilter);
        const rangePreset = normalizeSummaryRangePreset(luxriotSummaryRangeSelect ? luxriotSummaryRangeSelect.value : luxriotSummaryRangePreset);
        let fromTs = null;
        let toTs = null;
        if (rangePreset === 'custom') {
            fromTs = parseSummaryDatetimeInput(luxriotSummaryFromInput ? luxriotSummaryFromInput.value : '');
            toTs = parseSummaryDatetimeInput(luxriotSummaryToInput ? luxriotSummaryToInput.value : '');
        } else if (rangePreset !== 'all') {
            const bounds = getSummaryRangeBounds(rangePreset);
            fromTs = bounds.fromTs;
            toTs = bounds.toTs;
        }
        if (fromTs !== null && toTs !== null && fromTs > toTs) {
            const tmp = fromTs;
            fromTs = toTs;
            toTs = tmp;
        }
        return { run, fromTs, toTs, rangePreset };
    }

    function applySummaryFiltersFromInputs() {
        const filters = readSummaryFiltersFromInputs();
        luxriotSummaryRunFilter = filters.run;
        luxriotSummaryRangePreset = normalizeSummaryRangePreset(filters.rangePreset);
        luxriotSummaryFromTs = filters.fromTs;
        luxriotSummaryToTs = filters.toTs;
        if (luxriotSummaryRunSelect) {
            luxriotSummaryRunSelect.value = luxriotSummaryRunFilter;
        }
        syncSummaryRangeUI();
        if (luxriotSummaryFromInput) {
            luxriotSummaryFromInput.value = formatSummaryDatetimeInput(luxriotSummaryFromTs);
        }
        if (luxriotSummaryToInput) {
            luxriotSummaryToInput.value = formatSummaryDatetimeInput(luxriotSummaryToTs);
        }
    }

    function clearSummaryFilters() {
        luxriotSummaryRunFilter = 'latest';
        luxriotSummaryRangePreset = '6h';
        luxriotSummaryFromTs = null;
        luxriotSummaryToTs = null;
        if (luxriotSummaryRunSelect) {
            luxriotSummaryRunSelect.value = 'latest';
        }
        if (luxriotSummaryRangeSelect) {
            luxriotSummaryRangeSelect.value = '6h';
        }
        if (luxriotSummaryFromInput) {
            luxriotSummaryFromInput.value = '';
        }
        if (luxriotSummaryToInput) {
            luxriotSummaryToInput.value = '';
        }
        syncSummaryRangeUI();
    }

    function buildSummaryQueryParams(channelId) {
        const params = new URLSearchParams();
        params.set('channel_id', String(channelId));
        const run = normalizeSummaryRun(luxriotSummaryRunFilter);
        if (run) params.set('run', run);
        const preset = normalizeSummaryRangePreset(luxriotSummaryRangePreset);
        let fromTs = luxriotSummaryFromTs;
        let toTs = luxriotSummaryToTs;
        if (preset !== 'custom') {
            if (preset === 'all') {
                fromTs = null;
                toTs = null;
            } else {
                const bounds = getSummaryRangeBounds(preset);
                fromTs = bounds.fromTs;
                toTs = bounds.toTs;
            }
            luxriotSummaryFromTs = fromTs;
            luxriotSummaryToTs = toTs;
        }
        if (Number.isFinite(fromTs)) {
            params.set('from_ts', String(fromTs));
        }
        if (Number.isFinite(toTs)) {
            params.set('to_ts', String(toTs));
        }
        return params;
    }

    function syncSummaryRunSelectOptions(runs, selectedRun = null) {
        if (!luxriotSummaryRunSelect) return;
        const runItems = Array.isArray(runs) ? runs : [];
        const currentValue = normalizeSummaryRun(
            selectedRun || luxriotSummaryRunSelect.value || luxriotSummaryRunFilter || 'latest'
        );
        const baseOptions = [
            { value: 'latest', label: 'Latest run' },
            { value: 'live', label: 'Live run' },
            { value: 'all', label: 'All runs' },
        ];
        const seen = new Set(baseOptions.map((item) => item.value));
        const dynamicOptions = [];
        runItems.forEach((run) => {
            const runId = String(run?.run_id || '').trim();
            if (!runId || seen.has(runId)) return;
            seen.add(runId);
            const logCount = Number(run?.log_count || 0);
            const running = Boolean(run?.running);
            const stateLabel = running ? 'live' : 'saved';
            dynamicOptions.push({
                value: runId,
                label: `${runId} (${stateLabel}, ${logCount})`,
            });
        });
        const optionsHtml = baseOptions
            .concat(dynamicOptions)
            .map((item) => `<option value="${escapeHtml(item.value)}">${escapeHtml(item.label)}</option>`)
            .join('');
        luxriotSummaryRunSelect.innerHTML = optionsHtml;
        const hasCurrent = Array.from(luxriotSummaryRunSelect.options || [])
            .some((opt) => String(opt.value) === currentValue);
        luxriotSummaryRunSelect.value = hasCurrent ? currentValue : 'latest';
        luxriotSummaryRunFilter = normalizeSummaryRun(luxriotSummaryRunSelect.value);
    }

    function syncSummaryFiltersFromResponse(payload) {
        const data = payload && typeof payload === 'object' ? payload : {};
        if (Object.prototype.hasOwnProperty.call(data, 'selected_run')) {
            luxriotSummaryRunFilter = normalizeSummaryRun(data.selected_run);
        }
        if (Object.prototype.hasOwnProperty.call(data, 'from_ts')) {
            const rawFrom = data.from_ts;
            if (rawFrom === null || rawFrom === '' || typeof rawFrom === 'undefined') {
                luxriotSummaryFromTs = null;
            } else {
                const fromTs = Number(rawFrom);
                luxriotSummaryFromTs = Number.isFinite(fromTs) && fromTs > 0 ? fromTs : null;
            }
        }
        if (Object.prototype.hasOwnProperty.call(data, 'to_ts')) {
            const rawTo = data.to_ts;
            if (rawTo === null || rawTo === '' || typeof rawTo === 'undefined') {
                luxriotSummaryToTs = null;
            } else {
                const toTs = Number(rawTo);
                luxriotSummaryToTs = Number.isFinite(toTs) && toTs > 0 ? toTs : null;
            }
        }
        if (luxriotSummaryRunSelect) {
            const hasRunValue = Array.from(luxriotSummaryRunSelect.options || [])
                .some((opt) => String(opt.value) === luxriotSummaryRunFilter);
            luxriotSummaryRunSelect.value = hasRunValue ? luxriotSummaryRunFilter : 'latest';
            luxriotSummaryRunFilter = normalizeSummaryRun(luxriotSummaryRunSelect.value);
        }
        if (luxriotSummaryFromInput) {
            luxriotSummaryFromInput.value = formatSummaryDatetimeInput(luxriotSummaryFromTs);
        }
        if (luxriotSummaryToInput) {
            luxriotSummaryToInput.value = formatSummaryDatetimeInput(luxriotSummaryToTs);
        }
        syncSummaryRangeUI();
    }

    function normalizeSummaryLevel(value) {
        const text = String(value || '').trim().toUpperCase();
        if (text === 'L1' || text === 'L2' || text === 'L3') return text;
        return 'L0';
    }

    function setSummaryBaseLevel(level) {
        const normalized = normalizeSummaryLevel(level);
        luxriotSummaryLevel = normalized;
        luxriotSummaryRollupStack = [{ level: normalized, sourceIds: null, label: normalized }];
        if (luxriotSummaryLevelSelect) {
            luxriotSummaryLevelSelect.value = normalized;
        }
    }

    function getCurrentSummaryRollupContext() {
        if (!Array.isArray(luxriotSummaryRollupStack) || !luxriotSummaryRollupStack.length) {
            setSummaryBaseLevel(luxriotSummaryLevel);
        }
        const last = luxriotSummaryRollupStack[luxriotSummaryRollupStack.length - 1];
        if (!last || typeof last !== 'object') {
            setSummaryBaseLevel('L0');
            return luxriotSummaryRollupStack[luxriotSummaryRollupStack.length - 1] || null;
        }
        return last;
    }

    function isRollupViewActive() {
        const ctx = getCurrentSummaryRollupContext();
        if (!ctx) return false;
        const hasFilter = Array.isArray(ctx.sourceIds) && ctx.sourceIds.length > 0;
        return normalizeSummaryLevel(ctx.level) !== 'L0' || hasFilter;
    }

    function setSummaryUnread(count) {
        luxriotSummaryUnread = Math.max(0, Number.isFinite(count) ? count : 0);
        if (!luxriotSummaryJumpBtn) return;
        if (luxriotSummaryUnread > 0) {
            luxriotSummaryJumpBtn.classList.remove('is-hidden');
            luxriotSummaryJumpBtn.textContent = `⬇ Jump to latest (${luxriotSummaryUnread})`;
        } else {
            luxriotSummaryJumpBtn.classList.add('is-hidden');
            luxriotSummaryJumpBtn.textContent = '⬇ Jump to latest';
        }
    }

    function getSummaryCollapsedMap(channelId = getSelectedSummaryChannel()) {
        const key = String(channelId);
        if (!luxriotSummaryCollapsedByChannel[key] || typeof luxriotSummaryCollapsedByChannel[key] !== 'object') {
            luxriotSummaryCollapsedByChannel[key] = {};
        }
        return luxriotSummaryCollapsedByChannel[key];
    }

    function isSummaryCollapsed(channelId, logKey) {
        if (!logKey) return false;
        const map = getSummaryCollapsedMap(channelId);
        return Boolean(map[logKey]);
    }

    function setSummaryCollapsed(channelId, logKey, collapsed) {
        if (!logKey) return;
        const map = getSummaryCollapsedMap(channelId);
        if (collapsed) {
            map[logKey] = true;
        } else {
            delete map[logKey];
        }
    }

    function rollupSummaryKey(row, idx = 0) {
        const level = normalizeSummaryLevel(row?.level || '');
        const channelId = parseInt(String(row?.channel_id ?? ''), 10);
        const windowStart = Number(row?.window_start);
        const windowSecRaw = Number(row?.window_sec);
        const windowEnd = Number(row?.window_end);
        if (
            level !== 'L0'
            && Number.isFinite(channelId)
            && Number.isFinite(windowStart)
        ) {
            const startBucket = Math.floor(windowStart);
            let windowSec = Number.isFinite(windowSecRaw) ? Math.floor(windowSecRaw) : 0;
            if (!(windowSec > 0) && Number.isFinite(windowEnd)) {
                windowSec = Math.max(1, Math.floor(windowEnd - windowStart));
            }
            return `rollup:${level}:ch${channelId}:w${windowSec}:t${startBucket}`;
        }
        const rid = String(row?.rollup_id || '').trim();
        if (rid) return `rollup:${rid}`;
        return `rollup:idx-${idx}`;
    }

    function areAllSummariesCollapsed(channelId = getSelectedSummaryChannel()) {
        if (isRollupViewActive()) {
            const rows = Array.isArray(luxriotSummaryRollupRows) ? luxriotSummaryRollupRows : [];
            if (!rows.length) return false;
            return rows.every((row, idx) => isSummaryCollapsed(channelId, rollupSummaryKey(row, idx)));
        }
        const logs = Array.isArray(luxriotSummaryChannelCache[channelId]) ? luxriotSummaryChannelCache[channelId] : [];
        if (!logs.length) return false;
        return logs.every((log, idx) => isSummaryCollapsed(channelId, luxriotSummaryLogKey(log, idx)));
    }

    function collapseAllSummariesForChannel(channelId = getSelectedSummaryChannel(), collapsed = true) {
        const map = getSummaryCollapsedMap(channelId);
        if (isRollupViewActive()) {
            const rows = Array.isArray(luxriotSummaryRollupRows) ? luxriotSummaryRollupRows : [];
            if (!rows.length) return;
            if (collapsed) {
                rows.forEach((row, idx) => {
                    map[rollupSummaryKey(row, idx)] = true;
                });
            } else {
                rows.forEach((row, idx) => {
                    delete map[rollupSummaryKey(row, idx)];
                });
            }
            return;
        }
        const logs = Array.isArray(luxriotSummaryChannelCache[channelId]) ? luxriotSummaryChannelCache[channelId] : [];
        if (!logs.length) return;
        if (collapsed) {
            logs.forEach((log, idx) => {
                map[luxriotSummaryLogKey(log, idx)] = true;
            });
        } else {
            Object.keys(map).forEach((key) => {
                delete map[key];
            });
        }
    }

    function setSummaryCompactMode(enabled) {
        luxriotSummaryCompactMode = Boolean(enabled);
        if (luxriotSummaries) {
            luxriotSummaries.classList.toggle('compact', luxriotSummaryCompactMode);
        }
    }

    function copyTextToClipboard(text) {
        const value = String(text || '');
        if (!value) return Promise.reject(new Error('Nothing to copy'));
        if (navigator.clipboard && navigator.clipboard.writeText) {
            return navigator.clipboard.writeText(value);
        }
        const textarea = document.createElement('textarea');
        textarea.value = value;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        try {
            document.execCommand('copy');
            return Promise.resolve();
        } catch (err) {
            return Promise.reject(err);
        } finally {
            document.body.removeChild(textarea);
        }
    }

    function downloadTextFile(filename, content) {
        const blob = new Blob([String(content || '')], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = filename;
        document.body.appendChild(anchor);
        anchor.click();
        document.body.removeChild(anchor);
        URL.revokeObjectURL(url);
    }

    function updateSummaryControlsUI() {
        const rollupMode = isRollupViewActive();
        if (luxriotSummaryFollowBtn) {
            const liveOn = !rollupMode && luxriotSummaryAutoRefresh && luxriotSummaryFollowLive;
            luxriotSummaryFollowBtn.classList.toggle('primary', liveOn);
            luxriotSummaryFollowBtn.textContent = rollupMode
                ? '▶ Live n/a'
                : (liveOn ? '⏸ Live ON' : '▶ Live OFF');
            luxriotSummaryFollowBtn.disabled = rollupMode;
        }
        if (luxriotSummaryPauseBtn) {
            luxriotSummaryPauseBtn.classList.toggle('primary', !luxriotSummaryAutoRefresh);
            luxriotSummaryPauseBtn.textContent = luxriotSummaryAutoRefresh ? 'Pause updates' : 'Resume updates';
        }
        if (luxriotSummaryViewBtn) {
            luxriotSummaryViewBtn.classList.toggle('primary', luxriotSummaryCompactMode);
            luxriotSummaryViewBtn.textContent = luxriotSummaryCompactMode ? 'View: Compact' : 'View: Expanded';
        }
        if (luxriotSummaryCollapseAllBtn) {
            const collapsed = areAllSummariesCollapsed();
            luxriotSummaryCollapseAllBtn.classList.toggle('primary', collapsed);
            luxriotSummaryCollapseAllBtn.textContent = collapsed ? '⇵ Expand all' : '⇵ Collapse all';
            luxriotSummaryCollapseAllBtn.disabled = false;
        }
        if (luxriotSummaryBackBtn) {
            luxriotSummaryBackBtn.disabled = !Array.isArray(luxriotSummaryRollupStack) || luxriotSummaryRollupStack.length <= 1;
        }
        if (luxriotSummaryJumpBtn) {
            if (rollupMode) {
                luxriotSummaryJumpBtn.classList.add('is-hidden');
            } else if (luxriotSummaryUnread > 0) {
                luxriotSummaryJumpBtn.classList.remove('is-hidden');
            } else {
                luxriotSummaryJumpBtn.classList.add('is-hidden');
            }
        }
        if (luxriotSummaryApplyFiltersBtn) {
            luxriotSummaryApplyFiltersBtn.disabled = normalizeSummaryRangePreset(luxriotSummaryRangePreset) !== 'custom';
        }
    }

    function syncLuxriotSummaryChannelSelect() {
        if (!luxriotSummaryChannelSelect || !luxriotChannelSelect) return;
        const options = Array.from(luxriotChannelSelect.options || [])
            .map((opt) => {
                const value = String(opt.value || '').trim();
                const label = String(opt.textContent || '').trim();
                if (!value) return null;
                return `<option value="${value}">${escapeHtml(label)}</option>`;
            })
            .filter((opt) => Boolean(opt));
        if (!options.length) {
            luxriotSummaryChannelSelect.innerHTML = '<option value="">No channels</option>';
            return;
        }
        const selected = Number.isFinite(luxriotSummaryChannel) ? luxriotSummaryChannel : getSelectedLuxriotChannel();
        luxriotSummaryChannelSelect.innerHTML = options.join('');
        const exists = Array.from(luxriotSummaryChannelSelect.options || [])
            .some((opt) => parseInt(String(opt.value || ''), 10) === selected);
        if (exists) {
            luxriotSummaryChannelSelect.value = String(selected);
        } else {
            luxriotSummaryChannelSelect.selectedIndex = 0;
            const first = parseInt(luxriotSummaryChannelSelect.value || '', 10);
            luxriotSummaryChannel = Number.isFinite(first) ? first : getSelectedLuxriotChannel();
        }
    }

    function normalizeLuxriotChannelName(channel, channelId) {
        const raw = String(
            channel?.title
            || channel?.name
            || channel?.channel_name
            || channel?.label
            || ''
        ).trim();
        if (raw) return raw;
        if (Number.isFinite(channelId)) return `Channel #${channelId}`;
        return 'Unknown channel';
    }

    async function fetchLuxriotChannels(force = false) {
        if (!luxriotChannelSelect) return;
        luxriotChannelSelect.innerHTML = '<option>Loading...</option>';
        try {
            const response = await fetch(`/luxriot/channels${force ? '?force=1' : ''}`);
            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }
            const channels = data.channels || [];
            Object.keys(luxriotChannelNameById).forEach((key) => delete luxriotChannelNameById[key]);
            if (!channels.length) {
                luxriotChannelSelect.innerHTML = '<option value="">No channels</option>';
                if (luxriotSummaryChannelSelect) {
                    luxriotSummaryChannelSelect.innerHTML = '<option value="">No channels</option>';
                }
                setLuxriotStatus('No channels available', true);
                return;
            }
            const options = channels
                .map((ch) => {
                    const rawId = ch.id ?? ch.channel_id;
                    const id = parseInt(String(rawId || ''), 10);
                    if (!Number.isFinite(id)) return '';
                    const label = normalizeLuxriotChannelName(ch, id);
                    luxriotChannelNameById[String(id)] = label;
                    const selected = String(id) === String(luxriotActiveChannel) ? 'selected' : '';
                    return `<option value="${id}" ${selected}>${escapeHtml(label)}</option>`;
                })
                .filter((item) => Boolean(item));
            luxriotChannelSelect.innerHTML = options.join('');
            const channelIds = channels
                .map((ch) => parseInt(String(ch.id ?? ch.channel_id ?? ''), 10))
                .filter((id) => Number.isFinite(id));
            if (!channelIds.some((id) => String(id) === String(luxriotActiveChannel))) {
                luxriotActiveChannel = channelIds[0] || luxriotDefaults.channelId;
                luxriotChannelSelect.value = luxriotActiveChannel;
            }
            if (!Number.isFinite(luxriotSummaryChannel)) {
                luxriotSummaryChannel = luxriotActiveChannel;
            }
            syncLuxriotSummaryChannelSelect();
            if (!(String(luxriotActiveChannel) in luxriotCaptureRunningByChannel)) {
                luxriotCaptureRunningByChannel[String(luxriotActiveChannel)] = false;
            }
            updateLuxriotCaptureToggleButton(luxriotActiveChannel);
            setLuxriotStatus(`Loaded ${channels.length} channels`);
        } catch (err) {
            Object.keys(luxriotChannelNameById).forEach((key) => delete luxriotChannelNameById[key]);
            luxriotChannelSelect.innerHTML = '<option value="">Load failed</option>';
            if (luxriotSummaryChannelSelect) {
                luxriotSummaryChannelSelect.innerHTML = '<option value="">Load failed</option>';
            }
            updateLuxriotCaptureToggleButton();
            setLuxriotStatus('Channel load failed: ' + err.message, true);
        }
    }

    function getLuxriotChannelLabel(channelId) {
        if (!Number.isFinite(channelId)) return 'Unknown channel';
        const known = luxriotChannelNameById[String(channelId)];
        if (known) return known;
        if (!luxriotChannelSelect) return `Channel #${channelId}`;
        const options = Array.from(luxriotChannelSelect.options || []);
        const match = options.find((opt) => parseInt(opt.value || '', 10) === channelId);
        if (!match) return `Channel #${channelId}`;
        const label = String(match.textContent || '').trim();
        return label || `Channel #${channelId}`;
    }

    function setLuxriotCaptureRunning(channelId, running) {
        const parsed = parseInt(String(channelId || ''), 10);
        if (!Number.isFinite(parsed)) return;
        luxriotCaptureRunningByChannel[String(parsed)] = Boolean(running);
    }

    function isLuxriotCaptureRunning(channelId) {
        const parsed = parseInt(String(channelId || ''), 10);
        if (!Number.isFinite(parsed)) return false;
        return Boolean(luxriotCaptureRunningByChannel[String(parsed)]);
    }

    function updateLuxriotCaptureToggleButton(channelIdOverride = null) {
        if (!luxriotToggleCaptureBtn) return;
        const channelId = Number.isFinite(channelIdOverride) ? channelIdOverride : getSelectedLuxriotChannel();
        const running = isLuxriotCaptureRunning(channelId);
        luxriotToggleCaptureBtn.textContent = running ? 'Stop summaries' : 'Start summaries';
        luxriotToggleCaptureBtn.classList.toggle('primary', !running);
    }

    function getLuxriotPromptInputByTab(tab) {
        const normalized = String(tab || '').trim().toLowerCase();
        if (normalized === 'stream') return luxriotSystemPromptInput;
        if (normalized === 'l1') return luxriotRollupPromptL1Input;
        if (normalized === 'l2') return luxriotRollupPromptL2Input;
        if (normalized === 'l3') return luxriotRollupPromptL3Input;
        if (normalized === 'json') return luxriotJsonAlertPromptInput;
        return luxriotSystemPromptInput;
    }

    function getLuxriotPromptTabLabel(tab) {
        const normalized = String(tab || '').trim().toLowerCase();
        if (normalized === 'stream') return 'Stream system prompt';
        if (normalized === 'l1') return 'L1 rollup prompt';
        if (normalized === 'l2') return 'L2 rollup prompt';
        if (normalized === 'l3') return 'L3 rollup prompt';
        if (normalized === 'json') return 'JSON alert prompt';
        return 'System prompt';
    }

    function getLuxriotPromptTabMeta(tab) {
        const normalized = String(tab || '').trim().toLowerCase();
        if (normalized === 'stream') {
            return 'Editing stream system prompt used for live summaries.';
        }
        if (normalized === 'l1') {
            return 'Editing L1 rollup prompt (stored for rollup workflow tuning).';
        }
        if (normalized === 'l2') {
            return 'Editing L2 rollup prompt (stored for rollup workflow tuning).';
        }
        if (normalized === 'l3') {
            return 'Editing L3 rollup prompt (stored for rollup workflow tuning).';
        }
        if (normalized === 'json') {
            return 'Editing optional bookmark JSON block. It should only be emitted when a Task-defined trigger is observed.';
        }
        return 'Editing system prompt.';
    }

    function collectLuxriotPromptSettings() {
        return {
            stream_system_prompt: luxriotSystemPromptInput ? String(luxriotSystemPromptInput.value || '') : '',
            rollup_prompts: {
                L1: luxriotRollupPromptL1Input ? String(luxriotRollupPromptL1Input.value || '') : '',
                L2: luxriotRollupPromptL2Input ? String(luxriotRollupPromptL2Input.value || '') : '',
                L3: luxriotRollupPromptL3Input ? String(luxriotRollupPromptL3Input.value || '') : '',
            },
            json_alert_prompt: luxriotJsonAlertPromptInput ? String(luxriotJsonAlertPromptInput.value || '') : '',
            bookmark_enabled: luxriotBookmarkEnabledInput ? Boolean(luxriotBookmarkEnabledInput.checked) : false,
            bookmark_cooldown_sec: luxriotBookmarkCooldownInput
                ? Math.max(0, Number.parseFloat(String(luxriotBookmarkCooldownInput.value || '0')) || 0)
                : 0,
        };
    }

    function applyLuxriotPromptSettingsFromPayload(payload) {
        const settings = payload && typeof payload === 'object' ? payload : {};
        if (luxriotSystemPromptInput && Object.prototype.hasOwnProperty.call(settings, 'stream_system_prompt')) {
            luxriotSystemPromptInput.value = String(settings.stream_system_prompt || '');
        }
        const rollupPrompts = settings.rollup_prompts && typeof settings.rollup_prompts === 'object'
            ? settings.rollup_prompts
            : {};
        if (luxriotRollupPromptL1Input && Object.prototype.hasOwnProperty.call(rollupPrompts, 'L1')) {
            luxriotRollupPromptL1Input.value = String(rollupPrompts.L1 || '');
        }
        if (luxriotRollupPromptL2Input && Object.prototype.hasOwnProperty.call(rollupPrompts, 'L2')) {
            luxriotRollupPromptL2Input.value = String(rollupPrompts.L2 || '');
        }
        if (luxriotRollupPromptL3Input && Object.prototype.hasOwnProperty.call(rollupPrompts, 'L3')) {
            luxriotRollupPromptL3Input.value = String(rollupPrompts.L3 || '');
        }
        if (luxriotJsonAlertPromptInput && Object.prototype.hasOwnProperty.call(settings, 'json_alert_prompt')) {
            luxriotJsonAlertPromptInput.value = String(settings.json_alert_prompt || '');
        }
        if (luxriotBookmarkEnabledInput && Object.prototype.hasOwnProperty.call(settings, 'bookmark_enabled')) {
            luxriotBookmarkEnabledInput.checked = Boolean(settings.bookmark_enabled);
        }
        if (luxriotBookmarkCooldownInput && Object.prototype.hasOwnProperty.call(settings, 'bookmark_cooldown_sec')) {
            const cooldown = Number.parseFloat(String(settings.bookmark_cooldown_sec || '0'));
            luxriotBookmarkCooldownInput.value = Number.isFinite(cooldown) ? String(Math.max(0, cooldown)) : '0';
        }
        const activeInput = getLuxriotPromptInputByTab(luxriotPromptModalTab);
        if (luxriotPromptModalInput && activeInput) {
            luxriotPromptModalInput.value = String(activeInput.value || '');
        }
    }

    async function refreshLuxriotPromptSettings(showError = false, channelIdOverride = null) {
        const channelId = Number.isFinite(channelIdOverride)
            ? channelIdOverride
            : getSelectedLuxriotChannel();
        if (!Number.isFinite(channelId)) {
            return;
        }
        try {
            const params = new URLSearchParams();
            params.set('channel_id', String(channelId));
            const response = await fetch(`/luxriot/prompt_settings?${params.toString()}`);
            const data = await parseApiJson(response, 'Failed to load prompt settings');
            applyLuxriotPromptSettingsFromPayload(data);
        } catch (err) {
            if (showError) {
                setLuxriotStatus(err.message || 'Failed to load prompt settings', true);
            }
        }
    }

    async function persistLuxriotPromptSettings(channelIdOverride = null) {
        const channelId = Number.isFinite(channelIdOverride)
            ? channelIdOverride
            : getSelectedLuxriotChannel();
        if (!Number.isFinite(channelId)) {
            throw new Error('Select a channel first');
        }
        const payload = collectLuxriotPromptSettings();
        payload.channel_id = channelId;
        const response = await fetch('/luxriot/prompt_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await parseApiJson(response, 'Failed to save prompt settings');
        applyLuxriotPromptSettingsFromPayload(data);
    }

    function setLuxriotPromptModalTab(tab) {
        const normalized = String(tab || '').trim().toLowerCase();
        const previousInput = getLuxriotPromptInputByTab(luxriotPromptModalTab);
        if (luxriotPromptModalInput && previousInput) {
            previousInput.value = luxriotPromptModalInput.value || '';
        }
        const tabValue = normalized === 'stream' ? 'stream' : normalized.toUpperCase();
        luxriotPromptModalTab = tabValue;
        luxriotPromptTabButtons.forEach((button) => {
            const buttonTab = String(button.dataset.luxriotPromptTab || '').trim();
            button.classList.toggle('active', buttonTab.toLowerCase() === String(tabValue).toLowerCase());
        });
        if (luxriotPromptModalInput) {
            const sourceInput = getLuxriotPromptInputByTab(tabValue);
            luxriotPromptModalInput.value = sourceInput ? String(sourceInput.value || '') : '';
        }
        if (luxriotPromptModalMeta) {
            const channelLabel = getLuxriotChannelLabel(getSelectedLuxriotChannel());
            luxriotPromptModalMeta.textContent = `${getLuxriotPromptTabMeta(tabValue)} Channel: ${channelLabel}.`;
        }
    }

    function openLuxriotPromptModal() {
        if (!luxriotPromptModal) return;
        luxriotPromptModal.style.display = 'block';
        void refreshLuxriotPromptSettings(true);
        setLuxriotPromptModalTab(luxriotPromptModalTab || 'stream');
    }

    function closeLuxriotPromptModal() {
        if (!luxriotPromptModal) return;
        luxriotPromptModal.style.display = 'none';
    }

    async function applyLuxriotPromptModal() {
        const targetInput = getLuxriotPromptInputByTab(luxriotPromptModalTab);
        if (targetInput && luxriotPromptModalInput) {
            targetInput.value = luxriotPromptModalInput.value || '';
        }
        const channelId = getSelectedLuxriotChannel();
        await persistLuxriotPromptSettings(channelId);
        setLuxriotStatus(`${getLuxriotPromptTabLabel(luxriotPromptModalTab)} updated for ${getLuxriotChannelLabel(channelId)}`);
    }

    function startLuxriotPreview() {
        if (!luxriotPreviewImg) return;
        const channelId = getSelectedLuxriotChannel();
        if (!channelId) {
            setLuxriotStatus('Select a channel to preview', true);
            return;
        }
        const refresh = () => {
            if (luxriotOverlay) {
                luxriotOverlay.textContent = 'Loading...';
            }
            luxriotPreviewImg.src = `/luxriot/snapshot/${channelId}?t=${Date.now()}`;
        };
        luxriotPreviewImg.onload = () => {
            if (luxriotOverlay) luxriotOverlay.textContent = '';
            setLuxriotStatus(`Previewing channel ${channelId}`);
        };
        luxriotPreviewImg.onerror = () => {
            if (luxriotOverlay) luxriotOverlay.textContent = 'Preview failed';
            setLuxriotStatus('Preview failed', true);
        };
        stopLuxriotPreview();
        refresh();
        const intervalMs = Math.max(2000, (luxriotDefaults.snapshotInterval || 5) * 1000);
        luxriotPreviewTimer = setInterval(refresh, intervalMs);
    }

    function luxriotSummaryLogKey(log, idx = 0) {
        const createdRaw = Number(log?.created_at);
        const createdKey = Number.isFinite(createdRaw) ? createdRaw.toFixed(6) : `idx-${idx}`;
        const frameKey = Number(log?.frame_count || 0);
        const summaryKey = String(log?.summary || '').trim().slice(0, 160);
        return `${createdKey}|${frameKey}|${summaryKey}`;
    }

    function isSummaryNearBottom(threshold = 48) {
        if (!luxriotSummaries) return true;
        return (luxriotSummaries.scrollTop + luxriotSummaries.clientHeight) >= (luxriotSummaries.scrollHeight - threshold);
    }

    function scrollSummaryToLatest() {
        if (!luxriotSummaries) return;
        luxriotSummaries.scrollTop = luxriotSummaries.scrollHeight;
    }

    function setLuxriotSummaryMeta(text, isError = false) {
        if (!luxriotSummaryMeta) return;
        luxriotSummaryMeta.textContent = text;
        luxriotSummaryMeta.classList.toggle('error', Boolean(isError));
    }

    function withSummaryUpdatedMeta(text) {
        const base = String(text || '').trim();
        const stamp = new Date().toLocaleTimeString();
        return `${base} · updated ${stamp}`;
    }

    function formatSummaryWindowLabel(seconds) {
        const value = Number(seconds);
        if (!Number.isFinite(value) || value <= 0) return 'n/a';
        if (value % 86400 === 0) return `${Math.floor(value / 86400)}d`;
        if (value % 3600 === 0) return `${Math.floor(value / 3600)}h`;
        if (value % 60 === 0) return `${Math.floor(value / 60)}m`;
        return `${Math.floor(value)}s`;
    }

    function setSummaryRefreshButtonState(state = 'idle') {
        if (!luxriotRefreshSummariesBtn) return;
        if (state === 'busy') {
            luxriotRefreshSummariesBtn.disabled = true;
            luxriotRefreshSummariesBtn.textContent = '⟳ Refreshing...';
            return;
        }
        if (state === 'queued') {
            luxriotRefreshSummariesBtn.disabled = true;
            luxriotRefreshSummariesBtn.textContent = '⟳ Queued...';
            return;
        }
        luxriotRefreshSummariesBtn.disabled = false;
        luxriotRefreshSummariesBtn.textContent = '⟳ Refresh';
    }

    function renderLuxriotSummaries(logs, channelId = getSelectedSummaryChannel()) {
        if (!luxriotSummaries) return;
        const normalizedLogs = Array.isArray(logs) ? logs.slice() : [];
        normalizedLogs.sort((a, b) => Number(a?.created_at || 0) - Number(b?.created_at || 0));
        luxriotSummaryChannelCache[channelId] = normalizedLogs;
        setSummaryCompactMode(luxriotSummaryCompactMode);

        const prevKeys = Array.isArray(luxriotSummarySeenKeys[channelId]) ? luxriotSummarySeenKeys[channelId] : [];
        const prevKeySet = new Set(prevKeys);
        const newKeys = normalizedLogs.map((log, idx) => luxriotSummaryLogKey(log, idx));
        luxriotSummarySeenKeys[channelId] = newKeys;
        const newCount = newKeys.reduce((count, key) => count + (prevKeySet.has(key) ? 0 : 1), 0);
        const shouldStickBottom = luxriotSummaryFollowLive && isSummaryNearBottom();
        const prevScrollTop = luxriotSummaries.scrollTop;
        const hasInitialRender = luxriotSummaries.dataset.hasRender === '1';

        if (!normalizedLogs.length) {
            luxriotSummaryLogCache = [];
            luxriotSummaries.innerHTML = '<div class="loading">No summaries yet for this channel.</div>';
            if (luxriotSummaryFollowLive) {
                setSummaryUnread(0);
            }
            luxriotSummaries.dataset.hasRender = '1';
            updateSummaryControlsUI();
            return;
        }

        luxriotSummaryLogCache = normalizedLogs;
        const html = luxriotSummaryLogCache
            .map((log, idx) => {
                const logKey = luxriotSummaryLogKey(log, idx);
                const ts = Number(log.created_at) ? new Date(log.created_at * 1000) : null;
                const tsLabel = ts ? ts.toLocaleString() : 'n/a';
                const frameLabel = log.frame_count ? `${log.frame_count} frames` : '';
                const modelLabel = String(log.model || '').trim();
                const rowChannelId = parseInt(String(log?.channel_id ?? channelId), 10);
                const channelTag = Number.isFinite(rowChannelId) ? `#${rowChannelId}` : '#?';
                const channelLabel = Number.isFinite(rowChannelId)
                    ? getLuxriotChannelLabel(rowChannelId)
                    : 'Unknown channel';
                const summary = String(log.summary || '').trim();
                const summaryParts = splitSummaryAndJson(summary);
                const summaryMain = summaryParts.main || summary;
                const summaryJson = summaryParts.json;
                const canBookmark = summary.length > 0;
                const collapsed = isSummaryCollapsed(channelId, logKey);
                return `
                    <div class="luxriot-summary ${collapsed ? 'is-collapsed' : ''}" data-log-key="${escapeHtml(logKey)}">
                        <div class="luxriot-summary-head">
                            <div class="timestamp"><span class="luxriot-summary-channel-pill" title="${escapeHtml(channelLabel)}">${escapeHtml(channelTag)}</span> ${tsLabel}${frameLabel ? ` · ${frameLabel}` : ''}${modelLabel ? ` · ${escapeHtml(modelLabel)}` : ''}</div>
                            <div class="luxriot-summary-actions">
                                <button class="feature-btn luxriot-summary-action-btn" data-luxriot-collapse="${idx}">
                                    ${collapsed ? 'Expand' : 'Collapse'}
                                </button>
                                <button class="feature-btn luxriot-summary-action-btn" data-luxriot-copy="${idx}" ${canBookmark ? '' : 'disabled'}>
                                    Copy
                                </button>
                                <button class="feature-btn luxriot-summary-action-btn" data-luxriot-export="${idx}" ${canBookmark ? '' : 'disabled'}>
                                    Export
                                </button>
                                <button class="feature-btn luxriot-bookmark-btn" data-luxriot-bookmark="${idx}" ${canBookmark ? '' : 'disabled'}>
                                    Bookmark
                                </button>
                            </div>
                        </div>
                        <div class="summary-body">${renderMarkdown(summaryMain)}${summaryJson ? `<div class="summary-json-muted">${renderMarkdown(summaryJson)}</div>` : ''}</div>
                    </div>
                `;
            })
            .join('');
        luxriotSummaries.innerHTML = html;

        if (shouldStickBottom || !hasInitialRender) {
            scrollSummaryToLatest();
        } else {
            luxriotSummaries.scrollTop = prevScrollTop;
        }
        luxriotSummaries.dataset.hasRender = '1';

        if (luxriotSummaryFollowLive) {
            setSummaryUnread(0);
        } else if (newCount > 0) {
            setSummaryUnread(luxriotSummaryUnread + newCount);
        }
        updateSummaryControlsUI();
    }

    function formatRollupRange(windowStart, windowEnd) {
        const start = Number(windowStart);
        const end = Number(windowEnd);
        const startLabel = Number.isFinite(start) ? new Date(start * 1000).toLocaleString() : 'n/a';
        const endLabel = Number.isFinite(end) ? new Date(end * 1000).toLocaleString() : 'n/a';
        return `${startLabel} -> ${endLabel}`;
    }

    function formatLuxriotRollupExport(row) {
        const nl = String.fromCharCode(10);
        const channelId = Number(row?.channel_id) || getSelectedSummaryChannel() || luxriotDefaults.channelId;
        const level = normalizeSummaryLevel(row?.level || 'L0');
        const sourceLevel = String(row?.source_level || '').trim() || 'n/a';
        const rollupId = String(row?.rollup_id || '').trim() || 'n/a';
        const itemCount = Number(row?.item_count || 0);
        const frameCount = Number(row?.frame_count || 0);
        const runCount = Array.isArray(row?.run_ids) ? row.run_ids.length : 0;
        const range = formatRollupRange(row?.window_start, row?.window_end);
        const summary = String(row?.summary || '').trim();
        const header = [
            `Channel: ${channelId}`,
            `Level: ${level}`,
            `Rollup ID: ${rollupId}`,
            `Range: ${range}`,
            `Items: ${itemCount}`,
            `Frames: ${frameCount}`,
            `Runs: ${runCount}`,
            `Source level: ${sourceLevel}`,
        ].join(nl);
        return `${header}${nl}${nl}${summary}`;
    }

    async function copyLuxriotRollupFromRow(rowIndex, triggerBtn = null) {
        const idx = Number.isFinite(rowIndex) ? rowIndex : parseInt(String(rowIndex || ''), 10);
        if (!Number.isFinite(idx) || idx < 0 || idx >= luxriotSummaryRollupRows.length) {
            setLuxriotStatus('Invalid rollup selection', true);
            return;
        }
        const row = luxriotSummaryRollupRows[idx] || {};
        try {
            await copyTextToClipboard(formatLuxriotRollupExport(row));
            setLuxriotStatus('Rollup copied');
            if (triggerBtn) {
                const original = triggerBtn.textContent;
                triggerBtn.textContent = 'Copied';
                setTimeout(() => {
                    if (triggerBtn) triggerBtn.textContent = original || 'Copy';
                }, 1200);
            }
        } catch (_) {
            setLuxriotStatus('Failed to copy rollup', true);
        }
    }

    function exportLuxriotRollupFromRow(rowIndex) {
        const idx = Number.isFinite(rowIndex) ? rowIndex : parseInt(String(rowIndex || ''), 10);
        if (!Number.isFinite(idx) || idx < 0 || idx >= luxriotSummaryRollupRows.length) {
            setLuxriotStatus('Invalid rollup selection', true);
            return;
        }
        const row = luxriotSummaryRollupRows[idx] || {};
        const level = normalizeSummaryLevel(row?.level || 'L0');
        const stamp = Number.isFinite(Number(row?.window_start))
            ? new Date(Number(row.window_start) * 1000).toISOString().replace(/[:]/g, '-')
            : `entry-${idx + 1}`;
        const channelId = Number(row?.channel_id) || getSelectedSummaryChannel() || luxriotDefaults.channelId;
        const filename = `luxriot_rollup_${level.toLowerCase()}_ch${channelId}_${stamp}.txt`;
        downloadTextFile(filename, formatLuxriotRollupExport(row));
        setLuxriotStatus(`Exported ${filename}`);
    }

    function pushSummaryRollupContext(level, sourceIds = null, label = '') {
        const normalized = normalizeSummaryLevel(level);
        const ids = Array.isArray(sourceIds)
            ? sourceIds.map((id) => String(id || '').trim()).filter((id) => id.length > 0)
            : null;
        luxriotSummaryRollupStack.push({
            level: normalized,
            sourceIds: ids && ids.length ? ids : null,
            label: String(label || normalized).trim() || normalized,
        });
        luxriotSummaryLevel = normalized;
        if (luxriotSummaryLevelSelect) {
            luxriotSummaryLevelSelect.value = normalized;
        }
    }

    function popSummaryRollupContext() {
        if (!Array.isArray(luxriotSummaryRollupStack) || luxriotSummaryRollupStack.length <= 1) {
            return null;
        }
        luxriotSummaryRollupStack.pop();
        const ctx = getCurrentSummaryRollupContext();
        luxriotSummaryLevel = normalizeSummaryLevel(ctx?.level || 'L0');
        if (luxriotSummaryLevelSelect) {
            luxriotSummaryLevelSelect.value = luxriotSummaryLevel;
        }
        return ctx;
    }

    function renderLuxriotRollups(payload, channelId = getSelectedSummaryChannel()) {
        if (!luxriotSummaries) return 0;
        const data = payload && typeof payload === 'object' ? payload : {};
        const levels = data.levels && typeof data.levels === 'object' ? data.levels : {};
        const prevScrollTop = luxriotSummaries.scrollTop;
        const hasInitialRender = luxriotSummaries.dataset.hasRender === '1';
        const shouldStickBottom = isSummaryNearBottom();
        const ctx = getCurrentSummaryRollupContext();
        const level = normalizeSummaryLevel(ctx?.level || luxriotSummaryLevel);
        const sourceSet = Array.isArray(ctx?.sourceIds) && ctx.sourceIds.length
            ? new Set(ctx.sourceIds.map((id) => String(id || '').trim()).filter((id) => id))
            : null;
        const levelRows = Array.isArray(levels[level]) ? levels[level] : [];
        const rows = levelRows
            .filter((row) => {
                if (!sourceSet) return true;
                const rowId = String(row?.rollup_id || '').trim();
                return rowId && sourceSet.has(rowId);
            })
            .sort((a, b) => Number(a?.window_start || 0) - Number(b?.window_start || 0));

        luxriotSummaryRollupRows = rows;
        luxriotSummaryLogCache = [];
        if (!rows.length) {
            luxriotSummaries.innerHTML = `<div class="loading">No ${level} rollups available for this selection.</div>`;
            luxriotSummaries.dataset.hasRender = '1';
            setSummaryUnread(0);
            updateSummaryControlsUI();
            return 0;
        }

        const html = rows.map((row, idx) => {
            const rowLevel = normalizeSummaryLevel(row?.level || level);
            const rollupKey = rollupSummaryKey(row, idx);
            const collapsed = isSummaryCollapsed(channelId, rollupKey);
            const rangeLabel = formatRollupRange(row?.window_start, row?.window_end);
            const rowChannelId = parseInt(String(row?.channel_id ?? channelId), 10);
            const channelTag = Number.isFinite(rowChannelId) ? `#${rowChannelId}` : '#?';
            const channelLabel = Number.isFinite(rowChannelId)
                ? getLuxriotChannelLabel(rowChannelId)
                : 'Unknown channel';
            const itemCount = Number(row?.item_count || 0);
            const frameCount = Number(row?.frame_count || 0);
            const sourceTokens = Number(row?.source_tokens || 0);
            const runCount = Array.isArray(row?.run_ids) ? row.run_ids.length : 0;
            const sourceLevel = String(row?.source_level || '').trim();
            const sourceIds = Array.isArray(row?.source_ids) ? row.source_ids : [];
            const summary = String(row?.summary || '').trim();
            const summaryParts = splitSummaryAndJson(summary);
            const summaryMain = summaryParts.main || summary;
            const summaryJson = summaryParts.json;
            const canDrill = Boolean(sourceLevel && sourceIds.length > 0);
            const statsLabel = `${itemCount} items · ${frameCount} frames · ${runCount} runs${sourceTokens > 0 ? ` · ${sourceTokens} tok` : ''}`;
            const sourceLabel = canDrill ? `${sourceIds.length} from ${sourceLevel}` : 'source base';
            return `
                <div class="luxriot-summary ${collapsed ? 'is-collapsed' : ''}" data-log-key="${escapeHtml(rollupKey)}">
                    <div class="luxriot-summary-head">
                        <div class="timestamp"><span class="luxriot-summary-rollup-pill">${escapeHtml(rowLevel)}</span> <span class="luxriot-summary-channel-pill" title="${escapeHtml(channelLabel)}">${escapeHtml(channelTag)}</span> ${escapeHtml(rangeLabel)} · ${escapeHtml(statsLabel)} · ${escapeHtml(sourceLabel)}</div>
                        <div class="luxriot-summary-actions">
                            <button class="feature-btn luxriot-summary-action-btn" data-luxriot-rollup-collapse="${idx}">${collapsed ? 'Expand' : 'Collapse'}</button>
                            <button class="feature-btn luxriot-summary-action-btn" data-luxriot-rollup-copy="${idx}">Copy</button>
                            <button class="feature-btn luxriot-summary-action-btn" data-luxriot-rollup-export="${idx}">Export</button>
                            <button class="feature-btn luxriot-summary-action-btn" data-luxriot-rollup-drill="${idx}" ${canDrill ? '' : 'disabled'}>${canDrill ? `Drill ${escapeHtml(sourceLevel)}` : 'No source'}</button>
                        </div>
                    </div>
                    <div class="summary-body">${renderMarkdown(summaryMain)}${summaryJson ? `<div class="summary-json-muted">${renderMarkdown(summaryJson)}</div>` : ''}</div>
                </div>
            `;
        }).join('');

        luxriotSummaries.innerHTML = html;
        if (shouldStickBottom || !hasInitialRender) {
            scrollSummaryToLatest();
        } else {
            luxriotSummaries.scrollTop = prevScrollTop;
        }
        luxriotSummaries.dataset.hasRender = '1';
        setSummaryUnread(0);
        updateSummaryControlsUI();
        return rows.length;
    }

    async function refreshLuxriotRollups(channelId = getSelectedSummaryChannel(), force = false, allowRunFallback = true) {
        if (!channelId) return;
        if (!luxriotSummaryAutoRefresh && !force) return;
        try {
            const params = buildSummaryQueryParams(channelId);
            params.set('level_limit', '240');
            const resp = await fetch(`/luxriot/rollups?${params.toString()}`);
            const data = await resp.json();
            if (data.error) {
                throw new Error(data.error);
            }
            syncSummaryRunSelectOptions(data.runs, data.selected_run);
            syncSummaryFiltersFromResponse(data);
            const selectedRun = normalizeSummaryRun(luxriotSummaryRunFilter);
            if (
                allowRunFallback
                && !Boolean(data.running)
                && (selectedRun === 'live' || selectedRun === 'latest')
            ) {
                luxriotSummaryRunFilter = 'all';
                if (luxriotSummaryRunSelect) {
                    luxriotSummaryRunSelect.value = 'all';
                }
                refreshLuxriotSummaryView(channelId, true, false);
                return;
            }
            luxriotSummaryRollupCache[channelId] = data;
            const renderedCount = renderLuxriotRollups(data, channelId);
            const counts = data.source_counts && typeof data.source_counts === 'object' ? data.source_counts : {};
            const ctx = getCurrentSummaryRollupContext();
            const level = normalizeSummaryLevel(ctx?.level || luxriotSummaryLevel);
            const drillLabel = ctx?.sourceIds ? ` · drill ${ctx.sourceIds.length}` : '';
            const runLabel = luxriotSummaryRunFilter || 'latest';
            const countsLabel = `L1 ${Number(counts.L1 || 0)} · L2 ${Number(counts.L2 || 0)} · L3 ${Number(counts.L3 || 0)}`;
            const pendingCount = luxriotSummaryRollupRows
                .filter((row) => String(row?.summary_kind || '').trim() === 'pending_context')
                .length;
            const windowSecMap = data.window_sec && typeof data.window_sec === 'object' ? data.window_sec : {};
            const windowLabel = formatSummaryWindowLabel(windowSecMap[level]);
            const pendingLabel = pendingCount > 0 ? ` · pending ${pendingCount}` : '';
            const waitLabel = renderedCount === 0 ? ` · waiting for ${level} window ${windowLabel}` : '';
            const channelLabel = getLuxriotChannelLabel(channelId);
            setLuxriotSummaryMeta(withSummaryUpdatedMeta(`${channelLabel} · ${level}${drillLabel} · ${renderedCount} items${pendingLabel}${waitLabel} · run ${runLabel} · ${getSummaryRangeLabel()} · ${countsLabel}`));
            setLuxriotStatus(`Rollup view ${level} · ${renderedCount} entries`);
        } catch (err) {
            setLuxriotSummaryMeta('Failed to load rollups: ' + (err.message || 'Unknown error'), true);
            setLuxriotStatus('Failed to fetch rollups: ' + err.message, true);
        }
    }

    async function refreshLuxriotSummaryView(channelId = getSelectedSummaryChannel(), force = false, allowRunFallback = true) {
        if (!channelId) return;
        if (luxriotSummaryRefreshInFlight) {
            const next = luxriotSummaryRefreshQueued || {};
            luxriotSummaryRefreshQueued = {
                channelId,
                force: Boolean(force || next.force),
                allowRunFallback: Boolean((allowRunFallback !== false) || (next.allowRunFallback !== false)),
            };
            if (force) {
                setLuxriotStatus('Refresh queued...');
            }
            return false;
        }
        luxriotSummaryRefreshInFlight = true;
        try {
            if (isRollupViewActive()) {
                await refreshLuxriotRollups(channelId, force, allowRunFallback);
            } else {
                await refreshLuxriotSummaries(channelId, force, allowRunFallback);
            }
            return true;
        } finally {
            luxriotSummaryRefreshInFlight = false;
            if (luxriotSummaryRefreshQueued) {
                const next = luxriotSummaryRefreshQueued;
                luxriotSummaryRefreshQueued = null;
                void refreshLuxriotSummaryView(
                    next.channelId || getSelectedSummaryChannel(),
                    Boolean(next.force),
                    next.allowRunFallback !== false,
                );
            }
        }
    }

    function updateProbeChannelRuntime(payload, rerender = false) {
        const data = payload && typeof payload === 'object' ? payload : {};
        const pausedChannels = new Set(
            (Array.isArray(data.paused_analytics_channels) ? data.paused_analytics_channels : [])
                .map((val) => parseInt(String(val), 10))
                .filter((val) => Number.isFinite(val))
        );
        const analyticsStreams = Array.isArray(data.analytics_streams) ? data.analytics_streams : [];
        const nextState = {};
        analyticsStreams.forEach((stream) => {
            const channelId = parseInt(String(stream?.channel_id ?? ''), 10);
            if (!Number.isFinite(channelId)) return;
            if (stream?.running) {
                nextState[channelId] = 'running';
            } else if (pausedChannels.has(channelId)) {
                nextState[channelId] = 'paused';
            } else {
                nextState[channelId] = 'idle';
            }
        });
        pausedChannels.forEach((channelId) => {
            if (!(channelId in nextState)) {
                nextState[channelId] = 'paused';
            }
        });
        Object.keys(probeChannelRuntime).forEach((channelId) => {
            delete probeChannelRuntime[channelId];
        });
        Object.assign(probeChannelRuntime, nextState);
        Object.keys(probeCaptureState).forEach((channelId) => {
            delete probeCaptureState[channelId];
        });
        Object.entries(probeChannelRuntime).forEach(([channelId, state]) => {
            if (state === 'running') {
                probeCaptureState[channelId] = true;
                delete probeCaptureManualStop[channelId];
            }
        });
        updateProbeCaptureMeta(getSelectedProbeChannelId());
        syncProbePreview(getSelectedProbeChannelId());
        if (rerender) {
            renderProbeCards();
        }
    }

    async function refreshProbeRuntimeState(rerender = false) {
        try {
            const resp = await fetch('/luxriot/streams');
            const data = await resp.json();
            if (!resp.ok || data.error) {
                throw new Error(data.error || 'Failed to fetch runtime stream state');
            }
            updateProbeChannelRuntime(data, rerender);
        } catch (_) {
            // Keep previous runtime snapshot when stream endpoint is unavailable.
        }
    }

    function renderLuxriotStreams(payload, probes = probeCatalog) {
        if (!luxriotStreams) return;
        const data = payload && typeof payload === 'object' ? payload : {};
        const videoStreams = Array.isArray(data.video_streams) ? data.video_streams : [];
        const analyticsStreams = Array.isArray(data.analytics_streams) ? data.analytics_streams : [];
        const pausedChannels = new Set(
            (Array.isArray(data.paused_analytics_channels) ? data.paused_analytics_channels : [])
                .map((val) => parseInt(String(val), 10))
                .filter((val) => Number.isFinite(val))
        );
        const historyChannels = new Set(
            (Array.isArray(data.video_history_channels) ? data.video_history_channels : [])
                .map((val) => parseInt(String(val), 10))
                .filter((val) => Number.isFinite(val))
        );
        updateProbeChannelRuntime(data, probeList.length > 0);
        luxriotStreamsCache = [...videoStreams, ...analyticsStreams];
        const sortedVideo = videoStreams
            .slice()
            .sort((a, b) => (Number(a.channel_id) || 0) - (Number(b.channel_id) || 0));
        const sortedAnalytics = analyticsStreams
            .slice()
            .sort((a, b) => (Number(a.channel_id) || 0) - (Number(b.channel_id) || 0));
        const videoByChannel = new Map();
        sortedVideo.forEach((stream) => {
            const channelId = parseInt(String(stream?.channel_id ?? ''), 10);
            if (!Number.isFinite(channelId)) return;
            videoByChannel.set(channelId, stream);
        });
        const analyticsByChannel = new Map();
        sortedAnalytics.forEach((stream) => {
            const channelId = parseInt(String(stream?.channel_id ?? ''), 10);
            if (!Number.isFinite(channelId)) return;
            analyticsByChannel.set(channelId, stream);
        });
        const probeStatsByChannel = new Map();
        (Array.isArray(probes) ? probes : []).forEach((probe) => {
            const channelId = parseInt(String(probe?.channel_id ?? ''), 10);
            if (!Number.isFinite(channelId)) return;
            if (!probeStatsByChannel.has(channelId)) {
                probeStatsByChannel.set(channelId, { total: 0, enabled: 0, disabled: 0 });
            }
            const stats = probeStatsByChannel.get(channelId);
            stats.total += 1;
            if (probe?.enabled === false) stats.disabled += 1;
            else stats.enabled += 1;
        });
        const channelIds = new Set();
        sortedVideo.forEach((stream) => {
            const channelId = parseInt(String(stream?.channel_id ?? ''), 10);
            if (Number.isFinite(channelId)) channelIds.add(channelId);
        });
        sortedAnalytics.forEach((stream) => {
            const channelId = parseInt(String(stream?.channel_id ?? ''), 10);
            if (Number.isFinite(channelId)) channelIds.add(channelId);
        });
        pausedChannels.forEach((channelId) => channelIds.add(channelId));
        historyChannels.forEach((channelId) => channelIds.add(channelId));
        probeStatsByChannel.forEach((_, channelId) => channelIds.add(channelId));
        if (!channelIds.size) {
            luxriotStreams.innerHTML = '<div class="loading">No active channels.</div>';
            return;
        }
        const rows = Array.from(channelIds)
            .sort((a, b) => a - b)
            .map((channelId) => {
                const video = videoByChannel.get(channelId) || null;
                const analytics = analyticsByChannel.get(channelId) || null;
                const stats = probeStatsByChannel.get(channelId) || { total: 0, enabled: 0, disabled: 0 };
                const hasProbes = stats.total > 0;
                const enabledCount = stats.enabled || 0;
                const isVideoRunning = Boolean(video?.running);
                const isProbeRunning = Boolean(analytics?.running);
                const isProbePaused = pausedChannels.has(channelId);
                const hasVideoHistory = historyChannels.has(channelId);

                const videoParts = [];
                if (isVideoRunning) {
                    const batch = Number(video?.batch_size) || 0;
                    const queued = Number(video?.pending_frames) || 0;
                    const flushes = Number(video?.flush_count) || 0;
                    if (batch > 0) videoParts.push(`batch ${batch}`);
                    videoParts.push(`${queued} queued`);
                    if (flushes > 0) videoParts.push(`${flushes} flushes`);
                    if (video?.last_error) videoParts.push('error');
                }
                const videoLine = isVideoRunning
                    ? `Video summaries: active${videoParts.length ? ` · ${videoParts.join(' · ')}` : ''}`
                    : hasVideoHistory
                        ? 'Video summaries: stopped · history available'
                        : 'Video summaries: idle';

                const probeParts = [];
                if (isProbeRunning) {
                    const queued = Number(analytics?.pending_frames) || 0;
                    const intervalSec = Number(analytics?.interval_sec);
                    const fpsLabel = Number.isFinite(intervalSec) && intervalSec > 0 ? `${(1 / intervalSec).toFixed(2)} fps` : 'n/a fps';
                    probeParts.push(fpsLabel, `${queued} buffered`);
                    if (analytics?.last_error) probeParts.push('error');
                }
                const probeLine = isProbeRunning
                    ? `Probe capture: active${probeParts.length ? ` · ${probeParts.join(' · ')}` : ''}`
                    : isProbePaused
                        ? 'Probe capture: paused'
                        : enabledCount > 0
                            ? 'Probe capture: idle'
                            : hasProbes
                                ? 'Probe capture: all probes disabled'
                                : 'Probe capture: no probes configured';

                const probesLine = hasProbes
                    ? `${stats.total} probes · ${enabledCount} enabled${stats.disabled ? ` · ${stats.disabled} disabled` : ''}`
                    : 'No probes configured';
                const videoTag = isVideoRunning
                    ? '<span class="luxriot-stream-tag">video active</span>'
                    : '<span class="luxriot-stream-tag idle">video idle</span>';
                const probeTag = isProbeRunning
                    ? '<span class="luxriot-stream-tag">probes active</span>'
                    : isProbePaused
                        ? '<span class="luxriot-stream-tag paused">probes paused</span>'
                        : enabledCount > 0
                            ? '<span class="luxriot-stream-tag idle">probes idle</span>'
                            : hasProbes
                                ? '<span class="luxriot-stream-tag idle">probes disabled</span>'
                                : '<span class="luxriot-stream-tag idle">no probes</span>';
                const pauseLabel = isProbePaused ? 'Resume probes' : 'Pause probes';
                const pauseAction = isProbePaused ? 'resume' : 'pause';
                const canPauseProbes = !isProbePaused && (isProbeRunning || enabledCount > 0);
                const canResumeProbes = isProbePaused;
                const canProbeAction = canPauseProbes || canResumeProbes;
                const canStopAll = isVideoRunning || isProbeRunning;
                const channelLabel = getLuxriotChannelLabel(channelId);
                return `
                    <div class="luxriot-stream-item">
                        <div class="luxriot-stream-head">
                            <div class="luxriot-stream-title-wrap">
                                <div class="luxriot-stream-kind">Channel</div>
                                <div class="luxriot-stream-title">${escapeHtml(channelLabel)}</div>
                            </div>
                            <div class="luxriot-stream-tags">${videoTag} ${probeTag}</div>
                        </div>
                        <div class="luxriot-stream-stats">
                            <span class="luxriot-stream-stat">${escapeHtml(probesLine)}</span>
                            <span class="luxriot-stream-stat">${escapeHtml(videoLine)}</span>
                            <span class="luxriot-stream-stat">${escapeHtml(probeLine)}</span>
                        </div>
                        <div class="luxriot-stream-controls">
                            <button class="feature-btn" data-summary-channel="${channelId}" title="Open this channel in summaries panel">View summaries</button>
                            <button class="feature-btn" data-stream-stop="${channelId}" data-stream-type="analytics" data-stream-action="${pauseAction}" ${canProbeAction ? '' : 'disabled'}>${pauseLabel}</button>
                            <button class="feature-btn" data-stream-stop="${channelId}" data-stream-type="video" ${isVideoRunning ? '' : 'disabled'}>Stop video</button>
                            <button class="feature-btn" data-stream-stop="${channelId}" data-stream-type="both" ${canStopAll ? '' : 'disabled'}>Stop all</button>
                        </div>
                    </div>
                `;
            });
        luxriotStreams.innerHTML = rows.join('');
    }

    async function refreshLuxriotStreams() {
        if (!luxriotStreams) return;
        try {
            const resp = await fetch('/luxriot/streams');
            const data = await resp.json();
            if (!resp.ok || data.error) {
                throw new Error(data.error || 'Failed to fetch stream state');
            }
            const nextCaptureState = {};
            const videoStreams = Array.isArray(data.video_streams) ? data.video_streams : [];
            videoStreams.forEach((stream) => {
                const channelId = parseInt(String(stream?.channel_id ?? ''), 10);
                if (!Number.isFinite(channelId)) return;
                nextCaptureState[String(channelId)] = Boolean(stream?.running);
            });
            Object.keys(luxriotCaptureRunningByChannel).forEach((key) => {
                delete luxriotCaptureRunningByChannel[key];
            });
            Object.assign(luxriotCaptureRunningByChannel, nextCaptureState);
            const selectedChannelId = getSelectedLuxriotChannel();
            if (!(String(selectedChannelId) in luxriotCaptureRunningByChannel)) {
                luxriotCaptureRunningByChannel[String(selectedChannelId)] = false;
            }
            if (luxriotLiveModelInput && document.activeElement !== luxriotLiveModelInput) {
                const selectedVideoStream = videoStreams.find((stream) => parseInt(String(stream?.channel_id ?? ''), 10) === selectedChannelId);
                const liveModel = String(selectedVideoStream?.model || '').trim();
                if (liveModel) {
                    setModelSelectOptions(luxriotLiveModelInput, liveModel);
                    localStorage.setItem(LUXRIOT_LIVE_MODEL_STORAGE_KEY, liveModel);
                }
            }
            updateLuxriotCaptureToggleButton(selectedChannelId);
            try {
                const probesResp = await fetch('/probes/list');
                const probesData = await probesResp.json();
                if (probesResp.ok && !probesData.error && Array.isArray(probesData.probes)) {
                    probeCatalog = probesData.probes;
                }
            } catch (_) {
                // Keep previous probe catalog if probe listing fails.
            }
            renderLuxriotStreams(data, probeCatalog);
        } catch (err) {
            luxriotStreams.innerHTML = `<div class="loading">Stream state unavailable: ${escapeHtml(err.message || 'Unknown error')}</div>`;
        }
    }

    function guessProbeCaptureFps(channelId) {
        const targetChannel = parseInt(String(channelId || ''), 10);
        if (!Number.isFinite(targetChannel)) return 0;
        const fpsValues = (Array.isArray(probeCatalog) ? probeCatalog : [])
            .filter((probe) => parseInt(String(probe?.channel_id ?? ''), 10) === targetChannel)
            .map((probe) => Number.parseFloat(String(probe?.fps ?? '')))
            .filter((fps) => Number.isFinite(fps) && fps > 0);
        if (!fpsValues.length) return 0;
        return Math.max(...fpsValues);
    }

    async function resumeLuxriotProbeCapture(channelId) {
        const parsedChannelId = parseInt(String(channelId || ''), 10);
        if (!Number.isFinite(parsedChannelId)) {
            setLuxriotStatus('Invalid channel id for probe resume', true);
            return;
        }
        const fps = guessProbeCaptureFps(parsedChannelId);
        setLuxriotStatus(`Resuming probe capture on channel ${parsedChannelId}...`);
        try {
            const response = await fetch('/probes/start_capture', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    channel_id: parsedChannelId,
                    fps,
                    clear_pause: true,
                }),
            });
            await parseApiJson(response, 'Probe resume failed');
            await refreshLuxriotStreams();
            await refreshProbeStatus(parsedChannelId);
            setLuxriotStatus(`Probe capture resumed on channel ${parsedChannelId}`);
        } catch (err) {
            setLuxriotStatus(err.message || 'Failed to resume probe capture', true);
        }
    }

    async function stopLuxriotStream(channelId, streamType) {
        const parsedChannelId = parseInt(String(channelId || ''), 10);
        const normalizedType = String(streamType || '').trim().toLowerCase();
        if (!Number.isFinite(parsedChannelId)) {
            setLuxriotStatus('Invalid channel id for stream stop', true);
            return;
        }
        if (!['video', 'analytics', 'both'].includes(normalizedType)) {
            setLuxriotStatus('Invalid stream type', true);
            return;
        }
        const actionLabel = normalizedType === 'analytics'
            ? 'Pausing probe capture'
            : normalizedType === 'video'
                ? 'Stopping video summaries'
                : 'Stopping video and pausing probes';
        setLuxriotStatus(`${actionLabel} on channel ${parsedChannelId}...`);
        try {
            const response = await fetch('/luxriot/streams/stop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    channel_id: parsedChannelId,
                    stream_type: normalizedType,
                    pause_analytics: true,
                }),
            });
            await parseApiJson(response, 'Stream stop failed');
            await refreshLuxriotStreams();
            if (normalizedType === 'video' || normalizedType === 'both') {
                await refreshLuxriotSummaryView(getSelectedSummaryChannel(), true);
            }
            if (normalizedType === 'analytics' || normalizedType === 'both') {
                await refreshProbeStatus(parsedChannelId);
            }
            const doneLabel = normalizedType === 'analytics'
                ? 'Probe capture paused'
                : normalizedType === 'video'
                    ? 'Video summaries stopped'
                    : 'Video summaries stopped, probes paused';
            setLuxriotStatus(`${doneLabel} on channel ${parsedChannelId}`);
        } catch (err) {
            setLuxriotStatus(err.message || 'Failed to stop stream', true);
        }
    }

    async function stopAllLuxriotStreams(streamType) {
        const normalizedType = String(streamType || '').trim().toLowerCase();
        const stopVideo = normalizedType === 'video' || normalizedType === 'both';
        const stopAnalytics = normalizedType === 'analytics' || normalizedType === 'both';
        if (!stopVideo && !stopAnalytics) return;
        const actionLabel = stopVideo && stopAnalytics
            ? 'Stopping video summaries and pausing probes'
            : stopVideo
                ? 'Stopping all video summaries'
                : 'Pausing all probe capture streams';
        setLuxriotStatus(`${actionLabel}...`);
        try {
            const response = await fetch('/luxriot/streams/stop_all', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    stop_video: stopVideo,
                    stop_analytics: stopAnalytics,
                    pause_analytics: true,
                }),
            });
            await parseApiJson(response, 'Stop-all failed');
            await refreshLuxriotStreams();
            if (stopVideo) {
                stopLuxriotSummaryPoll();
                await refreshLuxriotSummaryView(getSelectedSummaryChannel(), true);
            }
            if (stopAnalytics) {
                await refreshProbeStatus();
            }
            const doneLabel = stopVideo && stopAnalytics
                ? 'Stopped video summaries and paused probes'
                : stopVideo
                    ? 'Stopped all video summaries'
                    : 'Paused all probe capture streams';
            setLuxriotStatus(doneLabel);
        } catch (err) {
            setLuxriotStatus(err.message || 'Failed to stop streams', true);
        }
    }

    async function sendLuxriotBookmarkFromLog(logIndex, triggerBtn = null) {
        const idx = Number.isFinite(logIndex) ? logIndex : parseInt(String(logIndex || ''), 10);
        if (!Number.isFinite(idx) || idx < 0 || idx >= luxriotSummaryLogCache.length) {
            setLuxriotStatus('Invalid summary selection', true);
            return;
        }
        const log = luxriotSummaryLogCache[idx] || {};
        const summaryText = String(log.summary || '').trim();
        if (!summaryText) {
            setLuxriotStatus('No summary text to bookmark', true);
            return;
        }
        const channelId = Number(log.channel_id) || getSelectedLuxriotChannel() || luxriotDefaults.channelId;
        const firstLine = summaryText.split(/\r?\n/, 1)[0].trim();
        const titleBase = firstLine || `Channel ${channelId} summary`;
        const title = titleBase.length > 80 ? `${titleBase.slice(0, 77)}...` : titleBase;
        const description = summaryText.length > 2400 ? `${summaryText.slice(0, 2397)}...` : summaryText;
        const createdAtSec = Number(log.created_at);
        const timestampMs = Number.isFinite(createdAtSec) ? Math.round(createdAtSec * 1000) : null;

        const button = triggerBtn;
        const originalLabel = button ? button.textContent : '';
        if (button) {
            button.disabled = true;
            button.textContent = 'Saving...';
        }

        try {
            const response = await fetch('/luxriot/bookmark', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    channel_id: channelId,
                    title: `Live summary: ${title}`,
                    description,
                    severity: 'normal',
                    state: 'new',
                    timestamp_ms: timestampMs
                }),
            });
            await parseApiJson(response, 'Bookmark failed');
            setLuxriotStatus(`Bookmark sent for channel ${channelId}`);
            if (button) {
                button.textContent = 'Bookmarked';
            }
        } catch (err) {
            setLuxriotStatus(err.message || 'Bookmark failed', true);
            if (button) {
                button.textContent = originalLabel || 'Bookmark';
            }
        } finally {
            if (button) {
                button.disabled = false;
            }
        }
    }

    function formatLuxriotSummaryExport(log) {
        const createdRaw = Number(log?.created_at);
        const ts = Number.isFinite(createdRaw) ? new Date(createdRaw * 1000) : null;
        const tsLabel = ts ? ts.toISOString() : 'n/a';
        const channelId = Number(log?.channel_id) || getSelectedSummaryChannel() || luxriotDefaults.channelId;
        const frameCount = Number(log?.frame_count || 0);
        const model = String(log?.model || '').trim();
        const summary = String(log?.summary || '').trim();
        const nl = String.fromCharCode(10);
        const header = [
            `Channel: ${channelId}`,
            `Timestamp: ${tsLabel}`,
            `Frames: ${frameCount || 'n/a'}`,
            `Model: ${model || 'n/a'}`,
        ].join(nl);
        return `${header}${nl}${nl}${summary}`;
    }

    async function copyLuxriotSummaryFromLog(logIndex, triggerBtn = null) {
        const idx = Number.isFinite(logIndex) ? logIndex : parseInt(String(logIndex || ''), 10);
        if (!Number.isFinite(idx) || idx < 0 || idx >= luxriotSummaryLogCache.length) {
            setLuxriotStatus('Invalid summary selection', true);
            return;
        }
        const log = luxriotSummaryLogCache[idx] || {};
        const summary = String(log.summary || '').trim();
        if (!summary) {
            setLuxriotStatus('Summary is empty', true);
            return;
        }
        try {
            await copyTextToClipboard(formatLuxriotSummaryExport(log));
            setLuxriotStatus('Summary copied');
            if (triggerBtn) {
                const original = triggerBtn.textContent;
                triggerBtn.textContent = 'Copied';
                setTimeout(() => {
                    if (triggerBtn) triggerBtn.textContent = original || 'Copy';
                }, 1200);
            }
        } catch (err) {
            setLuxriotStatus('Failed to copy summary', true);
        }
    }

    function exportLuxriotSummaryFromLog(logIndex) {
        const idx = Number.isFinite(logIndex) ? logIndex : parseInt(String(logIndex || ''), 10);
        if (!Number.isFinite(idx) || idx < 0 || idx >= luxriotSummaryLogCache.length) {
            setLuxriotStatus('Invalid summary selection', true);
            return;
        }
        const log = luxriotSummaryLogCache[idx] || {};
        const createdRaw = Number(log?.created_at);
        const stamp = Number.isFinite(createdRaw)
            ? new Date(createdRaw * 1000).toISOString().replace(/[:]/g, '-')
            : `entry-${idx + 1}`;
        const channelId = Number(log?.channel_id) || getSelectedSummaryChannel() || luxriotDefaults.channelId;
        const filename = `luxriot_summary_ch${channelId}_${stamp}.txt`;
        downloadTextFile(filename, formatLuxriotSummaryExport(log));
        setLuxriotStatus(`Exported ${filename}`);
    }

    function toggleLuxriotSummaryCollapse(logIndex) {
        const idx = Number.isFinite(logIndex) ? logIndex : parseInt(String(logIndex || ''), 10);
        if (!Number.isFinite(idx) || idx < 0 || idx >= luxriotSummaryLogCache.length) {
            return;
        }
        const channelId = getSelectedSummaryChannel();
        const log = luxriotSummaryLogCache[idx] || {};
        const key = luxriotSummaryLogKey(log, idx);
        const nextState = !isSummaryCollapsed(channelId, key);
        setSummaryCollapsed(channelId, key, nextState);
        renderLuxriotSummaries(luxriotSummaryLogCache, channelId);
    }

    async function refreshLuxriotSummaries(channelId = getSelectedSummaryChannel(), force = false, allowRunFallback = true) {
        if (!channelId) return;
        if (!luxriotSummaryAutoRefresh && !force) return;
        try {
            const params = buildSummaryQueryParams(channelId);
            params.set('limit', '240');
            const resp = await fetch(`/luxriot/session?${params.toString()}`);
            const data = await resp.json();
            if (data.error) {
                throw new Error(data.error);
            }
            syncSummaryRunSelectOptions(data.runs, data.selected_run);
            syncSummaryFiltersFromResponse(data);
            const selectedRun = normalizeSummaryRun(luxriotSummaryRunFilter);
            if (
                allowRunFallback
                && !Boolean(data.running)
                && (selectedRun === 'live' || selectedRun === 'latest')
            ) {
                luxriotSummaryRunFilter = 'all';
                if (luxriotSummaryRunSelect) {
                    luxriotSummaryRunSelect.value = 'all';
                }
                refreshLuxriotSummaryView(channelId, true, false);
                return;
            }
            renderLuxriotSummaries(data.logs || [], channelId);
            setLuxriotCaptureRunning(channelId, Boolean(data.running));
            if (channelId === getSelectedLuxriotChannel()) {
                updateLuxriotCaptureToggleButton(channelId);
            }
            const historyCount = Number(data.archived_log_count || 0);
            const totalCount = Array.isArray(data.logs) ? data.logs.length : 0;
            const stateLabel = data.running ? 'live' : 'stopped';
            const channelLabel = getLuxriotChannelLabel(channelId);
            const detailParts = [channelLabel, stateLabel, `${totalCount} entries`, `run ${luxriotSummaryRunFilter || 'latest'}`, getSummaryRangeLabel()];
            if (historyCount > 0) detailParts.push(`hist ${historyCount}`);
            if (typeof data.pending_frames === 'number' && data.pending_frames > 0) detailParts.push(`q ${data.pending_frames}`);
            if (data.last_error) detailParts.push('err');
            setLuxriotSummaryMeta(withSummaryUpdatedMeta(detailParts.join(' · ')), Boolean(data.last_error));
            let baseStatus = data.running ? `Summaries running · batch ${data.batch_size || ''}` : 'Summaries stopped';
            if (typeof data.pending_frames === 'number' && data.pending_frames > 0) {
                baseStatus += ` · ${data.pending_frames} frames queued`;
            }
            setLuxriotStatus(baseStatus, Boolean(data.last_error));
            if (data.last_error) {
                luxriotStatusLabel.title = data.last_error;
            }
        } catch (err) {
            setLuxriotSummaryMeta('Failed to load summaries: ' + (err.message || 'Unknown error'), true);
            setLuxriotStatus('Failed to fetch summaries: ' + err.message, true);
        }
    }

    function startLuxriotSummaryPoll() {
        stopLuxriotSummaryPoll();
        luxriotSummaryTimer = setInterval(() => {
            const channelId = getSelectedSummaryChannel();
            refreshLuxriotSummaryView(channelId);
            refreshLuxriotStreams();
        }, 8000);
    }

    async function startLuxriotCapture(channelIdOverride = null) {
        const channelId = Number.isFinite(channelIdOverride) ? channelIdOverride : getSelectedLuxriotChannel();
        if (!channelId) {
            setLuxriotStatus('Select a channel first', true);
            return;
        }
        await refreshLuxriotPromptSettings(false, channelId);
        const batchSize = luxriotBatchSizeSelect
            ? parseInt(luxriotBatchSizeSelect.value, 10)
            : luxriotDefaults.batchSize || 12;
        const prompt = luxriotPromptInput ? luxriotPromptInput.value.trim() : '';
        const systemPrompt = luxriotSystemPromptInput ? luxriotSystemPromptInput.value.trim() : '';
        const fallbackPrompt = videoPromptInput ? videoPromptInput.value.trim() : '';
        if (luxriotToggleCaptureBtn) {
            luxriotToggleCaptureBtn.disabled = true;
        }
        setLuxriotStatus('Starting summaries...');
        try {
            const resp = await fetch('/luxriot/start_capture', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    channel_id: channelId,
                    batch_size: batchSize,
                    prompt: prompt || fallbackPrompt,
                    model: luxriotLiveModelInput ? luxriotLiveModelInput.value.trim() : '',
                    system_prompt: systemPrompt
                })
            });
            const data = await resp.json();
            if (!resp.ok || data.error) {
                throw new Error(data.error || 'Luxriot start failed');
            }
            setLuxriotCaptureRunning(channelId, true);
            updateLuxriotCaptureToggleButton(channelId);
            const modelLabel = data?.session?.model || (luxriotLiveModelInput ? luxriotLiveModelInput.value.trim() : '') || '';
            setLuxriotStatus(`Summaries running on channel ${channelId} (batch ${batchSize}${modelLabel ? ` · ${modelLabel}` : ''})`);
            luxriotSummaryChannel = channelId;
            luxriotSummaryFollowLive = true;
            syncLuxriotSummaryChannelSelect();
            updateSummaryControlsUI();
            setSummaryUnread(0);
            refreshLuxriotSummaryView(channelId, true);
            refreshLuxriotStreams();
            startLuxriotSummaryPoll();
        } catch (err) {
            setLuxriotStatus(err.message, true);
        } finally {
            if (luxriotToggleCaptureBtn) {
                luxriotToggleCaptureBtn.disabled = false;
            }
        }
    }

    async function stopLuxriotCapture(channelIdOverride = null) {
        const channelId = Number.isFinite(channelIdOverride) ? channelIdOverride : getSelectedLuxriotChannel();
        if (luxriotToggleCaptureBtn) {
            luxriotToggleCaptureBtn.disabled = true;
        }
        setLuxriotStatus('Stopping...');
        try {
            const resp = await fetch('/luxriot/stop_capture', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ channel_id: channelId })
            });
            const data = await resp.json();
            if (data.error) {
                throw new Error(data.error);
            }
            setLuxriotCaptureRunning(channelId, false);
            updateLuxriotCaptureToggleButton(channelId);
            setLuxriotStatus('Summaries stopped');
            refreshLuxriotSummaryView(getSelectedSummaryChannel(), true);
            refreshLuxriotStreams();
        } catch (err) {
            setLuxriotStatus(err.message, true);
        } finally {
            if (luxriotToggleCaptureBtn) {
                luxriotToggleCaptureBtn.disabled = false;
            }
        }
    }

    async function toggleLuxriotCapture() {
        const channelId = getSelectedLuxriotChannel();
        if (!channelId) {
            setLuxriotStatus('Select a channel first', true);
            return;
        }
        if (isLuxriotCaptureRunning(channelId)) {
            await stopLuxriotCapture(channelId);
        } else {
            await startLuxriotCapture(channelId);
        }
    }

    async function flushLuxriotCapture() {
        const channelId = getSelectedLuxriotChannel();
        setLuxriotStatus('Flushing...');
        try {
            const resp = await fetch('/luxriot/flush_capture', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ channel_id: channelId })
            });
            const data = await resp.json();
            if (!resp.ok || data.error) {
                throw new Error(data.error || data.message || 'Flush failed');
            }
            setLuxriotStatus('Buffer flushed');
            if (data.status) {
                if (getSelectedSummaryChannel() === channelId) {
                    if (isRollupViewActive()) {
                        await refreshLuxriotSummaryView(channelId, true);
                    } else {
                        renderLuxriotSummaries(data.status.logs || [], channelId);
                    }
                }
            }
            refreshLuxriotStreams();
        } catch (err) {
            setLuxriotStatus(err.message, true);
        }
    }

    async function ensureLuxriotInit() {
        if (luxriotInitialized) return;
        luxriotInitialized = true;
        await fetchLuxriotChannels();
        await refreshLuxriotPromptSettings();
        updateLuxriotCaptureToggleButton(getSelectedLuxriotChannel());
        updateSummaryControlsUI();
        setSummaryUnread(0);
        syncLuxriotSummaryChannelSelect();
        startLuxriotPreview();
        refreshLuxriotSummaryView(getSelectedSummaryChannel(), true);
        refreshLuxriotStreams();
        startLuxriotSummaryPoll();
    }

    const savedVideoPrompt = localStorage.getItem('evs_video_prompt');
    if (savedVideoPrompt && videoPromptInput) {
        videoPromptInput.value = savedVideoPrompt;
        if (saveVideoPromptInput) {
            saveVideoPromptInput.checked = true;
        }
    }
    if (luxriotPromptInput && videoPromptInput && videoPromptInput.value && !luxriotPromptInput.value) {
        luxriotPromptInput.value = videoPromptInput.value;
    }
    updateLuxriotBatchInfo();
    setLuxriotPromptModalTab('stream');
    updateLuxriotCaptureToggleButton(luxriotActiveChannel);
    function syncProbeChannelSelect() {
        if (probeChannelSelect && luxriotChannelSelect && luxriotChannelSelect.innerHTML) {
            probeChannelSelect.innerHTML = luxriotChannelSelect.innerHTML;
            probeChannelSelect.value = luxriotChannelSelect.value || luxriotDefaults.channelId;
        }
    }
    syncProbeChannelSelect();
    syncSummaryRunSelectOptions([], luxriotSummaryRunFilter);
    applySummaryFiltersFromInputs();
    setSummaryBaseLevel(luxriotSummaryLevel);

    setMode(currentMode);
    
    // Settings modal elements
    const settingsBtn = document.getElementById('settingsBtn');
    const settingsModal = document.getElementById('settingsModal');
    const closeSettingsBtn = document.getElementById('closeSettings');
    const saveSettingsBtn = document.getElementById('saveSettings');
    const resetSettingsBtn = document.getElementById('resetSettings');
    const settingsStatus = document.getElementById('settingsStatus');
    const settingsScrollArea = document.getElementById('settingsScrollArea');
    const settingsNavButtons = Array.from(document.querySelectorAll('[data-settings-target]'));
    const envEditorInput = document.getElementById('envEditor');
    const reloadEnvBtn = document.getElementById('reloadEnvBtn');
    const saveEnvBtn = document.getElementById('saveEnvBtn');
    const thumbnailQualitySlider = document.getElementById('thumbnailQuality');
    const qualityValue = document.getElementById('qualityValue');
    const embedderSelect = document.getElementById('embedder');
    const fusionEnabledInput = document.getElementById('fusionEnabled');
    const fusionAlphaInput = document.getElementById('fusionAlpha');
    const fusionAlphaValue = document.getElementById('fusionAlphaValue');
    const dinoModelInput = document.getElementById('dinoModel');
    const dinoEmbedDimInput = document.getElementById('dinoEmbedDim');
    const dinoWeightsInput = document.getElementById('dinoWeightsPath');
    const indexModeSelect = document.getElementById('indexMode');
    const rerankEnabledInput = document.getElementById('rerankEnabled');
    const rerankTopKInput = document.getElementById('rerankTopK');
    const segmentsEnabledInput = document.getElementById('segmentsEnabled');
    const segmentMinPatchesInput = document.getElementById('segmentMinPatches');
    const segmentThresholdSlider = document.getElementById('segmentThresholdSlider');
    const segmentThresholdValueEl = document.getElementById('segmentThresholdValue');
    const segmentThresholdControl = document.getElementById('segmentThresholdControl');
    const luxriotBaseUrlInput = document.getElementById('luxriotBaseUrl');
    const luxriotUsernameInput = document.getElementById('luxriotUsername');
    const luxriotPasswordInput = document.getElementById('luxriotPassword');
    const luxriotDefaultChannelIdInput = document.getElementById('luxriotDefaultChannelId');
    const luxriotSnapshotIntervalInput = document.getElementById('luxriotSnapshotInterval');
    const luxriotSnapshotMaxEdgeInput = document.getElementById('luxriotSnapshotMaxEdge');
    const luxriotMaxBufferFramesInput = document.getElementById('luxriotMaxBufferFrames');
    const luxriotAutoBookmarksInput = document.getElementById('luxriotAutoBookmarks');
    const probeBookmarkCooldownSecInput = document.getElementById('probeBookmarkCooldownSec');
    const probeBookmarkDedupeWindowSecInput = document.getElementById('probeBookmarkDedupeWindowSec');
    const probeBookmarkSimHighInput = document.getElementById('probeBookmarkSimHigh');
    const probeBookmarkMarginDeltaInput = document.getElementById('probeBookmarkMarginDelta');
    const probeBookmarkScoreDeltaInput = document.getElementById('probeBookmarkScoreDelta');
    const probeBookmarkMaxFrameGapInput = document.getElementById('probeBookmarkMaxFrameGap');
    const luxriotSevInfoInput = document.getElementById('luxriotSevInfo');
    const luxriotSevLowInput = document.getElementById('luxriotSevLow');
    const luxriotSevNormalInput = document.getElementById('luxriotSevNormal');
    const luxriotSevHighInput = document.getElementById('luxriotSevHigh');
    const luxriotSevCriticalInput = document.getElementById('luxriotSevCritical');
    const adminUsersNavBtn = document.getElementById('adminUsersNavBtn');
    const adminUsersSection = document.getElementById('settings-section-users');
    const adminUsersDenied = document.getElementById('adminUsersDenied');
    const adminUsersPanel = document.getElementById('adminUsersPanel');
    const adminUsersStatus = document.getElementById('adminUsersStatus');
    const adminUsersRefreshBtn = document.getElementById('adminUsersRefreshBtn');
    const adminUsersNewBtn = document.getElementById('adminUsersNewBtn');
    const adminUsersList = document.getElementById('adminUsersList');
    const adminUserEditorTitle = document.getElementById('adminUserEditorTitle');
    const adminUsernameInput = document.getElementById('adminUsernameInput');
    const adminDisplayNameInput = document.getElementById('adminDisplayNameInput');
    const adminPasswordInput = document.getElementById('adminPasswordInput');
    const adminRolesList = document.getElementById('adminRolesList');
    const adminAllowedChannelsInput = document.getElementById('adminAllowedChannelsInput');
    const adminUserActiveInput = document.getElementById('adminUserActiveInput');
    const adminUserSaveBtn = document.getElementById('adminUserSaveBtn');
    const adminUserRevokeBtn = document.getElementById('adminUserRevokeBtn');
    const adminUserClearBtn = document.getElementById('adminUserClearBtn');
    const adminSessionsActiveOnlyInput = document.getElementById('adminSessionsActiveOnlyInput');
    const adminSessionsRefreshBtn = document.getElementById('adminSessionsRefreshBtn');
    const adminSessionsList = document.getElementById('adminSessionsList');
    const auditEventsNavBtn = document.getElementById('auditEventsNavBtn');
    const auditEventsSection = document.getElementById('settings-section-audit');
    const auditEventsDenied = document.getElementById('auditEventsDenied');
    const auditEventsPanel = document.getElementById('auditEventsPanel');
    const auditEventsStatus = document.getElementById('auditEventsStatus');
    const auditEventsRefreshBtn = document.getElementById('auditEventsRefreshBtn');
    const auditEventsNextBtn = document.getElementById('auditEventsNextBtn');
    const auditResultFilter = document.getElementById('auditResultFilter');
    const auditActionFilter = document.getElementById('auditActionFilter');
    const auditActorFilter = document.getElementById('auditActorFilter');
    const auditChannelFilter = document.getElementById('auditChannelFilter');
    const auditRequestFilter = document.getElementById('auditRequestFilter');
    const auditLimitSelect = document.getElementById('auditLimitSelect');
    const auditEventsList = document.getElementById('auditEventsList');
    
    let segmentThreshold = 0.7;
    let adminRoles = [];
    let adminUsers = [];
    let adminSessions = [];
    let selectedAdminUserId = null;
    let auditEvents = [];
    let auditNextCursor = null;
    let auditLastParams = null;

    function setActiveSettingsNav(targetId) {
        settingsNavButtons.forEach((btn) => {
            btn.classList.toggle('active', btn.dataset.settingsTarget === targetId);
        });
    }

    function scrollSettingsSectionIntoView(targetId, behavior = 'smooth') {
        if (!settingsScrollArea) return;
        const target = document.getElementById(targetId);
        if (!target) return;
        const offsetTop = Math.max(0, target.offsetTop - 8);
        settingsScrollArea.scrollTo({ top: offsetTop, behavior });
        setActiveSettingsNav(targetId);
    }

    function toBool(value, fallback = false) {
        if (typeof value === 'boolean') return value;
        if (value === null || value === undefined) return fallback;
        if (typeof value === 'string') {
            const normalized = value.trim().toLowerCase();
            if (['1', 'true', 'yes', 'on'].includes(normalized)) return true;
            if (['0', 'false', 'no', 'off'].includes(normalized)) return false;
            return fallback;
        }
        return Boolean(value);
    }

    function userCanManageUsers() {
        return Boolean(
            AUTH_ENABLED
            && authCurrentUser
            && Array.isArray(authCurrentUser.permissions)
            && authCurrentUser.permissions.includes('users:manage')
        );
    }

    function setAdminUsersStatus(message = '', type = '') {
        if (!adminUsersStatus) return;
        adminUsersStatus.textContent = message;
        adminUsersStatus.className = `admin-inline-status ${type || ''}`.trim();
    }

    function setAdminUsersAccess(allowed) {
        const enabled = Boolean(allowed);
        if (adminUsersNavBtn) adminUsersNavBtn.classList.toggle('is-hidden', !enabled);
        if (adminUsersSection) adminUsersSection.classList.toggle('is-hidden', !enabled);
        if (adminUsersDenied) adminUsersDenied.classList.toggle('is-hidden', enabled);
        if (adminUsersPanel) adminUsersPanel.classList.toggle('is-hidden', !enabled);
    }

    function syncAdminUsersAccess() {
        const allowed = userCanManageUsers();
        setAdminUsersAccess(allowed);
        return allowed;
    }

    function hasAllChannelAccess() {
        const channels = authCurrentUser && Array.isArray(authCurrentUser.allowedChannelIds)
            ? authCurrentUser.allowedChannelIds
            : [];
        return channels.some((value) => String(value).trim() === '*');
    }

    function userCanViewAudit() {
        return Boolean(
            AUTH_ENABLED
            && authCurrentUser
            && Array.isArray(authCurrentUser.permissions)
            && authCurrentUser.permissions.includes('audit:view')
            && hasAllChannelAccess()
        );
    }

    function setAuditAccess(allowed) {
        const enabled = Boolean(allowed);
        if (auditEventsNavBtn) auditEventsNavBtn.classList.toggle('is-hidden', !enabled);
        if (auditEventsSection) auditEventsSection.classList.toggle('is-hidden', !enabled);
        if (auditEventsDenied) auditEventsDenied.classList.toggle('is-hidden', enabled);
        if (auditEventsPanel) auditEventsPanel.classList.toggle('is-hidden', !enabled);
    }

    function syncAuditAccess() {
        const allowed = userCanViewAudit();
        setAuditAccess(allowed);
        return allowed;
    }

    function setAuditStatus(message = '', type = '') {
        if (!auditEventsStatus) return;
        auditEventsStatus.textContent = message;
        auditEventsStatus.className = `admin-inline-status ${type || ''}`.trim();
    }

    function effectiveAdminRoles() {
        if (adminRoles.length) return adminRoles;
        return ['admin', 'engineer', 'operator', 'viewer'].map((name) => ({
            name,
            permissions: [],
        }));
    }

    function formatAllowedChannels(channels) {
        const values = Array.isArray(channels) ? channels : [];
        if (values.some((value) => String(value).trim() === '*')) return '*';
        return values.map((value) => String(value).trim()).filter(Boolean).join(', ');
    }

    function parseAllowedChannelsText(text) {
        const raw = String(text || '').trim();
        if (!raw) return [];
        if (raw === '*') return ['*'];
        const seen = new Set();
        const parsed = [];
        raw.split(/[,\s]+/).forEach((item) => {
            if (seen.has('*')) return;
            const clean = item.trim();
            if (!clean) return;
            if (clean === '*') {
                seen.clear();
                parsed.length = 0;
                parsed.push('*');
                seen.add('*');
                return;
            }
            const value = Number.parseInt(clean, 10);
            if (!Number.isFinite(value) || value <= 0 || String(value) !== clean) {
                throw new Error('Channels must be "*" or positive numeric IDs.');
            }
            if (!seen.has(value)) {
                seen.add(value);
                parsed.push(value);
            }
        });
        return parsed;
    }

    function formatDateTime(value) {
        if (!value) return 'n/a';
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return String(value);
        return date.toLocaleString();
    }

    function selectedAdminRoles() {
        if (!adminRolesList) return [];
        return Array.from(adminRolesList.querySelectorAll('input[type="checkbox"]:checked'))
            .map((input) => String(input.value || '').trim())
            .filter(Boolean);
    }

    function renderAdminRoles(selectedRoles = []) {
        if (!adminRolesList) return;
        const selected = new Set(selectedRoles.map((role) => String(role || '').trim()));
        adminRolesList.innerHTML = effectiveAdminRoles().map((role) => {
            const name = String(role.name || '').trim();
            if (!name) return '';
            const permissions = Array.isArray(role.permissions) ? role.permissions : [];
            const title = permissions.length ? ` title="${escapeHtml(permissions.join(', '))}"` : '';
            const checked = selected.has(name) ? ' checked' : '';
            return `
                <label class="admin-role-item"${title}>
                    <input type="checkbox" value="${escapeHtml(name)}"${checked} />
                    <span>${escapeHtml(name)}</span>
                </label>
            `;
        }).join('');
    }

    function setAdminEditorUser(user = null) {
        const isExisting = Boolean(user && user.id);
        selectedAdminUserId = isExisting ? String(user.id) : null;
        if (adminUserEditorTitle) {
            adminUserEditorTitle.textContent = isExisting ? `Edit ${user.username || user.id}` : 'New User';
        }
        if (adminUsernameInput) {
            adminUsernameInput.value = isExisting ? String(user.username || '') : '';
            adminUsernameInput.disabled = isExisting;
        }
        if (adminDisplayNameInput) {
            adminDisplayNameInput.value = isExisting ? String(user.displayName || '') : '';
        }
        if (adminPasswordInput) {
            adminPasswordInput.value = '';
            adminPasswordInput.placeholder = isExisting ? 'Leave blank to keep current password' : 'Required for new user';
        }
        renderAdminRoles(isExisting ? (user.roles || []) : ['viewer']);
        if (adminAllowedChannelsInput) {
            adminAllowedChannelsInput.value = isExisting ? formatAllowedChannels(user.allowedChannelIds || []) : '*';
        }
        if (adminUserActiveInput) {
            adminUserActiveInput.checked = isExisting ? Boolean(user.isActive) : true;
        }
        if (adminUserRevokeBtn) {
            adminUserRevokeBtn.disabled = !isExisting;
        }
        renderAdminUsers();
    }

    function renderAdminUsers() {
        if (!adminUsersList) return;
        if (!adminUsers.length) {
            adminUsersList.innerHTML = '<div class="admin-empty">No users found.</div>';
            return;
        }
        const currentUserId = authCurrentUser ? String(authCurrentUser.id || '') : '';
        adminUsersList.innerHTML = adminUsers.map((user) => {
            const id = String(user.id || '');
            const selected = id && id === selectedAdminUserId ? ' selected' : '';
            const inactive = user.isActive ? '' : ' inactive';
            const selfBadge = id && id === currentUserId ? '<span class="admin-user-badge">you</span>' : '';
            const channels = formatAllowedChannels(user.allowedChannelIds || []) || 'none';
            return `
                <button type="button" class="admin-user-row${selected}${inactive}" data-user-id="${escapeHtml(id)}">
                    <span class="admin-user-main">
                        <span class="admin-user-name">${escapeHtml(user.username || id)}</span>
                        ${selfBadge}
                    </span>
                    <span class="admin-user-meta">${escapeHtml((user.roles || []).join(', ') || 'no roles')} · channels ${escapeHtml(channels)}</span>
                </button>
            `;
        }).join('');
    }

    function renderAdminSessions() {
        if (!adminSessionsList) return;
        if (!adminSessions.length) {
            adminSessionsList.innerHTML = '<div class="admin-empty">No sessions found.</div>';
            return;
        }
        adminSessionsList.innerHTML = `
            <table class="admin-session-table">
                <thead>
                    <tr>
                        <th>User</th>
                        <th>Last seen</th>
                        <th>Expires</th>
                        <th>Status</th>
                        <th></th>
                    </tr>
                </thead>
                <tbody>
                    ${adminSessions.map((session) => {
                        const revoked = Boolean(session.revokedAt);
                        const status = revoked ? `revoked${session.revokeReason ? `: ${session.revokeReason}` : ''}` : 'active';
                        const revokeButton = revoked
                            ? ''
                            : `<button type="button" class="settings-btn admin-row-btn" data-session-revoke="${escapeHtml(session.id || '')}">Revoke</button>`;
                        return `
                            <tr>
                                <td>
                                    <div class="admin-session-user">${escapeHtml(session.username || session.userId || '')}</div>
                                    <div class="admin-session-sub">${escapeHtml(session.clientIp || '')}</div>
                                </td>
                                <td>${escapeHtml(formatDateTime(session.lastSeenAt))}</td>
                                <td>${escapeHtml(formatDateTime(session.expiresAt))}</td>
                                <td><span class="admin-session-status ${revoked ? 'revoked' : 'active'}">${escapeHtml(status)}</span></td>
                                <td>${revokeButton}</td>
                            </tr>
                        `;
                    }).join('')}
                </tbody>
            </table>
        `;
    }

    async function loadAdminRoles() {
        const response = await fetch('/auth/roles', { cache: 'no-store' });
        const data = await parseApiJson(response, 'Failed to load roles');
        adminRoles = Array.isArray(data.roles) ? data.roles : [];
        const selected = selectedAdminRoles();
        renderAdminRoles(selected.length ? selected : ['viewer']);
    }

    async function loadAdminUsers() {
        const response = await fetch('/auth/users?includeInactive=true', { cache: 'no-store' });
        const data = await parseApiJson(response, 'Failed to load users');
        adminUsers = Array.isArray(data.users) ? data.users : [];
        adminUsers.sort((left, right) => String(left.username || '').localeCompare(String(right.username || '')));
        if (selectedAdminUserId && !adminUsers.some((user) => String(user.id) === selectedAdminUserId)) {
            selectedAdminUserId = null;
        }
        renderAdminUsers();
        if (selectedAdminUserId) {
            const selected = adminUsers.find((user) => String(user.id) === selectedAdminUserId);
            if (selected) setAdminEditorUser(selected);
        } else if (adminUsers.length && adminUserEditorTitle && adminUserEditorTitle.textContent !== 'New User') {
            setAdminEditorUser(adminUsers[0]);
        }
    }

    async function loadAdminSessions() {
        const activeOnly = adminSessionsActiveOnlyInput ? adminSessionsActiveOnlyInput.checked : true;
        const params = new URLSearchParams({ activeOnly: activeOnly ? 'true' : 'false' });
        const response = await fetch(`/auth/sessions?${params.toString()}`, { cache: 'no-store' });
        const data = await parseApiJson(response, 'Failed to load sessions');
        adminSessions = Array.isArray(data.sessions) ? data.sessions : [];
        renderAdminSessions();
    }

    async function loadAdminConsole() {
        if (!syncAdminUsersAccess()) return;
        if (!selectedAdminUserId) {
            setAdminEditorUser(null);
        }
        setAdminUsersStatus('Loading users...', 'loading');
        try {
            await loadAdminRoles();
            await Promise.all([loadAdminUsers(), loadAdminSessions()]);
            setAdminUsersStatus(`Loaded ${adminUsers.length} users and ${adminSessions.length} sessions.`, 'success');
        } catch (error) {
            const message = error.message || String(error);
            if (/permission|forbidden|required/i.test(message)) {
                setAdminUsersAccess(false);
            }
            setAdminUsersStatus(message, 'error');
        }
    }

    async function saveAdminUser() {
        if (!syncAdminUsersAccess()) return;
        const roles = selectedAdminRoles();
        if (!roles.length) {
            setAdminUsersStatus('Select at least one role.', 'error');
            return;
        }
        let allowedChannelIds;
        try {
            allowedChannelIds = parseAllowedChannelsText(adminAllowedChannelsInput ? adminAllowedChannelsInput.value : '');
        } catch (error) {
            setAdminUsersStatus(error.message || String(error), 'error');
            return;
        }
        const password = adminPasswordInput ? adminPasswordInput.value : '';
        const payload = {
            displayName: adminDisplayNameInput ? adminDisplayNameInput.value.trim() : '',
            roles,
            allowedChannelIds,
            isActive: adminUserActiveInput ? adminUserActiveInput.checked : true,
        };
        if (selectedAdminUserId) {
            if (password) payload.password = password;
        } else {
            payload.username = adminUsernameInput ? adminUsernameInput.value.trim() : '';
            payload.password = password;
            if (!payload.username || !payload.password) {
                setAdminUsersStatus('Username and password are required for a new user.', 'error');
                return;
            }
        }

        setButtonBusy(adminUserSaveBtn, true);
        setAdminUsersStatus('Saving user...', 'loading');
        try {
            const url = selectedAdminUserId ? `/auth/users/${encodeURIComponent(selectedAdminUserId)}` : '/auth/users';
            const response = await fetch(url, {
                method: selectedAdminUserId ? 'PATCH' : 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await parseApiJson(response, 'Failed to save user');
            const saved = data.user || null;
            if (saved && saved.id) {
                selectedAdminUserId = String(saved.id);
            }
            if (adminPasswordInput) adminPasswordInput.value = '';
            await loadAdminUsers();
            setAdminUsersStatus('User saved.', 'success');
        } catch (error) {
            setAdminUsersStatus(error.message || String(error), 'error');
        } finally {
            setButtonBusy(adminUserSaveBtn, false);
        }
    }

    async function revokeSelectedAdminUserSessions() {
        if (!selectedAdminUserId) return;
        const user = adminUsers.find((item) => String(item.id) === selectedAdminUserId);
        const label = user ? (user.username || user.id) : selectedAdminUserId;
        if (!window.confirm(`Revoke active sessions for ${label}?`)) return;
        setButtonBusy(adminUserRevokeBtn, true);
        setAdminUsersStatus('Revoking user sessions...', 'loading');
        try {
            const response = await fetch(`/auth/users/${encodeURIComponent(selectedAdminUserId)}/revoke-sessions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ reason: 'admin_ui' }),
            });
            const data = await parseApiJson(response, 'Failed to revoke sessions');
            await loadAdminSessions();
            setAdminUsersStatus(`Revoked ${data.revokedSessions || 0} sessions.`, 'success');
        } catch (error) {
            setAdminUsersStatus(error.message || String(error), 'error');
        } finally {
            setButtonBusy(adminUserRevokeBtn, false);
        }
    }

    async function revokeAdminSession(sessionId) {
        const clean = String(sessionId || '').trim();
        if (!clean) return;
        if (!window.confirm('Revoke this session?')) return;
        setAdminUsersStatus('Revoking session...', 'loading');
        try {
            const response = await fetch(`/auth/sessions/${encodeURIComponent(clean)}/revoke`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ reason: 'admin_ui' }),
            });
            await parseApiJson(response, 'Failed to revoke session');
            await loadAdminSessions();
            setAdminUsersStatus('Session revoked.', 'success');
        } catch (error) {
            setAdminUsersStatus(error.message || String(error), 'error');
        }
    }

    function auditParamsFromInputs() {
        const params = new URLSearchParams();
        const limit = auditLimitSelect ? auditLimitSelect.value : '50';
        params.set('limit', limit || '50');
        const filters = [
            ['result', auditResultFilter ? auditResultFilter.value : ''],
            ['action', auditActionFilter ? auditActionFilter.value.trim() : ''],
            ['actorUserId', auditActorFilter ? auditActorFilter.value.trim() : ''],
            ['channelId', auditChannelFilter ? auditChannelFilter.value.trim() : ''],
            ['requestId', auditRequestFilter ? auditRequestFilter.value.trim() : ''],
        ];
        filters.forEach(([name, value]) => {
            if (value) params.set(name, value);
        });
        return params;
    }

    function renderAuditEvents() {
        if (!auditEventsList) return;
        if (!auditEvents.length) {
            auditEventsList.innerHTML = '<div class="admin-empty">No audit events found.</div>';
            if (auditEventsNextBtn) auditEventsNextBtn.disabled = true;
            return;
        }
        auditEventsList.innerHTML = `
            <table class="audit-events-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Result</th>
                        <th>Action</th>
                        <th>Actor</th>
                        <th>Target</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
                    ${auditEvents.map((event) => {
                        const target = [event.targetType, event.targetId]
                            .map((value) => String(value || '').trim())
                            .filter(Boolean)
                            .join(': ');
                        const channel = event.channelId ? `channel ${event.channelId}` : '';
                        const details = JSON.stringify(event.details || {});
                        return `
                            <tr>
                                <td>
                                    <div class="audit-time">${escapeHtml(formatDateTime(event.occurredAt))}</div>
                                    <div class="admin-session-sub">${escapeHtml(event.requestId || '')}</div>
                                </td>
                                <td><span class="audit-result ${escapeHtml(event.result || '')}">${escapeHtml(event.result || '')}</span></td>
                                <td>${escapeHtml(event.action || '')}</td>
                                <td>
                                    <div class="audit-actor">${escapeHtml(event.actorUserId || 'system')}</div>
                                    <div class="admin-session-sub">${escapeHtml((event.actorRoles || []).join(', '))}</div>
                                </td>
                                <td>
                                    <div>${escapeHtml(target || 'route')}</div>
                                    <div class="admin-session-sub">${escapeHtml(channel)}</div>
                                </td>
                                <td><code class="audit-details">${escapeHtml(details)}</code></td>
                            </tr>
                        `;
                    }).join('')}
                </tbody>
            </table>
        `;
        if (auditEventsNextBtn) auditEventsNextBtn.disabled = !auditNextCursor;
    }

    async function loadAuditEvents({ append = false } = {}) {
        if (!syncAuditAccess()) return;
        const params = append && auditLastParams
            ? new URLSearchParams(auditLastParams.toString())
            : auditParamsFromInputs();
        if (append) {
            if (!auditNextCursor) return;
            params.set('cursor', auditNextCursor);
        } else {
            auditNextCursor = null;
            auditLastParams = new URLSearchParams(params.toString());
        }
        setAuditStatus(append ? 'Loading next audit page...' : 'Loading audit events...', 'loading');
        setButtonBusy(append ? auditEventsNextBtn : auditEventsRefreshBtn, true);
        try {
            const response = await fetch(`/audit/events?${params.toString()}`, { cache: 'no-store' });
            const data = await parseApiJson(response, 'Failed to load audit events');
            const rows = Array.isArray(data.events) ? data.events : [];
            auditEvents = append ? auditEvents.concat(rows) : rows;
            auditNextCursor = data.nextCursor || null;
            renderAuditEvents();
            setAuditStatus(`Loaded ${auditEvents.length} audit events.`, 'success');
        } catch (error) {
            setAuditStatus(error.message || String(error), 'error');
        } finally {
            setButtonBusy(append ? auditEventsNextBtn : auditEventsRefreshBtn, false);
            if (auditEventsNextBtn) auditEventsNextBtn.disabled = !auditNextCursor;
        }
    }

    function clampSegmentThreshold(value) {
        const numeric = Number.parseFloat(value);
        if (!Number.isFinite(numeric)) {
            return segmentThreshold;
        }
        return Math.min(0.99, Math.max(0.0, numeric));
    }

    function setSegmentThresholdFromPercent(percentValue) {
        const pct = Number.parseInt(percentValue, 10);
        const clamped = Math.min(99, Math.max(0, Number.isFinite(pct) ? pct : Math.round(segmentThreshold * 100)));
        segmentThreshold = clamped / 100;
        if (segmentThresholdSlider) {
            segmentThresholdSlider.value = String(clamped);
        }
        if (segmentThresholdValueEl) {
            segmentThresholdValueEl.textContent = `${clamped}%`;
        }
    }

    function formatPercent(value) {
        if (!Number.isFinite(value)) {
            return 'n/a';
        }
        return `${(value * 100).toFixed(1)}%`;
    }

    function buildSimilarityMetrics(result, isCommented = false) {
        if (isCommented) {
            const count = result.comment_count || 0;
            const latest = (result.latest_comment || '').toString();
            const trimmed = latest.length > 50 ? `${latest.substring(0, 50)}...` : latest;
            return `<div class="metric-line"><span class="metric-label">Comments:</span> ${count}${trimmed ? ` <span class="metric-note">Latest: ${trimmed}</span>` : ''}</div>`;
        }

        if (result && result.is_detection) {
            const probeName = String(result.probe_name || result.probe_id || result.filename || 'n/a').trim() || 'n/a';
            const ts = result.timestamp_ms ? new Date(result.timestamp_ms).toLocaleString() : 'n/a';
            const channelId = parseInt(String(result.channel_id ?? ''), 10);
            const channelName = Number.isFinite(channelId) ? getLuxriotChannelLabel(channelId) : (
                String(result.channel_name || result.channel_title || '').trim() || 'n/a'
            );
            const sev = result.severity ? escapeHtml(String(result.severity)) : 'n/a';
            const pos = Number.isFinite(result.pos_score) ? result.pos_score.toFixed(3) : 'n/a';
            const neg = Number.isFinite(result.neg_score) ? result.neg_score.toFixed(3) : 'n/a';
            const margin = Number.isFinite(result.margin) ? result.margin.toFixed(3) : 'n/a';
            const similarity = Number.isFinite(result.similarity) ? formatPercent(result.similarity) : null;
            const mode = String(result.search_mode || '').trim().toUpperCase();
            const clipSearch = Number.isFinite(result?.fusion?.clip_similarity) ? formatPercent(result.fusion.clip_similarity) : null;
            const dinoSearch = Number.isFinite(result?.fusion?.dino_similarity) ? formatPercent(result.fusion.dino_similarity) : null;
            const safeChannelName = escapeHtml(channelName);
            const safeProbeName = escapeHtml(probeName);
            const lines = [
                `<div class="metric-line metric-line-wrap"><span class="metric-label">Name:</span> <span class="metric-value metric-stream-name" title="${safeProbeName}">${safeProbeName}</span></div>`,
                `<div class="metric-line"><span class="metric-label">Time:</span> ${escapeHtml(ts)}</div>`,
                `<div class="metric-line metric-line-wrap"><span class="metric-label">Stream:</span> <span class="metric-value metric-stream-name" title="${safeChannelName}">${safeChannelName}</span></div>`,
                `<div class="metric-line"><span class="metric-label">Severity:</span> <span class="metric-value">${sev}</span></div>`,
                `<div class="metric-line"><span class="metric-label">Probe:</span> ${escapeHtml(pos)} / ${escapeHtml(neg)} / ${escapeHtml(margin)}</div>`,
            ];
            if (similarity) {
                const modeHint = mode ? ` <span class="metric-note">${escapeHtml(mode)}</span>` : '';
                lines.push(`<div class="metric-line"><span class="metric-label">Match:</span> ${escapeHtml(similarity)}${modeHint}</div>`);
            }
            if (clipSearch || dinoSearch) {
                lines.push(`<div class="metric-line"><span class="metric-label">Match C/D:</span> ${escapeHtml(clipSearch || 'n/a')} / ${escapeHtml(dinoSearch || 'n/a')}</div>`);
            }
            return lines.join('');
        }

        const lines = [];
        lines.push(`<div class="metric-line"><span class="metric-label">Final:</span> ${formatPercent(result.similarity)}</div>`);

        if (result.rerank) {
            const originalScore = formatPercent(result.rerank.original_score);
            if (Number.isFinite(result.rerank.original_score)) {
                lines.push(`<div class="metric-line"><span class="metric-label">Original:</span> ${originalScore}</div>`);
            }

            if (Number.isFinite(result.rerank.score)) {
                const rerankScore = formatPercent(result.rerank.score);
                const note = result.rerank.applied ? '' : '<span class="metric-note">fallback</span>';
                lines.push(`<div class="metric-line"><span class="metric-label">Rerank:</span> ${rerankScore}${note}</div>`);
            }
        }

        if (result.fusion) {
            if (Number.isFinite(result.fusion.clip_similarity)) {
                lines.push(`<div class="metric-line"><span class="metric-label">CLIP:</span> ${formatPercent(result.fusion.clip_similarity)}</div>`);
            }
            if (Number.isFinite(result.fusion.dino_similarity)) {
                lines.push(`<div class="metric-line"><span class="metric-label">DINO:</span> ${formatPercent(result.fusion.dino_similarity)}</div>`);
            }
            if (Number.isFinite(result.fusion.alpha)) {
                lines.push(`<div class="metric-line"><span class="metric-label">Fusion α:</span> ${result.fusion.alpha.toFixed(2)}</div>`);
            }
        }

        if (!lines.length) {
            lines.push(`<div class="metric-line"><span class="metric-label">Similarity:</span> ${formatPercent(result.similarity)}</div>`);
        }

        return lines.join('');
    }

    function buildResultBadges(result) {
        if (!result || typeof result !== 'object') return '';
        const badges = [];
        if (result.is_detection) {
            badges.push({ label: 'Detection', classes: '' });
        }

        const modeRaw = String(result.search_mode || '').trim().toLowerCase();
        if (modeRaw) {
            if (modeRaw === 'clip') {
                badges.push({ label: 'CLIP', classes: 'mode-clip' });
            } else if (modeRaw === 'fusion') {
                badges.push({ label: 'Fusion', classes: 'mode-fusion' });
            } else if (modeRaw === 'dino') {
                badges.push({ label: 'DINO', classes: 'mode-dino' });
            } else {
                badges.push({ label: modeRaw, classes: '' });
            }
        }

        const dinoFallback = Boolean(result.dino_fallback || result?.fusion?.dino_fallback);
        if (dinoFallback) {
            badges.push({ label: 'DINO fallback', classes: 'warning' });
        }

        if (!badges.length) return '';
        return `<div class="result-badges">${badges.map((badge) => {
            const cls = badge.classes ? ` result-badge ${badge.classes}` : 'result-badge';
            return `<span class="${cls}">${escapeHtml(String(badge.label || ''))}</span>`;
        }).join('')}</div>`;
    }

    function decorateDetectionSearchResults(results, modeUsed = '', modeRequested = '') {
        return (results || []).map((raw) => {
            if (!raw || typeof raw !== 'object') return raw;
            const item = { ...raw };
            if (modeUsed && !item.search_mode) {
                item.search_mode = String(modeUsed).trim().toLowerCase();
            }
            if (modeRequested && !item.mode_requested) {
                item.mode_requested = String(modeRequested).trim().toLowerCase();
            }
            if (item.dino_fallback === undefined && item.fusion && typeof item.fusion === 'object') {
                item.dino_fallback = Boolean(item.fusion.dino_fallback);
            }
            return item;
        });
    }

    function setArchiveDetectionsMeta(text, isError = false) {
        if (!archiveDetectionsMeta) return;
        archiveDetectionsMeta.textContent = text;
        archiveDetectionsMeta.style.color = isError ? '#ff8e8e' : '#9aa0ad';
    }

    function updateArchiveDetectionsNav() {
        if (archiveDetectionsPrevBtn) {
            archiveDetectionsPrevBtn.disabled = archiveDetectionsOffset <= 0;
        }
        if (archiveDetectionsNextBtn) {
            archiveDetectionsNextBtn.disabled = !archiveDetectionsHasMore;
        }
    }

    function applySelectOptions(selectEl, options, selected = '') {
        if (!selectEl) return;
        const previous = selected || selectEl.value || '';
        selectEl.innerHTML = options.map((opt) => `<option value="${escapeHtml(String(opt.value))}">${escapeHtml(String(opt.label))}</option>`).join('');
        const hasPrevious = options.some((opt) => String(opt.value) === String(previous));
        selectEl.value = hasPrevious ? String(previous) : String(options[0]?.value || '');
    }

    async function refreshArchiveChannelFilter() {
        if (!archiveChannelFilter) return;
        try {
            const response = await fetch('/luxriot/channels');
            const data = await parseApiJson(response, 'Failed to load channels');
            const channels = Array.isArray(data.channels) ? data.channels : [];
            const options = [{ value: '', label: 'All streams' }];
            channels.forEach((channel) => {
                const rawId = channel.channel_id ?? channel.id;
                const id = parseInt(String(rawId || ''), 10);
                if (!Number.isFinite(id)) return;
                const label = normalizeLuxriotChannelName(channel, id);
                luxriotChannelNameById[String(id)] = label;
                options.push({ value: String(id), label });
            });
            applySelectOptions(archiveChannelFilter, options, archiveChannelFilter.value);
        } catch (_) {
            applySelectOptions(archiveChannelFilter, [{ value: '', label: 'All streams' }], '');
        }
    }

    async function refreshArchiveProbeFilter() {
        if (!archiveProbeFilter) return;
        try {
            const params = new URLSearchParams({ hours: '168', limit: '300' });
            const channelId = archiveChannelFilter ? archiveChannelFilter.value.trim() : '';
            if (channelId) {
                params.set('channel_id', channelId);
            }
            const response = await fetch(`/detections/summary?${params.toString()}`);
            const data = await parseApiJson(response, 'Failed to load detection probes');
            const summary = Array.isArray(data.summary) ? data.summary : [];
            const options = [{ value: '', label: 'All probes' }];
            summary.forEach((item) => {
                const id = String(item.probe_id || '').trim();
                if (!id) return;
                const labelBase = item.probe_name ? String(item.probe_name) : id;
                const label = `${labelBase} (${item.hit_count || 0})`;
                options.push({ value: id, label });
            });
            applySelectOptions(archiveProbeFilter, options, archiveProbeFilter.value);
        } catch (_) {
            applySelectOptions(archiveProbeFilter, [{ value: '', label: 'All probes' }], '');
        }
    }

    async function refreshArchiveFilters() {
        archiveDetectionsOffset = 0;
        archiveDetectionsHasMore = false;
        updateArchiveDetectionsNav();
        await Promise.all([refreshArchiveChannelFilter(), refreshArchiveProbeFilter()]);
    }

    function normalizeDetectionResults(detections) {
        return (detections || []).map((det, idx) => {
            const ts = Number.isFinite(det?.timestamp_ms) ? det.timestamp_ms : null;
            const probeLabel = det?.probe_name || det?.probe_id || 'probe';
            return {
                filename: String(probeLabel),
                path: det?.image_path || det?.payload?.image_path || '',
                thumbnail: det?.thumbnail || '',
                is_detection: true,
                detection_id: det?.id,
                timestamp_ms: ts,
                channel_id: det?.channel_id,
                probe_id: det?.probe_id,
                probe_name: det?.probe_name,
                severity: det?.severity,
                pos_score: Number.isFinite(det?.pos_score) ? det.pos_score : _coerceNumeric(det?.pos_score),
                neg_score: Number.isFinite(det?.neg_score) ? det.neg_score : _coerceNumeric(det?.neg_score),
                margin: Number.isFinite(det?.margin) ? det.margin : _coerceNumeric(det?.margin),
                source: det?.source || '',
                _raw_index: idx,
            };
        });
    }

    function _coerceNumeric(value) {
        const parsed = Number.parseFloat(value);
        return Number.isFinite(parsed) ? parsed : 0;
    }

    async function loadDetectionsArchive(resetOffset = true) {
        if (!resultsContainer) return;
        if (resetOffset) {
            archiveDetectionsOffset = 0;
        }
        archiveDetectionsHasMore = false;
        updateArchiveDetectionsNav();
        const channelId = archiveChannelFilter ? archiveChannelFilter.value.trim() : '';
        const probeId = archiveProbeFilter ? archiveProbeFilter.value.trim() : '';
        const hoursRaw = archiveTimeFilter ? archiveTimeFilter.value : '24';
        const limitRaw = archiveDetectionsLimit ? archiveDetectionsLimit.value : '24';
        const params = new URLSearchParams();
        const parsedHours = Number.parseFloat(hoursRaw);
        if (Number.isFinite(parsedHours) && parsedHours > 0) {
            params.set('hours', String(parsedHours));
        } else {
            params.set('hours', '0');
        }
        if (channelId) params.set('channel_id', channelId);
        if (probeId) params.set('probe_id', probeId);
        const limit = Number.parseInt(limitRaw, 10);
        params.set('limit', String(Number.isFinite(limit) ? limit : 24));
        params.set('offset', String(Math.max(0, archiveDetectionsOffset)));

        resultsContainer.innerHTML = '<div class="loading"><div class="spinner"></div> Loading detections archive...</div>';
        setArchiveDetectionsMeta('Loading detections...');
        renderArchiveInspectorEmpty('Loading detections archive...');
        try {
            const response = await fetch(`/detections/list?${params.toString()}`);
            const data = await parseApiJson(response, 'Failed to load detections archive');
            const detections = Array.isArray(data.detections) ? data.detections : [];
            archiveDetectionsTotal = Number.isFinite(data.total) ? data.total : detections.length;
            archiveDetectionsHasMore = Boolean(data.has_more);
            const mapped = normalizeDetectionResults(detections);
            if (!mapped.length) {
                resultsContainer.innerHTML = '<div class="loading">No detections found for selected filters.</div>';
                setArchiveDetectionsMeta('No detections found for selected filters.');
                renderArchiveInspectorEmpty('No detections found for the selected filters.');
                updateArchiveDetectionsNav();
                return;
            }
            displayResults(mapped);
            const shownFrom = archiveDetectionsOffset + 1;
            const shownTo = archiveDetectionsOffset + mapped.length;
            setArchiveDetectionsMeta(`Showing detections ${shownFrom}-${shownTo} of ${archiveDetectionsTotal}.`);
            updateArchiveDetectionsNav();
        } catch (err) {
            resultsContainer.innerHTML = `<div class="loading">Error: ${escapeHtml(err.message || String(err))}</div>`;
            setArchiveDetectionsMeta(`Error loading detections: ${err.message || String(err)}`, true);
            renderArchiveInspectorEmpty(`Detections archive error: ${err.message || String(err)}`);
            archiveDetectionsHasMore = false;
            updateArchiveDetectionsNav();
        }
    }

    function buildDetectionSearchFilters() {
        const payload = {};
        const channelId = archiveChannelFilter ? archiveChannelFilter.value.trim() : '';
        const probeId = archiveProbeFilter ? archiveProbeFilter.value.trim() : '';
        const hoursRaw = archiveTimeFilter ? archiveTimeFilter.value : '24';
        if (channelId) payload.channel_id = channelId;
        if (probeId) payload.probe_id = probeId;
        const parsedHours = Number.parseFloat(hoursRaw);
        if (Number.isFinite(parsedHours)) {
            payload.hours = parsedHours;
        } else {
            payload.hours = 24;
        }
        return payload;
    }

    function isDetectionsScope() {
        return searchScopeSelect && searchScopeSelect.value === 'detections';
    }

    function updateSearchScopeUI() {
        if (!searchScopeSelect) return;
        if (isDetectionsScope()) {
            if (searchInput) {
                searchInput.placeholder = 'Describe detection scene (filtered by stream/probe/time)...';
            }
            setArchiveDetectionsMeta('Detections scope active: text/image search runs over filtered detection shards.');
        } else if (searchInput) {
            searchInput.placeholder = "Describe what you're looking for...";
        }
    }

    if (authTokenBtn) {
        authTokenBtn.addEventListener('click', async () => {
            if (AUTH_ENABLED) {
                if (!authCurrentUser) {
                    setAuthGateVisible(true);
                    return;
                }
                try {
                    await fetch('/auth/logout', { method: 'POST' });
                } finally {
                    authCurrentUser = null;
                    setAuthGateVisible(true);
                }
                return;
            }
            const existing = getAdminToken();
            const entered = window.prompt(
                'Set admin token (stored in this browser for mutating API calls). Leave empty to clear.',
                existing
            );
            if (entered === null) {
                return;
            }
            saveAdminToken(entered);
            const hasToken = !!getAdminToken();
            authTokenBtn.style.opacity = hasToken ? '1' : '0.6';
            indexStatus.textContent = hasToken ? 'Admin token saved in browser.' : 'Admin token cleared.';
            indexStatus.className = hasToken ? 'status success' : 'status warning';
        });
        authTokenBtn.style.opacity = AUTH_ENABLED || getAdminToken() ? '1' : '0.6';
    }
    
    // Settings modal functionality
    settingsNavButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
            const targetId = btn.dataset.settingsTarget;
            if (!targetId) return;
            scrollSettingsSectionIntoView(targetId);
        });
    });

    settingsBtn.addEventListener('click', () => {
        settingsModal.style.display = 'block';
        if (settingsStatus) {
            settingsStatus.textContent = '';
            settingsStatus.className = 'settings-status';
            settingsStatus.style.display = 'none';
        }
        loadSettings();
        loadEnvEditor();
        if (syncAdminUsersAccess()) {
            loadAdminConsole();
        }
        if (syncAuditAccess()) {
            loadAuditEvents();
        }
        if (settingsScrollArea) {
            settingsScrollArea.scrollTop = 0;
        }
        const firstTarget = settingsNavButtons[0]?.dataset.settingsTarget;
        if (firstTarget) {
            setActiveSettingsNav(firstTarget);
        }
    });
    
    closeSettingsBtn.addEventListener('click', () => {
        settingsModal.style.display = 'none';
    });

    if (closeImageLightboxBtn) {
        closeImageLightboxBtn.addEventListener('click', () => {
            closeImageLightbox();
        });
    }
    if (imageLightboxModal) {
        imageLightboxModal.addEventListener('click', (e) => {
            if (e.target === imageLightboxModal) {
                closeImageLightbox();
            }
        });
    }
    
    // Close modal when clicking outside
    settingsModal.addEventListener('click', (e) => {
        if (e.target === settingsModal) {
            settingsModal.style.display = 'none';
        }
    });

    document.addEventListener('keydown', (e) => {
        if (e.key !== 'Escape') return;
        if (imageLightboxModal && imageLightboxModal.style.display === 'block') {
            closeImageLightbox();
            return;
        }
        if (settingsModal && settingsModal.style.display === 'block') {
            settingsModal.style.display = 'none';
        }
    });

    if (adminUsersRefreshBtn) {
        adminUsersRefreshBtn.addEventListener('click', async () => {
            setButtonBusy(adminUsersRefreshBtn, true);
            try {
                await loadAdminConsole();
            } finally {
                setButtonBusy(adminUsersRefreshBtn, false);
            }
        });
    }
    if (adminUsersNewBtn) {
        adminUsersNewBtn.addEventListener('click', () => {
            setAdminEditorUser(null);
            setAdminUsersStatus('Creating a new user.', 'loading');
        });
    }
    if (adminUsersList) {
        adminUsersList.addEventListener('click', (event) => {
            const row = event.target.closest('[data-user-id]');
            if (!row) return;
            const userId = String(row.dataset.userId || '');
            const user = adminUsers.find((item) => String(item.id) === userId);
            if (user) {
                setAdminEditorUser(user);
                setAdminUsersStatus(`Selected ${user.username || user.id}.`, 'loading');
            }
        });
    }
    if (adminUserSaveBtn) {
        adminUserSaveBtn.addEventListener('click', () => {
            saveAdminUser();
        });
    }
    if (adminUserRevokeBtn) {
        adminUserRevokeBtn.addEventListener('click', () => {
            revokeSelectedAdminUserSessions();
        });
    }
    if (adminUserClearBtn) {
        adminUserClearBtn.addEventListener('click', () => {
            setAdminEditorUser(null);
            setAdminUsersStatus('', '');
        });
    }
    if (adminSessionsRefreshBtn) {
        adminSessionsRefreshBtn.addEventListener('click', async () => {
            setButtonBusy(adminSessionsRefreshBtn, true);
            setAdminUsersStatus('Refreshing sessions...', 'loading');
            try {
                await loadAdminSessions();
                setAdminUsersStatus(`Loaded ${adminSessions.length} sessions.`, 'success');
            } catch (error) {
                setAdminUsersStatus(error.message || String(error), 'error');
            } finally {
                setButtonBusy(adminSessionsRefreshBtn, false);
            }
        });
    }
    if (adminSessionsActiveOnlyInput) {
        adminSessionsActiveOnlyInput.addEventListener('change', () => {
            loadAdminSessions().catch((error) => {
                setAdminUsersStatus(error.message || String(error), 'error');
            });
        });
    }
    if (adminSessionsList) {
        adminSessionsList.addEventListener('click', (event) => {
            const button = event.target.closest('[data-session-revoke]');
            if (!button) return;
            revokeAdminSession(button.dataset.sessionRevoke);
        });
    }
    if (auditEventsRefreshBtn) {
        auditEventsRefreshBtn.addEventListener('click', () => {
            loadAuditEvents();
        });
    }
    if (auditEventsNextBtn) {
        auditEventsNextBtn.addEventListener('click', () => {
            loadAuditEvents({ append: true });
        });
    }
    [auditResultFilter, auditLimitSelect].forEach((control) => {
        if (!control) return;
        control.addEventListener('change', () => {
            loadAuditEvents();
        });
    });
    [auditActionFilter, auditActorFilter, auditChannelFilter, auditRequestFilter].forEach((control) => {
        if (!control) return;
        control.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                loadAuditEvents();
            }
        });
    });

    if (probeEditBtn && probeEditorModal) {
        probeEditBtn.addEventListener('click', () => {
            setProbeEditorModalVisibility(true);
        });
    }
    if (closeProbeEditorBtn && probeEditorModal) {
        closeProbeEditorBtn.addEventListener('click', () => {
            setProbeEditorModalVisibility(false);
        });
    }
    if (probeEditorCloseBtn && probeEditorModal) {
        probeEditorCloseBtn.addEventListener('click', () => {
            setProbeEditorModalVisibility(false);
        });
    }
    if (probeEditorModal) {
        probeEditorModal.addEventListener('click', (e) => {
            if (e.target === probeEditorModal) {
                setProbeEditorModalVisibility(false);
            }
        });
    }
    if (probeSnapBtn) {
        probeSnapBtn.addEventListener('click', () => {
            openProbeSnapModalFromPreview();
        });
    }
    if (closeProbeSnapBtn) {
        closeProbeSnapBtn.addEventListener('click', () => {
            setProbeSnapModalVisibility(false);
        });
    }
    if (probeSnapCloseBtn) {
        probeSnapCloseBtn.addEventListener('click', () => {
            setProbeSnapModalVisibility(false);
        });
    }
    if (probeSnapExportBtn) {
        probeSnapExportBtn.addEventListener('click', () => {
            exportProbeSnapshot();
        });
    }
    if (probeSnapUseBtn) {
        probeSnapUseBtn.addEventListener('click', () => {
            setProbeSnapshotAsImageProbe();
        });
    }
    if (probeSnapActualSizeInput) {
        probeSnapActualSizeInput.addEventListener('change', () => {
            updateProbeSnapScaleMode();
        });
    }
    if (probeSnapModal) {
        probeSnapModal.addEventListener('click', (e) => {
            if (e.target === probeSnapModal) {
                setProbeSnapModalVisibility(false);
            }
        });
    }

    // Thumbnail quality slider update
    thumbnailQualitySlider.addEventListener('input', (e) => {
        qualityValue.textContent = e.target.value;
    });

    fusionAlphaInput.addEventListener('input', () => {
        fusionAlphaValue.textContent = Number(fusionAlphaInput.value).toFixed(2);
    });

    if (segmentThresholdSlider) {
        segmentThresholdSlider.addEventListener('input', (e) => {
            setSegmentThresholdFromPercent(e.target.value);
        });
        setSegmentThresholdFromPercent(segmentThresholdSlider.value);
    }

    fusionEnabledInput.addEventListener('change', () => {
        updateFusionUI(fusionEnabledInput.checked);
    });

    rerankEnabledInput.addEventListener('change', () => {
        updateRerankUI(rerankEnabledInput.checked);
    });

    segmentsEnabledInput.addEventListener('change', () => {
        updateSegmentsUI(segmentsEnabledInput.checked);
        refreshSegmentsPanels();
    });

    async function loadEnvEditor() {
        if (!envEditorInput) return;
        try {
            const response = await fetch('/settings/env');
            const data = await response.json();
            if (data.success) {
                envEditorInput.value = String(data.envText || '');
            } else {
                showSettingsStatus('Error loading environment variables: ' + (data.error || 'Unknown error'), 'error');
            }
        } catch (error) {
            showSettingsStatus('Error loading environment variables: ' + error.message, 'error');
        }
    }

    async function saveEnvEditor() {
        if (!envEditorInput || !saveEnvBtn) return;
        setButtonBusy(saveEnvBtn, true);
        try {
            const response = await fetch('/settings/env', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    envText: envEditorInput.value || ''
                })
            });
            const data = await response.json();
            if (data.success) {
                showSettingsStatus(data.message || 'Environment variables saved.', 'success');
                await loadEnvEditor();
            } else {
                showSettingsStatus('Error saving environment variables: ' + (data.error || 'Unknown error'), 'error');
            }
        } catch (error) {
            showSettingsStatus('Error saving environment variables: ' + error.message, 'error');
        } finally {
            setButtonBusy(saveEnvBtn, false);
        }
    }

    // Load current settings
    async function loadSettings() {
        try {
            const response = await fetch('/settings');
            const data = await response.json();

            if (data.success) {
                const settings = data.settings;
                document.getElementById('host').value = settings.host;
                document.getElementById('port').value = settings.port;
                document.getElementById('debug').checked = toBool(settings.debug, false);
                embedderSelect.value = settings.embedder || 'clip';
                fusionEnabledInput.checked = toBool(settings.fusionEnabled, false);
                const parsedFusionAlpha = parseFloat(settings.fusionAlpha);
                const fusionAlpha = Number.isFinite(parsedFusionAlpha) ? parsedFusionAlpha : 0.7;
                fusionAlphaInput.value = fusionAlpha.toFixed(2);
                fusionAlphaValue.textContent = fusionAlpha.toFixed(2);
                dinoModelInput.value = settings.dinoModel || 'dinov3_vitb16';
                dinoEmbedDimInput.value = settings.dinoEmbedDim || 1280;
                dinoWeightsInput.value = settings.dinoWeightsPath || '';
                indexModeSelect.value = settings.indexMode || 'clip';
                updateFusionUI(fusionEnabledInput.checked);
                rerankEnabledInput.checked = toBool(settings.rerankEnabled, false);
                const parsedRerankTopK = parseInt(settings.rerankTopK, 10);
                rerankTopKInput.value = Number.isFinite(parsedRerankTopK) ? parsedRerankTopK : 50;
                updateRerankUI(rerankEnabledInput.checked);
                document.getElementById('clipModel').value = settings.clipModel;
                document.getElementById('minResults').value = settings.minResults;
                document.getElementById('maxResults').value = settings.maxResults;
                document.getElementById('defaultResults').value = settings.defaultResults;
                document.getElementById('batchSize').value = settings.batchSize;
                document.getElementById('thumbnailQuality').value = settings.thumbnailQuality;
                document.getElementById('qualityValue').textContent = settings.thumbnailQuality;
                document.getElementById('maxCommentLength').value = settings.maxCommentLength;
                document.getElementById('maxFileSize').value = settings.maxFileSize;
                document.getElementById('indexFolderName').value = settings.indexFolderName;
                if (luxriotBaseUrlInput) luxriotBaseUrlInput.value = settings.luxriotBaseUrl || '';
                if (luxriotUsernameInput) luxriotUsernameInput.value = settings.luxriotUsername || '';
                if (luxriotPasswordInput) luxriotPasswordInput.value = settings.luxriotPassword || '';
                if (luxriotDefaultChannelIdInput) luxriotDefaultChannelIdInput.value = settings.luxriotDefaultChannelId || '';
                if (luxriotSnapshotIntervalInput) luxriotSnapshotIntervalInput.value = settings.luxriotSnapshotInterval || 5;
                if (luxriotSnapshotMaxEdgeInput) luxriotSnapshotMaxEdgeInput.value = settings.luxriotSnapshotMaxEdge || 800;
                if (luxriotMaxBufferFramesInput) luxriotMaxBufferFramesInput.value = settings.luxriotMaxBufferFrames || 180;
                if (luxriotAutoBookmarksInput) luxriotAutoBookmarksInput.checked = toBool(settings.luxriotAutoBookmarks, false);
                if (probeBookmarkCooldownSecInput) probeBookmarkCooldownSecInput.value = settings.probeBookmarkCooldownSec ?? 8.0;
                if (probeBookmarkDedupeWindowSecInput) probeBookmarkDedupeWindowSecInput.value = settings.probeBookmarkDedupeWindowSec ?? 20.0;
                if (probeBookmarkSimHighInput) probeBookmarkSimHighInput.value = settings.probeBookmarkSimHigh ?? 0.985;
                if (probeBookmarkMarginDeltaInput) probeBookmarkMarginDeltaInput.value = settings.probeBookmarkMarginDelta ?? 0.08;
                if (probeBookmarkScoreDeltaInput) probeBookmarkScoreDeltaInput.value = settings.probeBookmarkScoreDelta ?? 0.08;
                if (probeBookmarkMaxFrameGapInput) probeBookmarkMaxFrameGapInput.value = settings.probeBookmarkMaxFrameGap ?? 8;
                if (settings.luxriotSeverityMap) {
                    if (luxriotSevInfoInput) luxriotSevInfoInput.value = settings.luxriotSeverityMap.info || 'info';
                    if (luxriotSevLowInput) luxriotSevLowInput.value = settings.luxriotSeverityMap.low || 'low';
                    if (luxriotSevNormalInput) luxriotSevNormalInput.value = settings.luxriotSeverityMap.normal || 'normal';
                    if (luxriotSevHighInput) luxriotSevHighInput.value = settings.luxriotSeverityMap.high || 'high';
                    if (luxriotSevCriticalInput) luxriotSevCriticalInput.value = settings.luxriotSeverityMap.critical || 'critical';
                }
                applyEmbedderUI(embedderSelect.value);
                segmentsEnabledInput.checked = toBool(settings.segmentsEnabled, segmentsEnabledInput.checked);
                segmentMinPatchesInput.value = settings.segmentMinPatches || 3;
                const thresholdRaw = clampSegmentThreshold(settings.segmentThreshold);
                const pctValue = Math.round(thresholdRaw * 100);
                setSegmentThresholdFromPercent(pctValue);
                updateSegmentsUI(segmentsEnabledInput.checked);
                refreshSegmentsPanels();
            } else {
                showSettingsStatus('Error loading settings: ' + data.error, 'error');
            }
        } catch (error) {
            showSettingsStatus('Error loading settings: ' + error.message, 'error');
        }
    }
    
    // Save settings
    saveSettingsBtn.addEventListener('click', async () => {
        try {
            const settings = {
                host: document.getElementById('host').value.trim(),
                port: parseInt(document.getElementById('port').value),
                debug: document.getElementById('debug').checked,
                embedder: embedderSelect.value,
                fusionEnabled: fusionEnabledInput.checked,
                fusionAlpha: parseFloat(fusionAlphaInput.value),
                rerankEnabled: rerankEnabledInput.checked,
                rerankTopK: parseInt(rerankTopKInput.value),
                segmentsEnabled: segmentsEnabledInput.checked,
                segmentMinPatches: parseInt(segmentMinPatchesInput.value),
                segmentThreshold: segmentThreshold,
                clipModel: document.getElementById('clipModel').value,
                dinoModel: dinoModelInput.value.trim(),
                dinoEmbedDim: parseInt(dinoEmbedDimInput.value),
                dinoWeightsPath: dinoWeightsInput.value.trim(),
                indexMode: indexModeSelect.value,
                minResults: parseInt(document.getElementById('minResults').value),
                maxResults: parseInt(document.getElementById('maxResults').value),
                defaultResults: parseInt(document.getElementById('defaultResults').value),
                batchSize: parseInt(document.getElementById('batchSize').value),
                thumbnailQuality: parseInt(document.getElementById('thumbnailQuality').value),
                maxCommentLength: parseInt(document.getElementById('maxCommentLength').value),
                maxFileSize: parseInt(document.getElementById('maxFileSize').value),
                indexFolderName: document.getElementById('indexFolderName').value.trim(),
                luxriotBaseUrl: luxriotBaseUrlInput.value.trim(),
                luxriotUsername: luxriotUsernameInput.value.trim(),
                luxriotPassword: luxriotPasswordInput ? luxriotPasswordInput.value : '',
                luxriotDefaultChannelId: parseInt(luxriotDefaultChannelIdInput ? luxriotDefaultChannelIdInput.value : config.LUXRIOT_DEFAULT_CHANNEL_ID),
                luxriotSnapshotInterval: parseInt(luxriotSnapshotIntervalInput ? luxriotSnapshotIntervalInput.value : config.LUXRIOT_SNAPSHOT_INTERVAL),
                luxriotSnapshotMaxEdge: parseInt(luxriotSnapshotMaxEdgeInput ? luxriotSnapshotMaxEdgeInput.value : config.LUXRIOT_SNAPSHOT_MAX_EDGE),
                luxriotMaxBufferFrames: parseInt(luxriotMaxBufferFramesInput ? luxriotMaxBufferFramesInput.value : config.LUXRIOT_MAX_BUFFER_FRAMES),
                luxriotAutoBookmarks: luxriotAutoBookmarksInput ? luxriotAutoBookmarksInput.checked : false,
                probeBookmarkCooldownSec: parseFloat(probeBookmarkCooldownSecInput ? probeBookmarkCooldownSecInput.value : '8'),
                probeBookmarkDedupeWindowSec: parseFloat(probeBookmarkDedupeWindowSecInput ? probeBookmarkDedupeWindowSecInput.value : '20'),
                probeBookmarkSimHigh: parseFloat(probeBookmarkSimHighInput ? probeBookmarkSimHighInput.value : '0.985'),
                probeBookmarkMarginDelta: parseFloat(probeBookmarkMarginDeltaInput ? probeBookmarkMarginDeltaInput.value : '0.08'),
                probeBookmarkScoreDelta: parseFloat(probeBookmarkScoreDeltaInput ? probeBookmarkScoreDeltaInput.value : '0.08'),
                probeBookmarkMaxFrameGap: parseInt(probeBookmarkMaxFrameGapInput ? probeBookmarkMaxFrameGapInput.value : '8'),
                luxriotSeverityMap: {
                    info: luxriotSevInfoInput ? (luxriotSevInfoInput.value.trim() || 'info') : 'info',
                    low: luxriotSevLowInput ? (luxriotSevLowInput.value.trim() || 'low') : 'low',
                    normal: luxriotSevNormalInput ? (luxriotSevNormalInput.value.trim() || 'normal') : 'normal',
                    high: luxriotSevHighInput ? (luxriotSevHighInput.value.trim() || 'high') : 'high',
                    critical: luxriotSevCriticalInput ? (luxriotSevCriticalInput.value.trim() || 'critical') : 'critical'
                }
            };
            
            // Basic validation
            if (!settings.host) {
                showSettingsStatus('Host cannot be empty', 'error');
                return;
            }
            
            if (settings.minResults >= settings.maxResults) {
                showSettingsStatus('Min results must be less than max results', 'error');
                return;
            }
            
            if (settings.defaultResults < settings.minResults || settings.defaultResults > settings.maxResults) {
                showSettingsStatus('Default results must be between min and max results', 'error');
                return;
            }

            if (!Number.isFinite(settings.dinoEmbedDim) || settings.dinoEmbedDim <= 0) {
                settings.dinoEmbedDim = parseInt(dinoEmbedDimInput.placeholder) || 1280;
            }

            if (!Number.isFinite(settings.fusionAlpha) || settings.fusionAlpha < 0 || settings.fusionAlpha > 1) {
                const defaultAlpha = parseFloat(fusionAlphaInput.defaultValue || '0.7');
                settings.fusionAlpha = Number.isFinite(defaultAlpha) ? defaultAlpha : 0.7;
            }

            if (!settings.fusionEnabled && settings.embedder === 'fusion') {
                settings.embedder = 'clip';
            }

            if (!Number.isFinite(settings.rerankTopK) || settings.rerankTopK < 1) {
                const defaultTopK = parseInt(rerankTopKInput.placeholder) || 50;
                settings.rerankTopK = Number.isFinite(defaultTopK) && defaultTopK > 0 ? defaultTopK : 50;
            }

            if (!Number.isFinite(settings.segmentMinPatches) || settings.segmentMinPatches < 1) {
                const defaultSegments = parseInt(segmentMinPatchesInput.placeholder) || 3;
                settings.segmentMinPatches = Number.isFinite(defaultSegments) && defaultSegments > 0 ? defaultSegments : 3;
            }

            if (!Number.isFinite(settings.probeBookmarkCooldownSec) || settings.probeBookmarkCooldownSec < 0) {
                settings.probeBookmarkCooldownSec = 8.0;
            }
            if (!Number.isFinite(settings.probeBookmarkDedupeWindowSec) || settings.probeBookmarkDedupeWindowSec < 0.5) {
                settings.probeBookmarkDedupeWindowSec = 20.0;
            }
            if (!Number.isFinite(settings.probeBookmarkSimHigh)) {
                settings.probeBookmarkSimHigh = 0.985;
            }
            settings.probeBookmarkSimHigh = Math.min(0.9999, Math.max(0.5, settings.probeBookmarkSimHigh));
            if (!Number.isFinite(settings.probeBookmarkMarginDelta) || settings.probeBookmarkMarginDelta < 0) {
                settings.probeBookmarkMarginDelta = 0.08;
            }
            if (!Number.isFinite(settings.probeBookmarkScoreDelta) || settings.probeBookmarkScoreDelta < 0) {
                settings.probeBookmarkScoreDelta = 0.08;
            }
            if (!Number.isFinite(settings.probeBookmarkMaxFrameGap) || settings.probeBookmarkMaxFrameGap < 1) {
                settings.probeBookmarkMaxFrameGap = 8;
            }

            settings.segmentThreshold = clampSegmentThreshold(settings.segmentThreshold);

            if (settings.embedder === 'dino' && !settings.dinoModel) {
                showSettingsStatus('DINO model name is required when DINO backend is selected', 'error');
                return;
            }
            
            setButtonBusy(saveSettingsBtn, true);
            
            const response = await fetch('/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            });
            
            const data = await response.json();
            
            if (data.success) {
                showSettingsStatus(data.message, 'success');
            } else {
                showSettingsStatus('Error saving settings: ' + data.error, 'error');
            }
            
        } catch (error) {
            showSettingsStatus('Error saving settings: ' + error.message, 'error');
        } finally {
            setButtonBusy(saveSettingsBtn, false);
        }
    });
    
    // Reset settings to defaults
    resetSettingsBtn.addEventListener('click', () => {
        if (confirm('Reset all settings to default values?')) {
            document.getElementById('host').value = '0.0.0.0';
            document.getElementById('port').value = '5000';
            document.getElementById('debug').checked = false;
            embedderSelect.value = 'clip';
            fusionEnabledInput.checked = false;
            fusionAlphaInput.value = '0.70';
            fusionAlphaValue.textContent = '0.70';
            rerankEnabledInput.checked = false;
            rerankTopKInput.value = '50';
            segmentsEnabledInput.checked = false;
            segmentMinPatchesInput.value = '3';
            setSegmentThresholdFromPercent(70);
            dinoModelInput.value = 'dinov3_vitb16';
            dinoEmbedDimInput.value = '1280';
            dinoWeightsInput.value = '';
            indexModeSelect.value = 'clip';
            document.getElementById('clipModel').value = 'ViT-B/32';
            document.getElementById('minResults').value = '3';
            document.getElementById('maxResults').value = '48';
            document.getElementById('defaultResults').value = '12';
            document.getElementById('batchSize').value = '32';
            document.getElementById('thumbnailQuality').value = '85';
            document.getElementById('qualityValue').textContent = '85';
            document.getElementById('maxCommentLength').value = '100';
            document.getElementById('maxFileSize').value = '50';
            document.getElementById('indexFolderName').value = '.clip_index';
            luxriotBaseUrlInput.value = 'http://luxriot-host:8080';
            luxriotUsernameInput.value = '';
            luxriotPasswordInput.value = '';
            luxriotDefaultChannelIdInput.value = '1';
            luxriotSnapshotIntervalInput.value = '5';
            luxriotSnapshotMaxEdgeInput.value = '800';
            luxriotMaxBufferFramesInput.value = '180';
            if (luxriotAutoBookmarksInput) luxriotAutoBookmarksInput.checked = false;
            if (probeBookmarkCooldownSecInput) probeBookmarkCooldownSecInput.value = '8.0';
            if (probeBookmarkDedupeWindowSecInput) probeBookmarkDedupeWindowSecInput.value = '20.0';
            if (probeBookmarkSimHighInput) probeBookmarkSimHighInput.value = '0.985';
            if (probeBookmarkMarginDeltaInput) probeBookmarkMarginDeltaInput.value = '0.08';
            if (probeBookmarkScoreDeltaInput) probeBookmarkScoreDeltaInput.value = '0.08';
            if (probeBookmarkMaxFrameGapInput) probeBookmarkMaxFrameGapInput.value = '8';
            if (luxriotSevInfoInput) luxriotSevInfoInput.value = 'info';
            if (luxriotSevLowInput) luxriotSevLowInput.value = 'low';
            if (luxriotSevNormalInput) luxriotSevNormalInput.value = 'normal';
            if (luxriotSevHighInput) luxriotSevHighInput.value = 'high';
            if (luxriotSevCriticalInput) luxriotSevCriticalInput.value = 'critical';
            updateFusionUI(false);
            updateRerankUI(false);
            updateSegmentsUI(false);
            refreshSegmentsPanels();
            applyEmbedderUI(embedderSelect.value);
        }
    });

    if (reloadEnvBtn) {
        reloadEnvBtn.addEventListener('click', () => {
            loadEnvEditor();
        });
    }

    if (saveEnvBtn) {
        saveEnvBtn.addEventListener('click', () => {
            saveEnvEditor();
        });
    }

    // Show settings status message
    function showSettingsStatus(message, type) {
        settingsStatus.textContent = message;
        settingsStatus.className = `settings-status ${type}`;
        settingsStatus.style.display = 'block';
        
        setTimeout(() => {
            settingsStatus.style.display = 'none';
        }, 5000);
    }

    function updateFusionUI(enabled) {
        fusionAlphaInput.disabled = !enabled;
        fusionAlphaValue.textContent = Number(fusionAlphaInput.value).toFixed(2);
        fusionAlphaValue.classList.toggle('disabled', !enabled);
        const fusionOption = embedderSelect.querySelector('option[value="fusion"]');
        if (fusionOption) {
            fusionOption.disabled = !enabled;
        }
        if (!enabled && embedderSelect.value === 'fusion') {
            embedderSelect.value = 'clip';
            applyEmbedderUI('clip');
        }
    }

    function updateRerankUI(enabled) {
        rerankTopKInput.disabled = !enabled;
        rerankTopKInput.classList.toggle('disabled', !enabled);
    }

    updateFusionUI(fusionEnabledInput.checked);
    updateRerankUI(rerankEnabledInput.checked);
    
    function updateSegmentsUI(enabled) {
        segmentMinPatchesInput.disabled = !enabled;
        segmentMinPatchesInput.classList.toggle('disabled', !enabled);
        updateSegmentControlsUI(enabled);
    }

    function updateSegmentControlsUI(enabled) {
        if (!segmentThresholdSlider || !segmentThresholdControl) return;
        segmentThresholdSlider.disabled = !enabled;
        segmentThresholdControl.classList.toggle('disabled', !enabled);
    }

    updateSegmentsUI(segmentsEnabledInput.checked);
    refreshSegmentsPanels();

    function applyEmbedderUI(embedder) {
        const showDino = embedder === 'dino' || embedder === 'fusion';
        const dinoRows = document.querySelectorAll('.backend-dino');
        dinoRows.forEach(row => {
            row.style.display = showDino ? 'flex' : 'none';
        });

        const clipRows = document.querySelectorAll('.backend-clip');
        clipRows.forEach(row => {
            row.style.display = embedder === 'dino' ? 'none' : 'flex';
        });

        const textSearchAvailable = embedder !== 'dino';
        searchInput.disabled = !textSearchAvailable;
        searchBtn.disabled = !textSearchAvailable;
        searchInput.placeholder = textSearchAvailable
            ? "Describe what you're looking for..."
            : 'Text search requires CLIP or Fusion backend.';
        searchBtn.title = textSearchAvailable ? '' : 'Text search is disabled when backend is DINO.';
    }

    embedderSelect.addEventListener('change', (event) => {
        applyEmbedderUI(event.target.value);
    });
    applyEmbedderUI(embedderSelect.value);
    if (luxriotRefreshChannelsBtn) {
        luxriotRefreshChannelsBtn.addEventListener('click', () => {
            fetchLuxriotChannels(true).then(syncProbeChannelSelect);
        });
    }
    if (luxriotToggleCaptureBtn) {
        luxriotToggleCaptureBtn.addEventListener('click', toggleLuxriotCapture);
    }
    if (luxriotFlushCaptureBtn) {
        luxriotFlushCaptureBtn.addEventListener('click', flushLuxriotCapture);
    }
    if (luxriotPromptSettingsBtn) {
        luxriotPromptSettingsBtn.addEventListener('click', openLuxriotPromptModal);
    }
    if (closeLuxriotPromptModalBtn) {
        closeLuxriotPromptModalBtn.addEventListener('click', closeLuxriotPromptModal);
    }
    if (luxriotPromptCloseBtn) {
        luxriotPromptCloseBtn.addEventListener('click', closeLuxriotPromptModal);
    }
    if (luxriotPromptApplyBtn) {
        luxriotPromptApplyBtn.addEventListener('click', async () => {
            try {
                await applyLuxriotPromptModal();
                closeLuxriotPromptModal();
            } catch (err) {
                setLuxriotStatus(err.message || 'Failed to save prompt settings', true);
            }
        });
    }
    if (luxriotPromptModalInput) {
        luxriotPromptModalInput.addEventListener('keydown', async (event) => {
            if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'enter') {
                event.preventDefault();
                try {
                    await applyLuxriotPromptModal();
                    closeLuxriotPromptModal();
                } catch (err) {
                    setLuxriotStatus(err.message || 'Failed to save prompt settings', true);
                }
            }
        });
    }
    luxriotPromptTabButtons.forEach((button) => {
        button.addEventListener('click', () => {
            const tab = button.dataset.luxriotPromptTab || 'stream';
            setLuxriotPromptModalTab(tab);
        });
    });
    if (luxriotPromptModal) {
        luxriotPromptModal.addEventListener('click', (event) => {
            if (event.target === luxriotPromptModal) {
                closeLuxriotPromptModal();
            }
        });
    }
    if (luxriotRefreshSummariesBtn) {
        luxriotRefreshSummariesBtn.addEventListener('click', async () => {
            setSummaryRefreshButtonState('busy');
            setLuxriotStatus('Refreshing summaries...');
            let queued = false;
            try {
                const started = await refreshLuxriotSummaryView(getSelectedSummaryChannel(), true);
                queued = started === false;
                if (queued) {
                    setSummaryRefreshButtonState('queued');
                }
            } finally {
                if (queued) {
                    setTimeout(() => setSummaryRefreshButtonState('idle'), 800);
                } else {
                    setSummaryRefreshButtonState('idle');
                }
            }
        });
    }
    if (luxriotSummaryChannelSelect) {
        luxriotSummaryChannelSelect.addEventListener('change', () => {
            luxriotSummaryChannel = getSelectedSummaryChannel();
            setSummaryBaseLevel(luxriotSummaryLevel);
            setSummaryUnread(0);
            refreshLuxriotSummaryView(luxriotSummaryChannel, true);
        });
    }
    if (luxriotSummaryRunSelect) {
        luxriotSummaryRunSelect.addEventListener('change', () => {
            applySummaryFiltersFromInputs();
            setSummaryUnread(0);
            refreshLuxriotSummaryView(getSelectedSummaryChannel(), true);
        });
    }
    if (luxriotSummaryRangeSelect) {
        luxriotSummaryRangeSelect.addEventListener('change', () => {
            luxriotSummaryRangePreset = normalizeSummaryRangePreset(luxriotSummaryRangeSelect.value);
            syncSummaryRangeUI();
            if (luxriotSummaryRangePreset === 'custom') {
                updateSummaryControlsUI();
                return;
            }
            applySummaryFiltersFromInputs();
            setSummaryUnread(0);
            refreshLuxriotSummaryView(getSelectedSummaryChannel(), true);
        });
    }
    if (luxriotSummaryLevelSelect) {
        luxriotSummaryLevelSelect.addEventListener('change', () => {
            setSummaryBaseLevel(luxriotSummaryLevelSelect.value);
            setSummaryUnread(0);
            updateSummaryControlsUI();
            refreshLuxriotSummaryView(getSelectedSummaryChannel(), true);
        });
    }
    if (luxriotSummaryApplyFiltersBtn) {
        luxriotSummaryApplyFiltersBtn.addEventListener('click', () => {
            applySummaryFiltersFromInputs();
            setSummaryUnread(0);
            refreshLuxriotSummaryView(getSelectedSummaryChannel(), true);
        });
    }
    if (luxriotSummaryFromInput) {
        luxriotSummaryFromInput.addEventListener('keydown', (event) => {
            if (event.key !== 'Enter') return;
            event.preventDefault();
            applySummaryFiltersFromInputs();
            setSummaryUnread(0);
            refreshLuxriotSummaryView(getSelectedSummaryChannel(), true);
        });
    }
    if (luxriotSummaryToInput) {
        luxriotSummaryToInput.addEventListener('keydown', (event) => {
            if (event.key !== 'Enter') return;
            event.preventDefault();
            applySummaryFiltersFromInputs();
            setSummaryUnread(0);
            refreshLuxriotSummaryView(getSelectedSummaryChannel(), true);
        });
    }
    if (luxriotSummaryBackBtn) {
        luxriotSummaryBackBtn.addEventListener('click', () => {
            const ctx = popSummaryRollupContext();
            if (!ctx) {
                updateSummaryControlsUI();
                return;
            }
            setSummaryUnread(0);
            const channelId = getSelectedSummaryChannel();
            const cached = luxriotSummaryRollupCache[channelId];
            if (isRollupViewActive() && cached) {
                renderLuxriotRollups(cached, channelId);
                return;
            }
            refreshLuxriotSummaryView(channelId, true);
        });
    }
    if (luxriotSummaryFollowBtn) {
        luxriotSummaryFollowBtn.addEventListener('click', () => {
            const enableLive = !(luxriotSummaryAutoRefresh && luxriotSummaryFollowLive);
            luxriotSummaryAutoRefresh = enableLive;
            luxriotSummaryFollowLive = enableLive;
            updateSummaryControlsUI();
            if (enableLive) {
                setSummaryUnread(0);
                scrollSummaryToLatest();
                refreshLuxriotSummaryView(getSelectedSummaryChannel(), true);
            }
        });
    }
    if (luxriotSummaryPauseBtn) {
        luxriotSummaryPauseBtn.addEventListener('click', () => {
            luxriotSummaryAutoRefresh = !luxriotSummaryAutoRefresh;
            updateSummaryControlsUI();
            if (luxriotSummaryAutoRefresh) {
                refreshLuxriotSummaryView(getSelectedSummaryChannel(), true);
            }
        });
    }
    if (luxriotSummaryViewBtn) {
        luxriotSummaryViewBtn.addEventListener('click', () => {
            setSummaryCompactMode(!luxriotSummaryCompactMode);
            updateSummaryControlsUI();
            if (isRollupViewActive()) {
                const channelId = getSelectedSummaryChannel();
                const cached = luxriotSummaryRollupCache[channelId];
                if (cached) {
                    renderLuxriotRollups(cached, channelId);
                }
            } else {
                renderLuxriotSummaries(luxriotSummaryLogCache, getSelectedSummaryChannel());
            }
        });
    }
    if (luxriotSummaryCollapseAllBtn) {
        luxriotSummaryCollapseAllBtn.addEventListener('click', () => {
            const channelId = getSelectedSummaryChannel();
            const allCollapsed = areAllSummariesCollapsed(channelId);
            collapseAllSummariesForChannel(channelId, !allCollapsed);
            if (isRollupViewActive()) {
                const cached = luxriotSummaryRollupCache[channelId];
                if (cached) {
                    renderLuxriotRollups(cached, channelId);
                }
            } else {
                renderLuxriotSummaries(luxriotSummaryLogCache, channelId);
            }
        });
    }
    if (luxriotSummaryJumpBtn) {
        luxriotSummaryJumpBtn.addEventListener('click', () => {
            if (isRollupViewActive()) return;
            luxriotSummaryFollowLive = true;
            setSummaryUnread(0);
            updateSummaryControlsUI();
            scrollSummaryToLatest();
        });
    }
    if (luxriotSummaries) {
        luxriotSummaries.addEventListener('scroll', () => {
            if (isRollupViewActive()) return;
            if (!luxriotSummaryFollowLive) return;
            if (!isSummaryNearBottom()) {
                luxriotSummaryFollowLive = false;
                updateSummaryControlsUI();
            }
        });
    }
    if (luxriotRefreshStreamsBtn) {
        luxriotRefreshStreamsBtn.addEventListener('click', () => refreshLuxriotStreams());
    }
    if (luxriotStopAllVideoBtn) {
        luxriotStopAllVideoBtn.addEventListener('click', () => stopAllLuxriotStreams('video'));
    }
    if (luxriotStopAllAnalyticsBtn) {
        luxriotStopAllAnalyticsBtn.addEventListener('click', () => stopAllLuxriotStreams('analytics'));
    }
    if (luxriotSummaries) {
        luxriotSummaries.addEventListener('click', (event) => {
            const target = event.target;
            if (!(target instanceof Element)) return;
            const rollupCollapseBtn = target.closest('[data-luxriot-rollup-collapse]');
            if (rollupCollapseBtn instanceof HTMLButtonElement) {
                const idx = parseInt(rollupCollapseBtn.dataset.luxriotRollupCollapse || '', 10);
                if (!Number.isFinite(idx) || idx < 0 || idx >= luxriotSummaryRollupRows.length) return;
                event.preventDefault();
                const row = luxriotSummaryRollupRows[idx] || {};
                const channelId = getSelectedSummaryChannel();
                const key = rollupSummaryKey(row, idx);
                const nextState = !isSummaryCollapsed(channelId, key);
                setSummaryCollapsed(channelId, key, nextState);
                const cached = luxriotSummaryRollupCache[channelId];
                if (cached) {
                    renderLuxriotRollups(cached, channelId);
                }
                return;
            }
            const rollupCopyBtn = target.closest('[data-luxriot-rollup-copy]');
            if (rollupCopyBtn instanceof HTMLButtonElement) {
                const idx = parseInt(rollupCopyBtn.dataset.luxriotRollupCopy || '', 10);
                if (!Number.isFinite(idx)) return;
                event.preventDefault();
                copyLuxriotRollupFromRow(idx, rollupCopyBtn);
                return;
            }
            const rollupExportBtn = target.closest('[data-luxriot-rollup-export]');
            if (rollupExportBtn instanceof HTMLButtonElement) {
                const idx = parseInt(rollupExportBtn.dataset.luxriotRollupExport || '', 10);
                if (!Number.isFinite(idx)) return;
                event.preventDefault();
                exportLuxriotRollupFromRow(idx);
                return;
            }
            const rollupDrillBtn = target.closest('[data-luxriot-rollup-drill]');
            if (rollupDrillBtn instanceof HTMLButtonElement) {
                const idx = parseInt(rollupDrillBtn.dataset.luxriotRollupDrill || '', 10);
                if (!Number.isFinite(idx) || idx < 0 || idx >= luxriotSummaryRollupRows.length) return;
                const row = luxriotSummaryRollupRows[idx] || {};
                const sourceLevel = String(row?.source_level || '').trim();
                const sourceIds = Array.isArray(row?.source_ids) ? row.source_ids : [];
                if (!sourceLevel || !sourceIds.length) return;
                event.preventDefault();
                pushSummaryRollupContext(sourceLevel, sourceIds, formatRollupRange(row?.window_start, row?.window_end));
                setSummaryUnread(0);
                const channelId = getSelectedSummaryChannel();
                const cached = luxriotSummaryRollupCache[channelId];
                if (cached) {
                    renderLuxriotRollups(cached, channelId);
                } else {
                    refreshLuxriotSummaryView(channelId, true);
                }
                return;
            }
            const collapseBtn = target.closest('[data-luxriot-collapse]');
            if (collapseBtn instanceof HTMLButtonElement) {
                const idx = parseInt(collapseBtn.dataset.luxriotCollapse || '', 10);
                if (!Number.isFinite(idx)) return;
                event.preventDefault();
                toggleLuxriotSummaryCollapse(idx);
                return;
            }
            const copyBtn = target.closest('[data-luxriot-copy]');
            if (copyBtn instanceof HTMLButtonElement) {
                const idx = parseInt(copyBtn.dataset.luxriotCopy || '', 10);
                if (!Number.isFinite(idx)) return;
                event.preventDefault();
                copyLuxriotSummaryFromLog(idx, copyBtn);
                return;
            }
            const exportBtn = target.closest('[data-luxriot-export]');
            if (exportBtn instanceof HTMLButtonElement) {
                const idx = parseInt(exportBtn.dataset.luxriotExport || '', 10);
                if (!Number.isFinite(idx)) return;
                event.preventDefault();
                exportLuxriotSummaryFromLog(idx);
                return;
            }
            const bookmarkBtn = target.closest('[data-luxriot-bookmark]');
            if (!(bookmarkBtn instanceof HTMLButtonElement)) return;
            const idx = parseInt(bookmarkBtn.dataset.luxriotBookmark || '', 10);
            if (!Number.isFinite(idx)) return;
            event.preventDefault();
            sendLuxriotBookmarkFromLog(idx, bookmarkBtn);
        });
    }
    if (luxriotStreams) {
        luxriotStreams.addEventListener('click', (event) => {
            const target = event.target;
            if (!(target instanceof Element)) return;
            const summaryBtn = target.closest('[data-summary-channel]');
            if (summaryBtn instanceof HTMLButtonElement) {
                const summaryChannelId = parseInt(summaryBtn.dataset.summaryChannel || '', 10);
                if (Number.isFinite(summaryChannelId)) {
                    luxriotSummaryChannel = summaryChannelId;
                    syncLuxriotSummaryChannelSelect();
                    setSummaryBaseLevel(luxriotSummaryLevel);
                    setSummaryUnread(0);
                    luxriotSummaryFollowLive = true;
                    updateSummaryControlsUI();
                    refreshLuxriotSummaryView(summaryChannelId, true);
                    if (!isRollupViewActive()) {
                        scrollSummaryToLatest();
                    }
                }
                event.preventDefault();
                return;
            }
            const button = target.closest('[data-stream-stop]');
            if (!(button instanceof HTMLButtonElement)) return;
            const channelId = parseInt(button.dataset.streamStop || '', 10);
            const streamType = (button.dataset.streamType || '').trim().toLowerCase();
            const streamAction = (button.dataset.streamAction || '').trim().toLowerCase();
            if (!Number.isFinite(channelId) || !streamType) return;
            event.preventDefault();
            if (streamType === 'analytics' && streamAction === 'resume') {
                resumeLuxriotProbeCapture(channelId);
            } else {
                stopLuxriotStream(channelId, streamType);
            }
        });
    }
    if (luxriotChannelSelect) {
        luxriotChannelSelect.addEventListener('change', () => {
            luxriotActiveChannel = getSelectedLuxriotChannel();
            syncProbeChannelSelect();
            syncLuxriotSummaryChannelSelect();
            setSummaryBaseLevel(luxriotSummaryLevel);
            updateLuxriotCaptureToggleButton(luxriotActiveChannel);
            void refreshLuxriotPromptSettings(false, luxriotActiveChannel);
            startLuxriotPreview();
            refreshLuxriotSummaryView(getSelectedSummaryChannel(), true);
            refreshLuxriotStreams();
        });
    }
    
    // -------- Monitoring / Probes --------
    function setProbeStatus(message, isError = false) {
        if (!probeStatus) return;
        probeStatus.textContent = message;
        probeStatus.classList.toggle('error', Boolean(isError));
    }

    function getSelectedProbeChannelId() {
        const parsed = parseInt(probeChannelSelect?.value || luxriotActiveChannel, 10);
        return Number.isFinite(parsed) ? parsed : luxriotActiveChannel;
    }

    function getProbeRuntimeState(channelId) {
        const state = probeChannelRuntime[channelId];
        if (state === 'running' || state === 'paused' || state === 'idle') {
            return state;
        }
        return 'idle';
    }

    function updateProbeStreamToggleButton(channelIdOverride = null) {
        if (!probeStreamToggleBtn) return;
        const channelId = Number.isFinite(channelIdOverride) ? channelIdOverride : getSelectedProbeChannelId();
        const runtimeState = getProbeRuntimeState(channelId);
        const enabled = probeEnableToggle ? probeEnableToggle.checked !== false : (runtimeState === 'running');
        probeStreamToggleBtn.textContent = enabled ? 'Stop Stream' : 'Start Stream';
        probeStreamToggleBtn.classList.toggle('primary', !enabled);
    }

    function updateProbeCaptureMeta(channelId, statusData = null) {
        const runtimeState = getProbeRuntimeState(channelId);
        const streamLabel = runtimeState === 'running' ? 'ok' : (runtimeState === 'paused' ? 'paused' : 'idle');
        let captureLabel = 'idle';
        const frameCount = Number(statusData?.frames);
        if (runtimeState === 'running') {
            captureLabel = Number.isFinite(frameCount) && frameCount > 0 ? 'ok' : 'warming';
        } else if (runtimeState === 'paused') {
            captureLabel = 'paused';
        }
        if (probeCaptureStatus) {
            probeCaptureStatus.textContent = `Stream: ${streamLabel} | Capture: ${captureLabel}`;
        }
        if (probeBufferInfo && statusData) {
            const lastTs = statusData.last_timestamp_ms ? new Date(statusData.last_timestamp_ms).toLocaleString() : 'n/a';
            probeBufferInfo.textContent = `Last snapshot: ${lastTs}`;
        }
        updateProbeStreamToggleButton(channelId);
    }

    function normalizeProbeRoiNorm(raw) {
        if (!raw || typeof raw !== 'object') return null;
        const x = Number.parseFloat(raw.x);
        const y = Number.parseFloat(raw.y);
        const w = Number.parseFloat(raw.w);
        const h = Number.parseFloat(raw.h);
        if (![x, y, w, h].every((value) => Number.isFinite(value))) return null;
        const minSide = 0.02;
        let nx = Math.min(1, Math.max(0, x));
        let ny = Math.min(1, Math.max(0, y));
        let nw = Math.min(1, Math.max(0, w));
        let nh = Math.min(1, Math.max(0, h));
        if (nw < minSide || nh < minSide) return null;
        if (nx + nw > 1) nx = Math.max(0, 1 - nw);
        if (ny + nh > 1) ny = Math.max(0, 1 - nh);
        return {
            x: Number(nx.toFixed(6)),
            y: Number(ny.toFixed(6)),
            w: Number(nw.toFixed(6)),
            h: Number(nh.toFixed(6)),
        };
    }

    function getProbePreviewGeometry() {
        if (!probePreviewViewport || !probePreviewImg) return null;
        const viewportRect = probePreviewViewport.getBoundingClientRect();
        const viewportWidth = viewportRect.width;
        const viewportHeight = viewportRect.height;
        if (!(viewportWidth > 1) || !(viewportHeight > 1)) return null;
        const naturalWidth = probePreviewImg.naturalWidth || 0;
        const naturalHeight = probePreviewImg.naturalHeight || 0;
        if (!(naturalWidth > 1) || !(naturalHeight > 1)) {
            return {
                viewportRect,
                viewportWidth,
                viewportHeight,
                imageWidth: viewportWidth,
                imageHeight: viewportHeight,
                imageOffsetX: 0,
                imageOffsetY: 0,
            };
        }
        const scale = Math.max(viewportWidth / naturalWidth, viewportHeight / naturalHeight);
        const imageWidth = naturalWidth * scale;
        const imageHeight = naturalHeight * scale;
        return {
            viewportRect,
            viewportWidth,
            viewportHeight,
            imageWidth,
            imageHeight,
            imageOffsetX: (viewportWidth - imageWidth) / 2,
            imageOffsetY: (viewportHeight - imageHeight) / 2,
        };
    }

    function viewportPointToProbeNorm(clientX, clientY) {
        const geom = getProbePreviewGeometry();
        if (!geom) return null;
        const px = clientX - geom.viewportRect.left;
        const py = clientY - geom.viewportRect.top;
        const nx = (px - geom.imageOffsetX) / geom.imageWidth;
        const ny = (py - geom.imageOffsetY) / geom.imageHeight;
        return {
            x: Math.min(1, Math.max(0, nx)),
            y: Math.min(1, Math.max(0, ny)),
        };
    }

    function probeNormToViewportRect(roiNorm) {
        const norm = normalizeProbeRoiNorm(roiNorm);
        const geom = getProbePreviewGeometry();
        if (!norm || !geom) return null;
        const left = geom.imageOffsetX + (norm.x * geom.imageWidth);
        const top = geom.imageOffsetY + (norm.y * geom.imageHeight);
        const right = left + (norm.w * geom.imageWidth);
        const bottom = top + (norm.h * geom.imageHeight);
        const clampedLeft = Math.max(0, Math.min(geom.viewportWidth, left));
        const clampedTop = Math.max(0, Math.min(geom.viewportHeight, top));
        const clampedRight = Math.max(0, Math.min(geom.viewportWidth, right));
        const clampedBottom = Math.max(0, Math.min(geom.viewportHeight, bottom));
        if (clampedRight - clampedLeft < 2 || clampedBottom - clampedTop < 2) return null;
        return {
            left: clampedLeft,
            top: clampedTop,
            width: clampedRight - clampedLeft,
            height: clampedBottom - clampedTop,
        };
    }

    function renderProbeRoiBox() {
        if (!probeRoiBox) return;
        const candidate = probeRoiDraftNorm || probeRoiNorm;
        if (!probeRoiEnabled || !candidate) {
            probeRoiBox.classList.remove('active');
            probeRoiBox.style.display = 'none';
            return;
        }
        const rect = probeNormToViewportRect(candidate);
        if (!rect) {
            probeRoiBox.classList.remove('active');
            probeRoiBox.style.display = 'none';
            return;
        }
        probeRoiBox.style.display = 'block';
        probeRoiBox.classList.add('active');
        probeRoiBox.style.left = `${rect.left}px`;
        probeRoiBox.style.top = `${rect.top}px`;
        probeRoiBox.style.width = `${rect.width}px`;
        probeRoiBox.style.height = `${rect.height}px`;
    }

    function updateProbeRoiUi() {
        const normalized = normalizeProbeRoiNorm(probeRoiNorm);
        if (probeRoiToggleBtn) {
            probeRoiToggleBtn.textContent = probeRoiEnabled ? 'ROI ON' : 'ROI OFF';
            probeRoiToggleBtn.classList.toggle('primary', probeRoiEnabled);
        }
        if (probeRoiClearBtn) {
            probeRoiClearBtn.disabled = !normalized;
        }
        if (probeRoiLayer) {
            probeRoiLayer.classList.toggle('active', probeRoiEnabled);
        }
        if (probeRoiInfo) {
            if (!probeRoiEnabled) {
                probeRoiInfo.textContent = 'Full frame matching';
            } else if (normalized) {
                const pct = (value) => `${Math.round(value * 100)}%`;
                probeRoiInfo.textContent = `ROI ${pct(normalized.w)} × ${pct(normalized.h)} @ ${pct(normalized.x)}, ${pct(normalized.y)}`;
            } else {
                probeRoiInfo.textContent = 'ROI enabled, draw on preview';
            }
        }
        renderProbeRoiBox();
    }

    function applyProbeRoiState(enabled, roiNorm) {
        const normalized = normalizeProbeRoiNorm(roiNorm);
        probeRoiEnabled = Boolean(enabled);
        probeRoiNorm = normalized;
        probeRoiDraftNorm = null;
        probeRoiDrawState = null;
        updateProbeRoiUi();
    }

    function clearProbeRoi(keepEnabled = true) {
        probeRoiNorm = null;
        probeRoiDraftNorm = null;
        probeRoiDrawState = null;
        probeRoiEnabled = Boolean(keepEnabled);
        updateProbeRoiUi();
    }

    function stopProbeRoiDraw(commit) {
        if (probeRoiLayer && probeRoiDrawState && Number.isFinite(probeRoiDrawState.pointerId)) {
            try {
                probeRoiLayer.releasePointerCapture(probeRoiDrawState.pointerId);
            } catch (_) {
                // ignore
            }
        }
        if (commit) {
            const normalized = normalizeProbeRoiNorm(probeRoiDraftNorm);
            if (probeRoiEnabled && normalized) {
                probeRoiNorm = normalized;
            }
        }
        probeRoiDraftNorm = null;
        probeRoiDrawState = null;
        updateProbeRoiUi();
    }

    function beginProbeRoiDraw(event) {
        if (!probeRoiEnabled || !probeRoiLayer) return;
        const point = viewportPointToProbeNorm(event.clientX, event.clientY);
        if (!point) return;
        event.preventDefault();
        probeRoiDrawState = {
            pointerId: event.pointerId,
            startX: point.x,
            startY: point.y,
            currentX: point.x,
            currentY: point.y,
        };
        probeRoiDraftNorm = {
            x: point.x,
            y: point.y,
            w: 0.001,
            h: 0.001,
        };
        probeRoiLayer.setPointerCapture(event.pointerId);
        renderProbeRoiBox();
    }

    function updateProbeRoiDraw(event) {
        if (!probeRoiEnabled || !probeRoiDrawState) return;
        const point = viewportPointToProbeNorm(event.clientX, event.clientY);
        if (!point) return;
        probeRoiDrawState.currentX = point.x;
        probeRoiDrawState.currentY = point.y;
        const x0 = Math.min(probeRoiDrawState.startX, probeRoiDrawState.currentX);
        const y0 = Math.min(probeRoiDrawState.startY, probeRoiDrawState.currentY);
        const x1 = Math.max(probeRoiDrawState.startX, probeRoiDrawState.currentX);
        const y1 = Math.max(probeRoiDrawState.startY, probeRoiDrawState.currentY);
        probeRoiDraftNorm = {
            x: x0,
            y: y0,
            w: x1 - x0,
            h: y1 - y0,
        };
        renderProbeRoiBox();
    }

    function setPreviewState(text, clearImage = false) {
        if (probePreviewOverlay) {
            probePreviewOverlay.style.display = text ? 'flex' : 'none';
            if (text) probePreviewOverlay.textContent = text;
        }
        if (clearImage && probePreviewImg) {
            probePreviewImg.src = '';
        }
        renderProbeRoiBox();
    }

    function stopProbePreview() {
        if (probePreviewTimer) {
            clearInterval(probePreviewTimer);
            probePreviewTimer = null;
        }
        probePreviewChannelId = null;
    }

    function startProbePreview(channelId) {
        if (!probePreviewImg) return;
        if (probePreviewTimer && probePreviewChannelId === channelId) return;
        stopProbePreview();
        if (!channelId && channelId !== 0) {
            setPreviewState('No channel', true);
            return;
        }
        const refresh = () => {
            if (probePreviewOverlay) probePreviewOverlay.textContent = 'Loading...';
            probePreviewImg.src = `/luxriot/snapshot/${channelId}?t=${Date.now()}`;
        };
        probePreviewImg.onload = () => {
            setPreviewState('');
            renderProbeRoiBox();
        };
        probePreviewImg.onerror = () => setPreviewState('Preview failed');
        probePreviewChannelId = channelId;
        refresh();
        const intervalMs = Math.max(2000, (luxriotDefaults.snapshotInterval || 5) * 1000);
        probePreviewTimer = setInterval(refresh, intervalMs);
    }

    function syncProbePreview(channelIdOverride = null) {
        const channelId = Number.isFinite(channelIdOverride) ? channelIdOverride : getSelectedProbeChannelId();
        if (!probeEditorModal || probeEditorModal.style.display !== 'block') {
            stopProbePreview();
            return;
        }
        if (!channelId && channelId !== 0) {
            stopProbePreview();
            setPreviewState('No channel', true);
            return;
        }
        const runtimeState = getProbeRuntimeState(channelId);
        const enabled = probeEnableToggle ? probeEnableToggle.checked !== false : true;
        if (enabled && runtimeState === 'running') {
            startProbePreview(channelId);
            setPreviewState('');
            return;
        }
        stopProbePreview();
        if (!enabled) {
            setPreviewState('Probe disabled');
            return;
        }
        if (runtimeState === 'paused') {
            setPreviewState('Paused');
            return;
        }
        setPreviewState('No stream');
    }

    function updateProbeSnapScaleMode() {
        if (!probeSnapPreview) return;
        const useActualSize = Boolean(probeSnapActualSizeInput?.checked);
        probeSnapPreview.classList.toggle('actual-size', useActualSize);
    }

    function _buildProbeSnapFilename(channelId, timestampMs, isRoi) {
        const dt = new Date(Number(timestampMs) || Date.now());
        const yyyy = dt.getFullYear();
        const mm = String(dt.getMonth() + 1).padStart(2, '0');
        const dd = String(dt.getDate()).padStart(2, '0');
        const hh = String(dt.getHours()).padStart(2, '0');
        const mi = String(dt.getMinutes()).padStart(2, '0');
        const ss = String(dt.getSeconds()).padStart(2, '0');
        const modeSuffix = isRoi ? '_roi' : '_full';
        return `probe_snap_ch${channelId}_${yyyy}${mm}${dd}_${hh}${mi}${ss}${modeSuffix}.jpg`;
    }

    function captureProbeSnapshotFromPreview() {
        if (!probePreviewImg || !probePreviewImg.complete) {
            throw new Error('Preview frame is not ready yet.');
        }
        const naturalWidth = probePreviewImg.naturalWidth || 0;
        const naturalHeight = probePreviewImg.naturalHeight || 0;
        if (!(naturalWidth > 1) || !(naturalHeight > 1)) {
            throw new Error('No preview frame available to capture.');
        }
        const roiNorm = probeRoiEnabled ? normalizeProbeRoiNorm(probeRoiNorm) : null;
        if (probeRoiEnabled && !roiNorm) {
            throw new Error('ROI is enabled. Draw ROI before snapping.');
        }
        const sx = roiNorm ? Math.max(0, Math.min(naturalWidth - 1, Math.round(roiNorm.x * naturalWidth))) : 0;
        const sy = roiNorm ? Math.max(0, Math.min(naturalHeight - 1, Math.round(roiNorm.y * naturalHeight))) : 0;
        const swRaw = roiNorm ? Math.round(roiNorm.w * naturalWidth) : naturalWidth;
        const shRaw = roiNorm ? Math.round(roiNorm.h * naturalHeight) : naturalHeight;
        const sw = Math.max(1, Math.min(naturalWidth - sx, swRaw));
        const sh = Math.max(1, Math.min(naturalHeight - sy, shRaw));
        if (sw < 2 || sh < 2) {
            throw new Error('Selected ROI is too small for a snapshot.');
        }
        const canvas = document.createElement('canvas');
        canvas.width = sw;
        canvas.height = sh;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            throw new Error('Failed to initialize snapshot buffer.');
        }
        ctx.drawImage(probePreviewImg, sx, sy, sw, sh, 0, 0, sw, sh);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
        const commaIdx = dataUrl.indexOf(',');
        const base64 = commaIdx >= 0 ? dataUrl.slice(commaIdx + 1) : '';
        if (!base64) {
            throw new Error('Failed to encode snapshot.');
        }
        const timestampMs = Date.now();
        const channelId = getSelectedProbeChannelId();
        return {
            dataUrl,
            base64,
            width: sw,
            height: sh,
            timestampMs,
            channelId,
            roi: Boolean(roiNorm),
            filename: _buildProbeSnapFilename(channelId, timestampMs, Boolean(roiNorm)),
        };
    }

    function openProbeSnapModalFromPreview() {
        try {
            const snap = captureProbeSnapshotFromPreview();
            probeSnapState = snap;
            if (probeSnapImg) {
                probeSnapImg.src = snap.dataUrl;
            }
            if (probeSnapMeta) {
                const mode = snap.roi ? 'ROI snapshot' : 'Full-frame snapshot';
                probeSnapMeta.textContent = `${mode} · ${snap.width}×${snap.height} · Channel #${snap.channelId}`;
            }
            if (probeSnapActualSizeInput) {
                probeSnapActualSizeInput.checked = false;
            }
            updateProbeSnapScaleMode();
            setProbeSnapModalVisibility(true);
        } catch (err) {
            setProbeStatus(err.message || 'Failed to capture snapshot.', true);
        }
    }

    function exportProbeSnapshot() {
        if (!probeSnapState?.dataUrl) {
            setProbeStatus('No snapshot to export.', true);
            return;
        }
        const anchor = document.createElement('a');
        anchor.href = probeSnapState.dataUrl;
        anchor.download = probeSnapState.filename || 'probe_snapshot.jpg';
        document.body.appendChild(anchor);
        anchor.click();
        document.body.removeChild(anchor);
        setProbeStatus('Snapshot exported.');
    }

    function setProbeSnapshotAsImageProbe() {
        if (!probeSnapState?.base64) {
            setProbeStatus('No snapshot to apply.', true);
            return;
        }
        probeImageState = {
            name: probeSnapState.filename || 'probe_snapshot.jpg',
            data: probeSnapState.base64,
        };
        applyImageThumb(probeSnapState.base64);
        updateImageProbeStatus(true);
        setProbeStatus('Snapshot set as image probe.');
        setProbeSnapModalVisibility(false);
    }

    function ensurePairsSeed() {
        if (!probePairsState || !probePairsState.length) {
            probePairsState = [
                { pos: '', neg: '' },
                { pos: '', neg: '' },
            ];
        }
    }

    function normalizeProbePairsForEditor(pairs) {
        if (!Array.isArray(pairs)) return [];
        return pairs.map((row) => ({
            pos: String(row?.pos ?? row?.positive ?? '').trim(),
            neg: String(row?.neg ?? row?.negative ?? '').trim(),
        }));
    }

    function buildProbePairsFromLists(positives, negatives) {
        const posList = Array.isArray(positives) ? positives : [];
        const negList = Array.isArray(negatives) ? negatives : [];
        const size = Math.max(posList.length, negList.length);
        const rows = [];
        for (let idx = 0; idx < size; idx += 1) {
            rows.push({
                pos: String(posList[idx] ?? '').trim(),
                neg: String(negList[idx] ?? '').trim(),
            });
        }
        return rows;
    }

    function serializeProbePairsForStorage(pairs) {
        return normalizeProbePairsForEditor(pairs)
            .filter((row) => row.pos || row.neg)
            .map((row) => ({
                positive: row.pos,
                negative: row.neg,
            }));
    }

    function renderPairs() {
        if (!probePairRows) return;
        ensurePairsSeed();
        const rows = probePairsState.map((row, idx) => {
            const canRemove = probePairsState.length > 1;
            const removeBtn = canRemove ? `<button class="feature-btn probe-remove-btn" data-remove="${idx}">×</button>` : '<div class="probe-pair-idx">–</div>';
            return `
                <div class="probe-pair-row" data-idx="${idx}">
                    <div class="probe-pair-idx">${idx + 1}.</div>
                    <input type="text" class="settings-input probe-pos" data-idx="${idx}" value="${escapeHtml(row.pos || '')}" placeholder="Positive probe ${idx + 1}">
                    <input type="text" class="settings-input probe-neg" data-idx="${idx}" value="${escapeHtml(row.neg || '')}" placeholder="Negative probe ${idx + 1}">
                    ${removeBtn}
                </div>
            `;
        }).join('');
        probePairRows.innerHTML = `
            ${rows}
            <div class="probe-pair-row probe-pair-add-row">
                <div class="probe-pair-idx">${probePairsState.length + 1}.</div>
                <button type="button" class="feature-btn probe-add-pair-btn" data-add-pair="1">Add pair</button>
                <div class="probe-add-empty"></div>
                <div class="probe-pairs-spacer">&nbsp;</div>
            </div>
        `;
    }

    function applyImageThumb(base64) {
        if (!probeImageThumb || !probeImageOverlay) return;
        if (base64) {
            probeImageThumb.src = `data:image/jpeg;base64,${base64}`;
            probeImageOverlay.style.display = 'none';
        } else {
            probeImageThumb.src = '';
            probeImageOverlay.style.display = 'flex';
        }
        if (probeImagePanel) {
            probeImagePanel.classList.toggle('no-image', !base64);
        }
        if (probeImageFileName) {
            const label = probeImageState?.name ? String(probeImageState.name) : 'No file selected';
            probeImageFileName.textContent = label;
            probeImageFileName.title = probeImageState?.name ? String(probeImageState.name) : '';
        }
    }

    function clearProbeImageSelection() {
        probeImageState = null;
        if (probeImageFile) probeImageFile.value = '';
        applyImageThumb('');
        updateImageProbeStatus(false);
    }

    function setArchiveUploadName(file) {
        if (!imageUploadName) return;
        const label = file?.name ? String(file.name) : 'No file selected';
        imageUploadName.textContent = label;
        imageUploadName.title = file?.name ? String(file.name) : '';
        imageUploadName.classList.toggle('is-hidden', !file?.name);
    }

    function setArchiveQueryPreview(file) {
        if (!queryImagePreview || !queryImageThumb) return;
        if (!file) {
            queryImageThumb.src = '';
            queryImagePreview.classList.add('is-empty');
            queryImagePreview.classList.add('is-hidden');
            if (imageQueryPanel) imageQueryPanel.classList.remove('has-image');
            return;
        }
        const reader = new FileReader();
        reader.onload = () => {
            const result = typeof reader.result === 'string' ? reader.result : '';
            if (!result) {
                queryImageThumb.src = '';
                queryImagePreview.classList.add('is-empty');
                queryImagePreview.classList.add('is-hidden');
                if (imageQueryPanel) imageQueryPanel.classList.remove('has-image');
                return;
            }
            queryImageThumb.src = result;
            queryImagePreview.classList.remove('is-empty');
            queryImagePreview.classList.remove('is-hidden');
            if (imageQueryPanel) imageQueryPanel.classList.add('has-image');
        };
        reader.onerror = () => {
            queryImageThumb.src = '';
            queryImagePreview.classList.add('is-empty');
            queryImagePreview.classList.add('is-hidden');
            if (imageQueryPanel) imageQueryPanel.classList.remove('has-image');
        };
        reader.readAsDataURL(file);
    }

    function updateImageProbeStatus(enabled) {
        const hasImage = Boolean(probeImageState?.data);
        imageProbeEnabled = Boolean(enabled && hasImage);
        if (probeImageEnableToggle) {
            probeImageEnableToggle.checked = imageProbeEnabled;
            probeImageEnableToggle.disabled = !hasImage;
        }
        if (probeImageClearBtn) {
            probeImageClearBtn.disabled = !hasImage;
        }
        if (probeImageClearRow) {
            probeImageClearRow.classList.toggle('is-hidden', !hasImage);
        }
        if (probeImageStatus) {
            const imageState = hasImage ? 'Ok' : 'Missing';
            probeImageStatus.textContent = `Probe status: ${imageProbeEnabled ? 'Enabled' : 'Disabled'}; Image: ${imageState}.`;
        }
    }

    function collectProbeForm() {
        const positives = [];
        const negatives = [];
        const normalizedRoi = normalizeProbeRoiNorm(probeRoiNorm);
        const roiActive = Boolean(probeRoiEnabled && normalizedRoi);
        ensurePairsSeed();
        const editorPairs = normalizeProbePairsForEditor(probePairsState);
        editorPairs.forEach((row) => {
            if (row.pos?.trim()) positives.push(row.pos.trim());
            if (row.neg?.trim()) negatives.push(row.neg.trim());
        });
        const channelId = getSelectedProbeChannelId();
        return {
            id: activeProbeId,
            name: (probeNameInput?.value || '').trim(),
            channel_id: Number.isFinite(channelId) ? channelId : luxriotActiveChannel,
            pairs: serializeProbePairsForStorage(editorPairs),
            positives,
            negatives,
            pos_floor: parseFloat(probePosFloorInput?.value) || 0.2,
            margin: Math.max(0, parseFloat(probeMarginInput?.value) || 0.05),
            top_k: parseInt(probeTopKInput?.value || '6', 10) || 6,
            window_sec: parseFloat(probeWindowSecInput?.value) || 300,
            fps: parseFloat(probeFpsInput?.value) || 0,
            severity: probeBookmarkSeverityInput ? probeBookmarkSeverityInput.value : 'info',
            bookmark: probeBookmarkToggle ? probeBookmarkToggle.checked : true,
            bookmark_cooldown_sec: probeBookmarkCooldownLocalInput ? (parseFloat(probeBookmarkCooldownLocalInput.value) || 0) : 8,
            bookmark_dedupe_window_sec: probeBookmarkDedupeWindowLocalInput ? (parseFloat(probeBookmarkDedupeWindowLocalInput.value) || 0.5) : 20,
            enabled: probeEnableToggle ? probeEnableToggle.checked : true,
            image_probe: {
                data: probeImageState?.data,
                name: probeImageState?.name,
                pos_floor: probeImagePosInput ? (parseFloat(probeImagePosInput.value) || 0.7) : 0.7,
                enabled: imageProbeEnabled,
            },
            roi_enabled: roiActive,
            roi_norm: roiActive ? normalizedRoi : null,
        };
    }

    function probeHitsKey(probeId = activeProbeId) {
        return probeId ? `probe:${probeId}` : 'probe:draft';
    }

    function renderProbeHitsSlice(hits) {
        if (!hits || !hits.length) {
            return '<div class="loading">No matches</div>';
        }
        return hits.map((hit) => {
            const ts = hit.timestamp_ms ? new Date(hit.timestamp_ms).toLocaleString() : 'n/a';
            return `
                <div class="probe-result">
                    ${hit.thumbnail ? `<img src="data:image/jpeg;base64,${hit.thumbnail}" alt="probe hit" />` : ''}
                    <div class="probe-result-time">${escapeHtml(ts)}</div>
                    <div class="probe-result-score">P ${(hit.pos_score || 0).toFixed(3)} · N ${(hit.neg_score || 0).toFixed(3)} · M ${(hit.margin || 0).toFixed(3)}</div>
                </div>
            `;
        }).join('');
    }

    function renderProbeHitsPage(key = probeHitsKey()) {
        const pageSize = 5;
        const allHits = probeHitsCacheByKey[key] || [];
        const total = allHits.length;
        if (!Number.isFinite(probeHitsOffsetByKey[key])) probeHitsOffsetByKey[key] = 0;
        if (probeHitsOffsetByKey[key] > Math.max(0, total - 1)) {
            probeHitsOffsetByKey[key] = 0;
        }
        const offset = probeHitsOffsetByKey[key];
        const pageSlice = allHits.slice(offset, offset + pageSize);
        if (probeResults) {
            probeResults.innerHTML = renderProbeHitsSlice(pageSlice);
        }
        lastProbeRefresh = probeHitsUpdatedByKey[key] || Date.now();
        if (probeHitsMeta) {
            const tsLabel = new Date(lastProbeRefresh).toLocaleTimeString();
            const pageIdx = total ? Math.floor(offset / pageSize) + 1 : 1;
            const pageCount = Math.max(1, Math.ceil(total / pageSize));
            const frames = probeFramesByKey[key] || 0;
            probeHitsMeta.textContent = `Frames: ${frames} · Hits: ${total} · Page: ${pageIdx}/${pageCount} · Updated: ${tsLabel}`;
        }
        if (probeDetLeftBtn) {
            probeDetLeftBtn.disabled = offset <= 0;
        }
        if (probeDetRightBtn) {
            probeDetRightBtn.disabled = offset + pageSize >= total;
        }
    }

    function renderProbeHits(hits = [], framesIndexed = 0, windowSec = null, options = {}) {
        const key = options.key || probeHitsKey();
        const replace = options.replace === true;
        const now = Date.now();
        const parsedWindow = Number.parseFloat(windowSec);
        const effectiveWindowSec = Number.isFinite(parsedWindow)
            ? parsedWindow
            : Number.parseFloat(probeWindowSecByKey[key]);
        if (Number.isFinite(effectiveWindowSec) && effectiveWindowSec > 0) {
            probeWindowSecByKey[key] = effectiveWindowSec;
        }
        const minTs = Number.isFinite(effectiveWindowSec) && effectiveWindowSec > 0
            ? now - (effectiveWindowSec * 1000)
            : null;
        const merged = new Map();
        const addHit = (hit) => {
            if (!hit) return;
            if (minTs && hit.timestamp_ms && hit.timestamp_ms < minTs) return;
            const dedupeKey = `${hit.timestamp_ms || 0}-${(hit.pos_score || 0).toFixed(3)}-${(hit.neg_score || 0).toFixed(3)}-${(hit.margin || 0).toFixed(3)}`;
            merged.set(dedupeKey, hit);
        };
        if (!replace) {
            (probeHitsCacheByKey[key] || []).forEach(addHit);
        }
        (hits || []).forEach(addHit);
        const combined = Array.from(merged.values())
            .sort((a, b) => (b.timestamp_ms || 0) - (a.timestamp_ms || 0))
            .slice(0, 50);
        probeHitsCacheByKey[key] = combined;
        probeFramesByKey[key] = Number.isFinite(framesIndexed) ? framesIndexed : (probeFramesByKey[key] || 0);
        probeHitsUpdatedByKey[key] = now;
        if (options.resetOffset !== false) {
            probeHitsOffsetByKey[key] = 0;
        }
        if (key === probeHitsKey()) {
            renderProbeHitsPage(key);
        }
    }

    function probeActionIcon(action) {
        const icons = {
            expand: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M240-240v-240h72v168h168v72H240Zm408-240v-168H480v-72h240v240h-72Z"/></svg>',
            run: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="m380-300 280-180-280-180v360Z"/></svg>',
            enable: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="m424-296 282-282-56-56-226 226-114-114-56 56 170 170Z"/></svg>',
            disable: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M520-200v-560h160v560H520Zm-240 0v-560h160v560H280Z"/></svg>',
            delete: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M280-120q-33 0-56.5-23.5T200-200v-520h-40v-80h200v-40h240v40h200v80h-40v520q0 33-23.5 56.5T680-120H280Zm400-600H280v520h400v-520ZM360-280h80v-360h-80v360Zm160 0h80v-360h-80v360Z"/></svg>',
            new: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M440-440H200v-80h240v-240h80v240h240v80H520v240h-80v-240Z"/></svg>',
        };
        return icons[action] || '';
    }

    function describeProbeBookmarkGate(rawGate, bookmarkEnabled) {
        if (bookmarkEnabled === false) {
            return { text: 'Gate: off', title: 'Bookmarks disabled for this probe' };
        }
        const gate = rawGate && typeof rawGate === 'object' ? rawGate : null;
        if (!gate) {
            return { text: 'Gate: n/a', title: 'No bookmark gate result yet' };
        }
        const reason = String(gate.reason || '').trim().toLowerCase();
        const dtMs = Number(gate.dt_ms);
        const sim = Number(gate.similarity);
        const frameGap = Number(gate.frame_gap);
        let text = 'Gate: n/a';
        if (reason === 'sent') {
            text = 'Gate: sent';
        } else if (reason === 'cooldown') {
            text = `Gate: cooldown${Number.isFinite(dtMs) ? ` (${(dtMs / 1000).toFixed(1)}s)` : ''}`;
        } else if (reason === 'similar_recent_hit') {
            text = `Gate: deduped${Number.isFinite(sim) ? ` (${(sim * 100).toFixed(1)}%)` : ''}`;
        } else if (reason === 'send_error') {
            text = 'Gate: send error';
        } else if (reason === 'bookmark_disabled') {
            text = 'Gate: off';
        } else if (reason) {
            text = `Gate: ${reason.replace(/_/g, ' ')}`;
        }
        const titleParts = [];
        if (reason) titleParts.push(`reason: ${reason}`);
        if (Number.isFinite(dtMs)) titleParts.push(`dt: ${(dtMs / 1000).toFixed(2)}s`);
        if (Number.isFinite(sim)) titleParts.push(`sim: ${sim.toFixed(4)}`);
        if (Number.isFinite(frameGap)) titleParts.push(`frame gap: ${frameGap.toFixed(2)}`);
        if (gate.error) titleParts.push(`error: ${String(gate.error)}`);
        return {
            text,
            title: titleParts.join(' · ') || 'No bookmark gate result yet',
        };
    }

    function renderMonitorProbeInspector() {
        if (!monitorProbeSummary) return;
        const probe = activeProbeId ? probeList.find((p) => String(p.id) === String(activeProbeId)) : null;
        if (!probe) {
            monitorProbeSummary.innerHTML = '<div class="studio-empty-state">Select a probe card to inspect its live state and operate it from here.</div>';
            if (monitorSelectionStatus) {
                monitorSelectionStatus.textContent = 'No probe selected';
            }
            return;
        }

        const channelId = parseInt(String(probe.channel_id || luxriotActiveChannel), 10);
        const runtimeState = Number.isFinite(channelId) ? probeChannelRuntime[channelId] : undefined;
        const status = probe.enabled === false
            ? 'disabled'
            : (runtimeState === 'running' ? 'running' : runtimeState === 'paused' ? 'paused' : 'idle');
        const pillClass = status === 'disabled'
            ? 'pill-disabled'
            : status === 'running'
                ? 'pill-running'
                : status === 'paused'
                    ? 'pill-paused'
                    : 'pill-idle';
        const last = probe.last_hit;
        const ts = last?.timestamp_ms ? new Date(last.timestamp_ms).toLocaleString() : 'No detections yet';
        const thumbSrc = last?.thumbnail || probe.image_probe?.data || '';
        const gateView = describeProbeBookmarkGate(probe.bookmark_gate, probe.bookmark !== false);
        const positiveCount = Array.isArray(probe.pairs) ? probe.pairs.filter((pair) => String(pair?.positive || '').trim()).length : 0;
        const negativeCount = Array.isArray(probe.pairs) ? probe.pairs.filter((pair) => String(pair?.negative || '').trim()).length : 0;
        const scores = `P: ${Number.isFinite(last?.pos_score) ? last.pos_score.toFixed(3) : '—'} · N: ${Number.isFinite(last?.neg_score) ? last.neg_score.toFixed(3) : '—'} · M: ${Number.isFinite(last?.margin) ? last.margin.toFixed(3) : '—'}`;

        if (monitorSelectionStatus) {
            monitorSelectionStatus.textContent = `${status.toUpperCase()} · Ch ${probe.channel_id || luxriotActiveChannel}`;
        }

        monitorProbeSummary.innerHTML = `
            <div class="monitor-probe-hero">
                <div class="monitor-probe-thumb ${thumbSrc ? '' : 'is-empty'}">
                    ${thumbSrc ? `<img src="data:image/jpeg;base64,${thumbSrc}" alt="${escapeHtml(probe.name || 'probe preview')}" />` : '<span>No preview</span>'}
                </div>
                <div class="monitor-probe-copy">
                    <div class="probe-status-pill ${pillClass}">${status}</div>
                    <div class="monitor-probe-name">${escapeHtml(probe.name || 'unnamed probe')}</div>
                    <div class="monitor-probe-meta">Channel ${escapeHtml(String(probe.channel_id || luxriotActiveChannel || 'n/a'))}</div>
                    <div class="monitor-probe-meta">Last event: ${escapeHtml(ts)}</div>
                </div>
            </div>
            <div class="monitor-probe-stats">
                <div class="monitor-probe-stat">
                    <span class="monitor-probe-stat-label">Scores</span>
                    <span class="monitor-probe-stat-value">${escapeHtml(scores)}</span>
                </div>
                <div class="monitor-probe-stat">
                    <span class="monitor-probe-stat-label">Bookmark gate</span>
                    <span class="monitor-probe-stat-value" title="${escapeHtml(gateView.title)}">${escapeHtml(gateView.text)}</span>
                </div>
                <div class="monitor-probe-stat">
                    <span class="monitor-probe-stat-label">Text pairs</span>
                    <span class="monitor-probe-stat-value">${positiveCount} positive · ${negativeCount} negative</span>
                </div>
                <div class="monitor-probe-stat">
                    <span class="monitor-probe-stat-label">Image probe</span>
                    <span class="monitor-probe-stat-value">${probe.image_probe?.enabled !== false && probe.image_probe?.data ? 'enabled' : 'off'}</span>
                </div>
            </div>
        `;
    }

    function renderProbeCards() {
        if (!probeCards) return;
        if (!probeList.length) {
            probeCards.innerHTML = `
                <div class="probe-mini-card new-probe-card">
                    <button class="probe-new-btn" data-action="new" aria-label="Create probe" title="Create probe">
                        ${probeActionIcon('new')}
                        <span>New Probe</span>
                    </button>
                </div>`;
            renderMonitorProbeInspector();
            return;
        }
        const cards = probeList.map((p) => {
            const last = p.last_hit;
            const ts = last?.timestamp_ms ? new Date(last.timestamp_ms).toLocaleTimeString() : 'n/a';
            const channelId = parseInt(String(p.channel_id || luxriotActiveChannel), 10);
            const runtimeState = Number.isFinite(channelId) ? probeChannelRuntime[channelId] : undefined;
            const status = p.enabled === false
                ? 'disabled'
                : (runtimeState === 'running' ? 'running' : runtimeState === 'paused' ? 'paused' : 'idle');
            const pillClass = status === 'disabled'
                ? 'pill-disabled'
                : status === 'running'
                    ? 'pill-running'
                    : status === 'paused'
                        ? 'pill-paused'
                        : 'pill-idle';
            const thumbSrc = last?.thumbnail || p.image_probe?.data || '';
            const toggleAction = status === 'disabled' ? 'enable' : 'disable';
            const toggleTitle = status === 'disabled' ? 'Enable probe' : 'Disable probe';
            const scores = `P: ${Number.isFinite(last?.pos_score) ? last.pos_score.toFixed(3) : '—'} · N: ${Number.isFinite(last?.neg_score) ? last.neg_score.toFixed(3) : '—'} · M: ${Number.isFinite(last?.margin) ? last.margin.toFixed(3) : '—'}`;
            const gateView = describeProbeBookmarkGate(p.bookmark_gate, p.bookmark !== false);
            return `
                <div class="probe-mini-card ${activeProbeId === p.id ? 'active' : ''}" data-probe-id="${p.id}">
                    <div class="probe-mini-thumb ${thumbSrc ? '' : 'is-empty'}">
                        ${thumbSrc ? `<img src="data:image/jpeg;base64,${thumbSrc}" alt="${escapeHtml(p.name || 'probe preview')}" />` : ''}
                        <div class="probe-mini-overlay">
                            <div class="probe-mini-top">
                                <div class="probe-status-pill ${pillClass}">${status}</div>
                                <div class="probe-mini-actions">
                                    <button class="probe-action-btn" data-action="expand" data-id="${p.id}" title="Open probe" aria-label="Open probe">${probeActionIcon('expand')}</button>
                                    <button class="probe-action-btn" data-action="run" data-id="${p.id}" title="Run probe" aria-label="Run probe">${probeActionIcon('run')}</button>
                                    <button class="probe-action-btn" data-action="${toggleAction}" data-id="${p.id}" title="${toggleTitle}" aria-label="${toggleTitle}">${probeActionIcon(toggleAction)}</button>
                                    <button class="probe-action-btn delete" data-action="delete" data-id="${p.id}" title="Delete probe" aria-label="Delete probe">${probeActionIcon('delete')}</button>
                                </div>
                            </div>
                            <div class="probe-mini-bottom">
                                <div class="probe-mini-name">${escapeHtml(p.name || 'unnamed')}</div>
                                <div class="probe-mini-meta">Ch ${p.channel_id || luxriotActiveChannel} · Last ${last ? ts : 'n/a'}</div>
                                <div class="probe-mini-score">${scores}</div>
                                <div class="probe-mini-gate" title="${escapeHtml(gateView.title)}">${escapeHtml(gateView.text)}</div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        cards.push(`
            <div class="probe-mini-card new-probe-card">
                <button class="probe-new-btn" data-action="new" aria-label="Create probe" title="Create probe">
                    ${probeActionIcon('new')}
                    <span>New Probe</span>
                </button>
            </div>
        `);
        probeCards.innerHTML = cards.join('');
        renderMonitorProbeInspector();
    }

    function setActiveProbe(probe) {
        activeProbeId = probe && probe.id ? probe.id : null;
        if (probeNameInput) probeNameInput.value = (probe && probe.name) || '';
        if (probeChannelSelect && probe && probe.channel_id) {
            probeChannelSelect.value = probe.channel_id;
            syncProbePreview(probe.channel_id);
        }
        if (probePosFloorInput) probePosFloorInput.value = probe?.pos_floor ?? 0.2;
        if (probeMarginInput) probeMarginInput.value = probe?.margin ?? 0.05;
        if (probeFpsInput) probeFpsInput.value = probe?.fps ?? 0;
        if (probeWindowSecInput) probeWindowSecInput.value = probe?.window_sec ?? 300;
        if (probeBookmarkSeverityInput) probeBookmarkSeverityInput.value = probe?.severity || 'info';
        if (probeBookmarkToggle) probeBookmarkToggle.checked = probe?.bookmark !== false;
        if (probeBookmarkCooldownLocalInput) probeBookmarkCooldownLocalInput.value = probe?.bookmark_cooldown_sec ?? 8;
        if (probeBookmarkDedupeWindowLocalInput) probeBookmarkDedupeWindowLocalInput.value = probe?.bookmark_dedupe_window_sec ?? 20;
        if (probeEnableToggle) probeEnableToggle.checked = probe?.enabled !== false;
        const normalizedPairs = normalizeProbePairsForEditor(probe?.pairs);
        probePairsState = normalizedPairs.length
            ? normalizedPairs
            : (probe ? buildProbePairsFromLists(probe?.positives, probe?.negatives) : probePairsState);
        if (probe?.image_probe?.data) {
            probeImageState = { data: probe.image_probe.data, name: probe.image_probe.name };
            applyImageThumb(probe.image_probe.data);
            if (probeImagePosInput) probeImagePosInput.value = probe.image_probe.pos_floor || 0.7;
            const enabled = probe.image_probe.enabled !== false;
            updateImageProbeStatus(enabled);
        } else {
            clearProbeImageSelection();
        }
        const legacyRoi = probe && probe.roi && typeof probe.roi === 'object' ? probe.roi : null;
        const savedRoiNorm = probe?.roi_norm || (legacyRoi ? (legacyRoi.norm || legacyRoi) : null);
        const hasSavedRoiNorm = Boolean(normalizeProbeRoiNorm(savedRoiNorm));
        const savedRoiEnabled = (probe?.roi_enabled === true)
            || (legacyRoi && legacyRoi.enabled === true)
            || (probe?.roi_enabled == null && (!legacyRoi || legacyRoi.enabled == null) && hasSavedRoiNorm);
        applyProbeRoiState(savedRoiEnabled, savedRoiNorm);
        renderPairs();
        const initialHits = Array.isArray(probe?.recent_hits) && probe.recent_hits.length
            ? probe.recent_hits
            : (probe?.last_hit ? [probe.last_hit] : []);
        const key = probeHitsKey(activeProbeId);
        renderProbeHits(
            initialHits,
            initialHits.length || (probe?.last_hit ? 1 : 0),
            probe?.window_sec ?? null,
            { key, replace: true, resetOffset: true }
        );
        updateProbeCaptureMeta(getSelectedProbeChannelId());
        renderProbeCards();
        setProbeStatus(activeProbeId ? `Editing: ${probe?.name || probe?.id}` : 'New probe');
    }

    function updateRunButton(running) {
        if (!probeRunBtn) return;
        probeRunBtn.textContent = running ? 'Stop probe' : 'Run probe';
        probeRunBtn.classList.toggle('primary', running);
    }

    async function persistProbeEnabled(enabled) {
        if (!activeProbeId) return;
        const payload = collectProbeForm();
        payload.id = activeProbeId;
        payload.enabled = enabled;
        try {
            const resp = await fetch('/probes/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await resp.json();
            if (!resp.ok || data.error) throw new Error(data.error || 'Save failed');
            await loadProbeList();
            if (probeEnableToggle) probeEnableToggle.checked = enabled;
        } catch (err) {
            setProbeStatus(err.message, true);
        }
    }

    function stopProbeRunLoop(message) {
        if (probeRunTimer) {
            clearInterval(probeRunTimer);
            probeRunTimer = null;
        }
        updateRunButton(false);
        if (message) setProbeStatus(message);
    }

    async function loadProbeList(showStatus = false) {
        try {
            const resp = await fetch(`/probes/list?t=${Date.now()}`, { cache: 'no-store' });
            const data = await resp.json();
            probeList = data.probes || [];
            probeCatalog = Array.isArray(probeList) ? [...probeList] : [];
            await refreshProbeRuntimeState(false);
            if (showStatus) setProbeStatus(`Loaded ${probeList.length} probes`);
            const match = activeProbeId ? probeList.find(p => p.id === activeProbeId) : null;
            if (match) {
                setActiveProbe(match);
            } else if (!activeProbeId && probeList.length) {
                setActiveProbe(probeList[0]);
            } else {
                renderProbeHits([], 0, null, { key: probeHitsKey(activeProbeId), replace: true, resetOffset: true });
                renderProbeCards();
            }
        } catch (err) {
            setProbeStatus('Failed to load probes: ' + err.message, true);
        }
    }

    async function ensureProbeCapture(channelId, quiet = false, options = null) {
        if (!channelId && channelId !== 0) return false;
        const forceStart = Boolean(options && options.forceStart);
        if (!forceStart && probeCaptureManualStop[channelId]) {
            probeChannelRuntime[channelId] = 'idle';
            updateProbeCaptureMeta(channelId);
            syncProbePreview(channelId);
            if (!quiet) {
                setProbeStatus('Stream stopped. Press Start Stream to resume.');
            }
            return false;
        }
        await refreshProbeRuntimeState(false);
        const runtimeState = probeChannelRuntime[channelId];
        if (runtimeState === 'running') {
            probeCaptureState[channelId] = true;
            delete probeCaptureManualStop[channelId];
            updateProbeCaptureMeta(channelId);
            setPreviewState('');
            syncProbePreview(channelId);
            if (!quiet) {
                await refreshProbeStatus(channelId);
            }
            return true;
        }
        try {
            channelCaptureConfig[channelId] = {
                fps: parseFloat(probeFpsInput?.value) || 0,
                windowSec: parseFloat(probeWindowSecInput?.value) || 300,
            };
            const resp = await fetch('/probes/start_capture', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    channel_id: channelId,
                    fps: channelCaptureConfig[channelId].fps,
                    clear_pause: true,
                })
            });
            const data = await resp.json();
            if (!resp.ok || data.error) throw new Error(data.error || 'Failed to start capture');
            probeCaptureState[channelId] = true;
            probeChannelRuntime[channelId] = 'running';
            delete probeCaptureManualStop[channelId];
            renderProbeCards();
            updateProbeCaptureMeta(channelId);
            setPreviewState('');
            syncProbePreview(channelId);
            if (!quiet) {
                await refreshProbeStatus(channelId);
            }
            return true;
        } catch (err) {
            if (probeCaptureStatus) probeCaptureStatus.textContent = 'Stream: error | Capture: error';
            if (!quiet) setProbeStatus(err.message, true);
            updateProbeStreamToggleButton(channelId);
            return false;
        }
    }

    async function stopProbeCapture(channelId, reason = 'stopped') {
        if (!channelId && channelId !== 0) return;
        try {
            const resp = await fetch('/probes/stop_capture', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    channel_id: channelId,
                    pause: reason === 'paused',
                })
            });
            const data = await resp.json();
            if (!resp.ok || data.error) throw new Error(data.error || 'Failed to stop capture');
            delete probeCaptureState[channelId];
            if (reason === 'paused') {
                probeChannelRuntime[channelId] = 'paused';
                delete probeCaptureManualStop[channelId];
                setPreviewState('Paused');
            } else {
                probeChannelRuntime[channelId] = 'idle';
                probeCaptureManualStop[channelId] = true;
                setPreviewState('No stream');
            }
            renderProbeCards();
            stopProbePreview();
            syncProbePreview(channelId);
            updateProbeCaptureMeta(channelId);
            await refreshProbeStatus(channelId);
        } catch (err) {
            if (probeCaptureStatus) probeCaptureStatus.textContent = 'Stream: error | Capture: error';
            setProbeStatus(err.message, true);
            updateProbeStreamToggleButton(channelId);
        }
    }

    async function refreshProbeStatus(channelIdOverride) {
        const channelId = channelIdOverride || getSelectedProbeChannelId();
        try {
            const resp = await fetch(`/probes/status?channel_id=${channelId}`);
            const data = await resp.json();
            if (data.error) {
                setProbeStatus(data.error, true);
                return;
            }
            const range = data.time_range_ms && data.time_range_ms.length === 2
                ? `${new Date(data.time_range_ms[0]).toLocaleTimeString()} - ${new Date(data.time_range_ms[1]).toLocaleTimeString()}`
                : 'n/a';
            setProbeStatus(`Frames: ${data.frames || 0} · Range: ${range}`);
            updateProbeCaptureMeta(channelId, data);
        } catch (err) {
            setProbeStatus('Status error: ' + err.message, true);
        }
    }

    async function saveActiveProbe() {
        const payload = collectProbeForm();
        const hasPos = payload.positives.length > 0 || (payload.image_probe?.enabled && payload.image_probe?.data);
        if (!hasPos) {
            setProbeStatus('Add a text positive or enable an image probe.', true);
            return;
        }
        setProbeStatus('Saving...');
        try {
            const resp = await fetch('/probes/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await resp.json();
            if (!resp.ok || data.error) {
                throw new Error(data.error || 'Save failed');
            }
            const saved = data.probe;
            activeProbeId = saved.id || activeProbeId;
            setProbeStatus(`Saved probe ${saved.name || saved.id}`);
            await loadProbeList();
            if (saved?.enabled !== false) {
                await ensureProbeCapture(saved.channel_id || payload.channel_id, true);
            } else {
                syncProbePreview(saved.channel_id || payload.channel_id);
            }
        } catch (err) {
            setProbeStatus(err.message, true);
        }
        return activeProbeId;
    }

    async function runActiveProbe(quiet = false) {
        if (probeRunInFlight) return;
        const payload = collectProbeForm();
        const hasPos = payload.positives.length > 0 || (payload.image_probe?.enabled && payload.image_probe?.data);
        if (!hasPos) {
            setProbeStatus('Add a text positive or enable an image probe.', true);
            if (probeRunTimer) stopProbeRunLoop();
            return;
        }
        const channelId = payload.channel_id;
        const captureReady = await ensureProbeCapture(channelId, true);
        if (!captureReady) {
            if (!quiet) setProbeStatus('Stream stopped. Press Start Stream to resume.');
            return;
        }
        if (!quiet) setProbeStatus('Running...');
        probeRunInFlight = true;
        try {
            let resp;
            if (activeProbeId) {
                resp = await fetch('/probes/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ id: activeProbeId })
                });
            } else {
                resp = await fetch('/probes/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
            }
            const data = await resp.json();
            if (!resp.ok || data.error) throw new Error(data.error || 'Probe failed');
            const hits = data.results || [];
            const framesCount = data.frames_indexed || data.status?.frames || 0;
            const persistedCount = Number.isFinite(data.persisted_hits) ? data.persisted_hits : hits.length;
            renderProbeHits(hits, framesCount, payload.window_sec);
            if (data.probe) {
                activeProbeId = data.probe.id || activeProbeId;
                await loadProbeList();
            } else {
                renderProbeCards();
            }
            if (!quiet) setProbeStatus(`Hits: ${hits.length} · Stored: ${persistedCount} · Frames: ${framesCount}`);
        } catch (err) {
            renderProbeHits([], 0);
            setProbeStatus(err.message, true);
        } finally {
            probeRunInFlight = false;
        }
    }

    function startProbeRunLoop(quiet = false) {
        stopProbeRunLoop();
        updateRunButton(true);
        runActiveProbe(quiet);
        const windowSec = parseFloat(probeWindowSecInput?.value) || 30;
        const intervalMs = Math.max(2000, Math.min(10000, (windowSec * 1000) / 2));
        probeRunTimer = setInterval(() => runActiveProbe(true), intervalMs);
        persistProbeEnabled(true);
    }

    function startProbeStatusPoll() {
        if (probeStatusTimer) return;
        refreshProbeStatus();
        void refreshProbeRuntimeState(true);
        probeStatusTimer = setInterval(() => {
            refreshProbeStatus();
            void refreshProbeRuntimeState(true);
        }, 8000);
    }

    function stopProbeStatusPoll() {
        if (probeStatusTimer) {
            clearInterval(probeStatusTimer);
            probeStatusTimer = null;
        }
    }

    async function deleteProbe(id) {
        if (!id) {
            setProbeStatus('No probe selected', true);
            return;
        }
        try {
            const resp = await fetch('/probes/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id })
            });
            const data = await resp.json();
            if (!resp.ok || data.error) throw new Error(data.error || 'Delete failed');
            if (activeProbeId === id) activeProbeId = null;
            setProbeStatus('Probe deleted');
            await loadProbeList(true);
            stopProbeRunLoop();
        } catch (err) {
            setProbeStatus(err.message, true);
        }
    }

    function resetProbeDraftEditor() {
        activeProbeId = null;
        probePairsState = [];
        clearProbeRoi(false);
        clearProbeImageSelection();
        renderPairs();
        renderProbeHits([], 0, null, { key: probeHitsKey(null), replace: true, resetOffset: true });
        if (probeNameInput) probeNameInput.value = '';
        if (probeEnableToggle) probeEnableToggle.checked = true;
        if (probeBookmarkToggle) probeBookmarkToggle.checked = true;
        if (probeBookmarkSeverityInput) probeBookmarkSeverityInput.value = 'info';
        if (probeBookmarkCooldownLocalInput) probeBookmarkCooldownLocalInput.value = '8';
        if (probeBookmarkDedupeWindowLocalInput) probeBookmarkDedupeWindowLocalInput.value = '20';
        if (probePosFloorInput) probePosFloorInput.value = '0.2';
        if (probeMarginInput) probeMarginInput.value = '0.05';
        if (probeFpsInput) probeFpsInput.value = '0';
        if (probeWindowSecInput) probeWindowSecInput.value = '300';
        updateProbeCaptureMeta(getSelectedProbeChannelId());
        renderMonitorProbeInspector();
    }

    function handleProbeCardClick(event) {
        const btn = event.target.closest('button[data-action]');
        if (!btn) {
            const card = event.target.closest('.probe-mini-card[data-probe-id]');
            if (!card) return;
            const probe = probeList.find((p) => String(p.id) === String(card.dataset.probeId || ''));
            if (probe) {
                setActiveProbe(probe);
            }
            return;
        }
        const id = btn.getAttribute('data-id');
        const action = btn.getAttribute('data-action');
        const probe = probeList.find(p => String(p.id) === String(id));
        if (!action) return;
        if (action === 'expand' && probe) {
            setActiveProbe(probe);
            if (probeEditorModal) {
                setProbeEditorModalVisibility(true);
            }
        } else if (action === 'run' && probe) {
            setActiveProbe(probe);
            startProbeRunLoop();
        } else if (action === 'enable' && probe) {
            setActiveProbe(probe);
            persistProbeEnabled(true);
            ensureProbeCapture(probe.channel_id || luxriotActiveChannel, true);
        } else if (action === 'delete') {
            deleteProbe(id);
        } else if (action === 'disable' && probe) {
            setActiveProbe(probe);
            persistProbeEnabled(false);
            stopProbeRunLoop();
        } else if (action === 'new') {
            resetProbeDraftEditor();
            setProbeStatus('New probe');
            if (probeEditorModal) {
                setProbeEditorModalVisibility(true);
            }
        }
    }

    if (probeRunBtn) {
        probeRunBtn.addEventListener('click', () => {
            if (probeRunTimer) {
                stopProbeRunLoop('Stopped probe loop');
                persistProbeEnabled(false);
            } else {
                if (!activeProbeId) {
                    saveActiveProbe().then(() => startProbeRunLoop());
                } else {
                    startProbeRunLoop();
                }
            }
        });
    }
    if (probeSaveBtn) {
        probeSaveBtn.addEventListener('click', async () => {
            const savedId = await saveActiveProbe();
            if (savedId && probeEditorModal) {
                setProbeEditorModalVisibility(false);
            }
        });
    }
    if (probeDeleteBtn) probeDeleteBtn.addEventListener('click', () => {
        if (activeProbeId) deleteProbe(activeProbeId);
        else {
            resetProbeDraftEditor();
            setProbeStatus('Cleared unsaved probe');
        }
    });
    if (probeStreamToggleBtn) {
        probeStreamToggleBtn.addEventListener('click', () => {
            if (probeEnableToggle) {
                probeEnableToggle.checked = !probeEnableToggle.checked;
                probeEnableToggle.dispatchEvent(new Event('change', { bubbles: true }));
                return;
            }
            const channelId = getSelectedProbeChannelId();
            ensureProbeCapture(channelId, false, { forceStart: true });
        });
    }
    if (probeChannelSelect) {
        probeChannelSelect.addEventListener('change', () => {
            const cid = getSelectedProbeChannelId();
            syncProbePreview(cid);
            updateProbeCaptureMeta(cid);
            refreshProbeStatus(cid);
        });
    }
    if (probeRoiToggleBtn) {
        probeRoiToggleBtn.addEventListener('click', () => {
            probeRoiEnabled = !probeRoiEnabled;
            if (!probeRoiEnabled) {
                probeRoiDraftNorm = null;
                probeRoiDrawState = null;
            }
            updateProbeRoiUi();
        });
    }
    if (probeRoiClearBtn) {
        probeRoiClearBtn.addEventListener('click', () => {
            clearProbeRoi(true);
        });
    }
    if (probeRoiLayer) {
        probeRoiLayer.addEventListener('pointerdown', (event) => {
            beginProbeRoiDraw(event);
        });
        probeRoiLayer.addEventListener('pointermove', (event) => {
            updateProbeRoiDraw(event);
        });
        probeRoiLayer.addEventListener('pointerup', () => {
            stopProbeRoiDraw(true);
        });
        probeRoiLayer.addEventListener('pointercancel', () => {
            stopProbeRoiDraw(false);
        });
    }
    window.addEventListener('resize', () => {
        renderProbeRoiBox();
    });
    if (probeCards) {
        probeCards.addEventListener('click', handleProbeCardClick);
    }
    if (probeNewBtn) {
        probeNewBtn.addEventListener('click', () => {
            resetProbeDraftEditor();
            setProbeStatus('New probe');
            if (probeEditorModal) {
                setProbeEditorModalVisibility(true);
            }
        });
    }
    if (probeReloadBtn) {
        probeReloadBtn.addEventListener('click', () => loadProbeList(true));
    }
    if (imageUpload) {
        imageUpload.addEventListener('change', () => {
            const file = imageUpload.files && imageUpload.files[0];
            setArchiveUploadName(file || null);
            setArchiveQueryPreview(file || null);
        });
    }
    if (probeImageFile) {
        probeImageFile.addEventListener('change', () => {
            const file = probeImageFile.files && probeImageFile.files[0];
            if (!file) {
                clearProbeImageSelection();
                return;
            }
            const reader = new FileReader();
            reader.onload = () => {
                const base64 = reader.result.split(',')[1];
                probeImageState = { name: file.name, data: base64 };
                applyImageThumb(base64);
                updateImageProbeStatus(imageProbeEnabled);
            };
            reader.readAsDataURL(file);
        });
    }
    if (probeImageClearBtn) {
        probeImageClearBtn.addEventListener('click', () => {
            clearProbeImageSelection();
        });
    }
    if (probePairsContainer) {
        probePairsContainer.addEventListener('input', (e) => {
            const target = e.target;
            const idx = parseInt(target.getAttribute('data-idx') || '-1', 10);
            if (!Number.isFinite(idx) || idx < 0 || idx >= probePairsState.length) return;
            if (target.classList.contains('probe-pos')) {
                probePairsState[idx].pos = target.value;
            } else if (target.classList.contains('probe-neg')) {
                probePairsState[idx].neg = target.value;
            }
        });
        probePairsContainer.addEventListener('click', (e) => {
            const addBtn = e.target.closest('button[data-add-pair]');
            if (addBtn) {
                probePairsState.push({ pos: '', neg: '' });
                renderPairs();
                return;
            }
            const btn = e.target.closest('button[data-remove]');
            if (!btn) return;
            const idx = parseInt(btn.getAttribute('data-remove') || '-1', 10);
            if (!Number.isFinite(idx) || idx < 0 || probePairsState.length <= 1) return;
            probePairsState.splice(idx, 1);
            renderPairs();
        });
    }
    if (probeDetLeftBtn && probeResults) {
        probeDetLeftBtn.addEventListener('click', () => {
            const key = probeHitsKey();
            const allHits = probeHitsCacheByKey[key] || [];
            if (!allHits.length) return;
            const pageSize = 5;
            const currentOffset = Number.isFinite(probeHitsOffsetByKey[key]) ? probeHitsOffsetByKey[key] : 0;
            probeHitsOffsetByKey[key] = Math.max(0, currentOffset - pageSize);
            renderProbeHitsPage(key);
        });
    }
    if (probeDetRightBtn && probeResults) {
        probeDetRightBtn.addEventListener('click', () => {
            const key = probeHitsKey();
            const allHits = probeHitsCacheByKey[key] || [];
            if (!allHits.length) return;
            const pageSize = 5;
            const currentOffset = Number.isFinite(probeHitsOffsetByKey[key]) ? probeHitsOffsetByKey[key] : 0;
            if (currentOffset + pageSize < allHits.length) {
                probeHitsOffsetByKey[key] = currentOffset + pageSize;
            }
            renderProbeHitsPage(key);
        });
    }
    if (probeEnableToggle) {
        probeEnableToggle.addEventListener('change', (e) => {
            const enabled = e.target.checked;
            persistProbeEnabled(enabled);
            if (enabled) {
                ensureProbeCapture(getSelectedProbeChannelId(), true);
                runActiveProbe(true);
            } else {
                stopProbeRunLoop('Probe disabled');
            }
            syncProbePreview(getSelectedProbeChannelId());
            updateProbeStreamToggleButton(getSelectedProbeChannelId());
        });
    }
    if (probeImageEnableToggle) {
        probeImageEnableToggle.addEventListener('change', () => {
            if (!probeImageState?.data) {
                updateImageProbeStatus(false);
                setProbeStatus('Select an image first.', true);
                return;
            }
            updateImageProbeStatus(Boolean(probeImageEnableToggle.checked));
        });
    }
    if (probeBenchBtn && probeBenchOutput) {
        probeBenchBtn.addEventListener('click', async () => {
            setButtonBusy(probeBenchBtn, true);
            probeBenchOutput.textContent = 'Benchmark running...';
            try {
                const resp = await fetch('/probes/bench');
                const data = await resp.json();
                if (!resp.ok || data.error) throw new Error(data.error || 'Benchmark failed');
                probeBenchOutput.textContent = `~${data.approx_fps} fps @ batch ${data.batch} on ${data.device} (elapsed ${data.elapsed_sec}s)`;
            } catch (err) {
                probeBenchOutput.textContent = `Benchmark failed: ${err.message}`;
            } finally {
                setButtonBusy(probeBenchBtn, false);
            }
        });
    }
    applyProbeRoiState(false, null);

    // Mode switching
    if (archiveModeBtn) archiveModeBtn.addEventListener('click', () => setMode('archive'));
    if (videoModeBtn) videoModeBtn.addEventListener('click', () => setMode('video'));
    if (monitorModeBtn) monitorModeBtn.addEventListener('click', () => setMode('monitor'));
    if (agentModeBtn) agentModeBtn.addEventListener('click', () => { setMode('agent'); agentInit(); });
    if (loadDetectionsBtn) {
        loadDetectionsBtn.addEventListener('click', () => {
            setMode('archive');
            loadDetectionsArchive(true);
        });
    }
    if (refreshDetectionsFiltersBtn) {
        refreshDetectionsFiltersBtn.addEventListener('click', () => refreshArchiveFilters());
    }
    if (archiveDetectionsPrevBtn) {
        archiveDetectionsPrevBtn.addEventListener('click', () => {
            const pageSize = Number.parseInt(archiveDetectionsLimit?.value || '24', 10);
            const size = Number.isFinite(pageSize) ? pageSize : 24;
            archiveDetectionsOffset = Math.max(0, archiveDetectionsOffset - size);
            loadDetectionsArchive(false);
        });
    }
    if (archiveDetectionsNextBtn) {
        archiveDetectionsNextBtn.addEventListener('click', () => {
            if (!archiveDetectionsHasMore) return;
            const pageSize = Number.parseInt(archiveDetectionsLimit?.value || '24', 10);
            const size = Number.isFinite(pageSize) ? pageSize : 24;
            archiveDetectionsOffset += Math.max(1, size);
            loadDetectionsArchive(false);
        });
    }
    if (archiveChannelFilter) {
        archiveChannelFilter.addEventListener('change', () => {
            archiveDetectionsOffset = 0;
            archiveDetectionsHasMore = false;
            updateArchiveDetectionsNav();
            refreshArchiveProbeFilter();
        });
    }
    if (archiveProbeFilter) {
        archiveProbeFilter.addEventListener('change', () => {
            archiveDetectionsOffset = 0;
            archiveDetectionsHasMore = false;
            updateArchiveDetectionsNav();
        });
    }
    if (archiveTimeFilter) {
        archiveTimeFilter.addEventListener('change', () => {
            archiveDetectionsOffset = 0;
            archiveDetectionsHasMore = false;
            updateArchiveDetectionsNav();
        });
    }
    if (archiveDetectionsLimit) {
        archiveDetectionsLimit.addEventListener('change', () => {
            archiveDetectionsOffset = 0;
            archiveDetectionsHasMore = false;
            updateArchiveDetectionsNav();
        });
    }
    if (searchScopeSelect) {
        searchScopeSelect.addEventListener('change', () => {
            updateSearchScopeUI();
        });
    }
    
    // Check index status
    async function checkIndexStatus(folder) {
        try {
            const response = await fetch('/check_index', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ folder })
            });
            return await response.json();
        } catch (error) {
            return { indexed: false, available_modes: [] };
        }
    }

    async function parseApiJson(response, fallbackMessage) {
        let data = {};
        try {
            data = await response.json();
        } catch (_) {
            data = {};
        }
        if (!response.ok || data.error) {
            const message = data.error || `${fallbackMessage} (${response.status})`;
            throw new Error(message);
        }
        return data;
    }
    
    // Index folder
    indexBtn.addEventListener('click', async () => {
        const folder = folderInput.value.trim();
        if (!folder) return;
        
        indexStatus.textContent = 'Indexing...';
        indexStatus.className = 'status';
        setButtonBusy(indexBtn, true);
        
        try {
            const response = await fetch('/index', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ folder })
            });
            
            const data = await response.json();
            
            if (data.success) {
                const counts = data.counts || {};
                const summary = Object.keys(counts).length > 0
                    ? Object.entries(counts).map(([mode, count]) => `${mode}: ${count}`).join(' | ')
                    : `Active: ${data.count || 0}`;
                indexStatus.textContent = `Indexed successfully (${summary})`;
                indexStatus.className = 'status success';
                currentFolder = folder;
                const modes = data.modes || [];
                if (modes.includes(embedderSelect.value)) {
                    applyEmbedderUI(embedderSelect.value);
                }
            } else {
                indexStatus.textContent = data.error || 'Indexing failed';
                indexStatus.className = 'status error';
            }
        } catch (error) {
            indexStatus.textContent = 'Error: ' + error.message;
            indexStatus.className = 'status error';
        } finally {
            setButtonBusy(indexBtn, false);
        }
    });
    
    // Text search
    searchBtn.addEventListener('click', async () => {
        setMode('archive');
        const query = searchInput.value.trim();
        const folder = folderInput.value.trim();
        const limit = resultLimitSelect.value;
        const sortBy = sortBySelect.value;
        const detectionsScope = isDetectionsScope();
        
        if (!query || (!detectionsScope && !folder)) return;
        
        setButtonBusy(searchBtn, true);
        resultsContainer.innerHTML = '<div class="loading"><div class="spinner"></div> Searching...</div>';
        renderArchiveInspectorEmpty('Searching archive...');
        
        try {
            let response;
            if (detectionsScope) {
                const payload = {
                    query,
                    limit,
                    sort_by: sortBy,
                    embedder: embedderSelect ? embedderSelect.value : 'clip',
                    ...buildDetectionSearchFilters(),
                };
                response = await fetch('/detections/search_text', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
            } else {
                response = await fetch('/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ folder, query, limit, sort_by: sortBy })
                });
            }
            
            const data = await parseApiJson(response, 'Text search failed');
            
            if (data.results && data.results.length > 0) {
                const renderedResults = detectionsScope
                    ? decorateDetectionSearchResults(data.results, data.mode_used, data.mode_requested)
                    : data.results;
                displayResults(renderedResults);
                if (detectionsScope && data.mode_requested && data.mode_used && data.mode_requested !== data.mode_used) {
                    indexStatus.textContent = `Detections text search uses ${data.mode_used.toUpperCase()} backend.`;
                    indexStatus.className = 'status warning';
                }
            } else {
                resultsContainer.innerHTML = '<div class="loading">No results found</div>';
                renderArchiveInspectorEmpty('No results found for this query.');
            }
        } catch (error) {
            resultsContainer.innerHTML = '<div class="loading">Error: ' + error.message + '</div>';
            renderArchiveInspectorEmpty(`Search error: ${error.message}`);
        } finally {
            setButtonBusy(searchBtn, false);
        }
    });
    
    // Image search
    imageSearchBtn.addEventListener('click', async () => {
        setMode('archive');
        const folder = folderInput.value.trim();
        const file = imageUpload.files[0];
        const limit = resultLimitSelect.value;
        const sortBy = sortBySelect.value;
        const detectionsScope = isDetectionsScope();
        
        if (!file) {
            alert('Please upload an image file.');
            return;
        }
        if (!detectionsScope && !folder) {
            alert('Please select a folder and upload an image file.');
            return;
        }
        
        setButtonBusy(imageSearchBtn, true);
        resultsContainer.innerHTML = '<div class="loading"><div class="spinner"></div> Searching by image...</div>';
        renderArchiveInspectorEmpty('Searching by image...');
        
        try {
            const formData = new FormData();
            formData.append('limit', limit);
            formData.append('sort_by', sortBy);
            formData.append('image', file);
            if (detectionsScope) {
                formData.append('embedder', embedderSelect ? embedderSelect.value : 'clip');
                const filters = buildDetectionSearchFilters();
                Object.entries(filters).forEach(([key, value]) => {
                    if (value !== undefined && value !== null && String(value).length > 0) {
                        formData.append(key, String(value));
                    }
                });
            } else {
                formData.append('folder', folder);
            }
            
            const response = await fetch(detectionsScope ? '/detections/search_image' : '/search_by_image', {
                method: 'POST',
                body: formData
            });
            
            const data = await parseApiJson(response, 'Image search failed');
            
            if (data.results && data.results.length > 0) {
                const renderedResults = detectionsScope
                    ? decorateDetectionSearchResults(data.results, data.mode_used, data.mode_requested)
                    : data.results;
                displayResults(renderedResults);
            } else {
                resultsContainer.innerHTML = '<div class="loading">No results found</div>';
                renderArchiveInspectorEmpty('No visual matches found for this reference image.');
            }
        } catch (error) {
            resultsContainer.innerHTML = '<div class="loading">Error: ' + error.message + '</div>';
            renderArchiveInspectorEmpty(`Image search error: ${error.message}`);
        } finally {
            setButtonBusy(imageSearchBtn, false);
        }
    });

    function renderVideoFrames(frames) {
        if (!videoFrames) return;
        if (!frames || !frames.length) {
            videoFrames.innerHTML = '';
            return;
        }
        const html = frames.map((frame, idx) => {
            const ts = typeof frame.time_sec === 'number' ? `${frame.time_sec.toFixed(2)}s` : 'n/a';
            return `<div title="Frame ${idx + 1} (${ts})"><img src="data:image/jpeg;base64,${frame.thumbnail}" alt="Frame ${idx + 1}" /></div>`;
        }).join('');
        videoFrames.innerHTML = html;
    }

    async function runVideoUnderstanding() {
        const videoPath = videoPathInput.value.trim();
        const frameCount = parseInt(videoFrameCount.value, 10) || 16;
        const sampleFpsValue = Number.parseFloat(videoSampleFpsInput.value);
        const prompt = videoPromptInput.value.trim();
        const modelId = videoModelInput ? videoModelInput.value.trim() : '';

        if (!videoPath) {
            videoStatus.textContent = 'Provide a video path.';
            videoStatus.className = 'video-status error';
            return;
        }

        if (saveVideoPromptInput && saveVideoPromptInput.checked) {
            localStorage.setItem('evs_video_prompt', prompt);
        } else {
            localStorage.removeItem('evs_video_prompt');
        }

        setButtonBusy(videoRunBtn, true);
        saveSummaryBtn.style.display = 'none';
        lastSummaryText = '';
        lastSummaryTarget = null;
        videoStatus.dataset.base = 'Sampling frames and querying the model...';
        videoStatus.textContent = videoStatus.dataset.base;
        videoStatus.className = 'video-status';
        videoOutput.style.display = 'none';
        videoOutput.innerHTML = '';
        renderVideoFrames([]);
        startVideoTimer();

        try {
            const payload = {
                video: videoPath,
                frame_count: frameCount,
                prompt,
            };
            if (modelId) {
                payload.model = modelId;
            }
            if (Number.isFinite(sampleFpsValue) && sampleFpsValue > 0) {
                payload.sample_fps = sampleFpsValue;
            }
            const response = await fetch('/video_understanding', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (!response.ok || data.error) {
                videoStatus.dataset.base = data.error || 'Video understanding request failed.';
                videoStatus.textContent = videoStatus.dataset.base;
                videoStatus.className = 'video-status error';
                stopVideoTimer();
                return;
            }
            const durationLabel = typeof data.duration_sec === 'number' ? ` · Duration: ${formatDuration(data.duration_sec)}` : '';
            videoStatus.dataset.base = `Model: ${data.model || modelId || 'LM Studio'} · Frames sent: ${(data.frames || []).length || frameCount}${durationLabel}`;
            videoStatus.textContent = videoStatus.dataset.base;
            if (data.summary) {
                videoOutput.style.display = 'block';
                videoOutput.innerHTML = renderMarkdown(data.summary);
                lastSummaryText = data.summary;
                lastSummaryTarget = null;
                saveSummaryBtn.style.display = 'inline-flex';
            } else {
                videoOutput.style.display = 'block';
                videoOutput.textContent = '(No summary returned)';
                lastSummaryText = '';
                lastSummaryTarget = null;
                saveSummaryBtn.style.display = 'none';
            }
            renderVideoFrames(data.frames || []);
            stopVideoTimer(true);
        } catch (error) {
            videoStatus.dataset.base = 'Error: ' + error.message;
            videoStatus.textContent = videoStatus.dataset.base;
            videoStatus.className = 'video-status error';
            stopVideoTimer(true);
        } finally {
            setButtonBusy(videoRunBtn, false);
        }
    }

    if (videoRunBtn) {
        videoRunBtn.addEventListener('click', runVideoUnderstanding);
    }

    async function saveSummaryAsComment() {
        if (!lastSummaryText || !lastSummaryTarget || !lastSummaryTarget.path) {
            alert('No summary or target image available to save.');
            return;
        }
        const folder = folderInput.value.trim();
        if (!folder) {
            alert('Please enter a folder path first.');
            return;
        }
        try {
            const response = await fetch('/comments', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    folder,
                    image_path: lastSummaryTarget.path,
                    comment: lastSummaryText,
                }),
            });
            const data = await response.json();
            if (data.success) {
                alert('Summary saved as comment.');
            } else {
                alert('Failed to save comment: ' + (data.error || 'Unknown error'));
            }
        } catch (err) {
            alert('Failed to save comment: ' + err.message);
        }
    }

    if (saveSummaryBtn) {
        saveSummaryBtn.addEventListener('click', saveSummaryAsComment);
    }
    
    // Show commented images
    showCommentedBtn.addEventListener('click', async () => {
        const folder = folderInput.value.trim();
        
        if (!folder) {
            alert('Please enter a folder path first');
            return;
        }
        setMode('archive');
        
        resultsContainer.innerHTML = '<div class="loading"><div class="spinner"></div> Loading commented images...</div>';
        renderArchiveInspectorEmpty('Loading commented images...');
        
        try {
            const response = await fetch('/commented_images', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ folder })
            });
            
            const data = await parseApiJson(response, 'Loading commented images failed');
            
            if (data.results && data.results.length > 0) {
                displayCommentedResults(data.results);
            } else {
                resultsContainer.innerHTML = '<div class="loading">No commented images found</div>';
                renderArchiveInspectorEmpty('No commented images found for the current archive.');
            }
        } catch (error) {
            resultsContainer.innerHTML = '<div class="loading">Error: ' + error.message + '</div>';
            renderArchiveInspectorEmpty(`Commented image load failed: ${error.message}`);
        }
    });
    
    function renderArchiveInspectorEmpty(message = 'Select a result to inspect the full image, metrics, comments, and segmentation tools.') {
        if (resultsContainer) {
            resultsContainer.classList.remove('results-grid--detections');
        }
        activeArchiveInspectorIndex = -1;
        if (archiveInspectorEmpty) {
            archiveInspectorEmpty.textContent = message;
            archiveInspectorEmpty.classList.remove('is-hidden');
        }
        if (archiveInspectorBody) {
            archiveInspectorBody.innerHTML = '';
            archiveInspectorBody.classList.add('is-hidden');
        }
        document.querySelectorAll('#results .result-item').forEach((item) => {
            item.classList.remove('selected');
        });
    }

    function syncArchiveResultsLayout(results) {
        if (!resultsContainer) return;
        const list = Array.isArray(results) ? results : [];
        const detectionOnly = list.length > 0 && list.every((result) => Boolean(result && result.is_detection));
        resultsContainer.classList.toggle('results-grid--detections', detectionOnly);
    }

    function highlightActiveArchiveResultCard(index) {
        document.querySelectorAll('#results .result-item').forEach((item) => {
            const itemIndex = Number.parseInt(item.dataset.resultIndex || '-1', 10);
            item.classList.toggle('selected', itemIndex === index);
        });
    }

    function openImageLightbox(src, meta = '') {
        if (!imageLightboxModal || !imageLightboxImg) return;
        imageLightboxImg.src = src || '';
        if (imageLightboxMeta) {
            imageLightboxMeta.textContent = meta || '';
        }
        imageLightboxModal.style.display = 'block';
    }

    function closeImageLightbox() {
        if (!imageLightboxModal) return;
        imageLightboxModal.style.display = 'none';
        if (imageLightboxImg) imageLightboxImg.src = '';
        if (imageLightboxMeta) imageLightboxMeta.textContent = '';
    }

    function showArchiveInspector(index) {
        if (!archiveInspectorBody || !Array.isArray(archiveRenderedResults) || !archiveRenderedResults.length) {
            renderArchiveInspectorEmpty();
            return;
        }
        const result = archiveRenderedResults[index];
        if (!result) {
            renderArchiveInspectorEmpty();
            return;
        }

        activeArchiveInspectorIndex = index;
        if (archiveInspectorEmpty) {
            archiveInspectorEmpty.classList.add('is-hidden');
        }
        archiveInspectorBody.classList.remove('is-hidden');
        const detailClasses = ['result-item', 'result-item--detail', 'expanded'];
        if (result.is_detection) {
            detailClasses.push('result-item--detection-detail');
        }
        archiveInspectorBody.innerHTML = `
            <div class="${detailClasses.join(' ')}" data-result-index="${index}">
                ${generateResultItemHTML(result, index, archiveRenderedCommented, 'detail')}
            </div>
        `;

        const detailItem = archiveInspectorBody.querySelector('.result-item');
        if (!detailItem) return;
        setupResultItemEventHandlers(detailItem, result, index, { variant: 'detail' });
        highlightActiveArchiveResultCard(index);
        archiveInspectorBody.scrollTop = 0;

        const detailImg = detailItem.querySelector('.thumbnail');
        const commentsContainer = document.getElementById(`comments-${index}`);
        const activeFolder = folderInput.value.trim();
        const canUseFolderComments = Boolean(result.path && activeFolder && String(result.path).startsWith(activeFolder));
        if (!result.is_detection) {
            if (activeFolder && result.path) {
                loadComments(index, result.path, activeFolder);
                prepareSegmentsPanel(detailItem, result, index);
            } else if (commentsContainer) {
                commentsContainer.innerHTML = '<div class="no-comments">Provide a folder path to load comments for indexed images.</div>';
            }
        } else if (commentsContainer) {
            if (canUseFolderComments) {
                loadComments(index, result.path, activeFolder);
            } else {
                commentsContainer.innerHTML = '<div class="no-comments">This detection can be described, but comments are only available when the image belongs to the active indexed folder.</div>';
            }
        }

        if (detailImg && segmentsEnabledInput.checked && !result.is_detection) {
            detailImg.classList.add('segment-enabled');
        }
    }

    // Generate common HTML structure for result items
    function generateResultItemHTML(result, index, isCommented = false, variant = 'card') {
        const similarityMarkup = buildSimilarityMetrics(result, isCommented);
        const badgesMarkup = buildResultBadges(result);
        const safeFilename = escapeHtml(result.filename || 'unnamed');
        const rawPath = String(result.path || '');
        const hasPath = rawPath.length > 0;
        const isDetectionResult = Boolean(result && result.is_detection);
        const showFilenameRow = !isDetectionResult;
        const activeFolder = folderInput ? folderInput.value.trim() : '';
        const canUseFolderComments = Boolean(hasPath && activeFolder && rawPath.startsWith(activeFolder));
        const safePath = escapeHtml(rawPath);
        const thumb = String(result.thumbnail || '').trim();
        const fallbackSvg = encodeURIComponent(
            '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="260">' +
            '<rect width="100%" height="100%" fill="#1f2026"/>' +
            '<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#9aa0ad" font-size="18">No thumbnail</text>' +
            '</svg>'
        );
        const thumbnailSrc = thumb ? `data:image/jpeg;base64,${thumb}` : `data:image/svg+xml;charset=utf-8,${fallbackSvg}`;
        const detailImageSrc = hasPath ? buildImageFetchUrl(rawPath, result) : thumbnailSrc;
        const overlayIcon = variant === 'detail'
            ? '<path d="M240-240v-200h80v120h120v80H240Zm400-400v-80h80v200H520v-80h120v-40Z"/>'
            : '<path d="M240-240v-240h72v168h168v72H240Zm408-240v-168H480v-72h240v240h-72Z"/>';

        if (variant === 'card') {
            return `
                <div class="image-container">
                    <img src="${thumbnailSrc}" class="thumbnail" alt="" />
                    <div class="image-overlay">
                        ${hasPath ? `
                            <div class="expand-collapse-icon" data-index="${index}" title="Inspect result">
                                <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="#e3e3e3">
                                    ${overlayIcon}
                                </svg>
                            </div>
                        ` : ''}
                    </div>
                </div>
                <div class="result-info">
                    ${showFilenameRow ? `
                        <div class="filename">
                            ${safeFilename}
                            <svg class="copy-icon" xmlns="http://www.w3.org/2000/svg" height="16px" viewBox="0 -960 960 960" width="16px" fill="#888">
                                <path d="M360-240q-29.7 0-50.85-21.15Q288-282.3 288-312v-480q0-29.7 21.15-50.85Q330.3-864 360-864h384q29.7 0 50.85 21.15Q816-821.7 816-792v480q0 29.7-21.15 50.85Q773.7-240 744-240H360Zm0-72h384v-480H360v480ZM216-96q-29.7 0-50.85-21.15Q144-138.3 144-168v-552h72v552h456v72H216Zm144-216v-480 480Z"/>
                            </svg>
                        </div>
                    ` : ''}
                    ${badgesMarkup}
                    <div class="similarity">${similarityMarkup}</div>
                    <div class="result-actions">
                        <button class="action-icon describe-icon" data-index="${index}" data-path="${safePath}" title="Describe with LM">
                            <svg xmlns="http://www.w3.org/2000/svg" height="18px" viewBox="0 -960 960 960" width="18px" fill="#e3e3e3">
                                <path d="M160-120q-33 0-56.5-23.5T80-200v-560q0-33 23.5-56.5T160-840h545q33 0 56.5 23.5T785-760v160h-80v-160H160v560h545v-160h80v160q0 33-23.5 56.5T705-120H160Zm520-240 57-57-143-143 143-143-57-57-143 143-143-143-57 57 143 143-143 143 57 57 143-143 143 143Z"/>
                            </svg>
                        </button>
                        <button class="action-icon find-similar-icon" data-index="${index}" data-path="${safePath}" title="Find similar">
                            <svg xmlns="http://www.w3.org/2000/svg" height="18px" viewBox="0 -960 960 960" width="18px" fill="#e3e3e3">
                                <path d="M784-120 532-372q-30 24-69 38t-83 14q-109 0-184.5-75.5T120-580q0-109 75.5-184.5T380-840q109 0 184.5 75.5T640-580q0 44-14 83t-38 69l252 252-56 56ZM380-400q75 0 127.5-52.5T560-580q0-75-52.5-127.5T380-760q-75 0-127.5 52.5T200-580q0 75 52.5 127.5T380-400Z"/>
                            </svg>
                        </button>
                    </div>
                </div>
            `;
        }

        const segmentsPanelMarkup = hasPath && !isDetectionResult ? `
            <div class="segments-panel" id="segments-${index}">
                <div class="segments-status warning">Segments disabled. Enable in settings to propose regions.</div>
            </div>
        ` : '';
        const commentsPanelMarkup = hasPath ? `
            <div class="comment-section">
                <div class="lm-description" id="lm-desc-${index}">
                    <div class="no-comments">No LLM description yet.</div>
                </div>
                <div class="lm-description-actions">
                    <button class="save-comment-btn is-hidden" id="lm-save-btn-${index}">Save LLM as comment</button>
                </div>
                ${!isDetectionResult || canUseFolderComments ? `
                    <div class="comments-list" id="comments-${index}">
                        <div class="comment-loading">Loading comments...</div>
                    </div>
                    <div class="comment-form">
                        <textarea class="comment-input" placeholder="Add a comment..." id="comment-input-${index}"></textarea>
                        <button class="save-comment-btn" id="save-btn-${index}">Save</button>
                    </div>
                ` : `
                    <div class="comments-list" id="comments-${index}">
                        <div class="no-comments">Comments can be saved only when this image is inside the active indexed folder.</div>
                    </div>
                `}
            </div>
        ` : '';

        return `
            <div class="image-container">
                <img src="${detailImageSrc}" class="thumbnail" alt="" />
                <div class="image-overlay">
                    ${hasPath ? `
                        <div class="expand-collapse-icon" data-index="${index}" title="Open full preview">
                            <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="#e3e3e3">
                                ${overlayIcon}
                            </svg>
                        </div>
                    ` : ''}
                </div>
            </div>
            <div class="result-info">
                ${showFilenameRow ? `
                    <div class="filename">
                        ${safeFilename}
                        <svg class="copy-icon" xmlns="http://www.w3.org/2000/svg" height="16px" viewBox="0 -960 960 960" width="16px" fill="#888">
                            <path d="M360-240q-29.7 0-50.85-21.15Q288-282.3 288-312v-480q0-29.7 21.15-50.85Q330.3-864 360-864h384q29.7 0 50.85 21.15Q816-821.7 816-792v480q0 29.7-21.15 50.85Q773.7-240 744-240H360Zm0-72h384v-480H360v480ZM216-96q-29.7 0-50.85-21.15Q144-138.3 144-168v-552h72v552h456v72H216Zm144-216v-480 480Z"/>
                        </svg>
                    </div>
                ` : ''}
                ${badgesMarkup}
                <div class="similarity">${similarityMarkup}</div>
                <div class="result-actions">
                    <button class="action-icon describe-icon" data-index="${index}" data-path="${safePath}" title="Describe with LM">
                        <svg xmlns="http://www.w3.org/2000/svg" height="18px" viewBox="0 -960 960 960" width="18px" fill="#e3e3e3">
                            <path d="M160-120q-33 0-56.5-23.5T80-200v-560q0-33 23.5-56.5T160-840h545q33 0 56.5 23.5T785-760v160h-80v-160H160v560h545v-160h80v160q0 33-23.5 56.5T705-120H160Zm520-240 57-57-143-143 143-143-57-57-143 143-143-143-57 57 143 143-143 143 57 57 143-143 143 143Z"/>
                        </svg>
                    </button>
                    <button class="action-icon find-similar-icon" data-index="${index}" data-path="${safePath}" title="Find similar">
                        <svg xmlns="http://www.w3.org/2000/svg" height="18px" viewBox="0 -960 960 960" width="18px" fill="#e3e3e3">
                            <path d="M784-120 532-372q-30 24-69 38t-83 14q-109 0-184.5-75.5T120-580q0-109 75.5-184.5T380-840q109 0 184.5 75.5T640-580q0 44-14 83t-38 69l252 252-56 56ZM380-400q75 0 127.5-52.5T560-580q0-75-52.5-127.5T380-760q-75 0-127.5 52.5T200-580q0 75 52.5 127.5T380-400Z"/>
                        </svg>
                    </button>
                </div>
            </div>
            ${segmentsPanelMarkup}
            ${commentsPanelMarkup}
        `;
    }

    // Setup event handlers for result item
    function setupResultItemEventHandlers(item, result, index, options = {}) {
        const variant = options.variant || 'card';

        const expandCollapseIcon = item.querySelector('.expand-collapse-icon');
        if (expandCollapseIcon && result.path) {
            expandCollapseIcon.addEventListener('click', (e) => {
                e.stopPropagation();
                if (variant === 'detail') {
                    openImageLightbox(buildImageFetchUrl(result.path, result), result.filename || result.path || 'Preview');
                } else {
                    showArchiveInspector(index);
                }
            });
        }

        if (variant === 'card') {
            item.addEventListener('click', (e) => {
                if (e.target.closest('button, .expand-collapse-icon, .copy-icon')) {
                    return;
                }
                showArchiveInspector(index);
            });
        }

        const copyIcon = item.querySelector('.copy-icon');
        if (copyIcon) {
            if (result.path) {
                copyIcon.addEventListener('click', (e) => {
                    e.stopPropagation();
                    copyImagePath(result.path);
                });
            } else {
                copyIcon.style.display = 'none';
            }
        }

        const findSimilarIcon = item.querySelector('.find-similar-icon');
        if (findSimilarIcon) {
            if (result.path) {
                findSimilarIcon.addEventListener('click', (e) => {
                    e.stopPropagation();
                    findSimilarImages(result.path, result);
                });
            } else {
                findSimilarIcon.style.display = 'none';
            }
        }

        const describeIcon = item.querySelector('.describe-icon');
        if (describeIcon) {
            if (result.path) {
                describeIcon.addEventListener('click', (e) => {
                    e.stopPropagation();
                    describeImageWithLM(index, result.path, item, result);
                });
            } else {
                describeIcon.style.display = 'none';
            }
        }

        const saveBtn = item.querySelector(`#save-btn-${index}`);
        const commentInput = item.querySelector(`#comment-input-${index}`);
        const activeFolder = folderInput ? folderInput.value.trim() : '';
        const canUseFolderComments = Boolean(result.path && activeFolder && String(result.path).startsWith(activeFolder));

        if (saveBtn) {
            if (canUseFolderComments || (result.path && !result.is_detection)) {
                saveBtn.addEventListener('click', () => {
                    saveComment(index, result.path, folderInput.value.trim(), commentInput.value.trim());
                });
            } else {
                saveBtn.disabled = true;
            }
        }

        const lmSaveBtn = item.querySelector(`#lm-save-btn-${index}`);
        if (lmSaveBtn) {
            if (canUseFolderComments || (result.path && !result.is_detection)) {
                lmSaveBtn.addEventListener('click', () => {
                    saveLmDescriptionAsComment(index, result.path);
                });
            } else {
                lmSaveBtn.style.display = 'none';
            }
        }

        const img = item.querySelector('.thumbnail');
        if (img && result.path && variant === 'detail') {
            if (!result.is_detection) {
                img.addEventListener('click', (e) => {
                    if (segmentsEnabledInput.checked) {
                        handleSegmentClick(e, result, index, item);
                        return;
                    }
                    openImageLightbox(buildImageFetchUrl(result.path, result), result.filename || result.path || 'Preview');
                });
            } else {
                img.addEventListener('click', () => {
                    openImageLightbox(buildImageFetchUrl(result.path, result), result.filename || result.path || 'Preview');
                });
            }
        }
    }

    // Display results
    function displayResults(results) {
        resultsContainer.innerHTML = '';
        segmentContextByIndex = {};
        archiveRenderedResults = Array.isArray(results) ? results : [];
        archiveRenderedCommented = false;
        syncArchiveResultsLayout(archiveRenderedResults);
        
        archiveRenderedResults.forEach((result, index) => {
            const item = document.createElement('div');
            item.className = 'result-item';
            if (result && result.is_detection) {
                item.classList.add('result-item--detection-card');
            }
            item.dataset.resultIndex = index;
            item.innerHTML = generateResultItemHTML(result, index, false, 'card');
            
            setupResultItemEventHandlers(item, result, index, { variant: 'card' });
            resultsContainer.appendChild(item);
        });

        if (archiveRenderedResults.length) {
            showArchiveInspector(0);
        } else {
            renderArchiveInspectorEmpty('Run a text search, image search, or load detections to populate the inspector.');
        }
    }
    
    // Display commented results (similar to displayResults but with comment info)
    function displayCommentedResults(results) {
        resultsContainer.innerHTML = '';
        segmentContextByIndex = {};
        archiveRenderedResults = Array.isArray(results) ? results : [];
        archiveRenderedCommented = true;
        syncArchiveResultsLayout(archiveRenderedResults);
        
        archiveRenderedResults.forEach((result, index) => {
            const item = document.createElement('div');
            item.className = 'result-item';
            if (result && result.is_detection) {
                item.classList.add('result-item--detection-card');
            }
            item.dataset.resultIndex = index;
            item.innerHTML = generateResultItemHTML(result, index, true, 'card');
            
            setupResultItemEventHandlers(item, result, index, { variant: 'card' });
            resultsContainer.appendChild(item);
        });

        if (archiveRenderedResults.length) {
            showArchiveInspector(0);
        } else {
            renderArchiveInspectorEmpty('No commented images found for the current archive.');
        }
    }
    
    // Comment functionality
    async function loadComments(index, imagePath, folder) {
        const commentsContainer = document.getElementById(`comments-${index}`);
        
        try {
            const response = await fetch(`/comments?folder=${encodeURIComponent(folder)}&image_path=${encodeURIComponent(imagePath)}`);
            const data = await response.json();
            
            if (data.comments && data.comments.length > 0) {
                displayComments(commentsContainer, data.comments);
            } else {
                commentsContainer.innerHTML = '<div class="no-comments">No comments yet. Be the first to add one!</div>';
            }
        } catch (error) {
            console.error('Error loading comments:', error);
            commentsContainer.innerHTML = '<div class="no-comments">Error loading comments</div>';
        }
    }
    
    function displayComments(container, comments) {
        container.innerHTML = '';
        comments.forEach(comment => {
            const commentDiv = document.createElement('div');
            commentDiv.className = 'comment-item';
            
            // Parse timestamp and comment text
            const timestampMatch = comment.match(/^\[(.*?)\] (.*)$/);
            if (timestampMatch) {
                const [, timestamp, text] = timestampMatch;
                commentDiv.innerHTML = `
                    <div class="comment-timestamp">${timestamp}</div>
                    <div class="comment-text">${escapeHtml(text)}</div>
                `;
            } else {
                commentDiv.innerHTML = `<div class="comment-text">${escapeHtml(comment)}</div>`;
            }
            
            container.appendChild(commentDiv);
        });
    }

    function renderLmDescription(index, summary, modelLabel = '') {
        const descContainer = document.getElementById(`lm-desc-${index}`);
        const saveBtn = document.getElementById(`lm-save-btn-${index}`);
        if (!descContainer || !saveBtn) return;

        const now = new Date().toLocaleString();
        const modelSuffix = modelLabel ? ` · ${escapeHtml(modelLabel)}` : '';
        descContainer.innerHTML = `
            <div class="comment-item lm-comment">
                <div class="comment-timestamp">LLM Description${modelSuffix} · ${escapeHtml(now)}</div>
                <div class="comment-text">${renderMarkdown(summary || '')}</div>
            </div>
        `;
        saveBtn.dataset.summary = summary || '';
        saveBtn.style.display = 'inline-flex';
        saveBtn.disabled = false;
        saveBtn.textContent = 'Save LLM as comment';
    }

    async function saveLmDescriptionAsComment(index, imagePath) {
        const saveBtn = document.getElementById(`lm-save-btn-${index}`);
        if (!saveBtn) return;
        const summary = (saveBtn.dataset.summary || '').trim();
        if (!summary) {
            alert('No LLM description to save yet.');
            return;
        }
        const folder = folderInput.value.trim();
        if (!folder) {
            alert('Please enter a folder path first.');
            return;
        }

        setButtonBusy(saveBtn, true);
        try {
            const response = await fetch('/comments', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    folder,
                    image_path: imagePath,
                    comment: summary,
                }),
            });
            const data = await parseApiJson(response, 'Saving LLM description failed');
            const commentsContainer = document.getElementById(`comments-${index}`);
            if (commentsContainer && Array.isArray(data.comments)) {
                displayComments(commentsContainer, data.comments);
            }
            indexStatus.textContent = 'LLM description saved as comment.';
            indexStatus.className = 'status success';
        } catch (err) {
            alert('Failed to save LLM description: ' + err.message);
        } finally {
            setButtonBusy(saveBtn, false);
        }
    }
    
    async function saveComment(index, imagePath, folder, comment) {
        if (!comment) return;
        
        const saveBtn = document.getElementById(`save-btn-${index}`);
        const commentInput = document.getElementById(`comment-input-${index}`);
        
        setButtonBusy(saveBtn, true);
        
        try {
            const response = await fetch('/comments', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    folder: folder,
                    image_path: imagePath,
                    comment: comment
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Clear input and reload comments
                commentInput.value = '';
                const commentsContainer = document.getElementById(`comments-${index}`);
                displayComments(commentsContainer, data.comments);
            } else {
                alert('Error saving comment: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            console.error('Error saving comment:', error);
            alert('Error saving comment: ' + error.message);
        } finally {
            setButtonBusy(saveBtn, false);
        }
    }
    
    function toggleImageExpansion(item, result, index) {
        showArchiveInspector(index);
    }

    function resetSegmentsPanel(item, index) {
        const panel = item.querySelector(`#segments-${index}`);
        if (!panel) return;
        if (!segmentsEnabledInput.checked) {
            panel.innerHTML = '<div class="segments-status warning">Segments disabled. Enable in settings to propose regions.</div>';
        } else {
            panel.innerHTML = '<div class="segments-status">Expand the image and click on an area to propose regions.</div>';
        }
    }

    function prepareSegmentsPanel(item, result, index) {
        const panel = item.querySelector(`#segments-${index}`);
        if (!panel) return;
        if (!segmentsEnabledInput.checked) {
            panel.innerHTML = '<div class="segments-status warning">Segments disabled. Enable in settings to propose regions.</div>';
            return;
        }
        panel.innerHTML = '<div class="segments-status">Click inside the image to propose a region near the selected point.</div>';
    }

    function refreshSegmentsPanels() {
        if (!archiveInspectorBody || activeArchiveInspectorIndex < 0) return;
        const detailItem = archiveInspectorBody.querySelector('.result-item');
        if (!detailItem) return;
        const img = detailItem.querySelector('.thumbnail');
        prepareSegmentsPanel(detailItem, null, activeArchiveInspectorIndex);
        if (img) {
            if (segmentsEnabledInput.checked) {
                img.classList.add('segment-enabled');
            } else {
                img.classList.remove('segment-enabled');
            }
        }
    }

    function clamp01(value) {
        if (!Number.isFinite(value)) return 0;
        return Math.min(1, Math.max(0, value));
    }

    function stripBase64Payload(rawValue) {
        const text = String(rawValue || '').trim();
        if (!text) return '';
        if (text.startsWith('data:')) {
            const comma = text.indexOf(',');
            return comma >= 0 ? text.slice(comma + 1) : '';
        }
        return text;
    }

    function extractSegmentMeta(segments) {
        const ids = [];
        const labels = {};
        (segments || []).forEach((segment) => {
            if (!segment || segment.segment_id === undefined || segment.segment_id === null) return;
            const segId = String(segment.segment_id).trim();
            if (!segId) return;
            ids.push(segId);
            if (segment.label !== undefined && segment.label !== null) {
                const label = String(segment.label).trim();
                if (label) {
                    labels[segId] = label;
                }
            }
        });
        return {
            segmentIds: [...new Set(ids)],
            segmentLabels: labels,
        };
    }

    function showSegmentPanelNotice(panel, message, level = 'success') {
        if (!panel) return;
        const safeLevel = ['success', 'warning', 'error'].includes(level) ? level : 'success';
        const notice = document.createElement('div');
        notice.className = `segments-status ${safeLevel}`;
        notice.textContent = message;
        panel.prepend(notice);
        setTimeout(() => {
            notice.remove();
        }, 5200);
    }

    function buildSegmentActionContext(result, data, xNorm, yNorm, baseImageSrc) {
        if (!result || !result.path) {
            return null;
        }
        const folder = folderInput.value.trim();
        if (!folder) {
            return null;
        }
        const overlay = data && data.overlay ? data.overlay : {};
        const maskBase64 = stripBase64Payload(overlay.mask_raw_png || overlay.mask_png || '');
        if (!maskBase64) {
            return null;
        }
        const segments = Array.isArray(data && data.segments) ? data.segments : [];
        const meta = extractSegmentMeta(segments);
        return {
            folder,
            imagePath: String(result.path),
            maskBase64,
            segmentIds: meta.segmentIds,
            segmentLabels: meta.segmentLabels,
            overlay,
            xNorm,
            yNorm,
            baseImageSrc: baseImageSrc || '',
        };
    }

    async function runMaskSearchFromSegment(index, panel, triggerBtn = null) {
        const context = segmentContextByIndex[index];
        if (!context || !context.maskBase64) {
            showSegmentPanelNotice(panel, 'Click the image to create a region mask first.', 'warning');
            return;
        }

        const payload = {
            folder: context.folder,
            image_path: context.imagePath,
            mask: context.maskBase64,
            limit: parseInt(resultLimitSelect.value, 10) || 12,
            sort_by: sortBySelect.value || 'similarity',
            targets: ['images', 'segments'],
        };
        if (context.segmentLabels && Object.keys(context.segmentLabels).length) {
            payload.segment_labels = context.segmentLabels;
        }

        const button = triggerBtn instanceof HTMLButtonElement ? triggerBtn : null;
        if (button) {
            setButtonBusy(button, true);
        }

        try {
            const response = await fetch('/search_by_mask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            let data = {};
            try {
                data = await response.json();
            } catch (_) {
                data = {};
            }
            if (!response.ok || data.error) {
                const hint = data.hint ? ` ${data.hint}` : '';
                throw new Error(`${data.error || 'Mask search failed'}${hint}`);
            }
            const segments = Array.isArray(data.segments) ? data.segments : [];
            const meta = extractSegmentMeta(segments);
            const refreshedContext = {
                ...context,
                segmentIds: meta.segmentIds,
                segmentLabels: meta.segmentLabels,
            };
            segmentContextByIndex[index] = refreshedContext;
            renderSegmentResponse(
                panel,
                { ...data, overlay: context.overlay || {} },
                context.xNorm,
                context.yNorm,
                context.baseImageSrc || '',
                { index, actionContext: refreshedContext, sourceLabel: 'Mask search' },
            );
            indexStatus.textContent = segments.length
                ? `Mask search returned ${segments.length} region candidate(s).`
                : 'Mask search returned no region candidates.';
            indexStatus.className = segments.length ? 'status success' : 'status warning';
        } catch (err) {
            showSegmentPanelNotice(panel, `Mask search failed: ${err.message || String(err)}`, 'error');
        } finally {
            if (button) {
                setButtonBusy(button, false);
            }
        }
    }

    async function indexSegmentsFromMask(index, panel, triggerBtn = null) {
        const context = segmentContextByIndex[index];
        if (!context || !context.maskBase64) {
            showSegmentPanelNotice(panel, 'Click the image to create a region mask first.', 'warning');
            return;
        }

        const payload = {
            folder: context.folder,
            image_path: context.imagePath,
            mask: context.maskBase64,
            segment_labels: context.segmentLabels || {},
        };

        const button = triggerBtn instanceof HTMLButtonElement ? triggerBtn : null;
        if (button) {
            setButtonBusy(button, true);
        }

        try {
            const response = await fetch('/index_segments', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            let data = {};
            try {
                data = await response.json();
            } catch (_) {
                data = {};
            }
            if (!response.ok || data.error) {
                const hint = data.hint ? ` ${data.hint}` : '';
                throw new Error(`${data.error || 'Segment indexing failed'}${hint}`);
            }
            const count = Array.isArray(data.segments_indexed)
                ? data.segments_indexed.length
                : Number(data.segment_count || 0);
            showSegmentPanelNotice(panel, `Indexed ${count} segment(s) for this image.`, 'success');
            const relaxedNote = data.min_patches_relaxed ? ' (min patch fallback used)' : '';
            indexStatus.textContent = `Segment index updated (${count} segment${count === 1 ? '' : 's'})${relaxedNote}.`;
            indexStatus.className = 'status success';
        } catch (err) {
            showSegmentPanelNotice(panel, `Segment indexing failed: ${err.message || String(err)}`, 'error');
            indexStatus.textContent = `Segment indexing failed: ${err.message || String(err)}`;
            indexStatus.className = 'status error';
        } finally {
            if (button) {
                setButtonBusy(button, false);
            }
        }
    }

    async function handleSegmentClick(event, result, index, item) {
        if (!segmentsEnabledInput.checked) return;
        if (!item.classList.contains('expanded')) return;

        const folder = folderInput.value.trim();
        if (!folder) {
            const panel = item.querySelector(`#segments-${index}`);
            if (panel) {
                panel.innerHTML = '<div class="segments-status error">Provide a folder path before running region proposals.</div>';
            }
            return;
        }

        if (item.dataset.segmentLoading === '1') {
            return;
        }

        const panel = item.querySelector(`#segments-${index}`);
        if (!panel) return;

        const img = event.currentTarget;
        const rect = img.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;

        const xNorm = clamp01((event.clientX - rect.left) / rect.width);
        const yNorm = clamp01((event.clientY - rect.top) / rect.height);

        const limitValue = parseInt(resultLimitSelect.value, 10);
        const payload = {
            folder,
            image_path: result.path,
            x: xNorm,
            y: yNorm,
            limit: Number.isFinite(limitValue) ? limitValue : 12,
            sort_by: sortBySelect.value || 'similarity',
            targets: ['images', 'segments'],
            threshold: segmentThreshold,
        };

        item.dataset.segmentLoading = '1';
        panel.innerHTML = '<div class="segments-status">Proposing region around the selected point...</div>';

        try {
            const response = await fetch('/segment_from_point', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (!response.ok || data.error) {
                throw new Error(data.error || 'Region proposal failed');
            }
            const actionContext = buildSegmentActionContext(result, data, xNorm, yNorm, img.currentSrc || img.src);
            if (actionContext) {
                segmentContextByIndex[index] = actionContext;
            } else {
                delete segmentContextByIndex[index];
            }
            renderSegmentResponse(panel, data, xNorm, yNorm, img.currentSrc || img.src, {
                index,
                actionContext,
                sourceLabel: 'Region proposal',
            });
        } catch (error) {
            panel.innerHTML = `<div class="segments-status error">Segment error: ${escapeHtml(error.message || String(error))}</div>`;
        } finally {
            delete item.dataset.segmentLoading;
        }
    }

    function renderSegmentResponse(panel, data, xNorm, yNorm, baseImageSrc, options = {}) {
        const segments = Array.isArray(data && data.segments) ? data.segments : [];
        const overlay = data && data.overlay ? data.overlay : {};
        const pctX = (xNorm * 100).toFixed(1);
        const pctY = (yNorm * 100).toFixed(1);
        const safeBaseSrc = baseImageSrc ? escapeHtml(baseImageSrc) : '';
        const sourceLabel = escapeHtml(String(options.sourceLabel || 'Region proposal'));
        const parsedIndex = Number.isFinite(options.index)
            ? Number(options.index)
            : parseInt(String(options.index || ''), 10);
        const actionContext = options.actionContext || null;
        const hasActions = Number.isFinite(parsedIndex)
            && actionContext
            && actionContext.maskBase64
            && actionContext.folder
            && actionContext.imagePath;

        const baseOverlayFigure = safeBaseSrc ? `
            <figure class="segment-overlay-figure">
                <div class="segment-overlay-stack">
                    <img src="${safeBaseSrc}" alt="Expanded image region" />
                    ${overlay.heatmap_png ? `<img class="overlay-layer overlay-heatmap" src="data:image/png;base64,${overlay.heatmap_png}" alt="Heatmap overlay" />` : ''}
                    ${overlay.mask_png ? `<img class="overlay-layer overlay-mask" src="data:image/png;base64,${overlay.mask_png}" alt="Refined mask overlay" />` : ''}
                    <div class="overlay-crosshair" style="left: ${pctX}%; top: ${pctY}%"></div>
                </div>
                <figcaption>Region overlay</figcaption>
            </figure>
        ` : '';

        const segmentationFigure = overlay.segmentation_png ? `
            <figure class="segment-segmap-figure">
                <img class="segment-segmap" src="data:image/png;base64,${overlay.segmentation_png}" alt="Semantic segmentation" />
                <figcaption>Mask2Former segmentation</figcaption>
            </figure>
        ` : '';

        const legendItems = Array.isArray(overlay.legend)
            ? overlay.legend.map((entry) => {
                const color = escapeHtml(String(entry.color || '#888'));
                const labelText = entry.label ? escapeHtml(String(entry.label)) : escapeHtml(String(entry.id || 'class'));
                const highlightClass = entry.highlight ? ' highlight' : '';
                return `<div class="segment-legend-item${highlightClass}"><span class="segment-legend-swatch" style="background:${color};"></span><span>${labelText}</span></div>`;
            }).join('')
            : '';

        const legendHtml = legendItems ? `<div class="segment-legend">${legendItems}</div>` : '';

        const overlayHtml = (baseOverlayFigure || segmentationFigure)
            ? `<div class="segment-overlay-grid">${baseOverlayFigure}${segmentationFigure}</div>${legendHtml}`
            : legendHtml;

        const listItems = segments.slice(0, 3).map((segment, idx) => {
            const segId = escapeHtml(String(segment.segment_id || `region-${idx + 1}`));
            const fraction = typeof segment.patch_fraction === 'number'
                ? `${(segment.patch_fraction * 100).toFixed(1)}% area`
                : 'Area n/a';
            const patchCount = typeof segment.patch_count === 'number'
                ? `${segment.patch_count} patch${segment.patch_count === 1 ? '' : 'es'}`
                : '';
            const humanLabel = segment.label ? ` · ${escapeHtml(String(segment.label))}` : '';

            const matches = Array.isArray(segment.image_results) ? segment.image_results.slice(0, 3) : [];
            const matchRows = matches.map((match, matchIdx) => {
                const label = escapeHtml(String(match.filename || match.path || `Match ${matchIdx + 1}`));
                const score = typeof match.similarity === 'number' ? `${(match.similarity * 100).toFixed(1)}%` : 'n/a';
                const thumb = match.thumbnail
                    ? `<img class="segment-match-thumb" src="data:image/jpeg;base64,${match.thumbnail}" alt="${label}" />`
                    : '<div class="segment-match-thumb placeholder"></div>';
                return `
                    <div class="segment-match-row">
                        ${thumb}
                        <div class="segment-match-meta">
                            <span>${label}</span>
                            <span>Similarity: ${score}</span>
                        </div>
                    </div>
                `;
            }).join('');

            const matchList = matchRows || '<div class="segments-status warning">No close matches for this region.</div>';

            return `
                <li>
                    <span class="segment-title">#${idx + 1} · ${segId}${humanLabel}</span>
                    <span class="segment-meta">${fraction}${patchCount ? ` · ${patchCount}` : ''}</span>
                    <div class="segment-match-list">
                        ${matchList}
                    </div>
                </li>
            `;
        }).join('');

        const refinementNote = overlay.refinement
            ? `<div class="segment-meta">Mask source: ${escapeHtml(String(overlay.refinement))}${overlay.refined_label ? ` · ${escapeHtml(String(overlay.refined_label))}` : ''}</div>`
            : '';
        const areaNote = typeof overlay.mask_fraction === 'number'
            ? `<div class="segment-meta">Refined mask coverage: ${(overlay.mask_fraction * 100).toFixed(1)}%</div>`
            : '';
        const resultsHtml = listItems
            ? `<ul class="segment-results-list">${listItems}</ul>`
            : '<div class="segments-status warning">Region proposals returned no matches.</div>';
        const actionsHtml = hasActions ? `
            <div class="segment-actions">
                <button class="segment-action-btn" data-segment-mask-search="${parsedIndex}">Search by mask</button>
                <button class="segment-action-btn primary" data-segment-index="${parsedIndex}">Index segments</button>
            </div>
        ` : '';

        panel.innerHTML = `
            <div class="segments-status success">${sourceLabel} near (${pctX}%, ${pctY}%) · ${segments.length} candidate(s)</div>
            ${actionsHtml}
            ${overlayHtml}
            ${refinementNote}
            ${typeof overlay.threshold === 'number' ? `<div class="segment-meta">Heatmap threshold: ${(overlay.threshold * 100).toFixed(1)}%</div>` : ''}
            ${areaNote}
            ${resultsHtml}
        `;

        if (hasActions) {
            const maskSearchBtn = panel.querySelector(`[data-segment-mask-search="${parsedIndex}"]`);
            if (maskSearchBtn) {
                maskSearchBtn.addEventListener('click', (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    runMaskSearchFromSegment(parsedIndex, panel, maskSearchBtn);
                });
            }
            const indexBtn = panel.querySelector(`[data-segment-index="${parsedIndex}"]`);
            if (indexBtn) {
                indexBtn.addEventListener('click', (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    indexSegmentsFromMask(parsedIndex, panel, indexBtn);
                });
            }
        }
    }
    
    async function copyImagePath(imagePath) {
        try {
            const textToCopy = imagePath;
            
            if (navigator.clipboard && window.isSecureContext) {
                // Use modern clipboard API
                await navigator.clipboard.writeText(textToCopy);
            } else {
                // Fallback for older browsers
                const textArea = document.createElement('textarea');
                textArea.value = textToCopy;
                textArea.style.position = 'fixed';
                textArea.style.left = '-999999px';
                textArea.style.top = '-999999px';
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();
                document.execCommand('copy');
                textArea.remove();
            }
            
            // Simple console feedback for now (could add toast notification)
            console.log('Copied to clipboard:', imagePath);
            
        } catch (error) {
            console.error('Failed to copy:', error);
        }
    }
    
    function buildImageFetchUrl(imagePath, result = null) {
        const params = new URLSearchParams();
        params.set('image_path', imagePath || '');
        if (result && result.is_detection) {
            const activeFolder = folderInput.value.trim();
            if (activeFolder && String(imagePath || '').startsWith(activeFolder)) {
                params.set('folder', activeFolder);
                return `/image?${params.toString()}`;
            }
            return `/detections/image?${params.toString()}`;
        }
        const activeFolder = folderInput.value.trim();
        if (activeFolder) {
            params.set('folder', activeFolder);
        }
        return `/image?${params.toString()}`;
    }

    async function findSimilarImages(imagePath, result = null) {
        const folder = folderInput.value.trim();
        const limit = resultLimitSelect.value;
        const sortBy = sortBySelect.value;
        const detectionResult = Boolean(result && result.is_detection);
        
        if (!detectionResult && !folder) {
            alert('Please enter a folder path first');
            return;
        }
        
        indexStatus.textContent = 'Finding similar images...';
        indexStatus.className = 'status';
        
        try {
            const imageResponse = await fetch(buildImageFetchUrl(imagePath, result));
            if (!imageResponse.ok) {
                throw new Error('Failed to load image file');
            }
            
            const imageBlob = await imageResponse.blob();
            const formData = new FormData();
            formData.append('image', imageBlob, 'reference_image.jpg');
            formData.append('limit', limit);
            formData.append('sort_by', sortBy);

            if (detectionResult || isDetectionsScope()) {
                const filters = buildDetectionSearchFilters();
                Object.entries(filters).forEach(([key, value]) => {
                    if (value !== undefined && value !== null && String(value).trim() !== '') {
                        formData.append(key, String(value));
                    }
                });
                formData.append('embedder', embedderSelect ? embedderSelect.value : 'clip');

                const response = await fetch('/detections/search_image', {
                    method: 'POST',
                    body: formData
                });
                const data = await parseApiJson(response, 'Detection image search failed');
                const rendered = decorateDetectionSearchResults(data.results, data.mode_used, data.mode_requested);
                if (!rendered.length) {
                    indexStatus.textContent = 'No similar detections found';
                    indexStatus.className = 'status warning';
                    return;
                }
                indexStatus.textContent = `Found ${rendered.length} similar detections`;
                indexStatus.className = 'status success';
                displayResults(rendered);
                return;
            }

            formData.append('folder', folder);
            const response = await fetch('/search_by_image', {
                method: 'POST',
                body: formData
            });
            const data = await parseApiJson(response, 'Image search failed');
            const results = Array.isArray(data.results) ? data.results : [];
            if (!results.length) {
                indexStatus.textContent = 'No similar images found';
                indexStatus.className = 'status warning';
                return;
            }
            indexStatus.textContent = `Found ${results.length} similar images`;
            indexStatus.className = 'status success';
            displayResults(results);
        } catch (error) {
            console.error('Find similar error:', error);
            indexStatus.textContent = 'Error finding similar images: ' + error.message;
            indexStatus.className = 'status error';
        }
    }

    async function describeImageWithLM(index, imagePath, item = null, result = null) {
        if (!imagePath) {
            alert('No filesystem path is available for this image.');
            return;
        }
        const detectionResult = Boolean(result && result.is_detection);
        const folder = folderInput.value.trim();
        const useFolderContext = !detectionResult || (folder && String(imagePath).startsWith(folder));
        if (!useFolderContext && !detectionResult) {
            alert('Please enter a folder path first.');
            return;
        }

        const prompt = videoPromptInput.value.trim();
        const modelId = videoModelInput ? videoModelInput.value.trim() : '';

        setMode('archive');
        showArchiveInspector(index);

        const descContainer = document.getElementById(`lm-desc-${index}`);
        const saveBtn = document.getElementById(`lm-save-btn-${index}`);
        if (!descContainer || !saveBtn) {
            alert('Unable to render LLM description panel for this result.');
            return;
        }

        descContainer.innerHTML = '<div class="comment-loading"><div class="spinner"></div> Generating LLM description...</div>';
        saveBtn.style.display = 'none';
        saveBtn.dataset.summary = '';

        try {
            const response = await fetch('/describe_image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(
                    useFolderContext
                        ? {
                            folder,
                            image_path: imagePath,
                            prompt,
                            model: modelId
                        }
                        : {
                            image_path: imagePath,
                            prompt,
                            model: modelId
                        }
                ),
            });
            const data = await parseApiJson(response, 'Describe request failed');
            if (data.summary) {
                renderLmDescription(index, data.summary, data.model || modelId || 'LM Studio');
                if (detectionResult && !useFolderContext) {
                    saveBtn.style.display = 'none';
                }
                return;
            }
            descContainer.innerHTML = '<div class="no-comments">(No description returned)</div>';
        } catch (err) {
            descContainer.innerHTML = `<div class="no-comments">Error: ${escapeHtml(err.message || String(err))}</div>`;
        }
    }

    updateArchiveDetectionsNav();
    updateSearchScopeUI();
    refreshArchiveFilters().catch(() => {
        setArchiveDetectionsMeta('Detection filters unavailable. Run probes to populate archive.');
    });
    
    // Enter key support
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') searchBtn.click();
    });
    
    folderInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') indexBtn.click();
    });
    
    // Check index on folder change
    folderInput.addEventListener('blur', async () => {
        const folder = folderInput.value.trim();
        if (folder) {
            const status = await checkIndexStatus(folder);
            if (status.indexed) {
                indexStatus.textContent = `Folder is indexed (${(status.available_modes || []).join(', ') || embedderSelect.value})`;
                indexStatus.className = 'status success';
            } else {
                const available = (status.available_modes || []).join(', ');
                indexStatus.textContent = available ? `Folder indexed for: ${available}` : 'Folder not indexed';
                indexStatus.className = available ? 'status warning' : 'status';
            }
        }
    });

    // =====================================================================
    // AGENT TAB
    // =====================================================================
    (function() {
        const AGENT_LS_SESSION = 'evs_agent_session_id';
        let _agentInitDone = false;
        let _agentCurrentSession = null;    // session_id string or null
        let _agentStreaming = false;
        let _agentPendingBubble = null;     // { el, bodyEl } for the current streaming bubble
        let _agentPendingImageB64 = null;   // base64 string of attached image

        // ---- DOM refs (safe — these are inside the IIFE scope) ----
        function q(id) { return document.getElementById(id); }
        const elSessionList = () => q('agentSessionList');
        const elMessages    = () => q('agentMessages');
        const elInput       = () => q('agentInput');
        const elSendBtn     = () => q('agentSendBtn');
        const elNewSession  = () => q('agentNewSessionBtn');
        const elProbeList   = () => q('agentProbeList');
        const elAgentModelInput = () => q('agentModelInput');
        const elAgentModelApplyBtn = () => q('agentModelApplyBtn');

        // ---- Helpers ----
        function fmtTime(isoOrTs) {
            try {
                const d = new Date(typeof isoOrTs === 'number' ? isoOrTs * 1000 : isoOrTs);
                return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            } catch(_) { return ''; }
        }

        function fmtDate(isoOrTs) {
            try {
                const d = new Date(typeof isoOrTs === 'number' ? isoOrTs * 1000 : isoOrTs);
                const today = new Date();
                if (d.toDateString() === today.toDateString()) return 'Today';
                const yesterday = new Date(today); yesterday.setDate(today.getDate() - 1);
                if (d.toDateString() === yesterday.toDateString()) return 'Yesterday';
                return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
            } catch(_) { return ''; }
        }

        async function agentLoadConfig() {
            try {
                await loadLmModelCatalog();
                const r = await fetch('/agent/config');
                if (!r.ok) return;
                const data = await r.json();
                const input = elAgentModelInput();
                if (input) {
                    setModelSelectOptions(input, data.model || data.default_model || '');
                    input.title = data.source === 'runtime_override'
                        ? `Runtime override active. Default: ${data.default_model || 'n/a'}`
                        : `Using default model: ${data.default_model || 'n/a'}`;
                }
            } catch(e) {
                console.warn('agent: failed to load config', e);
            }
        }

        async function agentSaveConfig() {
            const input = elAgentModelInput();
            if (!input || _agentStreaming) return;
            const applyBtn = elAgentModelApplyBtn();
            const model = (input.value || '').trim();
            const prevDisabled = applyBtn ? applyBtn.disabled : false;
            if (applyBtn) applyBtn.disabled = true;
            try {
                const r = await fetch('/agent/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model }),
                });
                const data = await r.json();
                if (!r.ok || data.error) {
                    throw new Error(data.error || 'Failed to save agent model');
                }
                if (input) {
                    setModelSelectOptions(input, data.model || model || data.default_model || '');
                    input.title = data.source === 'runtime_override'
                        ? `Runtime override active. Default: ${data.default_model || 'n/a'}`
                        : `Using default model: ${data.default_model || 'n/a'}`;
                }
                appendAgentNotice(`Agent model set to ${data.model || model || 'default'}`, 'success');
            } catch (e) {
                appendErrorToMessages(`Failed to set agent model: ${e.message}`);
            } finally {
                if (applyBtn) applyBtn.disabled = prevDisabled;
            }
        }

        // ---- Session list ----
        async function agentLoadSessions() {
            try {
                const r = await fetch('/agent/sessions');
                if (!r.ok) return;
                const data = await r.json();
                renderSessionList(data.sessions || []);
            } catch(e) {
                console.warn('agent: failed to load sessions', e);
            }
        }

        function renderSessionList(sessions) {
            const el = elSessionList();
            if (!el) return;
            if (!sessions.length) {
                el.innerHTML = '<div class="agent-session-empty">No sessions yet</div>';
                return;
            }
            el.innerHTML = sessions.map(s => {
                const active = s.id === _agentCurrentSession ? ' active' : '';
                const title = escapeHtml(s.title || 'Untitled session');
                const meta = `${fmtDate(s.updated_at)} · ${s.message_count || 0} msg`;
                return `<div class="agent-session-item${active}" data-sid="${escapeHtml(s.id)}">
                    <div class="si-title">${title}</div>
                    <div class="si-meta">
                        <span>${meta}</span>
                        <button class="si-delete" data-sid="${escapeHtml(s.id)}" title="Delete">&#x2715;</button>
                    </div>
                </div>`;
            }).join('');

            el.querySelectorAll('.agent-session-item').forEach(item => {
                item.addEventListener('click', (e) => {
                    if (e.target.closest('.si-delete')) return;
                    agentOpenSession(item.dataset.sid);
                });
            });
            el.querySelectorAll('.si-delete').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    agentDeleteSession(btn.dataset.sid);
                });
            });
        }

        async function agentOpenSession(sessionId) {
            if (_agentStreaming) return;
            try {
                const r = await fetch(`/agent/session/${sessionId}`);
                if (!r.ok) return;
                const data = await r.json();
                _agentCurrentSession = sessionId;
                localStorage.setItem(AGENT_LS_SESSION, sessionId);
                renderMessages(data.messages || []);
                highlightActiveSession(sessionId);
            } catch(e) {
                console.warn('agent: failed to open session', e);
            }
        }

        async function agentDeleteSession(sessionId) {
            try {
                const r = await fetch(`/agent/session/${sessionId}`, { method: 'DELETE' });
                if (!r.ok) return;
                if (_agentCurrentSession === sessionId) {
                    _agentCurrentSession = null;
                    localStorage.removeItem(AGENT_LS_SESSION);
                    showWelcome();
                }
                await agentLoadSessions();
            } catch(e) {
                console.warn('agent: failed to delete session', e);
            }
        }

        function highlightActiveSession(sessionId) {
            const el = elSessionList();
            if (!el) return;
            el.querySelectorAll('.agent-session-item').forEach(item => {
                item.classList.toggle('active', item.dataset.sid === sessionId);
            });
        }

        // ---- Messages ----
        function showWelcome() {
            const el = elMessages();
            if (!el) return;
            el.innerHTML = `<div class="agent-msg-welcome">
                <div class="agent-msg-welcome-title">EVA Agent</div>
                <div class="agent-msg-welcome-sub">Ask me about your camera streams, detections, and probes. I can search archives, analyze detections, and tune probe settings.</div>
            </div>`;
        }

        function renderMessages(messages) {
            const el = elMessages();
            if (!el) return;
            el.innerHTML = '';
            for (const msg of messages) {
                if (msg.role === 'user') {
                    appendUserBubble(msg.content, msg.created_at);
                } else if (msg.role === 'assistant') {
                    appendAssistantBubble(msg.content, msg.created_at);
                }
            }
            scrollToBottom(true);
        }

        function appendUserBubble(text, ts, imageB64) {
            const el = elMessages();
            if (!el) return;
            const div = document.createElement('div');
            div.className = 'agent-message user';
            let bodyContent = '';
            if (imageB64) {
                bodyContent += `<img class="agent-msg-image" src="data:image/jpeg;base64,${imageB64}" alt="attached image" />`;
            }
            bodyContent += escapeHtml(text);
            div.innerHTML = `<div class="agent-msg-header"><span class="agent-msg-ts">${fmtTime(ts || new Date().toISOString())}</span> Operator</div>
                <div class="agent-msg-body">${bodyContent}</div>`;
            el.appendChild(div);
            scrollToBottom(true);
        }

        function appendAssistantBubble(text, ts) {
            const el = elMessages();
            if (!el) return;
            const div = document.createElement('div');
            div.className = 'agent-message assistant';
            const bodyEl = document.createElement('div');
            bodyEl.className = 'agent-msg-body';
            const textEl = document.createElement('div');
            textEl.className = 'agent-msg-text';
            if (text) textEl.innerHTML = renderMarkdown ? renderMarkdown(text) : escapeHtml(text);
            bodyEl.appendChild(textEl);
            div.innerHTML = `<div class="agent-msg-header">EVA Agent <span class="agent-msg-ts">${fmtTime(ts || new Date().toISOString())}</span></div>`;
            div.appendChild(bodyEl);
            el.appendChild(div);
            scrollToBottom(true);
            return { el: div, bodyEl, textEl, traceEl: null, actionsEl: null, actionCount: 0, text: text || '' };
        }

        function isAgentNearBottom(threshold = 72) {
            const el = elMessages();
            if (!el) return true;
            return (el.scrollTop + el.clientHeight) >= (el.scrollHeight - threshold);
        }

        function setStreamingStatus(bubble, message, mode = 'thinking') {
            if (!bubble || !bubble.textEl || String(bubble.text || '').trim()) return;
            const safeMessage = escapeHtml(message || 'Thinking...');
            bubble.textEl.innerHTML = `<span class="agent-typing-indicator agent-typing-indicator-${mode}"><span class="agent-typing-dot"></span><span class="agent-typing-dot"></span><span class="agent-typing-dot"></span><span class="agent-typing-label">${safeMessage}</span></span>`;
        }

        function startStreamingBubble() {
            const el = elMessages();
            if (!el) return null;
            // Remove welcome if present
            const welcome = el.querySelector('.agent-msg-welcome');
            if (welcome) welcome.remove();

            const div = document.createElement('div');
            div.className = 'agent-message assistant';
            const bodyEl = document.createElement('div');
            bodyEl.className = 'agent-msg-body';
            const textEl = document.createElement('div');
            textEl.className = 'agent-msg-text';
            const traceEl = document.createElement('details');
            traceEl.className = 'agent-tool-trace';
            traceEl.open = true;
            const traceSummary = document.createElement('summary');
            traceSummary.className = 'agent-tool-trace-summary';
            traceSummary.textContent = 'Research trace';
            const actionsEl = document.createElement('div');
            actionsEl.className = 'agent-msg-actions';
            traceEl.appendChild(traceSummary);
            traceEl.appendChild(actionsEl);
            bodyEl.appendChild(textEl);
            bodyEl.appendChild(traceEl);
            div.innerHTML = `<div class="agent-msg-header">EVA Agent <span class="agent-msg-ts">${fmtTime(new Date().toISOString())}</span></div>`;
            div.appendChild(bodyEl);
            el.appendChild(div);
            const bubble = { el: div, bodyEl, textEl, traceEl, actionsEl, actionCount: 0, text: '' };
            setStreamingStatus(bubble, 'Thinking through the request...', 'thinking');
            scrollToBottom(true);
            return bubble;
        }

        function appendTokenToBubble(bubble, token) {
            const stickToBottom = isAgentNearBottom();
            bubble.text = (bubble.text || '') + token;
            const rendered = renderMarkdown ? renderMarkdown(bubble.text) : escapeHtml(bubble.text);
            if (bubble.textEl) {
                bubble.textEl.innerHTML = rendered;
            } else {
                bubble.bodyEl.innerHTML = rendered;
            }
            scrollToBottom(stickToBottom);
        }

        function appendActionCard(bubble, name, result) {
            const stickToBottom = isAgentNearBottom();
            const card = buildActionCard(name, result);
            if (!card) return;
            if (bubble.actionsEl) {
                bubble.actionsEl.appendChild(card);
                bubble.actionCount = (bubble.actionCount || 0) + 1;
                updateAgentTraceSummary(bubble);
            } else {
                bubble.bodyEl.appendChild(card);
            }
            scrollToBottom(stickToBottom);
        }

        function appendProgressNote(bubble, evt) {
            if (!bubble || !bubble.actionsEl) return;
            const stickToBottom = isAgentNearBottom();
            const note = document.createElement('div');
            note.className = 'agent-progress-note';
            const message = evt && evt.message ? String(evt.message) : 'Working...';
            note.innerHTML = `<span class="agent-progress-badge">In Progress</span><span class="agent-progress-text">${escapeHtml(message)}</span>`;
            bubble.actionsEl.appendChild(note);
            bubble.actionCount = (bubble.actionCount || 0) + 1;
            updateAgentTraceSummary(bubble);
            setStreamingStatus(bubble, message, 'working');
            scrollToBottom(stickToBottom);
        }

        function updateAgentTraceSummary(bubble) {
            const summaryEl = bubble && bubble.traceEl
                ? bubble.traceEl.querySelector('.agent-tool-trace-summary')
                : null;
            if (!summaryEl) return;
            const count = Number(bubble.actionCount || 0);
            summaryEl.textContent = count > 0 ? `Research trace · ${count} step${count === 1 ? '' : 's'}` : 'Research trace';
        }

        function scrollToBottom(force = false) {
            const el = elMessages();
            if (!el) return;
            if (!force && !isAgentNearBottom()) return;
            const bubble = _agentPendingBubble;
            const traceExpanded = Boolean(bubble && bubble.traceEl && bubble.traceEl.open);
            if (!force && traceExpanded && !isAgentNearBottom()) return;
            el.scrollTop = el.scrollHeight;
        }

        // ---- Action card builders ----
        // Helper: build a thumbnail element using image_url (backend-provided) or fallback
        function _makeThumb(item, cls, scoreVal) {
            const div = document.createElement('div');
            div.className = cls;
            const url = item.image_url || null;
            const score = scoreVal != null ? String(scoreVal) : '';
            const previewTitle = item.filename || item.name || item.path || item.image_path || item.id || 'Preview';
            if (url) {
                div.dataset.previewImage = String(url);
                div.dataset.previewTitle = String(previewTitle);
                div.title = String(previewTitle);
            }
            if (url) {
                div.innerHTML = `<img src="${escapeHtml(url)}" alt="" loading="lazy" />${score ? `<div class="${cls === 'agent-det-thumb' ? 'agent-det-score' : 'agent-search-score'}">${escapeHtml(score)}</div>` : ''}`;
            } else {
                div.textContent = item.id ? `#${item.id}` : '—';
                if (score) {
                    const badge = document.createElement('div');
                    badge.className = cls === 'agent-det-thumb' ? 'agent-det-score' : 'agent-search-score';
                    badge.textContent = score;
                    div.appendChild(badge);
                }
            }
            return div;
        }

        function appendApprovalControl(card, toolName, result) {
            const approval = result && result.approval;
            const planId = approval && approval.plan_id;
            if (!planId) return;
            const footer = document.createElement('div');
            footer.className = 'agent-approval-footer';
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'agent-approval-apply';
            button.textContent = 'Apply';
            button.addEventListener('click', async () => {
                if (button.disabled) return;
                button.disabled = true;
                button.textContent = 'Applying';
                try {
                    const body = {};
                    if (_agentCurrentSession) body.session_id = _agentCurrentSession;
                    const response = await fetch(`/agent/action-plans/${encodeURIComponent(planId)}/execute`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(body),
                    });
                    const data = await response.json().catch(() => ({}));
                    if (!response.ok || !data.success) {
                        throw new Error(data.error || `Server error ${response.status}`);
                    }
                    button.textContent = 'Applied';
                    footer.classList.add('is-applied');
                    const appliedCard = buildActionCard(toolName, data.result);
                    if (appliedCard) card.after(appliedCard);
                    if (['update_probe', 'create_probe', 'delete_probes'].includes(toolName)) {
                        void loadProbeList();
                    }
                } catch (err) {
                    button.disabled = false;
                    button.textContent = 'Apply';
                    appendAgentNotice(`Apply failed: ${err.message}`, 'error');
                }
            });
            footer.appendChild(button);
            card.appendChild(footer);
        }

        function buildActionCard(toolName, result) {
            const card = document.createElement('div');
            card.className = 'agent-action-card';

            if (toolName === 'search_archive') {
                // backend returns result.results (not result.hits)
                const hits = (result && (result.results || result.hits)) || [];
                const count = result && result.count != null ? result.count : hits.length;
                const scope = (result && result.scope) || '';
                const label = `SEARCH — ${count} result${count !== 1 ? 's' : ''}${scope ? ' · ' + scope : ''}`;
                card.innerHTML = `<div class="agent-action-card-head">&#9670; ${escapeHtml(label)}</div>`;
                const body = document.createElement('div');
                body.className = 'agent-action-card-body';
                if (hits.length) {
                    const grid = document.createElement('div');
                    grid.className = 'agent-search-results-grid';
                    hits.slice(0, 8).forEach(h => {
                        const score = h.score != null ? (h.score * 100).toFixed(0) + '%' : '';
                        grid.appendChild(_makeThumb(h, 'agent-search-thumb', score || null));
                    });
                    body.appendChild(grid);
                    if (hits.length > 8) {
                        const more = document.createElement('div');
                        more.style.cssText = 'margin-top:6px;font-size:12px;color:var(--muted)';
                        more.textContent = `+${hits.length - 8} more results`;
                        body.appendChild(more);
                    }
                } else {
                    body.innerHTML = '<div style="font-size:13px;color:var(--muted)">No results found.</div>';
                }
                card.appendChild(body);

            } else if (toolName === 'get_detections') {
                const detections = (result && result.detections) || [];
                const total = result && result.total_in_window;
                const label = total != null
                    ? `DETECTIONS — ${detections.length} shown of ${total} total`
                    : `DETECTIONS — ${detections.length} found`;
                card.innerHTML = `<div class="agent-action-card-head">&#9670; ${escapeHtml(label)}</div>`;
                const body = document.createElement('div');
                body.className = 'agent-action-card-body';
                if (detections.length) {
                    const grid = document.createElement('div');
                    grid.className = 'agent-det-grid';
                    detections.slice(0, 8).forEach(d => {
                        const score = d.score != null ? d.score.toFixed(3) : (d.margin != null ? d.margin.toFixed(3) : null);
                        grid.appendChild(_makeThumb(d, 'agent-det-thumb', score));
                    });
                    body.appendChild(grid);
                    if (detections.length > 8) {
                        const more = document.createElement('div');
                        more.style.cssText = 'margin-top:6px;font-size:12px;color:var(--muted)';
                        more.textContent = `+${detections.length - 8} more`;
                        body.appendChild(more);
                    }
                } else {
                    body.innerHTML = '<div style="font-size:13px;color:var(--muted)">No detections found.</div>';
                }
                card.appendChild(body);

            } else if (toolName === 'get_detection_summary') {
                const byProbe = (result && result.by_probe) || [];
                const total = (result && result.total_detections) || 0;
                const label = `DETECTIONS SUMMARY — ${total} total`;
                card.innerHTML = `<div class="agent-action-card-head">&#9670; ${escapeHtml(label)}</div>`;
                const body = document.createElement('div');
                body.className = 'agent-action-card-body';
                if (byProbe.length) {
                    let rows = byProbe.map(p => {
                        const name = escapeHtml(p.probe_name || p.probe_id || '?');
                        const hits = p.hit_count || 0;
                        return `<div class="agent-probe-update-field"><span class="agent-probe-update-key">${name}</span><span class="agent-probe-update-val">${hits} hits</span></div>`;
                    }).join('');
                    body.innerHTML = `<div class="agent-probe-update-row">${rows}</div>`;
                } else {
                    body.innerHTML = '<div style="font-size:13px;color:var(--muted)">No data.</div>';
                }
                card.appendChild(body);

            } else if (toolName === 'list_channels') {
                const channels = (result && result.channels) || [];
                const label = `CHANNELS — ${channels.length} available`;
                card.innerHTML = `<div class="agent-action-card-head">&#9670; ${escapeHtml(label)}</div>`;
                const body = document.createElement('div');
                body.className = 'agent-action-card-body';
                body.innerHTML = channels.length
                    ? channels.map((channel) => `<div class="agent-summary-entry"><span class="agent-summary-ts">CH ${escapeHtml(String(channel.id ?? '?'))}</span><span class="agent-summary-text">${escapeHtml(channel.title || 'Unnamed channel')}</span></div>`).join('')
                    : '<div style="font-size:13px;color:var(--muted)">No channels found.</div>';
                card.appendChild(body);

            } else if (toolName === 'list_probes') {
                const probes = (result && result.probes) || [];
                const label = `PROBES — ${probes.length} configured`;
                card.innerHTML = `<div class="agent-action-card-head">&#9670; ${escapeHtml(label)}</div>`;
                const body = document.createElement('div');
                body.className = 'agent-action-card-body';
                body.innerHTML = probes.length
                    ? probes.map((probe) => `<div class="agent-summary-entry"><span class="agent-summary-ts">${escapeHtml(probe.name || probe.id || '?')}</span><span class="agent-summary-text">CH ${escapeHtml(String(probe.channel_id ?? '?'))} · ${escapeHtml(String(probe.hit_count_24h ?? 0))} hits/24h</span></div>`).join('')
                    : '<div style="font-size:13px;color:var(--muted)">No probes configured.</div>';
                card.appendChild(body);

            } else if (toolName === 'survey_channels') {
                const channels = (result && result.channels) || [];
                const fastMode = Boolean(result && result.fast_mode);
                const label = `CHANNEL SURVEY${fastMode ? ' · FAST' : ''} — ${channels.length} channel${channels.length === 1 ? '' : 's'}`;
                card.innerHTML = `<div class="agent-action-card-head">&#9670; ${escapeHtml(label)}</div>`;
                const body = document.createElement('div');
                body.className = 'agent-action-card-body';
                body.innerHTML = channels.length
                    ? channels.map((channel) => {
                        const head = `CH ${channel.channel_id ?? '?'} · ${channel.title || 'Unnamed channel'}`;
                        const text = channel.error || channel.survey || 'No survey output.';
                        return `<div class="agent-summary-entry"><span class="agent-summary-ts">${escapeHtml(head)}</span><span class="agent-summary-text">${escapeHtml(text)}</span></div>`;
                    }).join('')
                    : '<div style="font-size:13px;color:var(--muted)">No survey data.</div>';
                card.appendChild(body);

            } else if (toolName === 'create_probe') {
                const isPreview = result && result.status === 'preview';
                const action = String(result && result.action || '');
                const label = isPreview
                    ? (action === 'update_existing' ? 'PROBE UPSERT PREVIEW' : 'PROBE CREATE PREVIEW')
                    : (action === 'update_existing' ? 'PROBE UPDATED VIA CREATE' : 'PROBE CREATED');
                card.innerHTML = `<div class="agent-action-card-head">&#9670; ${escapeHtml(label)}</div>`;
                const body = document.createElement('div');
                body.className = 'agent-action-card-body';
                const probe = (result && (result.proposed || result.probe)) || {};
                const conflicts = (result && result.conflicts) || [];
                let html = `<div class="agent-probe-update-row"><div class="agent-probe-update-field"><span class="agent-probe-update-key">Probe:</span><span class="agent-probe-update-val">${escapeHtml(probe.name || result.probe_name || 'unknown')}</span></div></div>`;
                if (action === 'update_existing') {
                    html += `<div style="margin-top:8px;font-size:12px;color:var(--muted)">Existing probe on this channel will be reused instead of creating a duplicate.</div>`;
                }
                if (conflicts.length) {
                    html += `<div style="margin-top:8px;font-size:12px;color:var(--warn)">Potential conflicts: ${escapeHtml(conflicts.map((item) => item.name || item.id || '?').join(', '))}</div>`;
                }
                body.innerHTML = html;
                card.appendChild(body);

            } else if (toolName === 'deploy_summary') {
                card.innerHTML = `<div class="agent-action-card-head">&#9670; DEPLOY SUMMARY</div>`;
                const body = document.createElement('div');
                body.className = 'agent-action-card-body';
                const mode = escapeHtml(String(result && result.mode || 'standard'));
                const wipe = Boolean(result && result.wipe);
                const overview = escapeHtml(String(result && result.overview || 'Deployment summary recorded.'));
                const elapsed = result && Number.isFinite(Number(result.elapsed_sec)) ? ` · ${Number(result.elapsed_sec).toFixed(1)}s` : '';
                const channels = Array.isArray(result && result.channels) ? result.channels : [];
                const probes = Array.isArray(result && result.probes) ? result.probes : [];
                const prompts = Array.isArray(result && result.prompt_targets) ? result.prompt_targets : [];
                const notes = Array.isArray(result && result.notes) ? result.notes : [];
                body.innerHTML = `
                    <div class="agent-summary-entry"><span class="agent-summary-ts">${mode.toUpperCase()}${wipe ? ' · WIPE' : ''}${elapsed}</span><span class="agent-summary-text">${overview}</span></div>
                    ${channels.length ? `<div class="agent-summary-entry"><span class="agent-summary-ts">CHANNELS</span><span class="agent-summary-text">${escapeHtml(channels.join(' · '))}</span></div>` : ''}
                    ${probes.length ? `<div class="agent-summary-entry"><span class="agent-summary-ts">PROBES</span><span class="agent-summary-text">${escapeHtml(probes.join(' · '))}</span></div>` : ''}
                    ${prompts.length ? `<div class="agent-summary-entry"><span class="agent-summary-ts">PROMPTS</span><span class="agent-summary-text">${escapeHtml(prompts.join(' · '))}</span></div>` : ''}
                    ${notes.length ? `<div class="agent-summary-entry"><span class="agent-summary-ts">NOTES</span><span class="agent-summary-text">${escapeHtml(notes.join(' · '))}</span></div>` : ''}
                `;
                card.appendChild(body);

            } else if (toolName === 'delete_probes') {
                const isPreview = result && result.status === 'preview';
                const targets = (result && result.targets) || [];
                const label = isPreview ? 'PROBE DELETE PREVIEW' : 'PROBES DELETED';
                card.innerHTML = `<div class="agent-action-card-head">&#9670; ${escapeHtml(label)}</div>`;
                const body = document.createElement('div');
                body.className = 'agent-action-card-body';
                body.innerHTML = targets.length
                    ? targets.map((probe) => `<div class="agent-summary-entry"><span class="agent-summary-ts">${escapeHtml(probe.name || probe.id || '?')}</span><span class="agent-summary-text">CH ${escapeHtml(String(probe.channel_id ?? '?'))}</span></div>`).join('')
                    : '<div style="font-size:13px;color:var(--muted)">No probes selected.</div>';
                card.appendChild(body);

            } else if (toolName === 'update_probe') {
                const isPreview = result && result.status === 'preview';
                const label = isPreview ? 'PROBE UPDATE PREVIEW' : 'PROBE UPDATED';
                card.innerHTML = `<div class="agent-action-card-head">&#9670; ${label}</div>`;
                const body = document.createElement('div');
                body.className = 'agent-action-card-body';
                const probeName = (result && result.probe_name) || 'unknown';
                const diff = (result && result.diff) || {};
                let html = `<div class="agent-probe-update-row"><div class="agent-probe-update-field"><span class="agent-probe-update-key">Probe:</span><span class="agent-probe-update-val">${escapeHtml(probeName)}</span></div>`;
                for (const [k, v] of Object.entries(diff)) {
                    const val = typeof v === 'object' ? JSON.stringify(v) : String(v);
                    html += `<div class="agent-probe-update-field"><span class="agent-probe-update-key">${escapeHtml(k)}:</span><span class="agent-probe-update-val">${escapeHtml(val)}</span></div>`;
                }
                html += '</div>';
                if (isPreview) {
                    html += `<div style="margin-top:8px;font-size:12px;color:var(--muted)">Preview only — confirm to apply.</div>`;
                }
                body.innerHTML = html;
                card.appendChild(body);

            } else if (toolName === 'update_prompt_settings') {
                const isPreview = result && result.status === 'preview';
                const label = isPreview ? 'PROMPT SETTINGS PREVIEW' : 'PROMPT SETTINGS UPDATED';
                card.innerHTML = `<div class="agent-action-card-head">&#9670; ${escapeHtml(label)}</div>`;
                const body = document.createElement('div');
                body.className = 'agent-action-card-body';
                const diff = (result && result.diff) || {};
                const rows = Object.keys(diff);
                body.innerHTML = rows.length
                    ? rows.map((key) => `<div class="agent-probe-update-field"><span class="agent-probe-update-key">${escapeHtml(key)}</span><span class="agent-probe-update-val">updated</span></div>`).join('')
                    : '<div style="font-size:13px;color:var(--muted)">No prompt changes.</div>';
                card.appendChild(body);

            } else if (toolName === 'describe_frame') {
                const source = (result && result.source) || '';
                const chId = result && result.channel_id;
                const headLabel = chId != null
                    ? `FRAME — CH ${chId} (LIVE)`
                    : 'FRAME DESCRIPTION';
                card.innerHTML = `<div class="agent-action-card-head">&#9670; ${escapeHtml(headLabel)}</div>`;
                const body = document.createElement('div');
                body.className = 'agent-action-card-body agent-describe-layout';
                const b64 = result && result.snapshot_b64;
                const imgPath = result && result.image_path;
                let imgSrc = null;
                if (b64) {
                    imgSrc = `data:image/jpeg;base64,${b64}`;
                } else if (imgPath) {
                    imgSrc = `/detections/image?image_path=${encodeURIComponent(imgPath)}`;
                }
                const desc = (result && result.description) || '';
                if (imgSrc) {
                    body.innerHTML = `<div class="agent-frame-img-wrap"><img class="agent-frame-img" src="${escapeHtml(imgSrc)}" alt="analyzed frame" data-preview-image="${escapeHtml(imgSrc)}" data-preview-title="Analyzed frame" /></div><div class="agent-describe-block">${escapeHtml(desc)}</div>`;
                } else {
                    body.innerHTML = `<div class="agent-describe-block">${escapeHtml(desc)}</div>`;
                }
                card.appendChild(body);

            } else if (toolName === 'get_video_summaries') {
                const entries = (result && result.entries) || [];
                const depth = (result && result.depth) || '';
                const ch = (result && result.channel_id) || '';
                const label = `VIDEO SUMMARIES — CH ${ch} · ${depth} · ${entries.length} entries`;
                card.innerHTML = `<div class="agent-action-card-head">&#9670; ${escapeHtml(label)}</div>`;
                const body = document.createElement('div');
                body.className = 'agent-action-card-body';
                if (entries.length) {
                    body.innerHTML = entries.map(e => {
                        const t = escapeHtml(e.time || '');
                        const s = escapeHtml(e.summary || '');
                        return `<div class="agent-summary-entry"><span class="agent-summary-ts">${t}</span><span class="agent-summary-text">${s}</span></div>`;
                    }).join('');
                } else {
                    body.innerHTML = '<div style="font-size:13px;color:var(--muted)">No summaries in this time range.</div>';
                }
                card.appendChild(body);

            } else if (toolName === 'create_bookmark') {
                card.innerHTML = `<div class="agent-action-card-head">&#9670; BOOKMARK CREATED</div>`;
                const body = document.createElement('div');
                body.className = 'agent-action-card-body';
                const bname = (result && (result.title || result.name || result.id)) || 'Bookmark';
                const bsev = (result && result.severity) || '';
                const bch = (result && result.channel_id) != null ? ` · CH ${result.channel_id}` : '';
                body.innerHTML = `<div class="agent-bookmark-row">
                    <span class="agent-bookmark-badge">${escapeHtml(bsev || 'bookmark')}</span>
                    <span>${escapeHtml(String(bname))}${escapeHtml(bch)}</span>
                </div>`;
                card.appendChild(body);

            } else if (toolName === 'generate_report') {
                card.innerHTML = `<div class="agent-action-card-head">&#9670; REPORT</div>`;
                const body = document.createElement('div');
                body.className = 'agent-action-card-body';
                const text = (result && (result.report || result.text || result.content)) || JSON.stringify(result, null, 2);
                body.innerHTML = `<div class="agent-report-block">${escapeHtml(text)}</div>`;
                card.appendChild(body);

            } else {
                // Generic fallback
                card.innerHTML = `<div class="agent-action-card-head">&#9670; ${escapeHtml(toolName.toUpperCase())}</div>`;
                const body = document.createElement('div');
                body.className = 'agent-action-card-body';
                body.innerHTML = `<div style="font-size:12px;color:var(--muted);white-space:pre-wrap;font-family:monospace">${escapeHtml(JSON.stringify(result, null, 2))}</div>`;
                card.appendChild(body);
            }

            appendApprovalControl(card, toolName, result);
            return card;
        }

        // ---- SSE chat ----
        async function agentSend(message) {
            if (_agentStreaming || !message.trim()) return;
            _agentStreaming = true;

            // Capture and clear pending image
            const imageB64 = _agentPendingImageB64 || null;
            _agentPendingImageB64 = null;
            clearImageAttachment();

            const sendBtn = elSendBtn();
            const inputEl = elInput();
            if (sendBtn) sendBtn.disabled = true;
            if (inputEl) { inputEl.disabled = true; inputEl.style.height = ''; }

            // Remove welcome screen and show user bubble
            const msgEl = elMessages();
            if (msgEl) {
                const welcome = msgEl.querySelector('.agent-msg-welcome');
                if (welcome) welcome.remove();
            }
            appendUserBubble(message, null, imageB64);

            // Start streaming assistant bubble
            _agentPendingBubble = startStreamingBubble();

            try {
                const body = { message };
                if (_agentCurrentSession) body.session_id = _agentCurrentSession;
                if (imageB64) body.image_b64 = imageB64;

                const streamController = new AbortController();
                const r = await fetch('/agent/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                    signal: streamController.signal
                });

                if (!r.ok) {
                    const errText = await r.text().catch(() => r.statusText);
                    finishStreamingBubble(_agentPendingBubble, null);
                    appendErrorToMessages(`Server error ${r.status}: ${errText}`);
                    return;
                }

                const reader = r.body.getReader();
                const decoder = new TextDecoder();
                let buf = '';
                let newSessionId = null;
                let sawDoneEvent = false;

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buf += decoder.decode(value, { stream: true });
                    // Parse SSE lines
                    const lines = buf.split('\n');
                    buf = lines.pop(); // keep incomplete last line
                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        const raw = line.slice(6).trim();
                        if (!raw) continue;
                        let evt;
                        try { evt = JSON.parse(raw); } catch(_) { continue; }
                        handleAgentEvent(evt, _agentPendingBubble);
                        if (evt.type === 'session' && evt.session_id) {
                            newSessionId = evt.session_id;
                        }
                        if (evt.type === 'done' && evt.session_id) {
                            newSessionId = evt.session_id;
                            sawDoneEvent = true;
                        }
                        if (evt.type === 'done') {
                            sawDoneEvent = true;
                        }
                    }
                    if (sawDoneEvent) {
                        try { reader.cancel().catch(() => {}); } catch(_) {}
                        try { streamController.abort(); } catch(_) {}
                        break;
                    }
                }

                finishStreamingBubble(_agentPendingBubble);
                if (newSessionId) {
                    _agentCurrentSession = newSessionId;
                    localStorage.setItem(AGENT_LS_SESSION, newSessionId);
                    highlightActiveSession(newSessionId);
                    void agentLoadSessions().then(() => highlightActiveSession(newSessionId));
                }

            } catch(e) {
                console.error('agent: stream error', e);
                finishStreamingBubble(_agentPendingBubble, null);
                appendErrorToMessages(`Connection error: ${e.message}`);
            } finally {
                _agentStreaming = false;
                _agentPendingBubble = null;
                if (sendBtn) sendBtn.disabled = false;
                if (inputEl) { inputEl.disabled = false; inputEl.focus(); }
            }
        }

        function handleAgentEvent(evt, bubble) {
            if (!bubble) return;
            switch (evt.type) {
                case 'token':
                case 'text':
                    setStreamingStatus(bubble, 'Writing response...', 'writing');
                    if (evt.content) appendTokenToBubble(bubble, evt.content);
                    break;
                case 'tool_call':
                    if (evt.name) setStreamingStatus(bubble, `Running ${evt.name}...`, 'working');
                    break;
                case 'tool_result':
                    appendActionCard(bubble, evt.name, evt.result);
                    if (
                        ((evt.name === 'update_probe') || (evt.name === 'create_probe') || (evt.name === 'delete_probes'))
                        && evt.result && evt.result.status === 'applied'
                    ) {
                        void loadProbeList();
                    }
                    break;
                case 'tool_progress':
                    appendProgressNote(bubble, evt);
                    break;
                case 'error':
                    appendErrorToMessages(evt.message || 'Unknown error');
                    break;
                case 'heartbeat':
                    setStreamingStatus(bubble, 'Still working...', 'thinking');
                    break;
                case 'tool_start':
                case 'done':
                    break;
            }
        }

        function finishStreamingBubble(bubble) {
            if (!bubble) return;
            // Remove typing indicator if still present (no tokens came)
            const indicator = (bubble.textEl || bubble.bodyEl).querySelector('.agent-typing-indicator');
            if (indicator) indicator.remove();
            updateAgentTraceSummary(bubble);
            if (bubble.traceEl) {
                const hasText = String(bubble.text || '').trim().length > 0;
                const hasActions = Number(bubble.actionCount || 0) > 0;
                bubble.traceEl.open = hasActions && !hasText;
            }
        }

        function appendAgentNotice(msg, tone = 'info') {
            const el = elMessages();
            if (!el) return;
            const stickToBottom = isAgentNearBottom();
            const div = document.createElement('div');
            div.className = `agent-inline-msg ${tone}`;
            div.textContent = msg;
            el.appendChild(div);
            scrollToBottom(stickToBottom);
        }

        function appendErrorToMessages(msg) {
            appendAgentNotice(msg, 'error');
        }

        // ---- Probe list (context sidebar) ----
        async function agentLoadProbes() {
            const el = elProbeList();
            if (!el) return;
            try {
                const r = await fetch(`/probes/list?t=${Date.now()}`, { cache: 'no-store' });
                if (!r.ok) return;
                const data = await r.json();
                const probes = data.probes || [];
                if (!probes.length) {
                    el.innerHTML = '<div class="agent-probe-empty">No probes configured</div>';
                    return;
                }
                el.innerHTML = probes.map(p => {
                    const running = p.status === 'running' || p.enabled !== false;
                    const dotCls = running ? 'on' : 'off';
                    const score = p.last_score != null ? p.last_score.toFixed(3) : '—';
                    return `<div class="agent-probe-mini">
                        <div class="agent-probe-dot ${dotCls}"></div>
                        <span class="agent-probe-name">${escapeHtml(p.name || 'unnamed')}</span>
                        <span class="agent-probe-score">${score}</span>
                    </div>`;
                }).join('');
            } catch(e) {
                el.innerHTML = '<div class="agent-probe-empty">Failed to load probes</div>';
            }
        }

        function closeAgentSkillModal() {
            if (agentSkillModal) agentSkillModal.style.display = 'none';
            agentSkillDraft = null;
        }

        function openAgentSkillModal(mode, skill = null) {
            agentSkillDraft = { mode, skill };
            if (agentSkillModalTitle) {
                agentSkillModalTitle.textContent = mode === 'create' ? 'Create Skill' : 'Edit Skill';
            }
            if (agentSkillNameInput) agentSkillNameInput.value = skill?.name || '';
            if (agentSkillSlugInput) {
                agentSkillSlugInput.value = skill?.slug || '';
                agentSkillSlugInput.disabled = mode !== 'create';
            }
            if (agentSkillMeta) {
                agentSkillMeta.textContent = mode === 'create'
                    ? 'Create a new playbook. It will immediately become available to the agent on the next message.'
                    : `Editing ${skill?.path || 'skill'}`;
            }
            if (agentSkillContentInput) {
                agentSkillContentInput.value = skill?.content || `# ${skill?.name || 'New Skill'}\n\nGoal: describe when this playbook should be used.\n\nDefault order:\n1. Clarify missing inputs if needed.\n2. Inspect the relevant context.\n3. Use the right tools in a safe order.\n4. Summarize the result for the operator.\n`;
            }
            if (agentSkillModal) agentSkillModal.style.display = 'block';
        }

        async function agentLoadSkills() {
            if (!agentSkillList) return;
            try {
                const r = await fetch('/agent/skills');
                if (!r.ok) return;
                const data = await r.json();
                const skills = Array.isArray(data.skills) ? data.skills : [];
                if (!skills.length) {
                    agentSkillList.innerHTML = '<div class="agent-probe-empty">No skills yet</div>';
                    return;
                }
                agentSkillList.innerHTML = skills.map((skill) => `
                    <div class="agent-skill-row" data-skill-slug="${escapeHtml(skill.slug || '')}">
                        <button
                            class="agent-skill-run"
                            type="button"
                            title="${escapeHtml(skill.summary || 'Run skill')}"
                            data-agent-skill-run="${escapeHtml(skill.slug || '')}">
                            <span class="agent-skill-run-title">${escapeHtml(skill.name || skill.slug || 'Unnamed skill')}</span>
                        </button>
                        <button
                            class="feature-btn agent-skill-edit"
                            type="button"
                            title="Edit skill"
                            aria-label="Edit skill ${escapeHtml(skill.name || skill.slug || 'Unnamed skill')}"
                            data-agent-skill-edit="${escapeHtml(skill.slug || '')}">&#9998;</button>
                    </div>
                `).join('');
            } catch(e) {
                agentSkillList.innerHTML = '<div class="agent-probe-empty">Failed to load skills</div>';
            }
        }

        async function agentOpenSkillEditor(slug) {
            try {
                const r = await fetch(`/agent/skills/${encodeURIComponent(slug)}`);
                const data = await r.json();
                if (!r.ok || data.error) throw new Error(data.error || 'Failed to load skill');
                openAgentSkillModal('edit', data);
            } catch (e) {
                appendErrorToMessages(`Failed to open skill: ${e.message}`);
            }
        }

        async function agentSaveSkill() {
            if (!agentSkillDraft) return;
            const payload = {
                name: (agentSkillNameInput?.value || '').trim(),
                slug: (agentSkillSlugInput?.value || '').trim(),
                content: agentSkillContentInput?.value || '',
            };
            try {
                const isCreate = agentSkillDraft.mode === 'create';
                const endpoint = isCreate
                    ? '/agent/skills/create'
                    : `/agent/skills/${encodeURIComponent(agentSkillDraft.skill.slug)}`;
                const r = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const data = await r.json();
                if (!r.ok || data.error) throw new Error(data.error || 'Failed to save skill');
                closeAgentSkillModal();
                await agentLoadSkills();
                appendAgentNotice(`Skill saved: ${(data.skill && data.skill.name) || payload.name || payload.slug}`, 'success');
            } catch (e) {
                appendErrorToMessages(`Failed to save skill: ${e.message}`);
            }
        }

        function agentRunSkill(slug) {
            const input = elInput();
            const existing = input ? (input.value || '').trim() : '';
            const prompt = existing
                ? `Use playbook "${slug}" for this operator request:\n${existing}`
                : `Use playbook "${slug}" for the current task. If required inputs are missing, ask only the minimum clarifying questions before acting.`;
            if (input) input.value = '';
            void agentSend(prompt);
        }

        function clearImageAttachment() {
            _agentPendingImageB64 = null;
            const preview = document.getElementById('agentImagePreview');
            const thumb = document.getElementById('agentImageThumb');
            if (preview) preview.classList.add('is-hidden');
            if (thumb) thumb.src = '';
        }

        // ---- Textarea auto-resize ----
        function setupTextarea() {
            const ta = elInput();
            if (!ta) return;
            function resize() {
                ta.style.height = 'auto';
                ta.style.height = Math.min(ta.scrollHeight, 110) + 'px';
            }
            ta.addEventListener('input', resize);
            ta.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    const msg = ta.value.trim();
                    if (msg) {
                        ta.value = '';
                        resize();
                        agentSend(msg);
                    }
                }
            });
        }

        // ---- Init ----
        function agentInit() {
            if (_agentInitDone) return;
            _agentInitDone = true;

            setupTextarea();
            void agentLoadConfig();
            void agentLoadSkills();

            const sendBtn = elSendBtn();
            if (sendBtn) {
                sendBtn.addEventListener('click', () => {
                    const ta = elInput();
                    const msg = ta ? ta.value.trim() : '';
                    if (msg) {
                        ta.value = '';
                        agentSend(msg);
                    }
                });
            }

            const newBtn = elNewSession();
            if (newBtn) {
                newBtn.addEventListener('click', () => {
                    if (_agentStreaming) return;
                    _agentCurrentSession = null;
                    localStorage.removeItem(AGENT_LS_SESSION);
                    showWelcome();
                    highlightActiveSession(null);
                });
            }

            const modelBtn = elAgentModelApplyBtn();
            if (modelBtn) {
                modelBtn.addEventListener('click', () => {
                    void agentSaveConfig();
                });
            }
            const modelInput = elAgentModelInput();
            if (modelInput) {
                modelInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        void agentSaveConfig();
                    }
                });
            }

            // Chip buttons (skip the file label)
            const box = document.getElementById('agentBox');
            if (box) {
                box.querySelectorAll('.agent-chip[data-prompt]').forEach(chip => {
                    chip.addEventListener('click', () => {
                        const ta = elInput();
                        if (ta && !_agentStreaming) {
                            ta.value = chip.dataset.prompt || chip.textContent;
                            ta.focus();
                        }
                    });
                });
            }

            if (agentSkillList) {
                agentSkillList.addEventListener('click', (event) => {
                    const target = event.target;
                    if (!(target instanceof Element)) return;
                    const runBtn = target.closest('[data-agent-skill-run]');
                    if (runBtn instanceof HTMLButtonElement) {
                        const slug = (runBtn.dataset.agentSkillRun || '').trim();
                        if (slug) agentRunSkill(slug);
                        return;
                    }
                    const editBtn = target.closest('[data-agent-skill-edit]');
                    if (editBtn instanceof HTMLButtonElement) {
                        const slug = (editBtn.dataset.agentSkillEdit || '').trim();
                        if (slug) void agentOpenSkillEditor(slug);
                    }
                });
            }

            const messagesEl = elMessages();
            if (messagesEl) {
                messagesEl.addEventListener('click', (event) => {
                    const target = event.target;
                    if (!(target instanceof Element)) return;
                    const previewEl = target.closest('[data-preview-image], .agent-search-thumb, .agent-det-thumb');
                    if (!(previewEl instanceof HTMLElement)) return;
                    let src = '';
                    let title = '';
                    if (previewEl.dataset.previewImage) {
                        src = previewEl.dataset.previewImage;
                        title = previewEl.dataset.previewTitle || previewEl.getAttribute('title') || '';
                    } else {
                        const img = previewEl.querySelector('img');
                        if (img) {
                            src = img.getAttribute('src') || '';
                            title = img.getAttribute('alt') || previewEl.getAttribute('title') || '';
                        }
                    }
                    if (!isPreviewableImageUrl(src)) return;
                    event.preventDefault();
                    openImageLightbox(src, title || 'Agent preview');
                });
            }

            if (agentCreateSkillBtn) {
                agentCreateSkillBtn.addEventListener('click', () => {
                    openAgentSkillModal('create', null);
                });
            }
            if (closeAgentSkillModalBtn) closeAgentSkillModalBtn.addEventListener('click', closeAgentSkillModal);
            if (agentSkillCancelBtn) agentSkillCancelBtn.addEventListener('click', closeAgentSkillModal);
            if (agentSkillSaveBtn) {
                agentSkillSaveBtn.addEventListener('click', () => {
                    void agentSaveSkill();
                });
            }

            // Image attachment
            const imageFileInput = q('agentImageFile');
            if (imageFileInput) {
                imageFileInput.addEventListener('change', () => {
                    const file = imageFileInput.files && imageFileInput.files[0];
                    if (!file) return;
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        // Strip data URI prefix, keep base64 only
                        const dataUrl = e.target.result || '';
                        const b64 = dataUrl.split(',')[1] || '';
                        if (b64) {
                            _agentPendingImageB64 = b64;
                            const preview = q('agentImagePreview');
                            const thumb = q('agentImageThumb');
                            if (preview) preview.classList.remove('is-hidden');
                            if (thumb) thumb.src = dataUrl;
                        }
                    };
                    reader.readAsDataURL(file);
                    // Reset so same file can be re-selected
                    imageFileInput.value = '';
                });
            }
            const imageClearBtn = q('agentImageClear');
            if (imageClearBtn) {
                imageClearBtn.addEventListener('click', clearImageAttachment);
            }

            // Try to restore last session
            const savedSession = localStorage.getItem(AGENT_LS_SESSION);

            agentLoadSessions().then(() => {
                if (savedSession) {
                    agentOpenSession(savedSession).catch(() => showWelcome());
                } else {
                    showWelcome();
                }
            });

            agentLoadProbes();
        }

        // Expose agentInit to outer scope
        window._agentInit = agentInit;
    })();

    function agentInit() {
        if (window._agentInit) window._agentInit();
    }
