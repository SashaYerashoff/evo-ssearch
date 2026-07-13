    const uiRuntimeParams = new URLSearchParams(window.location.search || '');
    const uiLiteMode = uiRuntimeParams.has('lite')
        || uiRuntimeParams.get('ui') === 'lite'
        || uiRuntimeParams.get('ui_lite') === '1';
    if (uiLiteMode) {
        document.documentElement.classList.add('ui-lite');
        const applyUiLiteClass = () => {
            if (document.body) document.body.classList.add('ui-lite');
        };
        applyUiLiteClass();
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', applyUiLiteClass, { once: true });
        }
    }

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
    const videoUploadInput = document.getElementById('videoUpload');
    const videoUploadName = document.getElementById('videoUploadName');
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
    const agentModelInput = document.getElementById('agentModelInput');
    const agentModelApplyBtn = document.getElementById('agentModelApplyBtn');
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
    const luxriotLiveIntervalInput = document.getElementById('luxriotLiveIntervalSec');
    const luxriotBatchInfo = document.getElementById('luxriotBatchInfo');
    const luxriotRuntimeConfigState = document.getElementById('luxriotRuntimeConfigState');
    const luxriotRuntimeConfigRunning = document.getElementById('luxriotRuntimeConfigRunning');
    const luxriotRuntimeConfigPending = document.getElementById('luxriotRuntimeConfigPending');
    const luxriotStatusLabel = document.getElementById('luxriotStatus');
    let luxriotPreviewImg = document.getElementById('luxriotPreview');
    const luxriotViewport = document.getElementById('luxriotViewport');
    const luxriotOverlay = document.getElementById('luxriotOverlay');
    const luxriotStreamName = document.getElementById('luxriotStreamName');
    const luxriotStreamState = document.getElementById('luxriotStreamState');
    const luxriotStreamChannel = document.getElementById('luxriotStreamChannel');
    const luxriotStreamResolution = document.getElementById('luxriotStreamResolution');
    const luxriotStreamCadence = document.getElementById('luxriotStreamCadence');
    const luxriotStreamBatch = document.getElementById('luxriotStreamBatch');
    const luxriotStreamModel = document.getElementById('luxriotStreamModel');
    const luxriotStreamQueue = document.getElementById('luxriotStreamQueue');
    const luxriotStreamProbesRow = document.getElementById('luxriotStreamProbesRow');
    const luxriotStreamProbes = document.getElementById('luxriotStreamProbes');
    const luxriotStreamLastFrame = document.getElementById('luxriotStreamLastFrame');
    const luxriotStreamDetail = document.getElementById('luxriotStreamDetail');
    const luxriotContextToggleCaptureBtn = document.getElementById('luxriotContextToggleCapture');
    const luxriotContextFlushCaptureBtn = document.getElementById('luxriotContextFlushCapture');
    const roadSceneGroundingBtns = Array.from(document.querySelectorAll('[data-road-scene-grounding]'));
    const roadSceneGroundingBtn = document.getElementById('roadSceneGroundingBtn') || roadSceneGroundingBtns[0] || null;
    const roadSceneGroundingPanel = document.getElementById('roadSceneGroundingPanel');
    const roadSceneGroundingImage = document.getElementById('roadSceneGroundingImage');
    const roadSceneGroundingTitle = document.getElementById('roadSceneGroundingTitle');
    const roadSceneGroundingConfidence = document.getElementById('roadSceneGroundingConfidence');
    const roadSceneGroundingMeta = document.getElementById('roadSceneGroundingMeta');
    const luxriotToggleCaptureBtn = document.getElementById('luxriotToggleCapture');
    const luxriotFlushCaptureBtn = document.getElementById('luxriotFlushCapture');
    const luxriotPromptSettingsBtn = document.getElementById('luxriotPromptSettingsBtn');
    const luxriotPromptModal = document.getElementById('luxriotPromptModal');
    const closeLuxriotPromptModalBtn = document.getElementById('closeLuxriotPromptModal');
    const luxriotPromptCloseBtn = document.getElementById('luxriotPromptCloseBtn');
    const luxriotPromptResetBtn = document.getElementById('luxriotPromptResetBtn');
    const luxriotPromptApplyBtn = document.getElementById('luxriotPromptApplyBtn');
    const luxriotPromptModalInput = document.getElementById('luxriotPromptModalInput');
    const luxriotPromptModalMeta = document.getElementById('luxriotPromptModalMeta');
    const luxriotPromptLayerDetails = document.getElementById('luxriotPromptLayerDetails');
    const luxriotPromptLayerContent = document.getElementById('luxriotPromptLayerContent');
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
    const luxriotSummaryDateNav = document.getElementById('luxriotSummaryDateNav');
    const luxriotSummaryDateLabel = document.getElementById('luxriotSummaryDateLabel');
    const luxriotSummaryPreviousPeriodBtn = document.getElementById('luxriotSummaryPreviousPeriod');
    const luxriotSummaryNextPeriodBtn = document.getElementById('luxriotSummaryNextPeriod');
    const luxriotSummaryLoadEarlierBtn = document.getElementById('luxriotSummaryLoadEarlierBtn');
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
    const luxriotAlertPolicyPromptInput = document.getElementById('luxriotAlertPolicyPrompt');
    const luxriotRollupPromptL1Input = document.getElementById('luxriotRollupPromptL1');
    const luxriotRollupPromptL2Input = document.getElementById('luxriotRollupPromptL2');
    const luxriotRollupPromptL3Input = document.getElementById('luxriotRollupPromptL3');
    const luxriotJsonAlertPromptInput = document.getElementById('luxriotJsonAlertPrompt');
    const luxriotBookmarkEnabledInput = document.getElementById('luxriotBookmarkEnabled');
    const luxriotBookmarkCooldownInput = document.getElementById('luxriotBookmarkCooldown');
    const luxriotSelectorBiasInput = document.getElementById('luxriotSelectorBias');
    const probeChannelSelect = document.getElementById('probeChannelSelect');
    const probeTopKInput = document.getElementById('probeTopK');
    const probePosFloorInput = document.getElementById('probePosFloor');
    const probeMarginInput = document.getElementById('probeMargin');
    const probeNameInput = document.getElementById('probeName');
    const probeRunBtn = document.getElementById('probeRunBtn');
    const probeSaveBtn = document.getElementById('probeSaveBtn');
    const probeCastBtn = document.getElementById('probeCastBtn');
    const probeDeleteBtn = document.getElementById('probeDeleteBtn');
    const probeEditBtn = document.getElementById('probeEditBtn');
    const probeEditorModal = document.getElementById('probeEditorModal');
    const closeProbeEditorBtn = document.getElementById('closeProbeEditor');
    const probeEditorCloseBtn = document.getElementById('probeEditorCloseBtn');
    const probeCastModal = document.getElementById('probeCastModal');
    const closeProbeCastBtn = document.getElementById('closeProbeCast');
    const probeCastCloseBtn = document.getElementById('probeCastCloseBtn');
    const probeCastApplyBtn = document.getElementById('probeCastApplyBtn');
    const probeCastChannelList = document.getElementById('probeCastChannelList');
    const probeCastSelectedMeta = document.getElementById('probeCastSelectedMeta');
    const probeCastStatus = document.getElementById('probeCastStatus');
    const probeCastAllBtn = document.getElementById('probeCastAllBtn');
    const probeCastNoneBtn = document.getElementById('probeCastNoneBtn');
    const probeCastCurrentBtn = document.getElementById('probeCastCurrentBtn');
    const probeCastConflictInput = document.getElementById('probeCastConflict');
    const probeCastEnableInput = document.getElementById('probeCastEnable');
    const probeCastCopyRoiInput = document.getElementById('probeCastCopyRoi');
    const probeCastStartStreamsInput = document.getElementById('probeCastStartStreams');
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
    let probePreviewImg = document.getElementById('probePreviewImg');
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
    const archiveProbeFilterGroup = archiveProbeFilter ? archiveProbeFilter.closest('.input-group') : null;
    const archiveSourceFilter = document.getElementById('archiveSourceFilter');
    const archiveTimeFilter = document.getElementById('archiveTimeFilter');
    const archiveFromTimeInput = document.getElementById('archiveFromTime');
    const archiveToTimeInput = document.getElementById('archiveToTime');
    const archiveDetectionsLimit = document.getElementById('archiveDetectionsLimit');
    const archiveScoreThresholdInput = document.getElementById('archiveScoreThreshold');
    const archiveScoreThresholdValue = document.getElementById('archiveScoreThresholdValue');
    const archiveScoreThresholdMeta = document.getElementById('archiveScoreThresholdMeta');
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
    const archiveReviewModal = document.getElementById('archiveReviewModal');
    const closeArchiveReviewBtn = document.getElementById('closeArchiveReview');
    const archiveReviewTitle = document.getElementById('archiveReviewTitle');
    const archiveReviewQuery = document.getElementById('archiveReviewQuery');
    const archiveReviewMatch = document.getElementById('archiveReviewMatch');
    const archiveReviewImg = document.getElementById('archiveReviewImg');
    const archiveReviewFrameContainer = archiveReviewImg ? archiveReviewImg.closest('.archive-review-frame') : null;
    const archiveReviewFrameEmpty = document.getElementById('archiveReviewFrameEmpty');
    const archiveReviewChannel = document.getElementById('archiveReviewChannel');
    const archiveReviewFrameRole = document.getElementById('archiveReviewFrameRole');
    const archiveReviewPrevFrameBtn = document.getElementById('archiveReviewPrevFrameBtn');
    const archiveReviewNextFrameBtn = document.getElementById('archiveReviewNextFrameBtn');
    const archiveReviewTimestamp = document.getElementById('archiveReviewTimestamp');
    const archiveReviewFilmstrip = document.getElementById('archiveReviewFilmstrip');
    const archiveReviewSummary = document.getElementById('archiveReviewSummary');
    const archiveReviewDescribeBtn = document.getElementById('archiveReviewDescribeBtn');
    const archiveReviewSimilarBtn = document.getElementById('archiveReviewSimilarBtn');
    const archiveReviewJumpBtn = document.getElementById('archiveReviewJumpBtn');
    const archiveReviewCopyBtn = document.getElementById('archiveReviewCopyBtn');
    
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
    let archiveInspectorRenderKey = '';
    let probeCardsRenderKey = '';
    let luxriotSummaryLogCache = [];
    const luxriotSummaryChannelCache = {};
    const luxriotSummarySeenKeys = {};
    let luxriotSummaryUnread = 0;
    let luxriotSummaryChannel = null;
    let luxriotSummaryRunFilter = 'live';
    let luxriotSummaryRangePreset = 'live';
    let luxriotSummaryFromTs = null;
    let luxriotSummaryToTs = null;
    let luxriotSummaryLevel = 'L0';
    let luxriotSummaryResolutionMode = 'AUTO';
    let luxriotSummaryArchiveOffset = 0;
    let luxriotSummaryArchiveHasMore = false;
    let luxriotSummaryArchiveEvidenceTotal = 0;
    let luxriotSummaryArchiveLoading = false;
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
        batchSize: {luxriot_batch_default}
    };
    let luxriotDisplayTimezone = 'UTC';
    try {
        luxriotDisplayTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
        new Intl.DateTimeFormat('en', { timeZone: luxriotDisplayTimezone }).format(new Date());
    } catch (_) {
        luxriotDisplayTimezone = 'UTC';
    }
    let luxriotActiveChannel = luxriotDefaults.channelId;
    let luxriotPreviewTimer = null;
    let luxriotPreviewRenewTimer = null;
    let luxriotPreviewStallTimer = null;
    let luxriotPreviewLoading = false;
    let luxriotPreviewRequestSeq = 0;
    let luxriotPreviewAbortController = null;
    let luxriotPreviewVideo = null;
    let luxriotPreviewRetryBtn = null;
    let luxriotPreviewNegotiation = null;
    let luxriotPreviewTransportBtn = null;
    let luxriotPreferFullOperatorMedia = false;
    let luxriotAttentionSwitchPending = false;
    let luxriotSummaryTimer = null;
    let luxriotSummaryRefreshInFlight = false;
    let luxriotSummaryRefreshQueued = null;
    let luxriotSummaryRequestGeneration = 0;
    let luxriotSummaryActiveRequest = null;
    let luxriotStreamsCache = [];
    let luxriotLiveIntervalDirtyChannelId = null;
    const luxriotChannelNameById = {};
    const luxriotChannelMetaById = {};
    const luxriotCaptureRunningByChannel = {};
    let luxriotCaptureBusy = false;
    let luxriotPreviewMeta = {
        width: 0,
        height: 0,
        loadedAt: 0,
        frameAgeSec: null,
        stale: false,
        staleAfterSec: 0,
        lostAfterSec: 0,
        failed: false,
        errorCode: '',
        errorText: '',
        mediaState: 'idle',
        mediaKind: '',
        degraded: false,
    };
    let luxriotPromptModalTab = 'stream';
    let luxriotPromptLayers = null;
    let luxriotPromptSettingSources = null;
    let luxriotPromptOverrideFields = [];
    let luxriotPromptPersistence = null;
    let luxriotPromptLoadedSettings = null;
    let luxriotPromptFormChannelId = null;
    let luxriotPromptRequestGeneration = 0;
    let luxriotPromptLoadAbortController = null;
    let luxriotPromptSaveAbortController = null;
    let luxriotInitialized = false;
    let luxriotInitPromise = null;
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
    let probeCastSelectedChannels = new Set();
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
    let probePreviewRenewTimer = null;
    let probePreviewStallTimer = null;
    let probePreviewChannelId = null;
    let probePreviewGeneration = 0;
    let probePreviewAbortController = null;
    let probePreviewVideo = null;
    let probePreviewRetryBtn = null;
    let probePreviewNegotiation = null;
    let probePreviewMediaState = 'idle';
    let lastProbeRefresh = 0;
    let probeStatusTimer = null;
    let archiveDetectionsOffset = 0;
    let archiveDetectionsTotal = 0;
    let archiveDetectionsHasMore = false;
    let archiveScoreThreshold = 0;
    let archiveScoreSliderPercent = 0;
    let archiveLastQueryText = '';
    let archiveReviewContext = null;
    let archiveEvidenceRequestGeneration = 0;
    let archiveEvidenceAbortController = null;
    let archiveEvidenceBusyButton = null;
    let archiveFilterRequestGeneration = 0;
    let archiveFilterAbortController = null;
    let archiveReviewRequestGeneration = 0;
    let archiveReviewAbortController = null;
    let archiveMediaRequestGeneration = 0;
    let archiveMediaAbortController = null;
    let archiveMediaLoadTimer = null;
    let archiveMediaLoopTimer = null;
    let archiveMediaObjectUrl = null;
    let archiveMediaVideo = null;
    let archiveMediaRetryBtn = null;
    let archiveMediaStatus = null;
    let archiveScoreRange = {
        count: 0,
        min: null,
        max: null,
        hasSpread: false,
    };
    const channelCaptureConfig = {};
    const channelFpsDesired = {};
    const ADMIN_TOKEN_STORAGE_KEY = 'evs_admin_token';
    const LUXRIOT_LIVE_MODEL_STORAGE_KEY = 'evs_luxriot_live_model';
    const LUXRIOT_LIVE_INTERVAL_STORAGE_KEY = 'evs_luxriot_live_interval_sec_by_channel';
    const VIDEO_MODEL_STORAGE_KEY = 'evs_video_model';
    const LM_AUTO_MODEL_SELECTOR = '__auto__';
    const LM_AUTO_MODEL_LABEL = 'Auto balance';
    let lmModelCatalog = {
        models: [],
        defaultModel: '',
        autoModelSelector: LM_AUTO_MODEL_SELECTOR,
        autoModelLabel: LM_AUTO_MODEL_LABEL,
        vlmBalancer: { enabled: false, profileIds: [] },
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

    function setModelSelectOptions(selectEl, selectedValue = '', fallbackValue = '', optionsConfig = {}) {
        if (!(selectEl instanceof HTMLSelectElement)) return;
        const includeAuto = Boolean(optionsConfig.includeAuto);
        const autoSelector = normalizeModelId(lmModelCatalog.autoModelSelector || LM_AUTO_MODEL_SELECTOR);
        const autoLabel = normalizeModelId(lmModelCatalog.autoModelLabel || LM_AUTO_MODEL_LABEL);
        const selected = normalizeModelId(selectedValue);
        const fallback = normalizeModelId(fallbackValue || lmModelCatalog.defaultModel);
        const options = includeAuto
            ? uniqueModelIds(autoSelector, lmModelCatalog.models || [], selected, fallback)
            : uniqueModelIds(lmModelCatalog.models || [], selected, fallback);
        const nextValue = selected || (includeAuto ? autoSelector : '') || fallback || options[0] || '';
        if (!options.length) {
            selectEl.innerHTML = '<option value="">No models available</option>';
            selectEl.value = '';
            return;
        }
        selectEl.innerHTML = options
            .map((modelId) => {
                const label = includeAuto && modelId === autoSelector ? autoLabel : modelId;
                return `<option value="${escapeHtml(modelId)}">${escapeHtml(label)}</option>`;
            })
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
        const offlineDefaultModel = normalizeModelId(lmModelCatalog.offlineDefaultModel || lmModelCatalog.agentDefaultModel || defaultModel);
        if (luxriotLiveModelInput) {
            const autoSelector = normalizeModelId(lmModelCatalog.autoModelSelector || LM_AUTO_MODEL_SELECTOR);
            const preferredLiveModel = normalizeModelId(luxriotLiveModelInput.value)
                || normalizeModelId(localStorage.getItem(LUXRIOT_LIVE_MODEL_STORAGE_KEY))
                || autoSelector
                || defaultModel;
            setModelSelectOptions(luxriotLiveModelInput, preferredLiveModel, defaultModel, { includeAuto: true });
        }
        if (videoModelInput) {
            const autoSelector = normalizeModelId(lmModelCatalog.autoModelSelector || LM_AUTO_MODEL_SELECTOR);
            const storedVideoModel = normalizeModelId(localStorage.getItem(VIDEO_MODEL_STORAGE_KEY));
            const preferredVideoModel = normalizeModelId(videoModelInput.value)
                || (storedVideoModel === autoSelector ? '' : storedVideoModel)
                || offlineDefaultModel
                || defaultModel;
            setModelSelectOptions(videoModelInput, preferredVideoModel, offlineDefaultModel || defaultModel, { includeAuto: true });
        }
        updateLuxriotStreamContext();
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
                    agentDefaultModel: normalizeModelId(data.agent_default_model),
                    offlineDefaultModel: normalizeModelId(data.offline_default_model),
                    autoModelSelector: normalizeModelId(data.auto_model_selector || LM_AUTO_MODEL_SELECTOR),
                    autoModelLabel: normalizeModelId(data.auto_model_label || LM_AUTO_MODEL_LABEL),
                    vlmBalancer: {
                        enabled: Boolean(data?.vlm_balancer?.enabled),
                        profileIds: uniqueModelIds(data?.vlm_balancer?.profile_ids || []),
                    },
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
                    agentDefaultModel: normalizeModelId(lmModelCatalog.agentDefaultModel || ''),
                    offlineDefaultModel: normalizeModelId(lmModelCatalog.offlineDefaultModel || ''),
                    autoModelSelector: normalizeModelId(lmModelCatalog.autoModelSelector || LM_AUTO_MODEL_SELECTOR),
                    autoModelLabel: normalizeModelId(lmModelCatalog.autoModelLabel || LM_AUTO_MODEL_LABEL),
                    vlmBalancer: lmModelCatalog.vlmBalancer || { enabled: false, profileIds: [] },
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
                syncUiAccess();
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
            syncUiAccess();
            if (authTokenBtn && authCurrentUser) {
                authTokenBtn.title = `${authCurrentUser.displayName || authCurrentUser.username} · Sign out`;
                authTokenBtn.style.opacity = '1';
            }
        } catch (error) {
            authCurrentUser = null;
            setAuthGateVisible(true, error.message || 'Sign in required');
            syncUiAccess();
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
        if (/\/detections\/thumbnail\/\d+/i.test(value)) return true;
        if (/^\/image\//i.test(value)) return true;
        if (/\/luxriot\/recent_frame/i.test(value)) return true;
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
            return { main: '', json: '', marker: '' };
        }

        const fenced = full.match(/```json\s*([\s\S]*?)```/i);
        if (fenced && fenced[1]) {
            const jsonBlock = String(fenced[1] || '').trim();
            const mainText = full.replace(fenced[0], '').trim();
            return { main: mainText, json: jsonBlock, marker: 'fenced_json' };
        }

        for (const marker of ['ALERTS_JSON:', 'MEMORY_UPDATE_JSON:']) {
            const markerIndex = full.toUpperCase().indexOf(marker);
            if (markerIndex >= 0) {
                const mainText = full.slice(0, markerIndex).trim();
                const jsonBlock = full.slice(markerIndex + marker.length).trim();
                if (jsonBlock) {
                    return { main: mainText, json: jsonBlock, marker };
                }
            }
        }

        const trailingStart = full.lastIndexOf('\n{');
        const startIndex = trailingStart >= 0 ? trailingStart + 1 : (full.startsWith('{') ? 0 : -1);
        if (startIndex >= 0) {
            const jsonCandidate = full.slice(startIndex).trim();
            const looksLikeAlerts = (jsonCandidate.includes('"alerts"') || jsonCandidate.includes("'alerts'"));
            if (looksLikeAlerts && jsonCandidate.startsWith('{') && jsonCandidate.endsWith('}')) {
                const mainText = full.slice(0, startIndex).trim();
                return { main: mainText, json: jsonCandidate, marker: 'trailing_json' };
            }
        }

        return { main: full, json: '', marker: '' };
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

    function userHasPermission(permission) {
        const clean = String(permission || '').trim();
        if (!clean) return true;
        if (!AUTH_ENABLED) return true;
        if (!authCurrentUser || !Array.isArray(authCurrentUser.permissions)) return false;
        return authCurrentUser.permissions.includes(clean);
    }

    function userHasAnyPermission(permissions = []) {
        const values = Array.isArray(permissions) ? permissions : [];
        if (!values.length) return true;
        return values.some((permission) => userHasPermission(permission));
    }

    function userHasAnyRole(roles = []) {
        const values = Array.isArray(roles) ? roles.map((role) => String(role || '').trim().toLowerCase()).filter(Boolean) : [];
        if (!values.length) return true;
        if (!AUTH_ENABLED) return true;
        if (!authCurrentUser || !Array.isArray(authCurrentUser.roles)) return false;
        const currentRoles = authCurrentUser.roles.map((role) => String(role || '').trim().toLowerCase()).filter(Boolean);
        return values.some((role) => currentRoles.includes(role));
    }

    function canViewVlmMachineJson() {
        return userHasAnyRole(['admin', 'engineer']);
    }

    function canUseProbeDiagnostics() {
        return userHasAnyRole(['admin', 'engineer'])
            || userHasAnyPermission(['probes:manage', 'diagnostics:view']);
    }

    function parseSummaryMachineJson(raw) {
        try {
            return JSON.parse(String(raw || '').trim());
        } catch (_) {
            return null;
        }
    }

    function shortMachineJsonTitle(value, fallback = 'System message') {
        const text = String(value || '').trim().replace(/\s+/g, ' ');
        if (!text) return fallback;
        return text.length > 72 ? `${text.slice(0, 69)}...` : text;
    }

    function summarizeMachineJson(raw, label = 'Machine JSON', marker = '') {
        const text = String(raw || '').trim();
        const lineCount = text.split(/\r?\n/).filter((line) => line.trim()).length || 1;
        const sizeLabel = `${lineCount} line${lineCount === 1 ? '' : 's'}`;
        const parsed = parseSummaryMachineJson(text);
        const markerText = String(marker || '').toUpperCase();
        const haystack = `${markerText}\n${text}`.toLowerCase();

        if (
            markerText.includes('MEMORY_UPDATE_JSON')
            || /\b(homeostasis|memory_update|memory update|routine_baseline|baseline|prior)\b/i.test(haystack)
        ) {
            return { label: 'Memory/homeostasis', meta: sizeLabel, kind: 'memory' };
        }

        const alerts = parsed && Array.isArray(parsed.alerts) ? parsed.alerts : null;
        if (alerts) {
            if (!alerts.length) {
                return { label: 'System message', meta: `no alerts · ${sizeLabel}`, kind: 'system' };
            }
            const first = alerts[0] || {};
            const title = shortMachineJsonTitle(
                first.title || first.alert_title || first.name || first.type || first.summary || first.description,
                'Alert event'
            );
            const severity = String(first.severity || first.level || '').trim();
            const countLabel = alerts.length > 1 ? ` +${alerts.length - 1}` : '';
            const meta = [
                severity || `${alerts.length} alert${alerts.length === 1 ? '' : 's'}`,
                sizeLabel,
            ].filter(Boolean).join(' · ');
            return { label: `${title}${countLabel}`, meta, kind: 'alert' };
        }

        if (lineCount <= 3) {
            return { label: 'System message', meta: sizeLabel, kind: 'system' };
        }
        return { label: label || 'Machine JSON', meta: sizeLabel, kind: 'machine' };
    }

    function renderSummaryMachineJson(jsonText, label = 'Machine JSON', marker = '') {
        const raw = String(jsonText || '').trim();
        if (!raw) return '';
        if (!canViewVlmMachineJson()) {
            return '<div class="summary-json-hidden" title="Visible to admin and engineer roles">Machine data hidden</div>';
        }
        const summary = summarizeMachineJson(raw, label, marker);
        return `
            <details class="summary-json-disclosure summary-json-${escapeHtml(summary.kind)}">
                <summary><span>${escapeHtml(summary.label)}</span><span class="summary-json-meta">${escapeHtml(summary.meta)}</span></summary>
                <div class="summary-json-muted">${renderMarkdown(raw)}</div>
            </details>
        `;
    }

    function canUseMode(mode) {
        if (!AUTH_ENABLED) return true;
        if (!authCurrentUser) return false;
        switch (mode) {
            case 'archive':
                return userHasPermission('detections:view');
            case 'video':
                return userHasPermission('streams:view');
            case 'monitor':
                return canUseProbeDiagnostics();
            case 'agent':
                return userHasPermission('agent:use');
            default:
                return false;
        }
    }

    function firstAllowedMode() {
        return ['archive', 'video', 'monitor', 'agent'].find((mode) => canUseMode(mode)) || 'archive';
    }

    function setMode(mode) {
        if (AUTH_ENABLED && authCurrentUser && !canUseMode(mode)) {
            mode = firstAllowedMode();
        }
        currentMode = mode;
        archiveModeBtn.classList.toggle('active', mode === 'archive');
        videoModeBtn.classList.toggle('active', mode === 'video');
        monitorModeBtn.classList.toggle('active', mode === 'monitor');
        if (agentModeBtn) agentModeBtn.classList.toggle('active', mode === 'agent');
        if (headerStatusText) {
            const statusByMode = {
                archive: 'Archive Research Ready',
                video: 'Live Video Ops',
                monitor: 'CLIP Probe Diagnostics',
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
        if (mode !== 'archive') {
            invalidateArchiveResultContext();
            cancelArchiveFilterRequest();
        }
        if (window._agentSetActive) {
            window._agentSetActive(mode === 'agent');
        }
        if (mode === 'video') {
            stopProbePreview();
            stopProbeRunLoop();
            stopProbeStatusPoll();
            if (probeEditorModal) {
                setProbeEditorModalVisibility(false);
            }
            setProbeCastModalVisibility(false);
            syncProbeChannelSelect();
            void ensureLuxriotInit()
                .then(() => {
                    if (currentMode !== 'video') return;
                    startLuxriotPreview();
                    refreshLuxriotSummaryView(getSelectedSummaryChannel(), true);
                    refreshLuxriotStreams();
                    startLuxriotSummaryPoll();
                })
                .catch((err) => {
                    setLuxriotStatus(`Luxriot init failed: ${err.message || err}`, true);
                });
        } else if (mode === 'monitor') {
            stopLuxriotPreview(true);
            stopLuxriotSummaryPoll();
            void ensureLuxriotInit()
                .then(() => {
                    if (currentMode !== 'monitor') return;
                    syncProbeChannelSelect();
                    syncProbePreview(getSelectedProbeChannelId());
                    refreshProbeStatus();
                    loadProbeList();
                    startProbeStatusPoll();
                })
                .catch((err) => {
                    setProbeStatus(`Luxriot init failed: ${err.message || err}`, true);
                });
        } else {
            stopLuxriotPreview(true);
            stopLuxriotSummaryPoll();
            stopProbePreview();
            stopProbeRunLoop();
            stopProbeStatusPoll();
            if (mode === 'archive') {
                refreshArchiveFilters().catch(() => {});
            }
            if (probeEditorModal) {
                setProbeEditorModalVisibility(false);
            }
            setProbeCastModalVisibility(false);
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

    function setProbeCastModalVisibility(visible) {
        if (!probeCastModal) return;
        probeCastModal.style.display = visible ? 'block' : 'none';
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

    function normalizeLuxriotLiveInterval(value) {
        const interval = Number.parseFloat(String(value ?? '').trim());
        if (!Number.isFinite(interval) || interval <= 0) return null;
        return Math.max(0.2, Math.min(300, interval));
    }

    function formatLuxriotLiveIntervalInput(intervalSec) {
        const interval = normalizeLuxriotLiveInterval(intervalSec)
            || normalizeLuxriotLiveInterval(luxriotDefaults.snapshotInterval)
            || 5;
        const decimals = interval < 10 && !Number.isInteger(interval) ? 1 : 0;
        return interval.toFixed(decimals).replace(/[.]0$/, '');
    }

    function readLuxriotLiveIntervalMap() {
        try {
            const parsed = JSON.parse(localStorage.getItem(LUXRIOT_LIVE_INTERVAL_STORAGE_KEY) || '{}');
            return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
        } catch (_) {
            return {};
        }
    }

    function getStoredLuxriotLiveInterval(channelId) {
        const parsedChannel = parseInt(String(channelId || ''), 10);
        if (!Number.isFinite(parsedChannel)) return null;
        const stored = readLuxriotLiveIntervalMap()[String(parsedChannel)];
        return normalizeLuxriotLiveInterval(stored);
    }

    function storeLuxriotLiveInterval(channelId, intervalSec) {
        const parsedChannel = parseInt(String(channelId || ''), 10);
        const interval = normalizeLuxriotLiveInterval(intervalSec);
        if (!Number.isFinite(parsedChannel) || interval === null) return;
        const map = readLuxriotLiveIntervalMap();
        map[String(parsedChannel)] = Number(interval.toFixed(3));
        localStorage.setItem(LUXRIOT_LIVE_INTERVAL_STORAGE_KEY, JSON.stringify(map));
    }

    function getLuxriotLiveIntervalInputValue() {
        const parsed = normalizeLuxriotLiveInterval(luxriotLiveIntervalInput ? luxriotLiveIntervalInput.value : null);
        return parsed
            || normalizeLuxriotLiveInterval(luxriotDefaults.snapshotInterval)
            || 5;
    }

    function markLuxriotLiveIntervalDirty() {
        const channelId = getSelectedLuxriotChannel();
        luxriotLiveIntervalDirtyChannelId = Number.isFinite(channelId) ? channelId : null;
    }

    function clearLuxriotLiveIntervalDirty(channelIdOverride = null) {
        const channelId = Number.isFinite(channelIdOverride)
            ? channelIdOverride
            : getSelectedLuxriotChannel();
        if (!Number.isFinite(channelId) || luxriotLiveIntervalDirtyChannelId === channelId) {
            luxriotLiveIntervalDirtyChannelId = null;
        }
    }

    function isLuxriotLiveIntervalDirty(channelIdOverride = null) {
        const channelId = Number.isFinite(channelIdOverride)
            ? channelIdOverride
            : getSelectedLuxriotChannel();
        return Number.isFinite(channelId) && luxriotLiveIntervalDirtyChannelId === channelId;
    }

    function formatLuxriotDuration(intervalSec) {
        const seconds = Number(intervalSec);
        if (!Number.isFinite(seconds) || seconds <= 0) return 'n/a';
        if (seconds < 60) {
            return `${seconds.toFixed(seconds < 10 && !Number.isInteger(seconds) ? 1 : 0).replace(/[.]0$/, '')}s`;
        }
        const minutes = seconds / 60;
        if (minutes < 60) {
            return `${minutes.toFixed(minutes < 10 && !Number.isInteger(minutes) ? 1 : 0).replace(/[.]0$/, '')}m`;
        }
        const hours = minutes / 60;
        return `${hours.toFixed(hours < 10 && !Number.isInteger(hours) ? 1 : 0).replace(/[.]0$/, '')}h`;
    }

    function syncLuxriotLiveIntervalInput(channelIdOverride = null, options = {}) {
        if (!luxriotLiveIntervalInput) return;
        const channelId = Number.isFinite(channelIdOverride)
            ? channelIdOverride
            : getSelectedLuxriotChannel();
        if (!Number.isFinite(channelId)) return;
        if (!options.force && (
            document.activeElement === luxriotLiveIntervalInput
            || isLuxriotLiveIntervalDirty(channelId)
        )) return;
        const videoStream = selectedLuxriotStream(channelId, 'video');
        const interval = normalizeLuxriotLiveInterval(videoStream?.interval_sec)
            || getStoredLuxriotLiveInterval(channelId)
            || normalizeLuxriotLiveInterval(luxriotDefaults.snapshotInterval)
            || 5;
        luxriotLiveIntervalInput.value = formatLuxriotLiveIntervalInput(interval);
        updateLuxriotBatchInfo();
    }

    function updateLuxriotBatchInfo() {
        if (!luxriotBatchInfo) return;
        const intervalSec = getLuxriotLiveIntervalInputValue();
        const batchSize = Number(luxriotBatchSizeSelect?.value) || Number(luxriotDefaults.batchSize) || 0;
        const fps = intervalSec > 0 ? (1 / intervalSec) : 0;
        const fpsLabel = fps >= 1 ? fps.toFixed(1).replace(/[.]0$/, '') : fps.toFixed(2);
        const summaryLabel = batchSize > 0 ? ` · batch ~${formatLuxriotDuration(intervalSec * batchSize)}` : '';
        luxriotBatchInfo.textContent = `~${fpsLabel} fps${summaryLabel} · ${luxriotDefaults.snapshotMaxEdge}px`;
        updateLuxriotRuntimeConfigHint();
        updateLuxriotStreamContext();
    }

    function updateLuxriotRuntimeConfigHint() {
        if (!luxriotRuntimeConfigState || !luxriotRuntimeConfigRunning || !luxriotRuntimeConfigPending) return;
        const channelId = getSelectedLuxriotChannel();
        const videoStream = selectedLuxriotStream(channelId, 'video');
        if (!videoStream?.running) {
            luxriotRuntimeConfigState.hidden = true;
            luxriotRuntimeConfigRunning.textContent = '';
            luxriotRuntimeConfigPending.hidden = true;
            return;
        }
        const runningBatch = Number(videoStream.batch_size);
        const runningInterval = normalizeLuxriotLiveInterval(videoStream.interval_sec);
        const selectedBatch = Number(luxriotBatchSizeSelect?.value);
        const selectedInterval = getLuxriotLiveIntervalInputValue();
        const batchLabel = Number.isFinite(runningBatch) && runningBatch > 0 ? String(runningBatch) : 'n/a';
        const intervalLabel = runningInterval !== null ? formatLuxriotDuration(runningInterval) : 'n/a';
        luxriotRuntimeConfigRunning.textContent = `running: batch ${batchLabel} · ${intervalLabel}`;
        const batchChanged = Number.isFinite(runningBatch)
            && runningBatch > 0
            && Number.isFinite(selectedBatch)
            && selectedBatch !== runningBatch;
        const intervalChanged = runningInterval !== null
            && Math.abs(selectedInterval - runningInterval) > 0.0005;
        luxriotRuntimeConfigPending.hidden = !(batchChanged || intervalChanged);
        luxriotRuntimeConfigState.hidden = false;
    }

    function setTextContentSafe(element, text) {
        if (!element) return;
        element.textContent = text;
    }

    function formatCompactCount(value) {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) return '0';
        return Math.round(numeric).toLocaleString();
    }

    function luxriotHealthCount(value) {
        const numeric = Number(value);
        if (!Number.isFinite(numeric) || numeric <= 0) return 0;
        return Math.round(numeric);
    }

    function truncateLuxriotHealthText(value, maxLength = 180) {
        const text = String(value || '').trim();
        if (!text || text.length <= maxLength) return text;
        return `${text.slice(0, Math.max(0, maxLength - 3))}...`;
    }

    function classifyLuxriotStreamIssue(stream) {
        const captureError = String(stream?.capture_last_error || '').trim();
        const probeError = String(stream?.probe_last_error || '').trim();
        const summaryError = String(stream?.summary_last_error || '').trim();
        const legacyError = String(stream?.last_error || '').trim();
        const effectiveSummaryError = summaryError || (
            /summary queue overflow|oldest pending batch dropped|lm admission queue timeout/i.test(legacyError)
                ? legacyError
                : ''
        );
        const backpressure = /summary queue overflow|oldest pending batch dropped|lm admission queue timeout/i.test(
            effectiveSummaryError
        );
        const hardSummaryError = backpressure ? '' : effectiveSummaryError;
        const classifiedSpecificError = captureError || probeError || hardSummaryError;
        const hardError = classifiedSpecificError || (
            legacyError && !backpressure && !summaryError && !captureError && !probeError
                ? legacyError
                : ''
        );
        return {
            backpressure,
            backpressureError: backpressure ? effectiveSummaryError : '',
            hardError,
        };
    }

    function getLuxriotStreamHealth(stream) {
        if (!stream || typeof stream !== 'object') return null;
        const pending = luxriotHealthCount(stream.pending_frames);
        const maxBuffer = luxriotHealthCount(stream.max_buffer_frames);
        const droppedFrames = luxriotHealthCount(stream.dropped_frames);
        const droppedBatches = luxriotHealthCount(stream.queue_dropped_batches);
        const logsTotalRaw = stream.logs_total ?? (Array.isArray(stream.logs) ? stream.logs.length : null);
        const logsTotal = logsTotalRaw === null || logsTotalRaw === undefined ? null : luxriotHealthCount(logsTotalRaw);
        const issue = classifyLuxriotStreamIssue(stream);
        const summaryQueueDepth = luxriotHealthCount(stream.summary_queue_depth);
        const summaryInflight = Boolean(stream.summary_inflight);
        const lastSnapshotLatency = Number(stream.last_snapshot_latency_sec);
        const avgSnapshotLatency = Number(stream.avg_snapshot_latency_sec);
        const snapshotSlowThreshold = Number(stream.snapshot_slow_threshold_sec);
        const slowSnapshots = luxriotHealthCount(stream.slow_snapshot_count);
        const activeCaptureSource = String(stream.active_capture_source || '').trim();
        const liveSegmentLatency = Number(stream.last_live_segment_latency_sec);
        const liveSegmentFrames = luxriotHealthCount(stream.last_live_segment_frames);
        const liveSegmentTargetSeconds = Number(stream.last_live_segment_target_seconds);
        const liveSegmentSummaryTargetSeconds = Number(stream.last_live_segment_summary_target_seconds);
        const liveSegmentRepresentedSeconds = Number(stream.last_live_segment_represented_seconds);
        const liveSegmentInflight = Boolean(stream.live_segment_inflight);
        const liveSegmentInflightTargetSeconds = Number(stream.live_segment_inflight_target_seconds);
        const liveSegmentInflightRawBudget = luxriotHealthCount(stream.live_segment_inflight_raw_frame_budget);
        const liveSegmentInflightFrames = luxriotHealthCount(stream.live_segment_inflight_frames);
        const liveSegmentInflightRepresentedSeconds = Number(stream.live_segment_inflight_represented_seconds);
        const frozenSignal = Boolean(stream.frozen_signal);
        const frozenAge = Number(stream.frozen_signal_age_sec);
        const frozenCount = luxriotHealthCount(stream.frozen_frame_count);
        const lagThreshold = maxBuffer > 0 ? Math.max(1, Math.ceil(maxBuffer * 0.8)) : 0;
        const titleParts = [
            `state ${stream.running ? 'running' : 'stopped'}`,
            maxBuffer > 0
                ? `pending ${formatCompactCount(pending)}/${formatCompactCount(maxBuffer)}`
                : `pending ${formatCompactCount(pending)}`,
        ];
        if (activeCaptureSource) titleParts.push(`source ${activeCaptureSource}`);
        if (Number.isFinite(lastSnapshotLatency) && lastSnapshotLatency > 0) {
            titleParts.push(`snapshot ${lastSnapshotLatency.toFixed(1)}s`);
        }
        if (Number.isFinite(avgSnapshotLatency) && avgSnapshotLatency > 0) {
            titleParts.push(`avg snapshot ${avgSnapshotLatency.toFixed(1)}s`);
        }
        if (slowSnapshots > 0) titleParts.push(`slow snapshots ${formatCompactCount(slowSnapshots)}`);
        if (Number.isFinite(liveSegmentLatency) && liveSegmentLatency > 0) {
            titleParts.push(`segment ${liveSegmentLatency.toFixed(1)}s/${formatCompactCount(liveSegmentFrames)} frames`);
        }
        const completedAttentionWindow = (
            activeCaptureSource === 'live_segment'
            && Number.isFinite(liveSegmentTargetSeconds)
            && liveSegmentTargetSeconds > 0
            && Number.isFinite(liveSegmentRepresentedSeconds)
            && liveSegmentRepresentedSeconds >= 0
        );
        const attentionRealtimeRatio = completedAttentionWindow
            && Number.isFinite(liveSegmentLatency)
            && liveSegmentLatency > 0
            ? liveSegmentRepresentedSeconds / liveSegmentLatency
            : null;
        const attentionUnderfilled = completedAttentionWindow
            && liveSegmentRepresentedSeconds + 0.25 < liveSegmentTargetSeconds * 0.8;
        const attentionBehindRealtime = Number.isFinite(attentionRealtimeRatio)
            && attentionRealtimeRatio < 0.8;
        if (completedAttentionWindow) {
            titleParts.push(
                `attention ${liveSegmentRepresentedSeconds.toFixed(1)}/${liveSegmentTargetSeconds.toFixed(1)}s`
                + (Number.isFinite(attentionRealtimeRatio) ? ` at ${attentionRealtimeRatio.toFixed(2)}x realtime` : '')
            );
        }
        if (Number.isFinite(liveSegmentSummaryTargetSeconds) && liveSegmentSummaryTargetSeconds > 0) {
            titleParts.push(`summary cadence ${liveSegmentSummaryTargetSeconds.toFixed(1)}s`);
        }
        if (liveSegmentInflight) {
            titleParts.push(
                `capturing ${Number.isFinite(liveSegmentInflightRepresentedSeconds) ? liveSegmentInflightRepresentedSeconds.toFixed(1) : '?'}s`
                + (Number.isFinite(liveSegmentInflightTargetSeconds) ? `/${liveSegmentInflightTargetSeconds.toFixed(1)}s` : '')
                + (liveSegmentInflightRawBudget > 0 ? ` · ${formatCompactCount(liveSegmentInflightFrames)}/${formatCompactCount(liveSegmentInflightRawBudget)} raw frames` : '')
            );
        }
        if (frozenSignal) {
            titleParts.unshift(`frozen ${Number.isFinite(frozenAge) && frozenAge > 0 ? formatLuxriotDuration(frozenAge) : ''}${frozenCount > 0 ? `/${formatCompactCount(frozenCount)} frames` : ''}`.trim());
            return { label: 'frozen', tone: 'error', title: titleParts.join(' | ') };
        }
        if (droppedFrames > 0) titleParts.push(`dropped frames ${formatCompactCount(droppedFrames)}`);
        if (droppedBatches > 0) titleParts.push(`queue drops ${formatCompactCount(droppedBatches)}`);
        if (logsTotal !== null) titleParts.push(`logs ${formatCompactCount(logsTotal)}`);
        if (issue.hardError) {
            titleParts.unshift(`error ${truncateLuxriotHealthText(issue.hardError)}`);
            return { label: 'error', tone: 'error', title: titleParts.join(' | ') };
        }
        if (attentionUnderfilled || attentionBehindRealtime) {
            return { label: 'apex-lag', tone: 'warning', title: titleParts.join(' | ') };
        }
        if (issue.backpressure || droppedBatches > 0) {
            if (issue.backpressureError) {
                titleParts.unshift(`aggregation backpressure ${truncateLuxriotHealthText(issue.backpressureError)}`);
            }
            return { label: 'backpressure', tone: 'warning', title: titleParts.join(' | ') };
        }
        if (
            activeCaptureSource !== 'live_segment'
            && Number.isFinite(lastSnapshotLatency)
            && Number.isFinite(snapshotSlowThreshold)
            && snapshotSlowThreshold > 0
            && lastSnapshotLatency >= snapshotSlowThreshold
        ) {
            return { label: 'slow', tone: 'warning', title: titleParts.join(' | ') };
        }
        if (droppedFrames > 0) {
            return { label: 'drops', tone: 'warning', title: titleParts.join(' | ') };
        }
        if (lagThreshold > 0 && pending >= lagThreshold) {
            titleParts.push(`lag threshold ${formatCompactCount(lagThreshold)}`);
            return { label: 'lag', tone: 'warning', title: titleParts.join(' | ') };
        }
        if (summaryInflight || summaryQueueDepth > 0) {
            titleParts.push(`aggregation ${summaryInflight ? 'inference active' : 'queued'}${summaryQueueDepth > 0 ? ` · ${formatCompactCount(summaryQueueDepth)} batches waiting` : ''}`);
            return { label: 'aggregating', tone: 'ok', title: titleParts.join(' | ') };
        }
        return { label: 'ok', tone: 'ok', title: titleParts.join(' | ') };
    }

    function renderLuxriotHealthBadge(health) {
        if (!health) return '';
        const label = health.label || 'ok';
        const tone = health.tone || 'ok';
        return `<span class="luxriot-health-badge ${escapeHtml(tone)} luxriot-health-${escapeHtml(label)}" title="${escapeHtml(health.title || label)}">${escapeHtml(label)}</span>`;
    }

    function updateLuxriotStreamHealthBadge(stream) {
        if (!luxriotStreamState) return;
        let badge = document.getElementById('luxriotStreamHealth');
        if (!badge) {
            badge = document.createElement('span');
            badge.id = 'luxriotStreamHealth';
            luxriotStreamState.insertAdjacentElement('afterend', badge);
        }
        const health = getLuxriotStreamHealth(stream);
        if (!health) {
            badge.className = 'luxriot-health-badge is-hidden';
            badge.textContent = '';
            badge.removeAttribute('title');
            return;
        }
        badge.className = `luxriot-health-badge ${health.tone || 'ok'} luxriot-health-${health.label || 'ok'}`;
        badge.textContent = health.label || 'ok';
        badge.title = health.title || health.label || 'ok';
    }

    function appendLuxriotStatusHealthBadge(stream) {
        if (!luxriotStatusLabel) return;
        const health = getLuxriotStreamHealth(stream);
        if (!health) return;
        const badge = document.createElement('span');
        badge.className = `luxriot-health-badge ${health.tone || 'ok'} luxriot-health-${health.label || 'ok'}`;
        badge.textContent = health.label || 'ok';
        badge.title = health.title || health.label || 'ok';
        luxriotStatusLabel.appendChild(badge);
    }

    function formatLuxriotCadence(intervalSec) {
        const interval = Number(intervalSec);
        if (!Number.isFinite(interval) || interval <= 0) return 'n/a';
        const fps = 1 / interval;
        const fpsLabel = fps >= 1 ? fps.toFixed(1).replace(/[.]0$/, '') : fps.toFixed(2);
        return `${fpsLabel} fps · ${interval.toFixed(interval >= 10 ? 0 : 1).replace(/[.]0$/, '')}s`;
    }

    function formatPreviewAge(timestampMs) {
        const ts = Number(timestampMs);
        if (!Number.isFinite(ts) || ts <= 0) return 'never';
        const ageSec = Math.max(0, Math.round((Date.now() - ts) / 1000));
        if (ageSec < 2) return 'just now';
        if (ageSec < 60) return `${ageSec}s ago`;
        return new Date(ts).toLocaleTimeString();
    }

    function selectedLuxriotStream(channelId, streamType) {
        const target = parseInt(String(channelId || ''), 10);
        if (!Number.isFinite(target)) return null;
        const type = String(streamType || '').trim().toLowerCase();
        return (Array.isArray(luxriotStreamsCache) ? luxriotStreamsCache : []).find((stream) => {
            const streamChannel = parseInt(String(stream?.channel_id ?? ''), 10);
            const streamKind = String(stream?.stream_type || '').trim().toLowerCase();
            return streamChannel === target && streamKind === type;
        }) || null;
    }

    function luxriotPreviewFreshnessLimits(channelId) {
        const videoStream = selectedLuxriotStream(channelId, 'video');
        const intervalSec = normalizeLuxriotLiveInterval(videoStream?.interval_sec)
            || getLuxriotLiveIntervalInputValue()
            || Number(luxriotDefaults.snapshotInterval)
            || 1;
        const batchSize = Number(videoStream?.batch_size) || Number(luxriotBatchSizeSelect?.value) || Number(luxriotDefaults.batchSize) || 1;
        const liveSegmentLatency = Number(videoStream?.last_live_segment_latency_sec);
        const sourceLatency = Number.isFinite(liveSegmentLatency) && liveSegmentLatency > 0 ? liveSegmentLatency : 0;
        const expectedCycleSec = Math.max(1, (Math.max(1, batchSize) * Math.max(0.25, intervalSec)) + sourceLatency);
        const staleAfterSec = Math.max(20, Math.min(75, expectedCycleSec + 10));
        const lostAfterSec = Math.max(staleAfterSec + 15, Math.min(180, (expectedCycleSec * 3) + 15));
        return { expectedCycleSec, staleAfterSec, lostAfterSec };
    }

    function probeStatsForChannel(channelId) {
        const target = parseInt(String(channelId || ''), 10);
        const stats = { total: 0, enabled: 0, disabled: 0 };
        if (!Number.isFinite(target)) return stats;
        (Array.isArray(probeCatalog) ? probeCatalog : []).forEach((probe) => {
            const probeChannel = parseInt(String(probe?.channel_id ?? ''), 10);
            if (probeChannel !== target) return;
            stats.total += 1;
            if (probe?.enabled === false) stats.disabled += 1;
            else stats.enabled += 1;
        });
        return stats;
    }

    function selectedLuxriotModelLabel(videoStream = null) {
        const streamModel = normalizeModelId(videoStream?.model || '');
        if (streamModel) return streamModel;
        const selected = normalizeModelId(luxriotLiveModelInput ? luxriotLiveModelInput.value : '');
        const autoSelector = normalizeModelId(lmModelCatalog.autoModelSelector || LM_AUTO_MODEL_SELECTOR);
        const autoLabel = normalizeModelId(lmModelCatalog.autoModelLabel || LM_AUTO_MODEL_LABEL);
        if (selected && selected === autoSelector) return autoLabel || selected;
        return selected || normalizeModelId(lmModelCatalog.defaultModel) || 'default';
    }

    function setLuxriotCaptureBusy(busy) {
        luxriotCaptureBusy = Boolean(busy);
        updateLuxriotCaptureToggleButton(getSelectedLuxriotChannel());
        updateLuxriotStreamContext();
    }

    function updateLuxriotStreamContext() {
        updateLuxriotRuntimeConfigHint();
        if (!luxriotStreamName && !luxriotStreamState) return;
        const channelId = getSelectedLuxriotChannel();
        const selectedRaw = luxriotChannelSelect ? String(luxriotChannelSelect.value || '').trim() : String(channelId || '');
        const hasChannel = Number.isFinite(channelId) && channelId > 0 && Boolean(selectedRaw);
        const channelLabel = hasChannel ? getLuxriotChannelLabel(channelId) : 'No channel selected';
        const videoStream = hasChannel ? selectedLuxriotStream(channelId, 'video') : null;
        const analyticsStream = hasChannel ? selectedLuxriotStream(channelId, 'analytics') : null;
        const showProbeDiagnostics = canUseProbeDiagnostics();
        const running = Boolean(videoStream?.running) || (hasChannel && isLuxriotCaptureRunning(channelId));
        const probeState = hasChannel ? String(probeChannelRuntime[String(channelId)] || '').trim() : '';
        const probeRunning = Boolean(analyticsStream?.running) || probeState === 'running';
        const probePaused = probeState === 'paused' || Boolean(analyticsStream?.paused);
        const mediaState = String(luxriotPreviewMeta.mediaState || 'idle');
        const stateClass = mediaState === 'error' || luxriotPreviewMeta.failed
            ? 'error'
            : mediaState === 'degraded' || luxriotPreviewMeta.stale
                ? 'slow'
            : mediaState === 'playing'
                ? 'running'
            : running
                ? 'running'
                : showProbeDiagnostics && probeRunning
                    ? 'running'
                    : showProbeDiagnostics && probePaused
                        ? 'paused'
                        : 'idle';
        const stateText = mediaState === 'loading'
            ? 'video loading'
            : mediaState === 'playing'
                ? (
                    luxriotPreviewMeta.mediaKind === 'attention'
                        ? 'attention playing'
                        : luxriotPreviewMeta.mediaKind === 'mjpeg'
                            ? 'mjpeg playing'
                            : 'video playing'
                )
            : mediaState === 'degraded'
                ? 'static fallback'
            : luxriotPreviewMeta.failed
            ? luxriotPreviewMeta.errorCode === 'signal_lost'
                ? 'signal lost'
                : luxriotPreviewMeta.errorCode === 'signal_frozen'
                    ? 'signal frozen'
                : 'preview error'
            : luxriotPreviewMeta.stale
                ? 'slow'
            : running
                ? 'summaries on'
                : showProbeDiagnostics && probeRunning
                    ? 'diagnostics on'
                    : showProbeDiagnostics && probePaused
                        ? 'diagnostics paused'
                        : 'idle';
        const previewWidth = Number(luxriotPreviewMeta.width);
        const previewHeight = Number(luxriotPreviewMeta.height);
        const resolution = previewWidth > 0 && previewHeight > 0
            ? `${previewWidth}x${previewHeight}`
            : mediaState === 'loading'
                ? 'loading video'
            : luxriotPreviewMeta.failed
                ? luxriotPreviewMeta.errorCode === 'signal_lost'
                    ? 'signal lost'
                    : luxriotPreviewMeta.errorCode === 'signal_frozen'
                        ? 'signal frozen'
                    : 'failed'
                : 'waiting';
        const intervalSec = normalizeLuxriotLiveInterval(videoStream?.interval_sec)
            || getLuxriotLiveIntervalInputValue()
            || Number(luxriotDefaults.snapshotInterval)
            || 0;
        const batchSize = Number(videoStream?.batch_size) || Number(luxriotBatchSizeSelect?.value) || Number(luxriotDefaults.batchSize) || 0;
        const queued = Number(videoStream?.pending_frames) || 0;
        const flushes = Number(videoStream?.flush_count) || 0;
        const dropped = Number(videoStream?.queue_dropped_batches) || 0;
        const probeStats = probeStatsForChannel(channelId);
        const analyticsQueued = Number(analyticsStream?.pending_frames) || 0;
        const analyticsInterval = Number(analyticsStream?.interval_sec);
        const probeCadence = Number.isFinite(analyticsInterval) && analyticsInterval > 0 ? formatLuxriotCadence(analyticsInterval) : '';
        const probeShared = Boolean(analyticsStream?.shared_capture);
        const lastSnapshotLatency = Number(videoStream?.last_snapshot_latency_sec);
        const avgSnapshotLatency = Number(videoStream?.avg_snapshot_latency_sec);
        const slowSnapshotCount = Number(videoStream?.slow_snapshot_count) || 0;
        const activeCaptureSource = String(videoStream?.active_capture_source || '').trim();
        const liveSegmentLatency = Number(videoStream?.last_live_segment_latency_sec);
        const liveSegmentFrames = Number(videoStream?.last_live_segment_frames) || 0;
        const liveSegmentTargetSeconds = Number(videoStream?.last_live_segment_target_seconds);
        const liveSegmentSummaryTargetSeconds = Number(videoStream?.last_live_segment_summary_target_seconds);
        const liveSegmentRepresentedSeconds = Number(videoStream?.last_live_segment_represented_seconds);
        const liveSegmentInflight = Boolean(videoStream?.live_segment_inflight);
        const liveSegmentInflightTargetSeconds = Number(videoStream?.live_segment_inflight_target_seconds);
        const liveSegmentInflightRawBudget = Number(videoStream?.live_segment_inflight_raw_frame_budget) || 0;
        const liveSegmentInflightFrames = Number(videoStream?.live_segment_inflight_frames) || 0;
        const liveSegmentInflightRepresentedSeconds = Number(videoStream?.live_segment_inflight_represented_seconds);
        const summaryQueueDepth = Number(videoStream?.summary_queue_depth) || 0;
        const summaryQueueFrames = Number(videoStream?.summary_queue_frame_count) || 0;
        const summaryInflight = Boolean(videoStream?.summary_inflight);
        const queueLabel = running
            ? `${formatCompactCount(queued)}/${formatCompactCount(batchSize)} frames · ${formatCompactCount(flushes)} flushes${dropped > 0 ? ` · ${formatCompactCount(dropped)} dropped` : ''}`
            : 'idle';
        const probeLabel = probeRunning
            ? probeShared
                ? `shared with summaries · ${formatCompactCount(analyticsQueued)} buffered`
                : `active · ${formatCompactCount(analyticsQueued)} buffered${probeCadence ? ` · ${probeCadence}` : ''}`
            : probePaused
                ? 'paused'
                : probeStats.enabled > 0
                    ? `${probeStats.enabled}/${probeStats.total} ready`
                    : probeStats.total > 0
                        ? 'configured, disabled'
                        : 'not configured';
        const detailParts = [];
        if (mediaState === 'loading') {
            detailParts.push(luxriotPreviewMeta.errorText || 'Negotiating browser-playable media through the same-origin broker.');
        }
        if (mediaState === 'playing') {
            detailParts.push(
                luxriotPreviewMeta.mediaKind === 'attention'
                    ? 'Model view: exact per-second EVA apex frames; no second recorder stream competes with analytics.'
                    : luxriotPreviewMeta.mediaKind === 'mjpeg'
                        ? 'Full operator media is a continuous MJPEG stream from Luxriot and may compete with analytics.'
                        : 'Full operator video is playing on a second recorder stream and may compete with analytics.'
            );
        }
        if (mediaState === 'degraded') {
            detailParts.push(luxriotPreviewMeta.errorText || 'Only one static fallback frame is shown; this is not video or a snapshot slideshow.');
        }
        if (luxriotPreviewMeta.failed && luxriotPreviewMeta.errorText) {
            detailParts.push(luxriotPreviewMeta.errorText);
        }
        if (!luxriotPreviewMeta.failed && luxriotPreviewMeta.stale) {
            const age = Number(luxriotPreviewMeta.frameAgeSec);
            const lostAfter = Number(luxriotPreviewMeta.lostAfterSec);
            detailParts.push(
                `Preview frame is delayed${Number.isFinite(age) ? ` (${age.toFixed(1)}s old)` : ''}; EVA is holding the last model-visible frame while the capture/VLM cycle catches up${Number.isFinite(lostAfter) && lostAfter > 0 ? `, signal loss after ${Math.round(lostAfter)}s` : ''}.`
            );
        }
        if (videoStream?.frozen_signal) {
            const age = Number(videoStream.frozen_signal_age_sec);
            detailParts.push(`Frozen source: repeated identical EVA frames${Number.isFinite(age) && age > 0 ? ` for ${formatLuxriotDuration(age)}` : ''}.`);
        }
        const videoIssue = classifyLuxriotStreamIssue(videoStream);
        if (videoIssue.backpressure) {
            detailParts.push('Aggregation backpressure: capture continues and the newest bounded batches are retained while older pending descriptions may be skipped.');
        } else if (videoIssue.hardError) {
            detailParts.push(`Summary error: ${videoIssue.hardError}`);
        }
        if (showProbeDiagnostics && analyticsStream?.last_error) detailParts.push(`Diagnostic probe error: ${analyticsStream.last_error}`);
        if (Number.isFinite(lastSnapshotLatency) && lastSnapshotLatency > 0) {
            detailParts.push(
                `Source snapshot ${lastSnapshotLatency.toFixed(1)}s${Number.isFinite(avgSnapshotLatency) && avgSnapshotLatency > 0 ? ` avg ${avgSnapshotLatency.toFixed(1)}s` : ''}${slowSnapshotCount > 0 ? ` · ${formatCompactCount(slowSnapshotCount)} slow` : ''}`
            );
        }
        if (activeCaptureSource === 'live_segment' && Number.isFinite(liveSegmentLatency) && liveSegmentLatency > 0) {
            const representedLabel = (
                Number.isFinite(liveSegmentRepresentedSeconds)
                && Number.isFinite(liveSegmentTargetSeconds)
                && liveSegmentTargetSeconds > 0
            )
                ? ` · attention ${liveSegmentRepresentedSeconds.toFixed(1)}/${liveSegmentTargetSeconds.toFixed(1)}s`
                : '';
            const realtimeRatio = Number.isFinite(liveSegmentRepresentedSeconds) && liveSegmentLatency > 0
                ? liveSegmentRepresentedSeconds / liveSegmentLatency
                : null;
            detailParts.push(
                `Analytics segment ${liveSegmentLatency.toFixed(1)}s · ${formatCompactCount(liveSegmentFrames)} dense frames${representedLabel}`
                + (Number.isFinite(realtimeRatio) ? ` · ${realtimeRatio.toFixed(2)}x realtime` : '')
                + (Number.isFinite(liveSegmentSummaryTargetSeconds) && liveSegmentSummaryTargetSeconds > 0 ? ` · descriptions every ${liveSegmentSummaryTargetSeconds.toFixed(1)}s` : '')
            );
        }
        if (activeCaptureSource === 'live_segment' && liveSegmentInflight) {
            detailParts.push(
                `Dense capture progress ${Number.isFinite(liveSegmentInflightRepresentedSeconds) ? liveSegmentInflightRepresentedSeconds.toFixed(1) : '?'}s`
                + (Number.isFinite(liveSegmentInflightTargetSeconds) ? ` / ${liveSegmentInflightTargetSeconds.toFixed(1)}s` : '')
                + (liveSegmentInflightRawBudget > 0 ? ` · ${formatCompactCount(liveSegmentInflightFrames)} / ${formatCompactCount(liveSegmentInflightRawBudget)} raw frames` : '')
                + '.'
            );
        }
        if (summaryInflight || summaryQueueDepth > 0) {
            detailParts.push(`VLM processing${summaryInflight ? ' active' : ''}${summaryQueueDepth > 0 ? ` · ${formatCompactCount(summaryQueueDepth)} queued batch${summaryQueueDepth === 1 ? '' : 'es'} / ${formatCompactCount(summaryQueueFrames)} frames` : ''}.`);
        }
        if (!detailParts.length && running) detailParts.push('Live summaries are collecting frames for the selected channel.');
        if (!detailParts.length && showProbeDiagnostics && probeRunning) {
            detailParts.push(probeShared
                ? 'Diagnostic probes are reading frames from the video-summary capture loop.'
                : 'Diagnostic capture is buffering frames for semantic/image probes.');
        }
        if (!detailParts.length) detailParts.push('Runtime state will update when the preview and stream status refresh.');

        setTextContentSafe(luxriotStreamName, channelLabel);
        setTextContentSafe(luxriotStreamChannel, hasChannel ? `#${channelId}` : '-');
        setTextContentSafe(luxriotStreamResolution, resolution);
        setTextContentSafe(luxriotStreamCadence, formatLuxriotCadence(intervalSec));
        setTextContentSafe(luxriotStreamBatch, batchSize > 0 ? formatCompactCount(batchSize) : '-');
        setTextContentSafe(luxriotStreamModel, selectedLuxriotModelLabel(videoStream));
        setTextContentSafe(luxriotStreamQueue, queueLabel);
        setElementHidden(luxriotStreamProbesRow, !showProbeDiagnostics);
        if (showProbeDiagnostics) {
            setTextContentSafe(luxriotStreamProbes, probeLabel);
        }
        setTextContentSafe(luxriotStreamLastFrame, formatPreviewAge(luxriotPreviewMeta.loadedAt));
        setTextContentSafe(luxriotStreamDetail, detailParts.join(' '));
        if (luxriotStreamState) {
            luxriotStreamState.className = `luxriot-stream-state ${stateClass}`;
            luxriotStreamState.textContent = stateText;
        }
        updateLuxriotStreamHealthBadge(videoStream);
        if (luxriotContextToggleCaptureBtn) {
            luxriotContextToggleCaptureBtn.textContent = running ? 'Stop summaries' : 'Start summaries';
            luxriotContextToggleCaptureBtn.classList.toggle('primary', !running);
            luxriotContextToggleCaptureBtn.disabled = !hasChannel || luxriotCaptureBusy;
        }
        if (luxriotContextFlushCaptureBtn) {
            luxriotContextFlushCaptureBtn.disabled = !hasChannel || luxriotCaptureBusy || !running;
        }
    }

    function abortLuxriotPreviewRequest() {
        if (!luxriotPreviewAbortController) return;
        try {
            luxriotPreviewAbortController.abort();
        } catch (err) {
            // Best-effort cleanup only.
        }
        luxriotPreviewAbortController = null;
    }

    function luxriotMediaBrokerUrl(mediaKind, channelId, options = {}) {
        if (mediaKind === 'attention') {
            // Dense capture fills bounded (default 60 s) incremental windows;
            // sparse sources legitimately go tens of seconds between apex
            // frames. A 15 s freshness bound turned that cadence into a
            // permanent 409/static fallback on underfilled channels.
            return `/luxriot/attention_stream/${encodeURIComponent(String(channelId))}?max_age_sec=60`;
        }
        const params = new URLSearchParams();
        params.set('stream', String(options.stream || 'mainStream'));
        if (mediaKind === 'archive' && Number.isFinite(Number(options.timeMs))) {
            params.set('time_ms', String(Math.trunc(Number(options.timeMs))));
            if (Number.isFinite(Number(options.durationSec))) {
                params.set('duration_sec', String(Math.max(1, Math.trunc(Number(options.durationSec)))));
            }
        }
        return `/luxriot/media/${encodeURIComponent(mediaKind)}/${encodeURIComponent(String(channelId))}?${params.toString()}`;
    }

    function parseLuxriotMediaResponse(response) {
        const mediaKind = String(response.headers.get('X-EVA-Media-Kind') || '').trim().toLowerCase();
        const errorCode = String(response.headers.get('X-EVA-Media-Error') || '').trim();
        if (!response.ok || !['video', 'mjpeg'].includes(mediaKind)) {
            const error = new Error(errorCode === 'snapshot_only'
                ? 'Luxriot returned a static image instead of video.'
                : `Browser-playable media is unavailable${errorCode ? ` (${errorCode})` : ''}.`);
            error.code = errorCode || 'media_unavailable';
            error.fallbackUrl = String(response.headers.get('X-EVA-Media-Fallback') || '').trim();
            throw error;
        }
        const numericHeader = (name) => {
            const rawValue = String(response.headers.get(name) || '').trim();
            if (!/^\d+$/.test(rawValue)) return null;
            const parsed = Number(rawValue);
            return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : null;
        };
        return {
            mediaKind,
            contentType: String(response.headers.get('Content-Type') || ''),
            bounded: response.headers.get('X-EVA-Media-Bounded') === '1',
            attentionPreview: response.headers.get('X-EVA-Attention-Preview') === '1',
            renewAfterMs: numericHeader('X-EVA-Media-Renew-After-Ms'),
            streamStartTimeMs: numericHeader('X-Stream-Start-Time'),
            streamEndTimeMs: numericHeader('X-Stream-End-Time'),
            lastSampleTimestampMs: numericHeader('X-Stream-Last-Sample-Timestamp'),
            requestedTimeMs: numericHeader('X-EVA-Archive-Requested-Time-Ms'),
            resolvedTimeMs: numericHeader('X-EVA-Archive-Resolved-Time-Ms'),
            durationSec: numericHeader('X-EVA-Archive-Duration-Seconds'),
            frameAlignment: String(response.headers.get('X-EVA-Archive-Frame-Alignment') || '').trim(),
            html5Compatibility: String(response.headers.get('X-EVA-HTML5-Compatible') || '').trim(),
        };
    }

    function assertSameOriginMediaUrl(mediaUrl) {
        const normalizedUrl = String(mediaUrl || '');
        if (
            !normalizedUrl.startsWith('/luxriot/media/')
            && !normalizedUrl.startsWith('/luxriot/attention_stream/')
        ) {
            throw new Error('Media broker URL must be same-origin.');
        }
        return normalizedUrl;
    }

    async function negotiateLuxriotMedia(mediaUrl, controller, timeoutMs = 6000) {
        const normalizedUrl = assertSameOriginMediaUrl(mediaUrl);
        const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
        try {
            const response = await fetch(normalizedUrl, {
                method: 'HEAD',
                cache: 'no-store',
                signal: controller.signal,
            });
            return parseLuxriotMediaResponse(response);
        } finally {
            window.clearTimeout(timeoutId);
        }
    }

    async function fetchLuxriotMediaBlob(mediaUrl, controller, timeoutMs) {
        const normalizedUrl = assertSameOriginMediaUrl(mediaUrl);
        const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
        try {
            const response = await fetch(normalizedUrl, {
                method: 'GET',
                cache: 'no-store',
                signal: controller.signal,
            });
            const metadata = parseLuxriotMediaResponse(response);
            const blob = await response.blob();
            if (!blob.size) throw new Error('The archive media response was empty.');
            return { ...metadata, blob };
        } finally {
            window.clearTimeout(timeoutId);
        }
    }

    function ensureLuxriotPreviewVideo() {
        if (luxriotPreviewVideo && luxriotPreviewVideo.isConnected) return luxriotPreviewVideo;
        if (!luxriotViewport) return null;
        const video = document.createElement('video');
        video.className = 'luxriot-operator-video';
        video.autoplay = true;
        video.muted = true;
        video.controls = true;
        video.playsInline = true;
        video.preload = 'metadata';
        video.setAttribute('aria-label', 'Luxriot operator live video');
        Object.assign(video.style, {
            width: '100%',
            height: '100%',
            maxHeight: 'none',
            objectFit: 'contain',
            background: '#000',
            display: 'none',
        });
        luxriotViewport.insertBefore(video, luxriotOverlay || null);
        luxriotPreviewVideo = video;
        return video;
    }

    function ensureLuxriotPreviewRetryButton() {
        if (luxriotPreviewRetryBtn && luxriotPreviewRetryBtn.isConnected) return luxriotPreviewRetryBtn;
        if (!luxriotViewport) return null;
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'feature-btn';
        button.textContent = 'Retry video';
        button.hidden = true;
        button.addEventListener('click', () => startLuxriotPreview());
        Object.assign(button.style, {
            position: 'absolute',
            right: '12px',
            bottom: '12px',
            zIndex: '5',
        });
        luxriotViewport.appendChild(button);
        luxriotPreviewRetryBtn = button;
        return button;
    }

    function ensureLuxriotPreviewTransportButton() {
        if (luxriotPreviewTransportBtn && luxriotPreviewTransportBtn.isConnected) return luxriotPreviewTransportBtn;
        if (!luxriotViewport) return null;
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'feature-btn';
        button.addEventListener('click', () => {
            luxriotPreferFullOperatorMedia = !luxriotPreferFullOperatorMedia;
            luxriotPreviewNegotiation = null;
            startLuxriotPreview();
        });
        Object.assign(button.style, {
            position: 'absolute',
            right: '12px',
            top: '12px',
            zIndex: '5',
        });
        luxriotViewport.appendChild(button);
        luxriotPreviewTransportBtn = button;
        return button;
    }

    function syncLuxriotPreviewTransportButton(channelId = getSelectedLuxriotChannel()) {
        const button = ensureLuxriotPreviewTransportButton();
        if (!button) return;
        const videoStream = selectedLuxriotStream(channelId, 'video');
        const attentionAvailable = Boolean(videoStream?.running);
        button.hidden = !attentionAvailable;
        button.textContent = luxriotPreferFullOperatorMedia ? 'Model view' : 'Full live';
        button.title = luxriotPreferFullOperatorMedia
            ? 'Return to the exact per-second EVA attention frames without opening another recorder stream.'
            : 'Open a second smooth recorder stream. This may reduce dense analytics throughput on constrained sources.';
    }

    function maybeSwitchLuxriotPreviewToAttention() {
        if (luxriotAttentionSwitchPending || luxriotPreferFullOperatorMedia || currentMode !== 'video') return;
        const channelId = getSelectedLuxriotChannel();
        const videoStream = selectedLuxriotStream(channelId, 'video');
        if (!Number.isFinite(channelId) || !videoStream?.running) return;
        const currentSource = String(
            luxriotPreviewVideo?.currentSrc
            || luxriotPreviewVideo?.getAttribute('src')
            || luxriotPreviewImg?.getAttribute('src')
            || ''
        );
        if (currentSource.includes('/luxriot/attention_stream/')) return;
        luxriotAttentionSwitchPending = true;
        queueMicrotask(() => {
            luxriotAttentionSwitchPending = false;
            if (luxriotPreferFullOperatorMedia || currentMode !== 'video') return;
            const selectedChannelId = getSelectedLuxriotChannel();
            if (selectedChannelId !== channelId || !selectedLuxriotStream(channelId, 'video')?.running) return;
            luxriotPreviewNegotiation = null;
            startLuxriotPreview();
        });
    }

    function setLuxriotOperatorMediaState(state, options = {}) {
        const normalized = ['idle', 'loading', 'playing', 'degraded', 'error'].includes(state) ? state : 'error';
        const failed = normalized === 'error';
        const degraded = normalized === 'degraded';
        luxriotPreviewMeta = {
            ...luxriotPreviewMeta,
            width: Number(options.width ?? luxriotPreviewMeta.width) || 0,
            height: Number(options.height ?? luxriotPreviewMeta.height) || 0,
            loadedAt: Number(options.loadedAt ?? luxriotPreviewMeta.loadedAt) || 0,
            frameAgeSec: null,
            stale: false,
            failed,
            errorCode: String(options.errorCode || (failed ? 'media_error' : '')),
            errorText: String(options.detail || ''),
            mediaState: normalized,
            mediaKind: Object.prototype.hasOwnProperty.call(options, 'mediaKind')
                ? String(options.mediaKind || '')
                : String(luxriotPreviewMeta.mediaKind || ''),
            degraded,
        };
        if (luxriotViewport) {
            luxriotViewport.dataset.mediaState = normalized;
            luxriotViewport.classList.toggle('is-signal-lost', failed);
            luxriotViewport.classList.toggle('is-signal-stale', degraded);
        }
        if (luxriotOverlay) {
            const defaultText = {
                idle: '',
                loading: 'Loading live video…',
                playing: '',
                degraded: 'Static frame fallback — not video',
                error: 'Video unavailable',
            };
            luxriotOverlay.textContent = String(options.overlay || defaultText[normalized] || '');
        }
        const retry = ensureLuxriotPreviewRetryButton();
        if (retry) retry.hidden = !(degraded || failed);
        syncLuxriotPreviewTransportButton();
        updateLuxriotStreamContext();
    }

    function clearLuxriotPreviewVideo() {
        if (!luxriotPreviewVideo) return;
        luxriotPreviewVideo.onloadedmetadata = null;
        luxriotPreviewVideo.oncanplay = null;
        luxriotPreviewVideo.onplaying = null;
        luxriotPreviewVideo.onwaiting = null;
        luxriotPreviewVideo.onstalled = null;
        luxriotPreviewVideo.onprogress = null;
        luxriotPreviewVideo.ontimeupdate = null;
        luxriotPreviewVideo.onended = null;
        luxriotPreviewVideo.onerror = null;
        try {
            luxriotPreviewVideo.pause();
        } catch (_) {
            // Best-effort media cleanup.
        }
        luxriotPreviewVideo.removeAttribute('src');
        try {
            luxriotPreviewVideo.load();
        } catch (_) {
            // Best-effort media cleanup.
        }
        luxriotPreviewVideo.style.display = 'none';
    }

    function replaceLuxriotPreviewImageElement() {
        if (!luxriotPreviewImg || !luxriotPreviewImg.parentNode) return;
        const previous = luxriotPreviewImg;
        previous.onload = null;
        previous.onerror = null;
        const replacement = previous.cloneNode(false);
        replacement.removeAttribute('src');
        replacement.style.display = 'none';
        previous.replaceWith(replacement);
        luxriotPreviewImg = replacement;
    }

    function stopLuxriotPreview(clearImage = false) {
        if (luxriotPreviewTimer) {
            clearTimeout(luxriotPreviewTimer);
            luxriotPreviewTimer = null;
        }
        if (luxriotPreviewRenewTimer) {
            clearTimeout(luxriotPreviewRenewTimer);
            luxriotPreviewRenewTimer = null;
        }
        if (luxriotPreviewStallTimer) {
            clearTimeout(luxriotPreviewStallTimer);
            luxriotPreviewStallTimer = null;
        }
        luxriotPreviewRequestSeq += 1;
        abortLuxriotPreviewRequest();
        luxriotPreviewLoading = false;
        clearLuxriotPreviewVideo();
        if (luxriotPreviewImg) {
            luxriotPreviewImg.onload = null;
            luxriotPreviewImg.onerror = null;
        }
        if (luxriotPreviewRetryBtn) luxriotPreviewRetryBtn.hidden = true;
        if (clearImage) {
            if (luxriotPreviewImg) {
                luxriotPreviewImg.removeAttribute('src');
                luxriotPreviewImg.style.display = 'none';
            }
            if (luxriotOverlay) {
                luxriotOverlay.textContent = '';
            }
            if (luxriotViewport) {
                luxriotViewport.classList.remove('is-signal-lost', 'is-signal-stale');
            }
            luxriotPreviewMeta = { width: 0, height: 0, loadedAt: 0, frameAgeSec: null, stale: false, staleAfterSec: 0, lostAfterSec: 0, failed: false, errorCode: '', errorText: '', mediaState: 'idle', mediaKind: '', degraded: false };
            if (luxriotViewport) delete luxriotViewport.dataset.mediaState;
            updateLuxriotStreamContext();
        }
    }

    function setLuxriotPreviewSignalLost(errorCode, errorText) {
        clearLuxriotPreviewVideo();
        if (luxriotPreviewImg) {
            luxriotPreviewImg.removeAttribute('src');
            luxriotPreviewImg.style.display = 'none';
        }
        setLuxriotOperatorMediaState('error', {
            errorCode: errorCode || 'preview_error',
            detail: errorText || 'Live video is unavailable.',
            overlay: errorCode === 'signal_lost'
                ? 'Signal lost'
                : 'Video unavailable',
        });
    }

    function stopLuxriotSummaryPoll() {
        if (luxriotSummaryTimer) {
            clearInterval(luxriotSummaryTimer);
            luxriotSummaryTimer = null;
        }
        luxriotSummaryRefreshQueued = null;
        cancelLuxriotSummaryRequest();
    }

    function luxriotSummaryRequestKey(channelId) {
        const context = getCurrentSummaryRollupContext();
        const range = normalizeSummaryRangePreset(luxriotSummaryRangePreset);
        return JSON.stringify({
            channelId: Number(channelId),
            view: isRollupViewActive() ? 'rollup' : 'summary',
            level: normalizeSummaryLevel(context?.level || luxriotSummaryLevel),
            sourceIds: Array.isArray(context?.sourceIds) ? context.sourceIds.map(String) : [],
            run: normalizeSummaryRun(luxriotSummaryRunFilter),
            range,
            fromTs: range === 'custom' && Number.isFinite(luxriotSummaryFromTs) ? luxriotSummaryFromTs : null,
            toTs: range === 'custom' && Number.isFinite(luxriotSummaryToTs) ? luxriotSummaryToTs : null,
        });
    }

    function cancelLuxriotSummaryRequest() {
        luxriotSummaryRequestGeneration += 1;
        const active = luxriotSummaryActiveRequest;
        if (active && active.controller) {
            try {
                active.controller.abort();
            } catch (_) {
                // Generation invalidation still prevents a stale render.
            }
        }
        luxriotSummaryActiveRequest = null;
    }

    function isCurrentLuxriotSummaryRequest(requestContext) {
        return Boolean(
            requestContext
            && requestContext.generation === luxriotSummaryRequestGeneration
            && luxriotSummaryActiveRequest === requestContext
            && !requestContext.controller.signal.aborted
            && currentMode === 'video'
            && getSelectedSummaryChannel() === requestContext.channelId
            && luxriotSummaryRequestKey(requestContext.channelId) === requestContext.requestKey
        );
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
        if (
            text === 'live'
            || text === 'today'
            || text === 'yesterday'
            || text === 'day_before_yesterday'
            || text === '6h'
            || text === '24h'
            || text === '3d'
            || text === '7d'
            || text === '30d'
            || text === 'all'
            || text === 'custom'
        ) {
            return text;
        }
        return 'live';
    }

    function summaryLocalDateParts(epochSec = null) {
        const hasEpoch = epochSec !== null && epochSec !== '' && Number.isFinite(Number(epochSec));
        const date = hasEpoch
            ? new Date(Number(epochSec) * 1000)
            : new Date();
        const parts = new Intl.DateTimeFormat('en-CA', {
            timeZone: luxriotDisplayTimezone,
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
        }).formatToParts(date);
        const values = {};
        parts.forEach((part) => {
            if (part.type !== 'literal') values[part.type] = Number(part.value);
        });
        return { year: values.year, month: values.month, day: values.day };
    }

    function shiftSummaryLocalDate(parts, dayDelta) {
        const shifted = new Date(Date.UTC(
            Number(parts.year),
            Number(parts.month) - 1,
            Number(parts.day) + Number(dayDelta || 0),
        ));
        return {
            year: shifted.getUTCFullYear(),
            month: shifted.getUTCMonth() + 1,
            day: shifted.getUTCDate(),
        };
    }

    function summaryLocalToEpoch(parts) {
        const desiredMs = Date.UTC(
            Number(parts.year),
            Number(parts.month) - 1,
            Number(parts.day),
            Number(parts.hour || 0),
            Number(parts.minute || 0),
            Number(parts.second || 0),
        );
        let guessMs = desiredMs;
        const formatter = new Intl.DateTimeFormat('en-CA', {
            timeZone: luxriotDisplayTimezone,
            hourCycle: 'h23',
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
        });
        for (let attempt = 0; attempt < 3; attempt += 1) {
            const observed = {};
            formatter.formatToParts(new Date(guessMs)).forEach((part) => {
                if (part.type !== 'literal') observed[part.type] = Number(part.value);
            });
            const observedMs = Date.UTC(
                observed.year,
                observed.month - 1,
                observed.day,
                observed.hour || 0,
                observed.minute || 0,
                observed.second || 0,
            );
            const correction = desiredMs - observedMs;
            guessMs += correction;
            if (Math.abs(correction) < 1000) break;
        }
        return guessMs / 1000;
    }

    function summaryLocalDayBounds(dayDelta = 0) {
        const today = summaryLocalDateParts();
        const startParts = shiftSummaryLocalDate(today, dayDelta);
        const endParts = shiftSummaryLocalDate(startParts, 1);
        return {
            fromTs: summaryLocalToEpoch(startParts),
            toTs: summaryLocalToEpoch(endParts) - 0.001,
        };
    }

    function getSummaryRangeBounds(rangePreset, nowSec = null) {
        const normalized = normalizeSummaryRangePreset(rangePreset);
        const now = Number.isFinite(nowSec) ? Number(nowSec) : Math.floor(Date.now() / 1000);
        const toTs = now;
        if (normalized === 'live' || normalized === 'all') return { fromTs: null, toTs: null };
        if (normalized === 'today') {
            const bounds = summaryLocalDayBounds(0);
            return { fromTs: bounds.fromTs, toTs: now };
        }
        if (normalized === 'yesterday') return summaryLocalDayBounds(-1);
        if (normalized === 'day_before_yesterday') return summaryLocalDayBounds(-2);
        if (normalized === '6h') return { fromTs: toTs - 6 * 3600, toTs };
        if (normalized === '24h') return { fromTs: toTs - 24 * 3600, toTs };
        if (normalized === '3d') return { fromTs: toTs - 3 * 24 * 3600, toTs };
        if (normalized === '7d') return { fromTs: toTs - 7 * 24 * 3600, toTs };
        if (normalized === '30d') return { fromTs: toTs - 30 * 24 * 3600, toTs };
        return { fromTs: null, toTs: null };
    }

    function getSummaryRangeLabel() {
        const preset = normalizeSummaryRangePreset(luxriotSummaryRangePreset);
        if (preset === 'live') return 'live';
        if (preset === 'today') return 'today';
        if (preset === 'yesterday') return 'yesterday';
        if (preset === 'day_before_yesterday') return 'day before yesterday';
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

    function formatSummaryLocalTimestamp(ts, options = {}) {
        if (ts === null || ts === '' || typeof ts === 'undefined') return 'n/a';
        const sec = Number(ts);
        if (!Number.isFinite(sec)) return 'n/a';
        return new Intl.DateTimeFormat(undefined, {
            timeZone: luxriotDisplayTimezone,
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: options.dateOnly ? undefined : '2-digit',
            minute: options.dateOnly ? undefined : '2-digit',
        }).format(new Date(sec * 1000));
    }

    function getSummaryEffectiveBounds() {
        const preset = normalizeSummaryRangePreset(luxriotSummaryRangePreset);
        if (preset === 'custom') {
            return { fromTs: luxriotSummaryFromTs, toTs: luxriotSummaryToTs };
        }
        return getSummaryRangeBounds(preset);
    }

    function summaryPeriodLabel() {
        const preset = normalizeSummaryRangePreset(luxriotSummaryRangePreset);
        const bounds = getSummaryEffectiveBounds();
        if (preset === 'today' || preset === 'yesterday' || preset === 'day_before_yesterday') {
            return formatSummaryLocalTimestamp(bounds.fromTs, { dateOnly: true });
        }
        if (Number.isFinite(bounds.fromTs) && Number.isFinite(bounds.toTs)) {
            return `${formatSummaryLocalTimestamp(bounds.fromTs)} – ${formatSummaryLocalTimestamp(bounds.toTs)}`;
        }
        return 'All retained history';
    }

    function syncSummaryRangeUI() {
        const preset = normalizeSummaryRangePreset(luxriotSummaryRangePreset);
        if (luxriotSummaryRangeSelect) {
            luxriotSummaryRangeSelect.value = preset;
        }
        if (luxriotSummaryCustomTime) {
            luxriotSummaryCustomTime.classList.toggle('is-hidden', preset !== 'custom');
        }
        if (luxriotSummaryDateNav) {
            luxriotSummaryDateNav.classList.toggle('is-hidden', preset === 'live');
        }
        if (luxriotSummaryDateLabel && preset !== 'live') {
            luxriotSummaryDateLabel.textContent = summaryPeriodLabel();
        }
        if (luxriotSummaryNextPeriodBtn) {
            const bounds = getSummaryEffectiveBounds();
            luxriotSummaryNextPeriodBtn.disabled = !Number.isFinite(bounds.toTs) || bounds.toTs >= (Date.now() / 1000) - 1;
        }
    }

    function parseSummaryDatetimeInput(value) {
        const text = String(value || '').trim();
        if (!text) return null;
        const match = text.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/);
        if (!match) return null;
        return summaryLocalToEpoch({
            year: Number(match[1]),
            month: Number(match[2]),
            day: Number(match[3]),
            hour: Number(match[4]),
            minute: Number(match[5]),
        });
    }

    function formatSummaryDatetimeInput(ts) {
        if (ts === null || ts === '' || typeof ts === 'undefined') return '';
        const sec = Number(ts);
        if (!Number.isFinite(sec)) return '';
        const values = {};
        new Intl.DateTimeFormat('en-CA', {
            timeZone: luxriotDisplayTimezone,
            hourCycle: 'h23',
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
        }).formatToParts(new Date(sec * 1000)).forEach((part) => {
            if (part.type !== 'literal') values[part.type] = part.value;
        });
        const yyyy = values.year;
        const mm = values.month;
        const dd = values.day;
        const hh = values.hour;
        const mi = values.minute;
        return `${yyyy}-${mm}-${dd}T${hh}:${mi}`;
    }

    function readSummaryFiltersFromInputs() {
        const rangePreset = normalizeSummaryRangePreset(luxriotSummaryRangeSelect ? luxriotSummaryRangeSelect.value : luxriotSummaryRangePreset);
        const currentRun = normalizeSummaryRun(luxriotSummaryRunSelect ? luxriotSummaryRunSelect.value : luxriotSummaryRunFilter);
        const isExplicitRun = !['latest', 'live', 'all'].includes(String(currentRun).toLowerCase());
        const run = rangePreset === 'live' ? 'live' : (isExplicitRun ? currentRun : 'all');
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
        const livePeriod = luxriotSummaryRangePreset === 'live';
        luxriotSummaryFollowLive = livePeriod;
        if (livePeriod) luxriotSummaryAutoRefresh = true;
        resetSummaryArchivePaging();
        applySummaryResolutionMode();
    }

    function clearSummaryFilters() {
        luxriotSummaryRunFilter = 'live';
        luxriotSummaryRangePreset = 'live';
        luxriotSummaryFromTs = null;
        luxriotSummaryToTs = null;
        if (luxriotSummaryRunSelect) {
            luxriotSummaryRunSelect.value = 'live';
        }
        if (luxriotSummaryRangeSelect) {
            luxriotSummaryRangeSelect.value = 'live';
        }
        if (luxriotSummaryFromInput) {
            luxriotSummaryFromInput.value = '';
        }
        if (luxriotSummaryToInput) {
            luxriotSummaryToInput.value = '';
        }
        syncSummaryRangeUI();
        luxriotSummaryFollowLive = true;
        luxriotSummaryAutoRefresh = true;
        resetSummaryArchivePaging();
        applySummaryResolutionMode();
    }

    function shiftSelectedSummaryPeriod(direction) {
        const delta = Number(direction) < 0 ? -1 : 1;
        const preset = normalizeSummaryRangePreset(luxriotSummaryRangePreset);
        const bounds = getSummaryEffectiveBounds();
        if (!Number.isFinite(bounds.fromTs) || !Number.isFinite(bounds.toTs)) return false;
        let fromTs;
        let toTs;
        if (preset === 'today' || preset === 'yesterday' || preset === 'day_before_yesterday') {
            const currentDate = summaryLocalDateParts(bounds.fromTs + 1);
            const nextDate = shiftSummaryLocalDate(currentDate, delta);
            const afterNext = shiftSummaryLocalDate(nextDate, 1);
            fromTs = summaryLocalToEpoch(nextDate);
            toTs = summaryLocalToEpoch(afterNext) - 0.001;
        } else {
            const durationSec = Math.max(60, bounds.toTs - bounds.fromTs);
            fromTs = bounds.fromTs + delta * durationSec;
            toTs = bounds.toTs + delta * durationSec;
        }
        const now = Date.now() / 1000;
        if (toTs > now) {
            const durationSec = toTs - fromTs;
            toTs = now;
            fromTs = Math.max(0, toTs - durationSec);
        }
        luxriotSummaryRangePreset = 'custom';
        luxriotSummaryRunFilter = 'all';
        luxriotSummaryFromTs = fromTs;
        luxriotSummaryToTs = toTs;
        luxriotSummaryFollowLive = false;
        if (luxriotSummaryRangeSelect) luxriotSummaryRangeSelect.value = 'custom';
        if (luxriotSummaryFromInput) luxriotSummaryFromInput.value = formatSummaryDatetimeInput(fromTs);
        if (luxriotSummaryToInput) luxriotSummaryToInput.value = formatSummaryDatetimeInput(toTs);
        resetSummaryArchivePaging();
        applySummaryResolutionMode();
        syncSummaryRangeUI();
        setSummaryUnread(0);
        return true;
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

    function normalizeSummaryResolutionMode(value) {
        const text = String(value || '').trim().toUpperCase();
        if (text === 'AUTO') return 'AUTO';
        return normalizeSummaryLevel(text);
    }

    function resolveAutoSummaryLevel() {
        const preset = normalizeSummaryRangePreset(luxriotSummaryRangePreset);
        if (preset === 'live') return 'L0';
        if (preset === 'today' || preset === 'yesterday' || preset === 'day_before_yesterday' || preset === '24h') {
            return 'L1';
        }
        if (preset === '3d' || preset === '7d') return 'L2';
        if (preset === '30d' || preset === 'all') return 'L3';
        const bounds = getSummaryEffectiveBounds();
        const durationSec = Number(bounds.toTs) - Number(bounds.fromTs);
        if (!Number.isFinite(durationSec) || durationSec <= 0) return 'L3';
        if (durationSec <= 8 * 3600) return 'L0';
        if (durationSec <= 36 * 3600) return 'L1';
        if (durationSec <= 8 * 24 * 3600) return 'L2';
        return 'L3';
    }

    function applySummaryResolutionMode() {
        const mode = normalizeSummaryResolutionMode(luxriotSummaryResolutionMode);
        const level = mode === 'AUTO' ? resolveAutoSummaryLevel() : normalizeSummaryLevel(mode);
        luxriotSummaryLevel = level;
        luxriotSummaryRollupStack = [{ level, sourceIds: null, label: level }];
        if (luxriotSummaryLevelSelect) {
            luxriotSummaryLevelSelect.value = mode;
        }
        updateSummaryControlsUI();
        return level;
    }

    function setSummaryBaseLevel(level, preserveResolutionMode = false) {
        const normalized = normalizeSummaryLevel(level);
        luxriotSummaryLevel = normalized;
        if (!preserveResolutionMode) luxriotSummaryResolutionMode = normalized;
        luxriotSummaryRollupStack = [{ level: normalized, sourceIds: null, label: normalized }];
        if (luxriotSummaryLevelSelect) {
            luxriotSummaryLevelSelect.value = preserveResolutionMode
                ? normalizeSummaryResolutionMode(luxriotSummaryResolutionMode)
                : normalized;
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
        const historicalMode = !isLiveSummaryPeriod();
        if (luxriotSummaryFollowBtn) {
            const liveOn = !historicalMode && !rollupMode && luxriotSummaryAutoRefresh && luxriotSummaryFollowLive;
            luxriotSummaryFollowBtn.classList.toggle('primary', liveOn);
            luxriotSummaryFollowBtn.textContent = historicalMode
                ? '▶ Go live'
                : rollupMode
                    ? '▶ Live observations'
                    : (liveOn ? '⏸ Live ON' : '▶ Live OFF');
            luxriotSummaryFollowBtn.disabled = false;
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
        updateSummaryArchivePagingUI();
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
            Object.keys(luxriotChannelMetaById).forEach((key) => delete luxriotChannelMetaById[key]);
            if (!channels.length) {
                luxriotChannelSelect.innerHTML = '<option value="">No channels</option>';
                if (luxriotSummaryChannelSelect) {
                    luxriotSummaryChannelSelect.innerHTML = '<option value="">No channels</option>';
                }
                updateLuxriotStreamContext();
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
                    luxriotChannelMetaById[String(id)] = ch;
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
            syncLuxriotLiveIntervalInput(luxriotActiveChannel);
            updateLuxriotCaptureToggleButton(luxriotActiveChannel);
            updateLuxriotStreamContext();
            setLuxriotStatus(`Loaded ${channels.length} channels`);
        } catch (err) {
            Object.keys(luxriotChannelNameById).forEach((key) => delete luxriotChannelNameById[key]);
            Object.keys(luxriotChannelMetaById).forEach((key) => delete luxriotChannelMetaById[key]);
            luxriotChannelSelect.innerHTML = '<option value="">Load failed</option>';
            if (luxriotSummaryChannelSelect) {
                luxriotSummaryChannelSelect.innerHTML = '<option value="">Load failed</option>';
            }
            updateLuxriotCaptureToggleButton();
            updateLuxriotStreamContext();
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
        updateLuxriotStreamContext();
    }

    function isLuxriotCaptureRunning(channelId) {
        const parsed = parseInt(String(channelId || ''), 10);
        if (!Number.isFinite(parsed)) return false;
        return Boolean(luxriotCaptureRunningByChannel[String(parsed)]);
    }

    function updateLuxriotCaptureToggleButton(channelIdOverride = null) {
        const channelId = Number.isFinite(channelIdOverride) ? channelIdOverride : getSelectedLuxriotChannel();
        const selectedRaw = luxriotChannelSelect ? String(luxriotChannelSelect.value || '').trim() : String(channelId || '');
        const hasChannel = Number.isFinite(channelId) && channelId > 0 && Boolean(selectedRaw);
        const running = isLuxriotCaptureRunning(channelId);
        [luxriotToggleCaptureBtn, luxriotContextToggleCaptureBtn].forEach((button) => {
            if (!button) return;
            button.textContent = running ? 'Stop summaries' : 'Start summaries';
            button.classList.toggle('primary', !running);
            button.disabled = luxriotCaptureBusy || !hasChannel;
        });
        if (luxriotContextFlushCaptureBtn) {
            luxriotContextFlushCaptureBtn.disabled = luxriotCaptureBusy || !hasChannel || !running;
        }
        updateLuxriotStreamContext();
    }

    function getLuxriotPromptInputByTab(tab) {
        const normalized = String(tab || '').trim().toLowerCase();
        if (normalized === 'stream') return luxriotSystemPromptInput;
        if (normalized === 'alerts') return luxriotAlertPolicyPromptInput;
        if (normalized === 'l1') return luxriotRollupPromptL1Input;
        if (normalized === 'l2') return luxriotRollupPromptL2Input;
        if (normalized === 'l3') return luxriotRollupPromptL3Input;
        if (normalized === 'json') return luxriotJsonAlertPromptInput;
        return luxriotSystemPromptInput;
    }

    function getLuxriotPromptTabLabel(tab) {
        const normalized = String(tab || '').trim().toLowerCase();
        if (normalized === 'stream') return 'Stream system prompt';
        if (normalized === 'alerts') return 'Alert criteria';
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
        if (normalized === 'alerts') {
            return 'Editing channel-specific alert criteria. EVA AI still keeps the general safety baseline active.';
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
            return 'Advanced: editing the machine-readable ALERTS_JSON contract. Do this only when changing parser/schema behavior.';
        }
        return 'Editing system prompt.';
    }

    function getLuxriotPromptLayerForTab(tab) {
        const layers = luxriotPromptLayers && typeof luxriotPromptLayers === 'object' ? luxriotPromptLayers : null;
        if (!layers) return null;
        const normalized = String(tab || '').trim().toLowerCase();
        if (normalized === 'stream') {
            return layers.stream && typeof layers.stream === 'object' ? layers.stream : null;
        }
        if (normalized === 'alerts') {
            return layers.alerts && typeof layers.alerts === 'object' ? layers.alerts : null;
        }
        if (normalized === 'json') {
            return layers.json && typeof layers.json === 'object' ? layers.json : null;
        }
        const rollups = layers.rollups && typeof layers.rollups === 'object' ? layers.rollups : {};
        const level = normalized.toUpperCase();
        return rollups[level] && typeof rollups[level] === 'object' ? rollups[level] : null;
    }

    function updateLuxriotPromptLayerDetails() {
        if (!luxriotPromptLayerDetails || !luxriotPromptLayerContent) return;
        const layer = getLuxriotPromptLayerForTab(luxriotPromptModalTab);
        if (!layer) {
            luxriotPromptLayerDetails.classList.add('is-hidden');
            luxriotPromptLayerContent.textContent = '';
            return;
        }
        const lines = [];
        const warnings = Array.isArray(layer.warnings) ? layer.warnings : [];
        warnings.forEach((warning) => {
            const text = String(warning || '').trim();
            if (text) lines.push(`WARNING: ${text}`);
        });
        const notes = Array.isArray(layer.notes) ? layer.notes : [];
        lines.push('Editable prompt: the text box above.');
        notes.forEach((note) => {
            const text = String(note || '').trim();
            if (text) lines.push(`- ${text}`);
        });
        const backendInstructions = String(layer.backend_instructions || '').trim();
        if (backendInstructions) {
            lines.push(`Backend instructions appended by EVA AI:\n${backendInstructions}`);
        }
        const backendMemory = String(layer.backend_memory || layer.active_memory || '').trim();
        if (backendMemory) {
            lines.push(`Active channel memory appended by EVA AI:\n${backendMemory}`);
        } else {
            lines.push('Active channel memory: none for the selected channel yet.');
        }
        luxriotPromptLayerContent.textContent = lines.join('\n\n');
        luxriotPromptLayerDetails.classList.remove('is-hidden');
    }

    function collectLuxriotPromptSettings() {
        const current = {
            stream_system_prompt: luxriotSystemPromptInput ? String(luxriotSystemPromptInput.value || '') : '',
            alert_policy_prompt: luxriotAlertPolicyPromptInput ? String(luxriotAlertPolicyPromptInput.value || '') : '',
            rollup_prompts: {
                L1: luxriotRollupPromptL1Input ? String(luxriotRollupPromptL1Input.value || '') : '',
                L2: luxriotRollupPromptL2Input ? String(luxriotRollupPromptL2Input.value || '') : '',
                L3: luxriotRollupPromptL3Input ? String(luxriotRollupPromptL3Input.value || '') : '',
            },
        };
        if (userHasPermission('bookmarks:create')) {
            current.json_alert_prompt = luxriotJsonAlertPromptInput ? String(luxriotJsonAlertPromptInput.value || '') : '';
            current.bookmark_enabled = luxriotBookmarkEnabledInput ? Boolean(luxriotBookmarkEnabledInput.checked) : false;
            current.bookmark_cooldown_sec = luxriotBookmarkCooldownInput
                ? Math.max(0, Number.parseFloat(String(luxriotBookmarkCooldownInput.value || '0')) || 0)
                : 0;
        }
        current.capture_selector_bias = luxriotSelectorBiasInput
            ? String(luxriotSelectorBiasInput.value || 'auto')
            : 'auto';
        const baseline = luxriotPromptLoadedSettings && typeof luxriotPromptLoadedSettings === 'object'
            ? luxriotPromptLoadedSettings
            : {};
        const payload = {};
        for (const field of ['stream_system_prompt', 'alert_policy_prompt']) {
            if (String(current[field] || '') !== String(baseline[field] || '')) {
                payload[field] = current[field];
            }
        }
        const changedRollups = {};
        for (const level of ['L1', 'L2', 'L3']) {
            const currentValue = String(current.rollup_prompts?.[level] || '');
            const baselineValue = String(baseline.rollup_prompts?.[level] || '');
            if (currentValue !== baselineValue) {
                changedRollups[level] = currentValue;
            }
        }
        if (Object.keys(changedRollups).length) {
            payload.rollup_prompts = changedRollups;
        }
        if (userHasPermission('bookmarks:create')) {
            if (String(current.json_alert_prompt || '') !== String(baseline.json_alert_prompt || '')) {
                payload.json_alert_prompt = current.json_alert_prompt;
            }
            if (Boolean(current.bookmark_enabled) !== Boolean(baseline.bookmark_enabled)) {
                payload.bookmark_enabled = current.bookmark_enabled;
            }
            if (Number(current.bookmark_cooldown_sec || 0) !== Number(baseline.bookmark_cooldown_sec || 0)) {
                payload.bookmark_cooldown_sec = current.bookmark_cooldown_sec;
            }
        }
        if (String(current.capture_selector_bias || 'auto') !== String(baseline.capture_selector_bias || 'auto')) {
            payload.capture_selector_bias = current.capture_selector_bias;
        }
        return payload;
    }

    function applyLuxriotPromptSettingsFromPayload(payload) {
        const settings = payload && typeof payload === 'object' ? payload : {};
        if (luxriotSystemPromptInput && Object.prototype.hasOwnProperty.call(settings, 'stream_system_prompt')) {
            luxriotSystemPromptInput.value = String(settings.stream_system_prompt || '');
        }
        if (luxriotAlertPolicyPromptInput && Object.prototype.hasOwnProperty.call(settings, 'alert_policy_prompt')) {
            luxriotAlertPolicyPromptInput.value = String(settings.alert_policy_prompt || '');
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
        if (luxriotLiveIntervalInput && Object.prototype.hasOwnProperty.call(settings, 'capture_interval_sec')) {
            const payloadChannel = parseInt(String(settings.channel_id ?? ''), 10);
            const selectedChannel = getSelectedLuxriotChannel();
            const interval = normalizeLuxriotLiveInterval(settings.capture_interval_sec);
            if (interval !== null && (!Number.isFinite(payloadChannel) || payloadChannel === selectedChannel)) {
                if (
                    document.activeElement !== luxriotLiveIntervalInput
                    && !isLuxriotLiveIntervalDirty(selectedChannel)
                ) {
                    luxriotLiveIntervalInput.value = formatLuxriotLiveIntervalInput(interval);
                    storeLuxriotLiveInterval(selectedChannel, interval);
                    updateLuxriotBatchInfo();
                }
            }
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
        if (luxriotSelectorBiasInput && Object.prototype.hasOwnProperty.call(settings, 'capture_selector_bias')) {
            const bias = String(settings.capture_selector_bias || 'auto').toLowerCase();
            luxriotSelectorBiasInput.value = ['auto', 'action', 'clarity'].includes(bias) ? bias : 'auto';
        }
        luxriotPromptLayers = settings.prompt_layers && typeof settings.prompt_layers === 'object'
            ? settings.prompt_layers
            : null;
        luxriotPromptSettingSources = settings.setting_sources && typeof settings.setting_sources === 'object'
            ? settings.setting_sources
            : null;
        luxriotPromptOverrideFields = Array.isArray(settings.override_fields)
            ? settings.override_fields.map((item) => String(item || ''))
            : [];
        luxriotPromptPersistence = settings.persistence && typeof settings.persistence === 'object'
            ? settings.persistence
            : null;
        luxriotPromptLoadedSettings = {
            stream_system_prompt: String(settings.stream_system_prompt || ''),
            alert_policy_prompt: String(settings.alert_policy_prompt || ''),
            rollup_prompts: {
                L1: String(rollupPrompts.L1 || ''),
                L2: String(rollupPrompts.L2 || ''),
                L3: String(rollupPrompts.L3 || ''),
            },
            json_alert_prompt: String(settings.json_alert_prompt || ''),
            bookmark_enabled: Boolean(settings.bookmark_enabled),
            bookmark_cooldown_sec: Math.max(0, Number(settings.bookmark_cooldown_sec || 0)),
            capture_selector_bias: String(settings.capture_selector_bias || 'auto').toLowerCase(),
        };
        const activeInput = getLuxriotPromptInputByTab(luxriotPromptModalTab);
        if (luxriotPromptModalInput && activeInput) {
            luxriotPromptModalInput.value = String(activeInput.value || '');
        }
        updateLuxriotPromptLayerDetails();
    }

    function getClearableLuxriotPromptOverrideFields() {
        const bookmarkFields = new Set([
            'bookmark_enabled',
            'bookmark_cooldown_sec',
            'json_alert_prompt',
        ]);
        return luxriotPromptOverrideFields.filter((field) => (
            !bookmarkFields.has(field) || userHasPermission('bookmarks:create')
        ));
    }

    function setLuxriotPromptApplyAvailability(loading = false) {
        const canManage = userHasPermission('prompts:manage');
        const selectedChannelId = getSelectedLuxriotChannel();
        const formMatchesChannel = Number.isFinite(luxriotPromptFormChannelId)
            && selectedChannelId === luxriotPromptFormChannelId;
        if (luxriotPromptApplyBtn) {
            luxriotPromptApplyBtn.disabled = Boolean(loading)
                || !canManage
                || !formMatchesChannel;
        }
        if (luxriotPromptResetBtn) {
            luxriotPromptResetBtn.disabled = Boolean(loading)
            || !canManage
                || !formMatchesChannel
                || getClearableLuxriotPromptOverrideFields().length === 0;
        }
    }

    function abortLuxriotPromptController(controller) {
        if (!controller) return;
        try {
            controller.abort();
        } catch (_) {
            // Abort is best-effort; generation checks remain authoritative.
        }
    }

    function invalidateLuxriotPromptRequests({ clearFormIdentity = false } = {}) {
        luxriotPromptRequestGeneration += 1;
        abortLuxriotPromptController(luxriotPromptLoadAbortController);
        abortLuxriotPromptController(luxriotPromptSaveAbortController);
        luxriotPromptLoadAbortController = null;
        luxriotPromptSaveAbortController = null;
        if (clearFormIdentity) {
            luxriotPromptFormChannelId = null;
            luxriotPromptLoadedSettings = null;
        }
        setLuxriotPromptApplyAvailability(false);
    }

    function isCurrentLuxriotPromptRequest(generation, controller, channelId) {
        return generation === luxriotPromptRequestGeneration
            && controller
            && !controller.signal.aborted
            && getSelectedLuxriotChannel() === channelId;
    }

    async function refreshLuxriotPromptSettings(showError = false, channelIdOverride = null) {
        const channelId = Number.isFinite(channelIdOverride)
            ? channelIdOverride
            : getSelectedLuxriotChannel();
        if (!Number.isFinite(channelId)) {
            return false;
        }
        invalidateLuxriotPromptRequests({ clearFormIdentity: luxriotPromptFormChannelId !== channelId });
        const generation = luxriotPromptRequestGeneration;
        const controller = new AbortController();
        luxriotPromptLoadAbortController = controller;
        setLuxriotPromptApplyAvailability(true);
        try {
            const params = new URLSearchParams();
            params.set('channel_id', String(channelId));
            const response = await fetch(`/luxriot/prompt_settings?${params.toString()}`, {
                signal: controller.signal,
            });
            const data = await parseApiJson(response, 'Failed to load prompt settings');
            if (!isCurrentLuxriotPromptRequest(generation, controller, channelId)) {
                return false;
            }
            const responseChannel = parseInt(String(data.channel_id ?? channelId), 10);
            if (Number.isFinite(responseChannel) && responseChannel !== channelId) {
                throw new Error(`Prompt settings response channel ${responseChannel} did not match requested channel ${channelId}`);
            }
            applyLuxriotPromptSettingsFromPayload(data);
            luxriotPromptFormChannelId = channelId;
            setLuxriotPromptModalTab(luxriotPromptModalTab || 'stream');
            return true;
        } catch (err) {
            if (err && err.name === 'AbortError') {
                return false;
            }
            if (showError && generation === luxriotPromptRequestGeneration) {
                setLuxriotStatus(err.message || 'Failed to load prompt settings', true);
            }
            return false;
        } finally {
            if (luxriotPromptLoadAbortController === controller) {
                luxriotPromptLoadAbortController = null;
            }
            if (generation === luxriotPromptRequestGeneration) {
                setLuxriotPromptApplyAvailability(false);
            }
        }
    }

    async function persistLuxriotPromptSettings(channelIdOverride = null, payloadOverride = null) {
        const channelId = Number.isFinite(channelIdOverride)
            ? channelIdOverride
            : getSelectedLuxriotChannel();
        if (!Number.isFinite(channelId)) {
            throw new Error('Select a channel first');
        }
        abortLuxriotPromptController(luxriotPromptLoadAbortController);
        abortLuxriotPromptController(luxriotPromptSaveAbortController);
        luxriotPromptLoadAbortController = null;
        const generation = ++luxriotPromptRequestGeneration;
        const controller = new AbortController();
        luxriotPromptSaveAbortController = controller;
        setLuxriotPromptApplyAvailability(true);
        const payload = payloadOverride && typeof payloadOverride === 'object'
            ? { ...payloadOverride }
            : collectLuxriotPromptSettings();
        payload.channel_id = channelId;
        try {
            const response = await fetch('/luxriot/prompt_settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: controller.signal,
            });
            const data = await parseApiJson(response, 'Failed to save prompt settings');
            if (!isCurrentLuxriotPromptRequest(generation, controller, channelId)) {
                throw new Error(`Prompt settings were saved for channel ${channelId}, but the selected channel changed; reload before editing again.`);
            }
            const responseChannel = parseInt(String(data.channel_id ?? channelId), 10);
            if (Number.isFinite(responseChannel) && responseChannel !== channelId) {
                throw new Error(`Prompt settings response channel ${responseChannel} did not match saved channel ${channelId}`);
            }
            applyLuxriotPromptSettingsFromPayload(data);
            clearLuxriotLiveIntervalDirty(channelId);
            luxriotPromptFormChannelId = channelId;
            return data;
        } finally {
            if (luxriotPromptSaveAbortController === controller) {
                luxriotPromptSaveAbortController = null;
            }
            if (generation === luxriotPromptRequestGeneration) {
                setLuxriotPromptApplyAvailability(false);
            }
        }
    }

    async function resetLuxriotPromptOverrides() {
        const selectedChannelId = getSelectedLuxriotChannel();
        const formChannelId = luxriotPromptFormChannelId;
        if (!Number.isFinite(formChannelId) || formChannelId !== selectedChannelId) {
            await refreshLuxriotPromptSettings(true, selectedChannelId);
            throw new Error('The selected channel changed. Its settings were reloaded; review them before resetting overrides.');
        }
        const clearOverrideFields = getClearableLuxriotPromptOverrideFields();
        if (!clearOverrideFields.length) {
            throw new Error('This channel has no prompt or bookmark overrides you can reset.');
        }
        const confirmed = window.confirm(
            `Remove ${clearOverrideFields.length} channel-specific override(s) for ${getLuxriotChannelLabel(formChannelId)} and use inherited defaults?`
        );
        if (!confirmed) return null;
        const result = await persistLuxriotPromptSettings(formChannelId, {
            clear_override_fields: clearOverrideFields,
        });
        setLuxriotPromptModalTab(luxriotPromptModalTab || 'stream');
        setLuxriotStatus(`Inherited defaults restored for ${getLuxriotChannelLabel(formChannelId)}`);
        return result;
    }

    function setLuxriotPromptModalTab(tab) {
        const normalized = String(tab || '').trim().toLowerCase();
        const previousInput = getLuxriotPromptInputByTab(luxriotPromptModalTab);
        if (luxriotPromptModalInput && previousInput) {
            previousInput.value = luxriotPromptModalInput.value || '';
        }
        const tabValue = (normalized === 'stream' || normalized === 'alerts' || normalized === 'json')
            ? normalized
            : normalized.toUpperCase();
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
            const channelId = Number.isFinite(luxriotPromptFormChannelId)
                ? luxriotPromptFormChannelId
                : getSelectedLuxriotChannel();
            const channelLabel = getLuxriotChannelLabel(channelId);
            const sourceKey = tabValue === 'stream'
                ? 'stream_system_prompt'
                : tabValue === 'alerts'
                    ? 'alert_policy_prompt'
                    : tabValue === 'json'
                        ? 'json_alert_prompt'
                        : null;
            const source = sourceKey
                ? String(luxriotPromptSettingSources?.[sourceKey] || '')
                : String(luxriotPromptSettingSources?.rollup_prompts?.[tabValue] || '');
            const sourceLabels = {
                channel_override: 'channel override',
                persisted_runtime_default: 'persisted runtime default',
                config_default: 'configuration default',
            };
            const sourceLabel = sourceLabels[source] || 'source unknown';
            const persistence = luxriotPromptPersistence;
            const persistenceLabel = persistence?.last_error
                ? `persistence error: ${String(persistence.last_error).slice(0, 160)}`
                : persistence?.persisted
                    ? `saved revision ${Number(persistence.revision || 0)}`
                    : 'not persisted yet';
            luxriotPromptModalMeta.textContent = `${getLuxriotPromptTabMeta(tabValue)} Channel: ${channelLabel}. Source: ${sourceLabel}; ${persistenceLabel}.`;
        }
        updateLuxriotPromptLayerDetails();
    }

    function openLuxriotPromptModal() {
        if (!luxriotPromptModal) return;
        invalidateLuxriotPromptRequests({ clearFormIdentity: true });
        const channelId = getSelectedLuxriotChannel();
        luxriotPromptModal.style.display = 'block';
        setLuxriotPromptModalTab(luxriotPromptModalTab || 'stream');
        void refreshLuxriotPromptSettings(true, channelId);
    }

    function closeLuxriotPromptModal() {
        if (!luxriotPromptModal) return;
        invalidateLuxriotPromptRequests({ clearFormIdentity: true });
        luxriotPromptModal.style.display = 'none';
    }

    async function applyLuxriotPromptModal() {
        const selectedChannelId = getSelectedLuxriotChannel();
        const formChannelId = luxriotPromptFormChannelId;
        if (!Number.isFinite(formChannelId) || formChannelId !== selectedChannelId) {
            await refreshLuxriotPromptSettings(true, selectedChannelId);
            throw new Error('The selected channel changed. Its prompt settings were reloaded; review them and Apply again.');
        }
        const targetInput = getLuxriotPromptInputByTab(luxriotPromptModalTab);
        if (targetInput && luxriotPromptModalInput) {
            targetInput.value = luxriotPromptModalInput.value || '';
        }
        await persistLuxriotPromptSettings(formChannelId);
        setLuxriotStatus(`${getLuxriotPromptTabLabel(luxriotPromptModalTab)} updated for ${getLuxriotChannelLabel(formChannelId)}`);
    }

    function isCurrentLuxriotMediaRequest(requestSeq, channelId) {
        return requestSeq === luxriotPreviewRequestSeq
            && currentMode === 'video'
            && getSelectedLuxriotChannel() === channelId;
    }

    function scheduleLuxriotPreviewRenewal(requestSeq, channelId, delayMs, detail) {
        if (luxriotPreviewRenewTimer) clearTimeout(luxriotPreviewRenewTimer);
        const parsedDelay = Number(delayMs);
        const safeDelay = delayMs !== null && delayMs !== undefined && Number.isFinite(parsedDelay) && parsedDelay > 0
            ? Math.max(750, Math.min(120000, Math.trunc(parsedDelay)))
            : 20000;
        luxriotPreviewRenewTimer = window.setTimeout(() => {
            luxriotPreviewRenewTimer = null;
            if (!isCurrentLuxriotMediaRequest(requestSeq, channelId)) return;
            setLuxriotOperatorMediaState('loading', {
                mediaKind: luxriotPreviewMeta.mediaKind,
                detail: detail || 'Renewing the bounded live media connection.',
                overlay: 'Reconnecting video…',
            });
            startLuxriotPreview({ reuseNegotiation: true });
        }, safeDelay);
    }

    function clearLuxriotPreviewStallWatchdog() {
        if (!luxriotPreviewStallTimer) return;
        clearTimeout(luxriotPreviewStallTimer);
        luxriotPreviewStallTimer = null;
    }

    function armLuxriotPreviewStallWatchdog(requestSeq, channelId) {
        clearLuxriotPreviewStallWatchdog();
        luxriotPreviewStallTimer = window.setTimeout(() => {
            luxriotPreviewStallTimer = null;
            if (!isCurrentLuxriotMediaRequest(requestSeq, channelId)) return;
            scheduleLuxriotPreviewRenewal(
                requestSeq,
                channelId,
                750,
                'The browser media clock stopped advancing; reconnecting without touching analytics.',
            );
        }, 5000);
    }

    function scheduleLuxriotAttentionRecovery(requestSeq, channelId) {
        if (luxriotPreferFullOperatorMedia) return;
        if (!selectedLuxriotStream(channelId, 'video')?.running) return;
        // Analytics keeps selecting apex frames while the preview sits in a
        // static fallback; retry the shared attention preview instead of
        // waiting for a manual Retry click.
        scheduleLuxriotPreviewRenewal(
            requestSeq,
            channelId,
            12000,
            'Retrying the shared EVA attention preview.',
        );
    }

    function showLuxriotStaticFrameFallback(requestSeq, channelId, reason, fallbackUrl = '') {
        if (!isCurrentLuxriotMediaRequest(requestSeq, channelId) || !luxriotPreviewImg) return;
        clearLuxriotPreviewVideo();
        luxriotPreviewLoading = true;
        const staticUrl = fallbackUrl && fallbackUrl.startsWith('/')
            ? fallbackUrl
            : `/luxriot/snapshot/${encodeURIComponent(String(channelId))}?stream=mainStream`;
        const separator = staticUrl.includes('?') ? '&' : '?';
        luxriotPreviewImg.onload = () => {
            if (!isCurrentLuxriotMediaRequest(requestSeq, channelId)) return;
            luxriotPreviewLoading = false;
            luxriotPreviewImg.style.display = 'block';
            setLuxriotOperatorMediaState('degraded', {
                width: Number(luxriotPreviewImg.naturalWidth) || 0,
                height: Number(luxriotPreviewImg.naturalHeight) || 0,
                loadedAt: Date.now(),
                mediaKind: 'static_frame',
                detail: `${reason || 'Browser-playable video is unavailable.'} One static fallback frame is shown; this is not video or a snapshot slideshow.`,
            });
            setLuxriotStatus('Static frame fallback shown; live video is unavailable.', true);
            scheduleLuxriotAttentionRecovery(requestSeq, channelId);
        };
        luxriotPreviewImg.onerror = () => {
            if (!isCurrentLuxriotMediaRequest(requestSeq, channelId)) return;
            luxriotPreviewLoading = false;
            setLuxriotPreviewSignalLost('media_error', reason || 'Live video and static fallback are unavailable.');
            scheduleLuxriotAttentionRecovery(requestSeq, channelId);
        };
        luxriotPreviewImg.style.display = 'block';
        luxriotPreviewImg.src = `${staticUrl}${separator}t=${Date.now()}`;
    }

    function startLuxriotPreview(options = {}) {
        if (!luxriotPreviewImg || !luxriotViewport) return;
        const channelId = getSelectedLuxriotChannel();
        if (!Number.isFinite(channelId) || channelId <= 0) {
            setLuxriotStatus('Select a channel to play video', true);
            return;
        }
        stopLuxriotPreview(false);
        clearLuxriotPreviewVideo();
        replaceLuxriotPreviewImageElement();
        const requestSeq = ++luxriotPreviewRequestSeq;
        const controller = new AbortController();
        const cachedNegotiation = options.reuseNegotiation
            && Number(luxriotPreviewNegotiation?.channelId) === channelId
            ? luxriotPreviewNegotiation.value
            : null;
        luxriotPreviewAbortController = cachedNegotiation ? null : controller;
        luxriotPreviewLoading = true;
        const videoStream = selectedLuxriotStream(channelId, 'video');
        const useAttentionPreview = Boolean(videoStream?.running) && !luxriotPreferFullOperatorMedia;
        const mediaUrl = luxriotMediaBrokerUrl(
            useAttentionPreview ? 'attention' : 'live',
            channelId,
            { stream: 'mainStream' },
        );
        syncLuxriotPreviewTransportButton(channelId);
        setLuxriotOperatorMediaState('loading', {
            width: 0,
            height: 0,
            loadedAt: 0,
            mediaKind: '',
            detail: useAttentionPreview
                ? 'Opening the shared EVA attention preview without another recorder stream.'
                : 'Negotiating full operator video through the same-origin broker.',
        });

        const negotiationRequest = cachedNegotiation
            ? Promise.resolve(cachedNegotiation)
            : negotiateLuxriotMedia(mediaUrl, controller);
        void negotiationRequest
            .then((negotiated) => {
                if (luxriotPreviewAbortController === controller) {
                    luxriotPreviewAbortController = null;
                }
                if (!isCurrentLuxriotMediaRequest(requestSeq, channelId)) return;
                luxriotPreviewNegotiation = { channelId, value: negotiated };
                const failToStatic = (detail) => {
                    if (luxriotPreviewTimer) {
                        clearTimeout(luxriotPreviewTimer);
                        luxriotPreviewTimer = null;
                    }
                    if (luxriotPreviewRenewTimer) {
                        clearTimeout(luxriotPreviewRenewTimer);
                        luxriotPreviewRenewTimer = null;
                    }
                    clearLuxriotPreviewStallWatchdog();
                    showLuxriotStaticFrameFallback(requestSeq, channelId, detail);
                };
                luxriotPreviewTimer = window.setTimeout(() => {
                    if (!isCurrentLuxriotMediaRequest(requestSeq, channelId)) return;
                    failToStatic('Live media load timed out.');
                }, 12000);
                if (negotiated.mediaKind === 'mjpeg') {
                    luxriotPreviewImg.onload = () => {
                        if (!isCurrentLuxriotMediaRequest(requestSeq, channelId)) return;
                        if (luxriotPreviewTimer) {
                            clearTimeout(luxriotPreviewTimer);
                            luxriotPreviewTimer = null;
                        }
                        luxriotPreviewLoading = false;
                        luxriotPreviewImg.style.display = 'block';
                        setLuxriotOperatorMediaState('playing', {
                            width: Number(luxriotPreviewImg.naturalWidth) || 0,
                            height: Number(luxriotPreviewImg.naturalHeight) || 0,
                            loadedAt: Date.now(),
                            mediaKind: negotiated.attentionPreview ? 'attention' : 'mjpeg',
                            detail: negotiated.attentionPreview
                                ? 'The exact selected per-second EVA attention frames are playing without a second Evo stream.'
                                : 'Continuous MJPEG operator media is playing.',
                        });
                        setLuxriotStatus(
                            negotiated.attentionPreview
                                ? `Playing EVA attention preview for channel ${channelId}`
                                : `Playing MJPEG video for channel ${channelId}`
                        );
                    };
                    luxriotPreviewImg.onerror = () => failToStatic('The MJPEG media stream could not be decoded.');
                    luxriotPreviewImg.style.display = 'block';
                    scheduleLuxriotPreviewRenewal(
                        requestSeq,
                        channelId,
                        negotiated.renewAfterMs,
                        negotiated.attentionPreview
                            ? 'Renewing the bounded EVA attention preview.'
                            : 'Renewing the bounded MJPEG connection before its server lease expires.',
                    );
                    luxriotPreviewImg.src = `${mediaUrl}&request=${Date.now()}`;
                    return;
                }

                const video = ensureLuxriotPreviewVideo();
                if (!video) {
                    failToStatic('The browser video element could not be initialized.');
                    return;
                }
                video.style.display = 'block';
                const markPlayable = (detail) => {
                    if (!isCurrentLuxriotMediaRequest(requestSeq, channelId)) return;
                    if (luxriotPreviewTimer) {
                        clearTimeout(luxriotPreviewTimer);
                        luxriotPreviewTimer = null;
                    }
                    clearLuxriotPreviewStallWatchdog();
                    luxriotPreviewLoading = false;
                    setLuxriotOperatorMediaState('playing', {
                        width: Number(video.videoWidth) || 0,
                        height: Number(video.videoHeight) || 0,
                        loadedAt: Date.now(),
                        mediaKind: 'video',
                        detail,
                    });
                };
                video.onloadedmetadata = () => {
                    if (!isCurrentLuxriotMediaRequest(requestSeq, channelId)) return;
                    luxriotPreviewMeta.width = Number(video.videoWidth) || 0;
                    luxriotPreviewMeta.height = Number(video.videoHeight) || 0;
                    updateLuxriotStreamContext();
                };
                video.oncanplay = () => {
                    if (!isCurrentLuxriotMediaRequest(requestSeq, channelId)) return;
                    markPlayable('Browser-playable operator video is ready.');
                    const playPromise = video.play();
                    if (playPromise && typeof playPromise.catch === 'function') {
                        playPromise.catch(() => {
                            if (isCurrentLuxriotMediaRequest(requestSeq, channelId)) {
                                markPlayable('Video is ready; press Play if browser autoplay is blocked.');
                            }
                        });
                    }
                };
                video.onplaying = () => {
                    markPlayable('Operator video is playing independently of EVA analytics.');
                    setLuxriotStatus(`Playing live video for channel ${channelId}`);
                };
                const markBuffering = () => {
                    if (!isCurrentLuxriotMediaRequest(requestSeq, channelId)) return;
                    setLuxriotOperatorMediaState('loading', {
                        mediaKind: 'video',
                        detail: 'Live video transport is buffering.',
                        overlay: 'Buffering video…',
                    });
                    armLuxriotPreviewStallWatchdog(requestSeq, channelId);
                };
                video.onwaiting = markBuffering;
                video.onstalled = markBuffering;
                video.onprogress = clearLuxriotPreviewStallWatchdog;
                video.ontimeupdate = clearLuxriotPreviewStallWatchdog;
                video.onerror = () => failToStatic('The browser rejected the Luxriot video container or codec.');
                video.onended = () => {
                    if (!isCurrentLuxriotMediaRequest(requestSeq, channelId)) return;
                    clearLuxriotPreviewStallWatchdog();
                    setLuxriotOperatorMediaState('loading', {
                        mediaKind: 'video',
                        detail: 'The bounded media segment ended; reconnecting.',
                        overlay: 'Reconnecting video…',
                    });
                    scheduleLuxriotPreviewRenewal(requestSeq, channelId, 750, 'The bounded media segment ended; reconnecting.');
                };
                scheduleLuxriotPreviewRenewal(
                    requestSeq,
                    channelId,
                    negotiated.renewAfterMs,
                    'Renewing the bounded video connection before its server lease expires.',
                );
                video.src = `${mediaUrl}&request=${Date.now()}`;
                video.load();
            })
            .catch((error) => {
                if (luxriotPreviewAbortController === controller) {
                    luxriotPreviewAbortController = null;
                }
                if (!isCurrentLuxriotMediaRequest(requestSeq, channelId)) return;
                const timedOut = controller.signal.aborted;
                showLuxriotStaticFrameFallback(
                    requestSeq,
                    channelId,
                    timedOut ? 'Live media negotiation timed out.' : (error.message || 'Live video is unavailable.'),
                    error && error.fallbackUrl,
                );
            });
    }

    function setRoadSceneGroundingConfidence(value) {
        if (!roadSceneGroundingConfidence) return;
        const normalized = String(value || 'idle').toLowerCase();
        roadSceneGroundingConfidence.classList.remove('low', 'medium', 'high', 'error');
        if (['low', 'medium', 'high', 'error'].includes(normalized)) {
            roadSceneGroundingConfidence.classList.add(normalized);
        }
        roadSceneGroundingConfidence.textContent = normalized;
    }

    function renderRoadSceneGrounding(payload) {
        if (!roadSceneGroundingPanel) return;
        roadSceneGroundingPanel.hidden = false;
        const scene = payload?.scene || {};
        const sceneCard = scene.scene_card || {};
        const zones = Array.isArray(sceneCard.zones) ? sceneCard.zones : [];
        const zone = zones[0] || {};
        const channelId = payload?.channel_id || getSelectedLuxriotChannel();
        const label = getLuxriotChannelLabel(channelId);
        if (roadSceneGroundingImage && payload?.overlay_b64) {
            roadSceneGroundingImage.src = `data:image/png;base64,${payload.overlay_b64}`;
        }
        setTextContentSafe(roadSceneGroundingTitle, label);
        setRoadSceneGroundingConfidence(scene.confidence || 'unknown');
        const flow = Array.isArray(zone.expected_flow)
            ? `flow ${zone.expected_flow.map((v) => Number(v).toFixed(2)).join(', ')}`
            : 'flow not inferred';
        const area = Number(scene.zone_area_ratio);
        const areaText = Number.isFinite(area) ? `${Math.round(area * 100)}% zone` : 'zone n/a';
        const budget = payload?.budget || {};
        const meta = [
            scene.reason || 'No scene reason returned.',
            `${scene.frame_count || 0} frames · ${scene.motion_pair_count || 0} motion pairs · ${scene.scene_cut_count || 0} scene cuts`,
            `${areaText} · ${flow}`,
            `Budget: ${budget.seconds || '?'}s · max ${budget.frames || '?'} frames`,
        ].join(' | ');
        setTextContentSafe(roadSceneGroundingMeta, meta);
    }

    function resetRoadSceneGrounding() {
        if (!roadSceneGroundingPanel) return;
        roadSceneGroundingPanel.hidden = true;
        if (roadSceneGroundingImage) roadSceneGroundingImage.removeAttribute('src');
        setTextContentSafe(roadSceneGroundingTitle, 'No preview yet');
        setRoadSceneGroundingConfidence('idle');
        setTextContentSafe(roadSceneGroundingMeta, 'Generate a bounded preview to inspect the inferred motion zone.');
    }

    function setRoadSceneGroundingBusy(busy) {
        roadSceneGroundingBtns.forEach((button) => setButtonBusy(button, busy));
    }

    async function refreshRoadSceneGrounding() {
        const channelId = getSelectedLuxriotChannel();
        if (!channelId) {
            setLuxriotStatus('Select a channel before grounding road mask', true);
            return;
        }
        if (roadSceneGroundingPanel) roadSceneGroundingPanel.hidden = false;
        setTextContentSafe(roadSceneGroundingTitle, getLuxriotChannelLabel(channelId));
        setRoadSceneGroundingConfidence('idle');
        setTextContentSafe(roadSceneGroundingMeta, 'Capturing a short bounded live segment and inferring motion zone...');
        setRoadSceneGroundingBusy(true);
        try {
            const params = new URLSearchParams({
                stream: 'mainStream',
                seconds: '8',
                frames: '60',
                every_n: '6',
                mb: '8',
            });
            const response = await fetch(`/road/scene_overlay/${channelId}?${params.toString()}`, { cache: 'no-store' });
            const payload = await parseApiJson(response, 'Road mask preview failed');
            renderRoadSceneGrounding(payload);
            setLuxriotStatus(`Road mask grounded for ${getLuxriotChannelLabel(channelId)}`);
        } catch (error) {
            if (roadSceneGroundingPanel) roadSceneGroundingPanel.hidden = false;
            setRoadSceneGroundingConfidence('error');
            setTextContentSafe(roadSceneGroundingTitle, getLuxriotChannelLabel(channelId));
            setTextContentSafe(roadSceneGroundingMeta, error.message || 'Road mask preview failed');
            if (roadSceneGroundingImage) roadSceneGroundingImage.removeAttribute('src');
            setLuxriotStatus(error.message || 'Road mask preview failed', true);
        } finally {
            setRoadSceneGroundingBusy(false);
        }
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

    function scrollLuxriotSummaryToTimestamp(targetMs) {
        if (!luxriotSummaries) return false;
        const target = Number(targetMs);
        if (!Number.isFinite(target) || target <= 0) return false;
        const cards = Array.from(luxriotSummaries.querySelectorAll('.luxriot-summary[data-summary-created-ms]'));
        let bestCard = null;
        let bestDistance = Number.POSITIVE_INFINITY;
        cards.forEach((card) => {
            const createdMs = Number(card.dataset.summaryCreatedMs);
            const batchStartMs = Number(card.dataset.summaryBatchStartMs);
            const batchEndMs = Number(card.dataset.summaryBatchEndMs);
            const candidates = [];
            if (Number.isFinite(createdMs) && createdMs > 0) candidates.push(createdMs);
            if (Number.isFinite(batchStartMs) && batchStartMs > 0) candidates.push(batchStartMs);
            if (Number.isFinite(batchEndMs) && batchEndMs > 0) candidates.push(batchEndMs);
            let distance = candidates.length
                ? Math.min(...candidates.map((value) => Math.abs(value - target)))
                : Number.POSITIVE_INFINITY;
            if (
                Number.isFinite(batchStartMs)
                && Number.isFinite(batchEndMs)
                && target >= Math.min(batchStartMs, batchEndMs)
                && target <= Math.max(batchStartMs, batchEndMs)
            ) {
                distance = 0;
            }
            if (distance < bestDistance) {
                bestDistance = distance;
                bestCard = card;
            }
        });
        if (!bestCard) return false;
        luxriotSummaries.querySelectorAll('.luxriot-summary.is-jump-target').forEach((card) => {
            card.classList.remove('is-jump-target');
        });
        bestCard.classList.add('is-jump-target');
        bestCard.scrollIntoView({ block: 'center', behavior: uiLiteMode ? 'auto' : 'smooth' });
        return true;
    }

    function setLuxriotSummaryMeta(text, isError = false) {
        if (!luxriotSummaryMeta) return;
        luxriotSummaryMeta.textContent = text;
        luxriotSummaryMeta.classList.toggle('error', Boolean(isError));
    }

    function isLiveSummaryPeriod() {
        return normalizeSummaryRangePreset(luxriotSummaryRangePreset) === 'live';
    }

    function resetSummaryArchivePaging() {
        luxriotSummaryArchiveOffset = 0;
        luxriotSummaryArchiveHasMore = false;
        luxriotSummaryArchiveEvidenceTotal = 0;
        luxriotSummaryArchiveLoading = false;
        if (luxriotSummaryLoadEarlierBtn) {
            luxriotSummaryLoadEarlierBtn.classList.add('is-hidden');
            luxriotSummaryLoadEarlierBtn.disabled = false;
            luxriotSummaryLoadEarlierBtn.textContent = '← Load earlier';
        }
    }

    function updateSummaryArchivePagingUI() {
        if (!luxriotSummaryLoadEarlierBtn) return;
        const show = !isLiveSummaryPeriod()
            && !isRollupViewActive()
            && (luxriotSummaryArchiveHasMore || luxriotSummaryArchiveLoading);
        luxriotSummaryLoadEarlierBtn.classList.toggle('is-hidden', !show);
        luxriotSummaryLoadEarlierBtn.disabled = luxriotSummaryArchiveLoading;
        luxriotSummaryLoadEarlierBtn.textContent = luxriotSummaryArchiveLoading
            ? '← Loading earlier…'
            : '← Load earlier';
    }

    function showLuxriotSummaryLoading(label = 'Loading descriptions…', preserveRows = false) {
        if (!luxriotSummaries) return;
        luxriotSummaries.setAttribute('aria-busy', 'true');
        luxriotSummaries.classList.add('is-loading-history');
        setLuxriotSummaryMeta(`${label} · ${summaryPeriodLabel()}`);
        if (!preserveRows) {
            luxriotSummaries.innerHTML = `
                <div class="luxriot-feed-loading" role="status" aria-live="polite">
                    <span class="luxriot-feed-loading-spinner" aria-hidden="true"></span>
                    <strong>${escapeHtml(label)}</strong>
                    <span>${escapeHtml(summaryPeriodLabel())}</span>
                </div>
            `;
        }
    }

    function finishLuxriotSummaryLoading() {
        if (!luxriotSummaries) return;
        luxriotSummaries.removeAttribute('aria-busy');
        luxriotSummaries.classList.remove('is-loading-history');
    }

    function archiveSummaryBatchKey(log) {
        return [
            String(log?.run_id || ''),
            Number(log?.batch_start_ms || 0),
            Number(log?.batch_end_ms || 0),
        ].join('|');
    }

    async function refreshLuxriotArchivedSummaries(channelId, requestContext = null, append = false) {
        if (!channelId || luxriotSummaryArchiveLoading) return false;
        const requestKey = luxriotSummaryRequestKey(channelId);
        const offset = append ? luxriotSummaryArchiveOffset : 0;
        const pageLimit = 120;
        const bounds = getSummaryEffectiveBounds();
        const params = new URLSearchParams({
            channel_id: String(channelId),
            limit: String(pageLimit),
            offset: String(offset),
        });
        if (Number.isFinite(bounds.fromTs)) params.set('from_ts', String(bounds.fromTs));
        if (Number.isFinite(bounds.toTs)) params.set('to_ts', String(bounds.toTs));
        luxriotSummaryArchiveLoading = true;
        updateSummaryArchivePagingUI();
        showLuxriotSummaryLoading(append ? 'Loading earlier descriptions…' : 'Loading archived descriptions…', append);
        try {
            const response = await fetch(`/luxriot/history?${params.toString()}`, {
                signal: requestContext?.controller?.signal,
            });
            const data = await parseApiJson(response, 'Failed to load archived descriptions');
            const stillCurrent = requestContext
                ? isCurrentLuxriotSummaryRequest(requestContext)
                : (
                    currentMode === 'video'
                    && Number(getSelectedSummaryChannel()) === Number(channelId)
                    && luxriotSummaryRequestKey(channelId) === requestKey
                );
            if (!stillCurrent) return false;
            const pageLogs = Array.isArray(data.logs) ? data.logs : [];
            const combined = new Map();
            if (append) {
                (Array.isArray(luxriotSummaryLogCache) ? luxriotSummaryLogCache : []).forEach((log) => {
                    combined.set(archiveSummaryBatchKey(log), log);
                });
            }
            pageLogs.forEach((log) => combined.set(archiveSummaryBatchKey(log), log));
            luxriotSummaryArchiveOffset = offset + pageLogs.length;
            luxriotSummaryArchiveEvidenceTotal = Math.max(0, Number(data.total || 0));
            luxriotSummaryArchiveHasMore = Boolean(data.has_more);
            renderLuxriotSummaries(Array.from(combined.values()), channelId);
            const channelLabel = getLuxriotChannelLabel(channelId);
            const batchCount = combined.size;
            const pageLabel = luxriotSummaryArchiveHasMore ? ' · more available' : ' · complete period';
            setLuxriotSummaryMeta(withSummaryUpdatedMeta(
                `${channelLabel} · Observations · ${batchCount} loaded of ${luxriotSummaryArchiveEvidenceTotal} archived batches${pageLabel} · all runs · ${getSummaryRangeLabel()}`
            ));
            setLuxriotStatus(`Loaded ${batchCount} archived description batches`);
            return true;
        } catch (error) {
            if (error && error.name === 'AbortError') return false;
            setLuxriotSummaryMeta(`Failed to load archived descriptions: ${error.message || 'Unknown error'}`, true);
            setLuxriotStatus(error.message || 'Failed to load archived descriptions', true);
            return false;
        } finally {
            luxriotSummaryArchiveLoading = false;
            finishLuxriotSummaryLoading();
            updateSummaryArchivePagingUI();
        }
    }

    const SUMMARY_ALERT_SEVERITIES = ['critical', 'high', 'normal', 'low', 'info'];

    function normalizeSummaryAlertCounts(row) {
        const counts = {};
        const rawCounts = row && typeof row === 'object'
            ? (row.alert_counts || row.alertCounts || {})
            : {};
        if (rawCounts && typeof rawCounts === 'object' && !Array.isArray(rawCounts)) {
            SUMMARY_ALERT_SEVERITIES.forEach((severity) => {
                const value = Number(rawCounts[severity] || 0);
                if (Number.isFinite(value) && value > 0) {
                    counts[severity] = Math.floor(value);
                }
            });
        }
        if (!Object.keys(counts).length) {
            const total = Number(row?.alert_total || row?.alertTotal || 0);
            if (Number.isFinite(total) && total > 0) {
                const severity = SUMMARY_ALERT_SEVERITIES.includes(String(row?.severity || '').toLowerCase())
                    ? String(row.severity).toLowerCase()
                    : 'normal';
                counts[severity] = Math.floor(total);
            }
        }
        return counts;
    }

    function renderSummaryAlertBadges(row, levelLabel = '') {
        const counts = normalizeSummaryAlertCounts(row);
        const parts = SUMMARY_ALERT_SEVERITIES
            .filter((severity) => counts[severity] > 0)
            .map((severity) => {
                const count = counts[severity];
                return `<span class="summary-alert-chip severity-${escapeHtml(severity)}" title="${escapeHtml(severity)} alerts">${escapeHtml(severity)} <strong>${count}</strong></span>`;
            });
        if (!parts.length) return '';
        const level = String(levelLabel || row?.level || 'L0').trim().toUpperCase();
        const title = `${level || 'Summary'} alerts`;
        return `<span class="summary-alert-badges" title="${escapeHtml(title)}"><span class="summary-alert-level">${escapeHtml(level || 'L0')}</span>${parts.join('')}</span>`;
    }

    function summaryBurstAttention(row) {
        const attention = row?.vector_signal?.capture_attention;
        const seconds = attention && typeof attention === 'object' && Array.isArray(attention.seconds)
            ? attention.seconds
            : [];
        const bursts = seconds.filter((item) => (
            item && typeof item === 'object' && String(item.mode || '').trim().toLowerCase() === 'burst'
        ));
        if (!bursts.length) return null;
        const activityValues = bursts
            .map((item) => Number(item.activity_x))
            .filter((value) => Number.isFinite(value) && value >= 0);
        const snapshots = bursts
            .map((item) => String(item.snapshot ?? '').trim())
            .filter(Boolean);
        return {
            count: bursts.length,
            maxActivity: activityValues.length ? Math.max(...activityValues) : null,
            snapshots,
        };
    }

    function renderSummaryBurstAttentionChip(row) {
        const burst = summaryBurstAttention(row);
        if (!burst) return '';
        const maxLabel = Number.isFinite(burst.maxActivity) ? ` (max ${burst.maxActivity.toFixed(1)}×)` : '';
        const label = `⚡ burst ×${burst.count}${maxLabel}`;
        const snapshotLabel = burst.snapshots.length ? burst.snapshots.join(', ') : 'n/a';
        const title = `Motion far above this channel's measured norm; snapshot numbers: ${snapshotLabel}`;
        return `<span class="summary-attention-chip" title="${escapeHtml(title)}">${escapeHtml(label)}</span>`;
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
                const createdAtSec = Number(log.created_at);
                const createdMs = Number.isFinite(createdAtSec) && createdAtSec > 0 ? Math.round(createdAtSec * 1000) : 0;
                const batchStartMs = Number(log.batch_start_ms || log.window_start_ms || 0);
                const batchEndMs = Number(log.batch_end_ms || log.window_end_ms || 0);
                const tsLabel = createdMs ? formatSummaryLocalTimestamp(createdAtSec) : 'n/a';
                const frameLabel = log.frame_count ? `${log.frame_count} frames` : '';
                const modelLabel = String(log.model || '').trim();
                const rowChannelId = parseInt(String(log?.channel_id ?? channelId), 10);
                const channelTag = Number.isFinite(rowChannelId) ? `#${rowChannelId}` : '#?';
                const channelLabel = Number.isFinite(rowChannelId)
                    ? getLuxriotChannelLabel(rowChannelId)
                    : 'Unknown channel';
                if (log.coverage_gap) {
                    const gapWindow = Number.isFinite(batchStartMs) && batchStartMs > 0 && Number.isFinite(batchEndMs) && batchEndMs > batchStartMs
                        ? `${formatSummaryLocalTimestamp(batchStartMs / 1000)}–${formatSummaryLocalTimestamp(batchEndMs / 1000)}`
                        : tsLabel;
                    const gapReason = String(log.gap_reason || 'dropped batch').replace(/_/g, ' ');
                    return `
                        <div class="luxriot-summary luxriot-summary-gap" data-log-key="${escapeHtml(logKey)}" data-summary-index="${idx}" data-summary-created-ms="${createdMs || ''}" data-summary-batch-start-ms="${Number.isFinite(batchStartMs) && batchStartMs > 0 ? batchStartMs : ''}" data-summary-batch-end-ms="${Number.isFinite(batchEndMs) && batchEndMs > 0 ? batchEndMs : ''}">
                            <div class="timestamp"><span class="luxriot-summary-channel-pill" title="${escapeHtml(channelLabel)}">${escapeHtml(channelTag)}</span> ${escapeHtml(gapWindow)} · <strong>coverage gap</strong> — no description exists for this window (${escapeHtml(gapReason)})</div>
                        </div>
                    `;
                }
                const coalescedBatches = Number(log.coalesced?.batches || 0);
                const coalescedLabel = coalescedBatches > 1
                    ? ` · coalesced ×${coalescedBatches}`
                    : '';
                const summary = String(log.summary || '').trim();
                const summaryParts = splitSummaryAndJson(summary);
                const summaryMain = summaryParts.main || summary;
                const summaryJson = summaryParts.json;
                const hasSummaryText = summary.length > 0;
                const canBookmark = userHasPermission('bookmarks:create');
                const collapsed = isSummaryCollapsed(channelId, logKey);
                const alertBadges = renderSummaryAlertBadges(log, 'L0');
                const attentionBadge = renderSummaryBurstAttentionChip(log);
                const bookmarkButton = canBookmark
                    ? `<button class="feature-btn luxriot-bookmark-btn" data-luxriot-bookmark="${idx}" ${hasSummaryText ? '' : 'disabled'}>Bookmark</button>`
                    : '';
                return `
                    <div class="luxriot-summary ${collapsed ? 'is-collapsed' : ''}" data-log-key="${escapeHtml(logKey)}" data-summary-index="${idx}" data-summary-created-ms="${createdMs || ''}" data-summary-batch-start-ms="${Number.isFinite(batchStartMs) && batchStartMs > 0 ? batchStartMs : ''}" data-summary-batch-end-ms="${Number.isFinite(batchEndMs) && batchEndMs > 0 ? batchEndMs : ''}">
                        <div class="luxriot-summary-head">
                            <div class="timestamp"><span class="luxriot-summary-channel-pill" title="${escapeHtml(channelLabel)}">${escapeHtml(channelTag)}</span> ${tsLabel}${frameLabel ? ` · ${frameLabel}` : ''}${coalescedLabel}${modelLabel ? ` · ${escapeHtml(modelLabel)}` : ''}${alertBadges}${attentionBadge}</div>
                            <div class="luxriot-summary-actions">
                                <button class="feature-btn luxriot-summary-action-btn" data-luxriot-collapse="${idx}">
                                    ${collapsed ? 'Expand' : 'Collapse'}
                                </button>
                                <button class="feature-btn luxriot-summary-action-btn" data-luxriot-copy="${idx}" ${hasSummaryText ? '' : 'disabled'}>
                                    Copy
                                </button>
                                <button class="feature-btn luxriot-summary-action-btn" data-luxriot-export="${idx}" ${hasSummaryText ? '' : 'disabled'}>
                                    Export
                                </button>
                                ${bookmarkButton}
                            </div>
                        </div>
                        <div class="summary-body">${renderMarkdown(summaryMain)}${renderSummaryMachineJson(summaryJson, 'Machine JSON', summaryParts.marker)}</div>
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
        const startLabel = Number.isFinite(start) ? formatSummaryLocalTimestamp(start) : 'n/a';
        const endLabel = Number.isFinite(end) ? formatSummaryLocalTimestamp(end) : 'n/a';
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

    async function generateLuxriotSemanticRollup(rowIndex, triggerBtn = null) {
        const idx = Number.isFinite(rowIndex) ? rowIndex : parseInt(String(rowIndex || ''), 10);
        if (!Number.isFinite(idx) || idx < 0 || idx >= luxriotSummaryRollupRows.length) return;
        const row = luxriotSummaryRollupRows[idx] || {};
        const channelId = Number(row?.channel_id) || getSelectedSummaryChannel();
        const level = normalizeSummaryLevel(row?.level || luxriotSummaryLevel);
        const windowStart = Number(row?.window_start);
        const windowEnd = Number(row?.window_end);
        if (!Number.isFinite(channelId) || !Number.isFinite(windowStart) || !Number.isFinite(windowEnd)) return;
        const params = new URLSearchParams({
            channel_id: String(channelId),
            run: 'all',
            from_ts: String(windowStart),
            to_ts: String(Math.max(windowStart, windowEnd - 0.001)),
            level_limit: '1',
            target_level: level,
            synthesize: '1',
        });
        const originalLabel = triggerBtn ? triggerBtn.textContent : '';
        if (triggerBtn) {
            triggerBtn.disabled = true;
            triggerBtn.textContent = 'Generating…';
        }
        setLuxriotStatus(`Generating ${level} semantic summary for ${getLuxriotChannelLabel(channelId)}…`);
        const controller = new AbortController();
        const timeoutId = window.setTimeout(() => controller.abort(), 60000);
        try {
            const response = await fetch(`/luxriot/rollups?${params.toString()}`, {
                cache: 'no-store',
                signal: controller.signal,
            });
            const data = await parseApiJson(response, 'Semantic rollup generation failed');
            const rows = Array.isArray(data?.levels?.[level]) ? data.levels[level] : [];
            const generated = rows.find((candidate) => String(candidate?.rollup_id || '') === String(row?.rollup_id || '')) || rows[0];
            const kind = String(generated?.summary_kind || '').trim().toLowerCase();
            if (!['llm', 'llm_cached'].includes(kind)) {
                const reason = String(generated?.generation_error || generated?.generation_status || 'LM unavailable');
                throw new Error(`Semantic pass did not complete: ${reason}`);
            }
            setLuxriotStatus(`${level} semantic summary generated`);
            await refreshLuxriotSummaryView(getSelectedSummaryChannel(), true, false);
        } catch (error) {
            if (error?.name === 'AbortError') {
                setLuxriotStatus('Semantic pass remains queued behind live descriptions and will appear automatically.');
            } else {
                setLuxriotStatus(error.message || 'Semantic rollup generation failed', true);
            }
        } finally {
            window.clearTimeout(timeoutId);
            if (triggerBtn) {
                triggerBtn.disabled = false;
                triggerBtn.textContent = originalLabel || 'Retry semantic';
            }
        }
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
        luxriotSummaryResolutionMode = normalized;
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
        luxriotSummaryResolutionMode = luxriotSummaryLevel;
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
            const summaryKind = String(row?.summary_kind || 'queued').trim().toLowerCase();
            const generationStatus = String(row?.generation_status || summaryKind).trim().toLowerCase();
            const summaryParts = splitSummaryAndJson(summary);
            const summaryMain = summaryParts.main || summary;
            const summaryJson = summaryParts.json;
            const canDrill = Boolean(sourceLevel && sourceIds.length > 0);
            const statsLabel = `${itemCount} items · ${frameCount} frames · ${runCount} runs${sourceTokens > 0 ? ` · ${sourceTokens} tok` : ''}`;
            const sourceLabel = canDrill ? `${sourceIds.length} from ${sourceLevel}` : 'source base';
            const alertBadges = renderSummaryAlertBadges(row, rowLevel);
            const semanticReady = ['llm', 'llm_cached', 'legacy_cached'].includes(summaryKind);
            const legacySemantic = summaryKind === 'legacy_cached';
            const pending = summaryKind === 'pending_context' || generationStatus === 'pending';
            const queued = summaryKind === 'queued' || generationStatus === 'queued' || generationStatus === 'deferred';
            const semanticRefreshPending = generationStatus === 'refresh_pending' || row?.semantic_refresh_pending === true;
            const statusLabel = semanticReady
                ? (legacySemantic
                    ? 'semantic · legacy'
                    : semanticRefreshPending
                    ? 'semantic · refreshing'
                    : (summaryKind === 'llm_cached' ? 'semantic · cached' : 'semantic'))
                : pending
                    ? 'aggregation pending'
                    : queued
                        ? 'semantic queued'
                        : 'semantic retry available';
            const statusClass = semanticReady ? 'ready' : pending ? 'pending' : queued ? 'queued' : 'degraded';
            const statusTitle = semanticRefreshPending
                ? 'Showing the last completed semantic narrative while newly retained source observations are folded into this closed window.'
                : legacySemantic
                ? 'Imported from the pre-0.8.4 semantic history without regenerating the original operator narrative.'
                : statusClass === 'degraded'
                ? 'The semantic pass did not complete; source observations remain available and the window can be retried.'
                : statusClass === 'queued'
                    ? 'Background semantic aggregation is queued behind live descriptions.'
                : statusLabel;
            const generateButton = !semanticReady && !pending
                ? `<button class="feature-btn luxriot-summary-action-btn" data-luxriot-rollup-generate="${idx}">${queued ? 'Generate now' : 'Retry semantic'}</button>`
                : '';
            return `
                <div class="luxriot-summary ${collapsed ? 'is-collapsed' : ''}" data-log-key="${escapeHtml(rollupKey)}">
                    <div class="luxriot-summary-head">
                        <div class="timestamp"><span class="luxriot-summary-rollup-pill">${escapeHtml(rowLevel)}</span> <span class="luxriot-rollup-status ${escapeHtml(statusClass)}" title="${escapeHtml(statusTitle)}">${escapeHtml(statusLabel)}</span> <span class="luxriot-summary-channel-pill" title="${escapeHtml(channelLabel)}">${escapeHtml(channelTag)}</span> ${escapeHtml(rangeLabel)} · ${escapeHtml(statsLabel)} · ${escapeHtml(sourceLabel)}${alertBadges}</div>
                        <div class="luxriot-summary-actions">
                            <button class="feature-btn luxriot-summary-action-btn" data-luxriot-rollup-collapse="${idx}">${collapsed ? 'Expand' : 'Collapse'}</button>
                            <button class="feature-btn luxriot-summary-action-btn" data-luxriot-rollup-copy="${idx}">Copy</button>
                            <button class="feature-btn luxriot-summary-action-btn" data-luxriot-rollup-export="${idx}">Export</button>
                            ${generateButton}
                            <button class="feature-btn luxriot-summary-action-btn" data-luxriot-rollup-drill="${idx}" ${canDrill ? '' : 'disabled'}>${canDrill ? `Drill ${escapeHtml(sourceLevel)}` : 'No source'}</button>
                        </div>
                    </div>
                    <div class="summary-body">${renderMarkdown(summaryMain)}${renderSummaryMachineJson(summaryJson)}</div>
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

    async function refreshLuxriotRollups(channelId = getSelectedSummaryChannel(), force = false, allowRunFallback = true, requestContext = null) {
        if (!channelId) return;
        if (!luxriotSummaryAutoRefresh && !force) return;
        const progressStartedAt = Date.now();
        let progressTimer = null;
        let admissionHint = '';
        const rollupContext = getCurrentSummaryRollupContext();
        const targetLevel = normalizeSummaryLevel(rollupContext?.level || luxriotSummaryLevel);
        const aggregationLabel = targetLevel === 'L0'
            ? 'Loading L0 rollup source…'
            : `Aggregating ${targetLevel}…`;
        const renderAggregationProgress = () => {
            if (!isCurrentLuxriotSummaryRequest(requestContext)) return;
            const elapsedSec = Math.max(0, (Date.now() - progressStartedAt) / 1000);
            const detail = `${aggregationLabel} ${elapsedSec.toFixed(1)}s${admissionHint ? ` · ${admissionHint}` : ''}`;
            setLuxriotSummaryMeta(detail);
            setLuxriotStatus(detail);
        };
        try {
            showLuxriotSummaryLoading(aggregationLabel);
            const params = buildSummaryQueryParams(channelId);
            params.set('level_limit', '240');
            params.set('target_level', targetLevel);
            renderAggregationProgress();
            progressTimer = window.setInterval(renderAggregationProgress, 500);
            void fetch(`/lm/admission?t=${Date.now()}`, {
                cache: 'no-store',
                signal: requestContext?.controller?.signal,
            }).then(async (response) => {
                const admission = await response.json().catch(() => ({}));
                if (!response.ok || !isCurrentLuxriotSummaryRequest(requestContext)) return;
                const active = Number(admission.active || 0);
                const queued = Number(admission.queued || 0);
                admissionHint = queued > 0
                    ? `shared LM queue ${queued} queued · ${active} active`
                    : active > 0
                        ? `shared LM ${active} active`
                        : '';
                renderAggregationProgress();
            }).catch(() => {
                // Admission diagnostics are optional; the rollup request remains authoritative.
            });
            const resp = await fetch(`/luxriot/rollups?${params.toString()}`, {
                signal: requestContext?.controller?.signal,
            });
            const data = await resp.json();
            if (!isCurrentLuxriotSummaryRequest(requestContext)) return false;
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
                if (currentMode === 'video') {
                    refreshLuxriotSummaryView(channelId, true, false);
                }
                return false;
            }
            luxriotSummaryRollupCache[channelId] = data;
            const renderedCount = renderLuxriotRollups(data, channelId);
            if (
                renderedCount === 0
                && normalizeSummaryResolutionMode(luxriotSummaryResolutionMode) === 'AUTO'
                && !isLiveSummaryPeriod()
                && !(Array.isArray(rollupContext?.sourceIds) && rollupContext.sourceIds.length)
            ) {
                luxriotSummaryLevel = 'L0';
                luxriotSummaryRollupStack = [{ level: 'L0', sourceIds: null, label: 'L0' }];
                if (luxriotSummaryLevelSelect) luxriotSummaryLevelSelect.value = 'AUTO';
                setLuxriotSummaryMeta('No precomputed rollup covers this period · loading archived observations…');
                window.setTimeout(() => {
                    if (currentMode === 'video' && Number(getSelectedSummaryChannel()) === Number(channelId)) {
                        void refreshLuxriotSummaryView(channelId, true, false);
                    }
                }, 0);
                return true;
            }
            const counts = data.source_counts && typeof data.source_counts === 'object' ? data.source_counts : {};
            const ctx = getCurrentSummaryRollupContext();
            const level = normalizeSummaryLevel(ctx?.level || luxriotSummaryLevel);
            const drillLabel = ctx?.sourceIds ? ` · drill ${ctx.sourceIds.length}` : '';
            const runLabel = luxriotSummaryRunFilter || 'latest';
            const computedLevels = new Set(
                Array.isArray(data.computed_levels)
                    ? data.computed_levels.map((item) => String(item || '').toUpperCase())
                    : ['L0', 'L1', 'L2', 'L3']
            );
            const countLabel = (rollupLevel) => computedLevels.has(rollupLevel)
                ? String(Number(counts[rollupLevel] || 0))
                : '—';
            const countsLabel = `L1 ${countLabel('L1')} · L2 ${countLabel('L2')} · L3 ${countLabel('L3')}`;
            const pendingCount = luxriotSummaryRollupRows
                .filter((row) => String(row?.summary_kind || '').trim() === 'pending_context')
                .length;
            const queuedCount = luxriotSummaryRollupRows
                .filter((row) => {
                    const kind = String(row?.summary_kind || '').trim().toLowerCase();
                    const status = String(row?.generation_status || '').trim().toLowerCase();
                    return kind === 'queued' || status === 'queued' || status === 'deferred';
                })
                .length;
            const retryCount = luxriotSummaryRollupRows
                .filter((row) => {
                    const kind = String(row?.summary_kind || '').trim().toLowerCase();
                    return kind && !['llm', 'llm_cached', 'legacy_cached', 'pending_context', 'queued'].includes(kind);
                })
                .length;
            const windowSecMap = data.window_sec && typeof data.window_sec === 'object' ? data.window_sec : {};
            const windowLabel = formatSummaryWindowLabel(windowSecMap[level]);
            const pendingLabel = pendingCount > 0 ? ` · pending ${pendingCount}` : '';
            const queuedLabel = queuedCount > 0 ? ` · semantic queued ${queuedCount}` : '';
            const retryLabel = retryCount > 0 ? ` · retry available ${retryCount}` : '';
            const waitLabel = renderedCount === 0 ? ` · waiting for ${level} window ${windowLabel}` : '';
            const aggregationElapsed = Number(data.aggregation?.elapsed_sec);
            const elapsedLabel = Number.isFinite(aggregationElapsed)
                ? ` · aggregated ${aggregationElapsed.toFixed(1)}s`
                : '';
            const channelLabel = getLuxriotChannelLabel(channelId);
            setLuxriotSummaryMeta(withSummaryUpdatedMeta(`${channelLabel} · ${level}${drillLabel} · ${renderedCount} items${pendingLabel}${queuedLabel}${retryLabel}${waitLabel}${elapsedLabel} · run ${runLabel} · ${getSummaryRangeLabel()} · ${countsLabel}`));
            setLuxriotStatus(`Rollup view ${level} · ${renderedCount} entries`);
            return true;
        } catch (err) {
            if (err && err.name === 'AbortError') return false;
            if (!isCurrentLuxriotSummaryRequest(requestContext)) return false;
            setLuxriotSummaryMeta('Failed to load rollups: ' + (err.message || 'Unknown error'), true);
            setLuxriotStatus('Failed to fetch rollups: ' + err.message, true);
            return false;
        } finally {
            if (progressTimer) {
                clearInterval(progressTimer);
                progressTimer = null;
            }
            finishLuxriotSummaryLoading();
        }
    }

    async function refreshLuxriotSummaryView(channelId = getSelectedSummaryChannel(), force = false, allowRunFallback = true) {
        if (currentMode !== 'video') return false;
        if (!channelId) return;
        const requestKey = luxriotSummaryRequestKey(channelId);
        if (luxriotSummaryRefreshInFlight) {
            const next = luxriotSummaryRefreshQueued || {};
            luxriotSummaryRefreshQueued = {
                channelId,
                force: Boolean(force || next.force),
                allowRunFallback: Boolean((allowRunFallback !== false) || (next.allowRunFallback !== false)),
            };
            const active = luxriotSummaryActiveRequest;
            if (active && (force || active.requestKey !== requestKey)) {
                cancelLuxriotSummaryRequest();
            }
            if (force) {
                setLuxriotStatus('Refresh queued...');
            }
            return false;
        }
        luxriotSummaryRefreshInFlight = true;
        const controller = new AbortController();
        const requestContext = {
            generation: ++luxriotSummaryRequestGeneration,
            controller,
            channelId: Number(channelId),
            requestKey,
        };
        luxriotSummaryActiveRequest = requestContext;
        try {
            if (isRollupViewActive()) {
                return await refreshLuxriotRollups(channelId, force, allowRunFallback, requestContext);
            } else {
                return await refreshLuxriotSummaries(channelId, force, allowRunFallback, requestContext);
            }
        } finally {
            if (luxriotSummaryActiveRequest === requestContext) {
                luxriotSummaryActiveRequest = null;
            }
            luxriotSummaryRefreshInFlight = false;
            if (currentMode === 'video' && luxriotSummaryRefreshQueued) {
                const next = luxriotSummaryRefreshQueued;
                luxriotSummaryRefreshQueued = null;
                void refreshLuxriotSummaryView(
                    next.channelId || getSelectedSummaryChannel(),
                    Boolean(next.force),
                    next.allowRunFallback !== false,
                );
            } else if (currentMode !== 'video') {
                luxriotSummaryRefreshQueued = null;
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
        updateLuxriotStreamContext();
        if (rerender) {
            renderProbeCards();
        }
    }

    async function refreshProbeRuntimeState(rerender = false) {
        try {
            const resp = await fetch('/luxriot/streams');
            const data = await parseApiJson(resp, 'Failed to fetch runtime stream state');
            updateProbeChannelRuntime(data, rerender);
        } catch (_) {
            // Keep previous runtime snapshot when stream endpoint is unavailable.
        }
    }

    function renderLuxriotStreams(payload, probes = probeCatalog) {
        if (!luxriotStreams) return;
        const showProbeDiagnostics = canUseProbeDiagnostics();
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
        updateLuxriotStreamContext();
        maybeSwitchLuxriotPreviewToAttention();
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
        if (showProbeDiagnostics) {
            sortedAnalytics.forEach((stream) => {
                const channelId = parseInt(String(stream?.channel_id ?? ''), 10);
                if (!Number.isFinite(channelId)) return;
                analyticsByChannel.set(channelId, stream);
            });
        }
        const probeStatsByChannel = new Map();
        if (showProbeDiagnostics) {
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
        }
        const channelIds = new Set();
        sortedVideo.forEach((stream) => {
            const channelId = parseInt(String(stream?.channel_id ?? ''), 10);
            if (Number.isFinite(channelId)) channelIds.add(channelId);
        });
        if (showProbeDiagnostics) {
            sortedAnalytics.forEach((stream) => {
                const channelId = parseInt(String(stream?.channel_id ?? ''), 10);
                if (Number.isFinite(channelId)) channelIds.add(channelId);
            });
            pausedChannels.forEach((channelId) => channelIds.add(channelId));
        }
        historyChannels.forEach((channelId) => channelIds.add(channelId));
        if (showProbeDiagnostics) {
            probeStatsByChannel.forEach((_, channelId) => channelIds.add(channelId));
        }
        if (!channelIds.size) {
            luxriotStreams.innerHTML = '<div class="loading">No active channels.</div>';
            updateLuxriotStreamContext();
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
                    const snapshotLatency = Number(video?.last_snapshot_latency_sec);
                    const source = String(video?.active_capture_source || '').trim();
                    const segmentLatency = Number(video?.last_live_segment_latency_sec);
                    const segmentFrames = Number(video?.last_live_segment_frames) || 0;
                    const segmentTargetSeconds = Number(video?.last_live_segment_target_seconds);
                    const segmentSummaryTargetSeconds = Number(video?.last_live_segment_summary_target_seconds);
                    const segmentRepresentedSeconds = Number(video?.last_live_segment_represented_seconds);
                    const segmentInflight = Boolean(video?.live_segment_inflight);
                    const segmentInflightTargetSeconds = Number(video?.live_segment_inflight_target_seconds);
                    const segmentInflightRawBudget = Number(video?.live_segment_inflight_raw_frame_budget) || 0;
                    const segmentInflightFrames = Number(video?.live_segment_inflight_frames) || 0;
                    const segmentInflightRepresentedSeconds = Number(video?.live_segment_inflight_represented_seconds);
                    if (batch > 0) videoParts.push(`batch ${batch}`);
                    videoParts.push(`${queued} queued`);
                    if (source) videoParts.push(source);
                    if (Number.isFinite(snapshotLatency) && snapshotLatency > 0) videoParts.push(`snapshot ${snapshotLatency.toFixed(1)}s`);
                    const currentSnapshotSlow = source !== 'live_segment'
                        && Number.isFinite(snapshotLatency)
                        && snapshotLatency >= Math.max(2, Number(video?.snapshot_slow_threshold_sec) || 0);
                    if (currentSnapshotSlow) videoParts.push('slow snapshot');
                    if (Number.isFinite(segmentLatency) && segmentLatency > 0) videoParts.push(`segment ${segmentLatency.toFixed(1)}s/${segmentFrames}f`);
                    if (
                        Number.isFinite(segmentRepresentedSeconds)
                        && Number.isFinite(segmentTargetSeconds)
                        && segmentTargetSeconds > 0
                    ) {
                        const rate = segmentLatency > 0 ? segmentRepresentedSeconds / segmentLatency : null;
                        videoParts.push(
                            `attention ${segmentRepresentedSeconds.toFixed(1)}/${segmentTargetSeconds.toFixed(1)}s`
                            + (Number.isFinite(rate) ? ` ${rate.toFixed(2)}x` : '')
                        );
                    }
                    if (Number.isFinite(segmentSummaryTargetSeconds) && segmentSummaryTargetSeconds > 0) {
                        videoParts.push(`describe every ${segmentSummaryTargetSeconds.toFixed(1)}s`);
                    }
                    if (segmentInflight) {
                        videoParts.push(
                            `capturing ${Number.isFinite(segmentInflightRepresentedSeconds) ? segmentInflightRepresentedSeconds.toFixed(1) : '?'}s`
                            + (Number.isFinite(segmentInflightTargetSeconds) ? `/${segmentInflightTargetSeconds.toFixed(1)}s` : '')
                            + (segmentInflightRawBudget > 0 ? ` · ${segmentInflightFrames}/${segmentInflightRawBudget} raw` : '')
                        );
                    }
                    if (flushes > 0) videoParts.push(`${flushes} flushes`);
                    const issue = classifyLuxriotStreamIssue(video);
                    if (issue.hardError) videoParts.push('error');
                    else if (issue.backpressure) videoParts.push('aggregation backpressure');
                    else if (Boolean(video?.summary_inflight) || Number(video?.summary_queue_depth) > 0) videoParts.push('aggregating');
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
                    if (analytics?.shared_capture) {
                        probeParts.push('shared with video', `${queued} buffered`);
                    } else {
                        probeParts.push(fpsLabel, `${queued} buffered`);
                    }
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
                const videoHealthBadge = renderLuxriotHealthBadge(getLuxriotStreamHealth(video));
                const probeTag = isProbeRunning
                    ? '<span class="luxriot-stream-tag">probes active</span>'
                    : isProbePaused
                        ? '<span class="luxriot-stream-tag paused">probes paused</span>'
                        : enabledCount > 0
                            ? '<span class="luxriot-stream-tag idle">probes idle</span>'
                            : hasProbes
                                ? '<span class="luxriot-stream-tag idle">probes disabled</span>'
                                : '<span class="luxriot-stream-tag idle">no probes</span>';
                const pauseLabel = isProbePaused ? 'Resume CLIP probes' : 'Pause CLIP probes';
                const pauseAction = isProbePaused ? 'resume' : 'pause';
                const canPauseProbes = !isProbePaused && (isProbeRunning || enabledCount > 0);
                const canResumeProbes = isProbePaused;
                const canProbeAction = canPauseProbes || canResumeProbes;
                const canStopAll = isVideoRunning || isProbeRunning;
                const channelLabel = getLuxriotChannelLabel(channelId);
                const diagnosticsTags = showProbeDiagnostics ? ` ${probeTag}` : '';
                const diagnosticsStats = showProbeDiagnostics
                    ? `<span class="luxriot-stream-stat">${escapeHtml(probesLine)}</span>
                            <span class="luxriot-stream-stat">${escapeHtml(probeLine)}</span>`
                    : '';
                const diagnosticsControls = showProbeDiagnostics
                    ? `<button class="feature-btn" data-stream-stop="${channelId}" data-stream-type="analytics" data-stream-action="${pauseAction}" ${canProbeAction ? '' : 'disabled'}>${pauseLabel}</button>
                            <button class="feature-btn" data-stream-stop="${channelId}" data-stream-type="both" ${canStopAll ? '' : 'disabled'}>Stop all</button>`
                    : '';
                return `
                    <div class="luxriot-stream-item">
                        <div class="luxriot-stream-head">
                            <div class="luxriot-stream-title-wrap">
                                <div class="luxriot-stream-kind">Channel</div>
                                <div class="luxriot-stream-title">${escapeHtml(channelLabel)}</div>
                            </div>
                            <div class="luxriot-stream-tags">${videoTag} ${videoHealthBadge}${diagnosticsTags}</div>
                        </div>
                        <div class="luxriot-stream-stats">
                            <span class="luxriot-stream-stat">${escapeHtml(videoLine)}</span>
                            ${diagnosticsStats}
                        </div>
                        <div class="luxriot-stream-controls">
                            <button class="feature-btn" data-summary-channel="${channelId}" title="Open this channel in summaries panel">View summaries</button>
                            <button class="feature-btn" data-stream-stop="${channelId}" data-stream-type="video" ${isVideoRunning ? '' : 'disabled'}>Stop video</button>
                            ${diagnosticsControls}
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
            const data = await parseApiJson(resp, 'Failed to fetch stream state');
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
                const currentLiveModel = normalizeModelId(luxriotLiveModelInput.value);
                const autoSelector = normalizeModelId(lmModelCatalog.autoModelSelector || LM_AUTO_MODEL_SELECTOR);
                if (liveModel && currentLiveModel !== autoSelector) {
                    setModelSelectOptions(luxriotLiveModelInput, liveModel, '', { includeAuto: true });
                    localStorage.setItem(LUXRIOT_LIVE_MODEL_STORAGE_KEY, liveModel);
                }
            }
            updateLuxriotCaptureToggleButton(selectedChannelId);
            if (canUseProbeDiagnostics()) {
                try {
                    const probesResp = await fetch('/probes/list');
                    const probesData = await parseApiJson(probesResp, 'Failed to load CLIP probes');
                    if (Array.isArray(probesData.probes)) {
                        probeCatalog = probesData.probes;
                    }
                } catch (_) {
                    // Keep previous probe catalog if probe listing fails.
                }
            } else {
                probeCatalog = [];
            }
            renderLuxriotStreams(data, probeCatalog);
            syncLuxriotLiveIntervalInput(selectedChannelId);
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

    async function refreshLuxriotSummaries(channelId = getSelectedSummaryChannel(), force = false, allowRunFallback = true, requestContext = null) {
        if (!channelId) return;
        if (!isLiveSummaryPeriod()) {
            if (!force) return false;
            return refreshLuxriotArchivedSummaries(channelId, requestContext, false);
        }
        if (!luxriotSummaryAutoRefresh && !force) return;
        try {
            showLuxriotSummaryLoading(
                'Loading live descriptions…',
                Array.isArray(luxriotSummaryLogCache) && luxriotSummaryLogCache.length > 0,
            );
            const params = buildSummaryQueryParams(channelId);
            params.set('limit', '240');
            params.set('view', 'feed');
            const resp = await fetch(`/luxriot/session?${params.toString()}`, {
                signal: requestContext?.controller?.signal,
            });
            const data = await resp.json();
            if (!isCurrentLuxriotSummaryRequest(requestContext)) return false;
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
                if (currentMode === 'video') {
                    refreshLuxriotSummaryView(channelId, true, false);
                }
                return false;
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
            const streamIssue = classifyLuxriotStreamIssue(data);
            if (streamIssue.hardError) detailParts.push('err');
            else if (streamIssue.backpressure) detailParts.push('backpressure');
            else if (Boolean(data.summary_inflight) || Number(data.summary_queue_depth) > 0) detailParts.push('aggregating');
            setLuxriotSummaryMeta(withSummaryUpdatedMeta(detailParts.join(' · ')), Boolean(streamIssue.hardError));
            let baseStatus = data.running ? `Summaries running · batch ${data.batch_size || ''}` : 'Summaries stopped';
            if (typeof data.pending_frames === 'number' && data.pending_frames > 0) {
                baseStatus += ` · ${data.pending_frames} frames queued`;
            }
            if (streamIssue.backpressure) baseStatus += ' · aggregation backpressure';
            else if (Boolean(data.summary_inflight) || Number(data.summary_queue_depth) > 0) baseStatus += ' · aggregating';
            setLuxriotStatus(baseStatus, Boolean(streamIssue.hardError));
            if (streamIssue.hardError || streamIssue.backpressureError) {
                luxriotStatusLabel.title = streamIssue.hardError || streamIssue.backpressureError;
            }
            appendLuxriotStatusHealthBadge(data);
            return true;
        } catch (err) {
            if (err && err.name === 'AbortError') return false;
            if (!isCurrentLuxriotSummaryRequest(requestContext)) return false;
            setLuxriotSummaryMeta('Failed to load summaries: ' + (err.message || 'Unknown error'), true);
            setLuxriotStatus('Failed to fetch summaries: ' + err.message, true);
            return false;
        } finally {
            finishLuxriotSummaryLoading();
        }
    }

    function startLuxriotSummaryPoll() {
        if (luxriotSummaryTimer) {
            clearInterval(luxriotSummaryTimer);
            luxriotSummaryTimer = null;
        }
        luxriotSummaryTimer = setInterval(() => {
            if (currentMode !== 'video') return;
            const channelId = getSelectedSummaryChannel();
            if (isLiveSummaryPeriod()) refreshLuxriotSummaryView(channelId);
            refreshLuxriotStreams();
        }, 8000);
    }

    async function startLuxriotCapture(channelIdOverride = null) {
        const channelId = Number.isFinite(channelIdOverride) ? channelIdOverride : getSelectedLuxriotChannel();
        if (!channelId) {
            setLuxriotStatus('Select a channel first', true);
            return;
        }
        const promptSettingsReady = await refreshLuxriotPromptSettings(false, channelId);
        const selectedChannelId = getSelectedLuxriotChannel();
        if (!promptSettingsReady || selectedChannelId !== channelId) {
            const message = selectedChannelId !== channelId
                ? 'Channel changed while prompt settings were loading; start summaries again for the selected channel.'
                : 'Prompt settings could not be verified for this channel; summary start was cancelled.';
            setLuxriotStatus(message, true);
            return;
        }
        const batchSize = luxriotBatchSizeSelect
            ? parseInt(luxriotBatchSizeSelect.value, 10)
            : luxriotDefaults.batchSize || 12;
        const intervalSec = getLuxriotLiveIntervalInputValue();
        if (luxriotLiveIntervalInput) {
            luxriotLiveIntervalInput.value = formatLuxriotLiveIntervalInput(intervalSec);
        }
        storeLuxriotLiveInterval(channelId, intervalSec);
        const prompt = luxriotPromptInput ? luxriotPromptInput.value.trim() : '';
        const systemPrompt = luxriotSystemPromptInput ? luxriotSystemPromptInput.value.trim() : '';
        const fallbackPrompt = videoPromptInput ? videoPromptInput.value.trim() : '';
        setLuxriotCaptureBusy(true);
        setLuxriotStatus('Starting summaries...');
        try {
            const resp = await fetch('/luxriot/start_capture', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    channel_id: channelId,
                    batch_size: batchSize,
                    interval_sec: intervalSec,
                    prompt: prompt || fallbackPrompt,
                    model: luxriotLiveModelInput ? luxriotLiveModelInput.value.trim() : '',
                    system_prompt: systemPrompt
                })
            });
            const data = await resp.json();
            if (!resp.ok || data.error) {
                throw new Error(data.error || 'Luxriot start failed');
            }
            clearLuxriotLiveIntervalDirty(channelId);
            setLuxriotCaptureRunning(channelId, true);
            updateLuxriotCaptureToggleButton(channelId);
            const modelLabel = data?.session?.model || (luxriotLiveModelInput ? luxriotLiveModelInput.value.trim() : '') || '';
            const appliedInterval = normalizeLuxriotLiveInterval(data?.session?.interval_sec) || intervalSec;
            setLuxriotStatus(`Summaries running on channel ${channelId} (${formatLuxriotCadence(appliedInterval)} · batch ${batchSize}${modelLabel ? ` · ${modelLabel}` : ''})`);
            luxriotSummaryChannel = channelId;
            luxriotSummaryFollowLive = true;
            syncLuxriotSummaryChannelSelect();
            updateSummaryControlsUI();
            setSummaryUnread(0);
            refreshLuxriotSummaryView(channelId, true);
            refreshLuxriotStreams();
            if (currentMode === 'video') {
                startLuxriotSummaryPoll();
            }
        } catch (err) {
            setLuxriotStatus(err.message, true);
        } finally {
            setLuxriotCaptureBusy(false);
        }
    }

    async function stopLuxriotCapture(channelIdOverride = null) {
        const channelId = Number.isFinite(channelIdOverride) ? channelIdOverride : getSelectedLuxriotChannel();
        setLuxriotCaptureBusy(true);
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
            setLuxriotCaptureBusy(false);
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
        if (luxriotInitPromise) return luxriotInitPromise;
        luxriotInitPromise = (async () => {
            await fetchLuxriotChannels();
            await refreshLuxriotPromptSettings();
            updateLuxriotCaptureToggleButton(getSelectedLuxriotChannel());
            updateSummaryControlsUI();
            setSummaryUnread(0);
            syncLuxriotSummaryChannelSelect();
            refreshLuxriotStreams();
            luxriotInitialized = true;
        })();
        try {
            await luxriotInitPromise;
        } catch (err) {
            luxriotInitialized = false;
            luxriotInitPromise = null;
            throw err;
        }
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
    syncLuxriotLiveIntervalInput(luxriotActiveChannel, { force: true });
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
    applySummaryResolutionMode();

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
    const clipModelSelect = document.getElementById('clipModel');
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
    const luxriotSummaryRetentionDaysInput = document.getElementById('luxriotSummaryRetentionDays');
    const luxriotSummaryHistoryLimitInput = document.getElementById('luxriotSummaryHistoryLimit');
    const archiveRetentionEnabledInput = document.getElementById('archiveRetentionEnabled');
    const archiveRowRetentionDaysInput = document.getElementById('archiveRowRetentionDays');
    const archiveThumbnailRetentionDaysInput = document.getElementById('archiveThumbnailRetentionDays');
    const archiveMaxRecordsInput = document.getElementById('archiveMaxRecords');
    const archiveEstimateChannelsInput = document.getElementById('archiveEstimateChannels');
    const archiveEstimateFramesPerBatchInput = document.getElementById('archiveEstimateFramesPerBatch');
    const archiveEstimateAvgJpegKbInput = document.getElementById('archiveEstimateAvgJpegKb');
    const archiveEstimateProbeRowsInput = document.getElementById('archiveEstimateProbeRows');
    const archiveCapacitySummary = document.getElementById('archiveCapacitySummary');
    const luxriotAutoBookmarksInput = document.getElementById('luxriotAutoBookmarks');
    let experimentalEmbeddersEnabled = false;
    let productionClipModel = 'ViT-B/32';
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
    const adminChannelList = document.getElementById('adminChannelList');
    const adminChannelsAllBtn = document.getElementById('adminChannelsAllBtn');
    const adminChannelsNoneBtn = document.getElementById('adminChannelsNoneBtn');
    const adminChannelsRefreshBtn = document.getElementById('adminChannelsRefreshBtn');
    const adminUserActiveInput = document.getElementById('adminUserActiveInput');
    const adminUserStateSummary = document.getElementById('adminUserStateSummary');
    const adminUserSaveBtn = document.getElementById('adminUserSaveBtn');
    const adminUserResetPasswordBtn = document.getElementById('adminUserResetPasswordBtn');
    const adminUserToggleActiveBtn = document.getElementById('adminUserToggleActiveBtn');
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
    let adminChannelCatalog = [];
    let selectedAdminUserId = null;
    let auditEvents = [];
    let auditNextCursor = null;
    let auditLastParams = null;

    function setActiveSettingsNav(targetId) {
        settingsNavButtons.forEach((btn) => {
            const active = btn.dataset.settingsTarget === targetId;
            btn.classList.toggle('active', active);
            if (active && typeof btn.scrollIntoView === 'function') {
                btn.scrollIntoView({ block: 'nearest', inline: 'nearest' });
            }
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

    function setElementHidden(element, hidden) {
        if (!element) return;
        element.classList.toggle('is-hidden', Boolean(hidden));
    }

    function setPermissionHidden(element, hidden) {
        if (!element) return;
        element.classList.toggle('permission-hidden', Boolean(hidden));
    }

    function setControlDisabled(element, disabled) {
        if (!element) return;
        element.disabled = Boolean(disabled);
    }

    function settingsTargetAllowed(targetId) {
        if (!targetId) return false;
        if (targetId === 'settings-section-users') return userCanManageUsers();
        if (targetId === 'settings-section-audit') return userCanViewAudit();
        if (targetId === 'settings-section-env') return userHasPermission('settings:manage');
        return userHasAnyPermission(['settings:view', 'settings:manage']);
    }

    function firstVisibleSettingsTarget() {
        const button = settingsNavButtons.find((btn) => !btn.classList.contains('is-hidden'));
        return button ? button.dataset.settingsTarget : '';
    }

    function syncSettingsAccess() {
        settingsNavButtons.forEach((btn) => {
            const targetId = btn.dataset.settingsTarget;
            const allowed = settingsTargetAllowed(targetId);
            setElementHidden(btn, !allowed);
            setElementHidden(document.getElementById(targetId), !allowed);
        });
        setAdminUsersAccess(userCanManageUsers());
        setAuditAccess(userCanViewAudit());

        const canManageSettings = userHasPermission('settings:manage');
        setElementHidden(saveSettingsBtn, !canManageSettings);
        setElementHidden(resetSettingsBtn, !canManageSettings);
        setElementHidden(saveEnvBtn, !canManageSettings);
        setElementHidden(reloadEnvBtn, !canManageSettings);
        setControlDisabled(saveSettingsBtn, !canManageSettings);
        setControlDisabled(resetSettingsBtn, !canManageSettings);
        setControlDisabled(saveEnvBtn, !canManageSettings);

        const firstTarget = firstVisibleSettingsTarget();
        setElementHidden(settingsBtn, !firstTarget);
        if (!firstTarget && settingsModal && settingsModal.style.display === 'block') {
            settingsModal.style.display = 'none';
        }
        const activeNav = settingsNavButtons.find((btn) => btn.classList.contains('active'));
        if (firstTarget && (!activeNav || activeNav.classList.contains('is-hidden'))) {
            setActiveSettingsNav(firstTarget);
        }
        return Boolean(firstTarget);
    }

    function syncUiAccess() {
        const archiveAllowed = canUseMode('archive');
        const videoAllowed = canUseMode('video');
        const monitorAllowed = canUseMode('monitor');
        const agentAllowed = canUseMode('agent');
        setElementHidden(archiveModeBtn, !archiveAllowed);
        setElementHidden(videoModeBtn, !videoAllowed);
        setElementHidden(monitorModeBtn, !monitorAllowed);
        setElementHidden(agentModeBtn, !agentAllowed);

        if (AUTH_ENABLED && authCurrentUser && !canUseMode(currentMode)) {
            setMode(firstAllowedMode());
        }

        const canCapture = userHasPermission('capture:manage');
        const canPromptManage = userHasPermission('prompts:manage');
        const canProbeRun = userHasPermission('probes:run');
        const canProbeManage = userHasPermission('probes:manage');
        const canDiagnostics = userHasPermission('diagnostics:view');
        const canModelsManage = userHasPermission('models:manage');
        const canBookmarks = userHasPermission('bookmarks:create');
        const canRoadGround = (
            userHasAnyRole(['admin', 'engineer'])
            || (canDiagnostics && userHasPermission('streams:view'))
        );

        [
            luxriotToggleCaptureBtn,
            luxriotFlushCaptureBtn,
            luxriotStopAllVideoBtn,
            probeStreamToggleBtn,
        ].forEach((element) => {
            setElementHidden(element, !canCapture);
            setControlDisabled(element, !canCapture);
        });
        const canOperateProbeDiagnostics = canCapture && (canProbeManage || canDiagnostics);
        setElementHidden(luxriotStopAllAnalyticsBtn, !canOperateProbeDiagnostics);
        setControlDisabled(luxriotStopAllAnalyticsBtn, !canOperateProbeDiagnostics);
        setElementHidden(luxriotPromptSettingsBtn, !canPromptManage);
        setControlDisabled(luxriotPromptSettingsBtn, !canPromptManage);
        setControlDisabled(luxriotPromptApplyBtn, !canPromptManage);
        setControlDisabled(luxriotPromptResetBtn, !canPromptManage);
        [luxriotBookmarkEnabledInput, luxriotBookmarkCooldownInput].forEach((element) => {
            setControlDisabled(element, !canBookmarks || !canPromptManage);
        });
        luxriotPromptTabButtons.forEach((button) => {
            const tab = String(button.dataset.luxriotPromptTab || '').trim().toLowerCase();
            if (tab === 'json') setElementHidden(button, !canBookmarks);
        });
        if (!canBookmarks && String(luxriotPromptModalTab || '').trim().toLowerCase() === 'json') {
            setLuxriotPromptModalTab('stream');
        }
        setLuxriotPromptApplyAvailability(false);
        setPermissionHidden(saveSummaryBtn, !canBookmarks);
        setControlDisabled(saveSummaryBtn, !canBookmarks);

        [probeRunBtn, probeImageEnableToggle].forEach((element) => {
            setElementHidden(element, !canProbeRun);
            setControlDisabled(element, !canProbeRun);
        });
        [
            probeSaveBtn,
            probeCastBtn,
            probeDeleteBtn,
            probeEditBtn,
            probeNewBtn,
            probeSnapUseBtn,
            probeRoiToggleBtn,
            probeRoiClearBtn,
        ].forEach((element) => {
            setElementHidden(element, !canProbeManage);
            setControlDisabled(element, !canProbeManage);
        });
        setElementHidden(probeBenchBtn, !canDiagnostics);
        setControlDisabled(probeBenchBtn, !canDiagnostics);
        roadSceneGroundingBtns.forEach((button) => {
            setElementHidden(button, !canRoadGround);
            setControlDisabled(button, !canRoadGround);
        });
        [
            probeBookmarkToggle,
            probeBookmarkSeverityInput,
            probeBookmarkCooldownLocalInput,
            probeBookmarkDedupeWindowLocalInput,
        ].forEach((element) => {
            setControlDisabled(element, !canBookmarks || !canProbeManage);
        });

        setElementHidden(agentCreateSkillBtn, !canPromptManage);
        setControlDisabled(agentCreateSkillBtn, !canPromptManage);
        setControlDisabled(agentSkillSaveBtn, !canPromptManage);
        setControlDisabled(agentModelApplyBtn, !canModelsManage);
        if (agentModelInput) {
            agentModelInput.title = canModelsManage ? '' : 'Model changes require models:manage.';
        }
        syncArchiveDiagnosticSourceVisibility();

        syncSettingsAccess();
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

    function selectedAdminChannelsFromInput() {
        try {
            return parseAllowedChannelsText(adminAllowedChannelsInput ? adminAllowedChannelsInput.value : '');
        } catch (_) {
            return [];
        }
    }

    function normalizeAdminChannelCatalog(channels = []) {
        const seen = new Set();
        return (Array.isArray(channels) ? channels : [])
            .map((channel) => {
                const id = Number.parseInt(String(channel?.channel_id ?? channel?.id ?? ''), 10);
                if (!Number.isFinite(id) || id <= 0 || seen.has(id)) return null;
                seen.add(id);
                return {
                    id,
                    label: normalizeLuxriotChannelName(channel, id),
                };
            })
            .filter((channel) => Boolean(channel))
            .sort((left, right) => left.id - right.id);
    }

    function renderAdminChannelPicker(selectedChannels = null) {
        if (!adminChannelList) return;
        const selected = Array.isArray(selectedChannels) ? selectedChannels : selectedAdminChannelsFromInput();
        const allSelected = selected.some((value) => String(value).trim() === '*');
        const selectedSet = new Set(selected.map((value) => String(value).trim()));
        if (!adminChannelCatalog.length) {
            adminChannelList.innerHTML = '<div class="admin-empty">Channel list unavailable. Use IDs above.</div>';
            return;
        }
        adminChannelList.innerHTML = adminChannelCatalog.map((channel) => {
            const value = String(channel.id);
            const checked = allSelected || selectedSet.has(value) ? ' checked' : '';
            return `
                <label class="admin-channel-item" title="${escapeHtml(channel.label)}">
                    <input type="checkbox" value="${escapeHtml(value)}"${checked} />
                    <span class="admin-channel-id">#${escapeHtml(value)}</span>
                    <span class="admin-channel-name">${escapeHtml(channel.label)}</span>
                </label>
            `;
        }).join('');
    }

    function setAdminChannelCatalog(channels = []) {
        adminChannelCatalog = normalizeAdminChannelCatalog(channels);
        renderAdminChannelPicker();
    }

    function syncAdminChannelTextFromPicker() {
        if (!adminChannelList || !adminAllowedChannelsInput) return;
        const checked = Array.from(adminChannelList.querySelectorAll('input[type="checkbox"]:checked'))
            .map((input) => Number.parseInt(String(input.value || ''), 10))
            .filter((id) => Number.isFinite(id) && id > 0);
        if (checked.length && checked.length === adminChannelCatalog.length) {
            adminAllowedChannelsInput.value = '*';
        } else {
            adminAllowedChannelsInput.value = checked.join(', ');
        }
        renderAdminChannelPicker(checked.length === adminChannelCatalog.length ? ['*'] : checked);
    }

    function syncAdminChannelPickerFromText() {
        if (!adminAllowedChannelsInput) return;
        try {
            renderAdminChannelPicker(parseAllowedChannelsText(adminAllowedChannelsInput.value));
        } catch (error) {
            setAdminUsersStatus(error.message || String(error), 'error');
        }
    }

    async function loadAdminChannelCatalog(force = false) {
        if (!adminChannelList) return;
        if (!force && adminChannelCatalog.length) {
            renderAdminChannelPicker();
            return;
        }
        adminChannelList.innerHTML = '<div class="admin-empty">Loading channels...</div>';
        try {
            const response = await fetch(`/luxriot/channels${force ? '?force=1' : ''}`, { cache: 'no-store' });
            const data = await parseApiJson(response, 'Failed to load channels');
            setAdminChannelCatalog(data.channels || []);
            if (!adminChannelCatalog.length) {
                adminChannelList.innerHTML = '<div class="admin-empty">No channels returned by Luxriot.</div>';
            }
        } catch (error) {
            adminChannelCatalog = [];
            adminChannelList.innerHTML = `<div class="admin-empty">Channel list failed: ${escapeHtml(error.message || String(error))}</div>`;
        }
    }

    function formatDateTime(value) {
        if (!value) return 'n/a';
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return String(value);
        return date.toLocaleString();
    }

    function currentAdminUserId() {
        return authCurrentUser ? String(authCurrentUser.id || '') : '';
    }

    function selectedAdminUser() {
        if (!selectedAdminUserId) return null;
        return adminUsers.find((item) => String(item.id || '') === selectedAdminUserId) || null;
    }

    function selectedAdminUserIsSelf(user = selectedAdminUser()) {
        const currentUserId = currentAdminUserId();
        return Boolean(user && currentUserId && String(user.id || '') === currentUserId);
    }

    function setAdminUserStateSummary(user = null) {
        if (!adminUserStateSummary) return;
        if (!user || !user.id) {
            adminUserStateSummary.classList.add('is-hidden');
            adminUserStateSummary.innerHTML = '';
            return;
        }
        const active = Boolean(user.isActive);
        const self = selectedAdminUserIsSelf(user);
        const statusClass = active ? 'active' : 'inactive';
        const statusText = active ? 'Active' : 'Disabled';
        const selfText = self ? 'Current session account' : 'Managed account';
        adminUserStateSummary.classList.remove('is-hidden');
        adminUserStateSummary.innerHTML = `
            <span class="admin-user-status-badge ${statusClass}">${statusText}</span>
            <span>${escapeHtml(selfText)}</span>
        `;
    }

    function syncAdminUserActionButtons(user = selectedAdminUser()) {
        const isExisting = Boolean(user && user.id);
        const isSelf = selectedAdminUserIsSelf(user);
        if (adminUserResetPasswordBtn) {
            adminUserResetPasswordBtn.disabled = !isExisting;
        }
        if (adminUserToggleActiveBtn) {
            adminUserToggleActiveBtn.disabled = !isExisting || (isSelf && Boolean(user?.isActive));
            adminUserToggleActiveBtn.textContent = user && !user.isActive ? 'Enable User' : 'Disable User';
            adminUserToggleActiveBtn.classList.toggle('danger', !user || Boolean(user.isActive));
            adminUserToggleActiveBtn.title = isSelf && Boolean(user?.isActive)
                ? 'Cannot disable the current account.'
                : '';
        }
        if (adminUserRevokeBtn) {
            adminUserRevokeBtn.disabled = !isExisting;
        }
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
            syncAdminChannelPickerFromText();
        }
        if (adminUserActiveInput) {
            adminUserActiveInput.checked = isExisting ? Boolean(user.isActive) : true;
        }
        setAdminUserStateSummary(isExisting ? user : null);
        syncAdminUserActionButtons(isExisting ? user : null);
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
            const stateBadge = user.isActive
                ? '<span class="admin-user-status-badge active">active</span>'
                : '<span class="admin-user-status-badge inactive">disabled</span>';
            const channels = formatAllowedChannels(user.allowedChannelIds || []) || 'none';
            return `
                <button type="button" class="admin-user-row${selected}${inactive}" data-user-id="${escapeHtml(id)}">
                    <span class="admin-user-main">
                        <span class="admin-user-name">${escapeHtml(user.username || id)}</span>
                        ${stateBadge}
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
            await loadAdminChannelCatalog(false);
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

    function adminUserLabel(user = selectedAdminUser()) {
        if (!user) return selectedAdminUserId || 'selected user';
        return user.username || user.displayName || user.id || 'selected user';
    }

    async function refreshAdminIdentityPanels() {
        await Promise.all([loadAdminUsers(), loadAdminSessions()]);
    }

    async function patchAdminUser(userId, payload, errorMessage = 'Failed to update user') {
        const clean = String(userId || '').trim();
        if (!clean) throw new Error('Select a user first.');
        const response = await fetch(`/auth/users/${encodeURIComponent(clean)}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        return parseApiJson(response, errorMessage);
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
        const existingUser = selectedAdminUser();
        if (selectedAdminUserId) {
            if (!existingUser) {
                setAdminUsersStatus('Selected user was not found. Refresh and try again.', 'error');
                return;
            }
            if (password) payload.password = password;
        } else {
            payload.username = adminUsernameInput ? adminUsernameInput.value.trim() : '';
            payload.password = password;
            if (!payload.username || !payload.password) {
                setAdminUsersStatus('Username and password are required for a new user.', 'error');
                return;
            }
        }
        if (existingUser) {
            const warnings = [];
            if (existingUser.isActive && !payload.isActive) {
                if (selectedAdminUserIsSelf(existingUser)) {
                    setAdminUsersStatus('You cannot disable your own active account.', 'error');
                    return;
                }
                warnings.push('disable the account and revoke active sessions');
            }
            if (password) {
                warnings.push('reset the password and revoke active sessions');
            }
            if (warnings.length) {
                const label = adminUserLabel(existingUser);
                if (!window.confirm(`Save changes for ${label}? This will ${warnings.join(' and ')}.`)) {
                    return;
                }
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
            await refreshAdminIdentityPanels();
            setAdminUsersStatus('User saved.', 'success');
        } catch (error) {
            setAdminUsersStatus(error.message || String(error), 'error');
        } finally {
            setButtonBusy(adminUserSaveBtn, false);
        }
    }

    async function resetSelectedAdminUserPassword() {
        if (!syncAdminUsersAccess()) return;
        const user = selectedAdminUser();
        if (!user || !selectedAdminUserId) {
            setAdminUsersStatus('Select a user first.', 'error');
            return;
        }
        const password = adminPasswordInput ? adminPasswordInput.value : '';
        if (!password) {
            setAdminUsersStatus('Enter a replacement password first.', 'error');
            return;
        }
        const label = adminUserLabel(user);
        if (!window.confirm(`Reset password for ${label}? Active sessions will be revoked.`)) return;

        setButtonBusy(adminUserResetPasswordBtn, true);
        setAdminUsersStatus('Resetting password...', 'loading');
        try {
            await patchAdminUser(
                selectedAdminUserId,
                { password },
                'Failed to reset password'
            );
            if (adminPasswordInput) adminPasswordInput.value = '';
            await refreshAdminIdentityPanels();
            setAdminUsersStatus('Password reset. Active sessions revoked.', 'success');
        } catch (error) {
            setAdminUsersStatus(error.message || String(error), 'error');
        } finally {
            setButtonBusy(adminUserResetPasswordBtn, false);
        }
    }

    async function toggleSelectedAdminUserActive() {
        if (!syncAdminUsersAccess()) return;
        const user = selectedAdminUser();
        if (!user || !selectedAdminUserId) {
            setAdminUsersStatus('Select a user first.', 'error');
            return;
        }
        const nextActive = !Boolean(user.isActive);
        const label = adminUserLabel(user);
        if (!nextActive && selectedAdminUserIsSelf(user)) {
            setAdminUsersStatus('You cannot disable your own active account.', 'error');
            return;
        }
        const message = nextActive
            ? `Enable ${label}?`
            : `Disable ${label}? Active sessions will be revoked.`;
        if (!window.confirm(message)) return;

        setButtonBusy(adminUserToggleActiveBtn, true);
        setAdminUsersStatus(nextActive ? 'Enabling user...' : 'Disabling user...', 'loading');
        try {
            const data = await patchAdminUser(
                selectedAdminUserId,
                { isActive: nextActive },
                nextActive ? 'Failed to enable user' : 'Failed to disable user'
            );
            const saved = data.user || null;
            if (saved && saved.id) {
                selectedAdminUserId = String(saved.id);
            }
            await refreshAdminIdentityPanels();
            setAdminUsersStatus(nextActive ? 'User enabled.' : 'User disabled. Active sessions revoked.', 'success');
        } catch (error) {
            setAdminUsersStatus(error.message || String(error), 'error');
        } finally {
            setButtonBusy(adminUserToggleActiveBtn, false);
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

    function parseArchiveLocalDateTimeMs(value) {
        const raw = String(value || '').trim();
        if (!raw) return null;
        const parsed = new Date(raw);
        const ms = parsed.getTime();
        return Number.isFinite(ms) ? ms : null;
    }

    function readArchiveTimeWindow() {
        const sinceMs = parseArchiveLocalDateTimeMs(archiveFromTimeInput ? archiveFromTimeInput.value : '');
        const untilMs = parseArchiveLocalDateTimeMs(archiveToTimeInput ? archiveToTimeInput.value : '');
        if (sinceMs !== null && untilMs !== null && sinceMs > untilMs) {
            throw new Error('Archive time window is invalid: From is later than To.');
        }
        return {
            since_ms: sinceMs,
            until_ms: untilMs,
            absolute: sinceMs !== null || untilMs !== null,
        };
    }

    function applyArchiveTimeFilters(target) {
        const window = readArchiveTimeWindow();
        if (window.since_ms !== null) {
            target.set ? target.set('since_ms', String(window.since_ms)) : (target.since_ms = window.since_ms);
        }
        if (window.until_ms !== null) {
            target.set ? target.set('until_ms', String(window.until_ms)) : (target.until_ms = window.until_ms);
        }
        if (!window.absolute) {
            const hoursRaw = archiveTimeFilter ? archiveTimeFilter.value : '24';
            const parsedHours = Number.parseFloat(hoursRaw);
            target.set
                ? target.set('hours', Number.isFinite(parsedHours) && parsedHours > 0 ? String(parsedHours) : '0')
                : (target.hours = Number.isFinite(parsedHours) ? parsedHours : 24);
        }
        return window;
    }

    function normalizeArchiveThresholdPercent(value) {
        const parsed = Number.parseFloat(value);
        return Math.min(100, Math.max(0, Number.isFinite(parsed) ? parsed : 0));
    }

    function getArchiveResultScore(result) {
        if (!result || typeof result !== 'object') return null;
        const candidates = [
            result.similarity,
            result.score,
            result.match_score,
            result.final_score,
            result?.fusion?.clip_similarity,
            result?.fusion?.dino_similarity,
        ];
        for (const candidate of candidates) {
            const score = Number.parseFloat(candidate);
            if (Number.isFinite(score)) {
                if (score > 1 && score <= 100) return score / 100;
                return Math.min(1, Math.max(0, score));
            }
        }
        return null;
    }

    function formatArchiveScoreValue(score) {
        if (!Number.isFinite(score)) return 'n/a';
        return score.toFixed(3);
    }

    function computeArchiveScoreRange(results) {
        const scores = (Array.isArray(results) ? results : [])
            .map(getArchiveResultScore)
            .filter((score) => Number.isFinite(score))
            .sort((a, b) => a - b);
        if (!scores.length) {
            return { count: 0, min: null, max: null, hasSpread: false };
        }
        const min = scores[0];
        const max = scores[scores.length - 1];
        return {
            count: scores.length,
            min,
            max,
            hasSpread: (max - min) > 0.000001,
        };
    }

    function archiveThresholdFromSlider(percent) {
        const sliderPercent = normalizeArchiveThresholdPercent(percent);
        if (sliderPercent <= 0 || !archiveScoreRange.hasSpread) return 0;
        return archiveScoreRange.min + ((archiveScoreRange.max - archiveScoreRange.min) * (sliderPercent / 100));
    }

    function refreshArchiveScoreScale(results) {
        archiveScoreRange = computeArchiveScoreRange(results);
        archiveScoreThreshold = archiveThresholdFromSlider(archiveScoreSliderPercent);
        updateArchiveThresholdUi();
    }

    function archiveResultPassesThreshold(result) {
        if (archiveScoreThreshold <= 0) return true;
        const score = getArchiveResultScore(result);
        return Number.isFinite(score) && score >= archiveScoreThreshold;
    }

    function visibleArchiveResultIndexes() {
        return archiveRenderedResults
            .map((result, index) => (archiveResultPassesThreshold(result) ? index : -1))
            .filter((index) => index >= 0);
    }

    function updateArchiveThresholdUi() {
        const percent = Math.round(archiveScoreSliderPercent);
        if (archiveScoreThresholdInput) {
            archiveScoreThresholdInput.value = String(percent);
            archiveScoreThresholdInput.disabled = !archiveScoreRange.hasSpread;
        }
        if (archiveScoreThresholdValue) {
            if (!archiveScoreRange.count) {
                archiveScoreThresholdValue.textContent = 'No scores';
            } else if (!archiveScoreRange.hasSpread) {
                archiveScoreThresholdValue.textContent = `All @ ${formatArchiveScoreValue(archiveScoreRange.min)}`;
            } else if (archiveScoreThreshold <= 0) {
                archiveScoreThresholdValue.textContent = 'All';
            } else {
                archiveScoreThresholdValue.textContent = `≥ ${formatArchiveScoreValue(archiveScoreThreshold)}`;
            }
        }
    }

    function applyArchiveScoreThreshold({ selectFirstVisible = false } = {}) {
        updateArchiveThresholdUi();
        const items = Array.from(document.querySelectorAll('#results .result-item'));
        let visibleCount = 0;
        let firstVisibleIndex = -1;
        items.forEach((item) => {
            const index = Number.parseInt(item.dataset.resultIndex || '-1', 10);
            const result = archiveRenderedResults[index];
            const visible = archiveResultPassesThreshold(result);
            item.classList.toggle('is-score-hidden', !visible);
            if (visible) {
                visibleCount += 1;
                if (firstVisibleIndex < 0) firstVisibleIndex = index;
            }
        });
        if (archiveScoreThresholdMeta) {
            const total = archiveRenderedResults.length;
            if (!total) {
                archiveScoreThresholdMeta.textContent = 'No results loaded.';
            } else if (!archiveScoreRange.count) {
                archiveScoreThresholdMeta.textContent = `Showing ${visibleCount}/${total} results. No match scores in this batch.`;
            } else if (!archiveScoreRange.hasSpread) {
                archiveScoreThresholdMeta.textContent = `Showing ${visibleCount}/${total} results. All scored matches are ${formatArchiveScoreValue(archiveScoreRange.min)}.`;
            } else if (archiveScoreThreshold <= 0) {
                archiveScoreThresholdMeta.textContent = `Showing ${visibleCount}/${total} results. Range ${formatArchiveScoreValue(archiveScoreRange.min)}-${formatArchiveScoreValue(archiveScoreRange.max)}.`;
            } else {
                archiveScoreThresholdMeta.textContent = `Showing ${visibleCount}/${total} results at ≥ ${formatArchiveScoreValue(archiveScoreThreshold)} (${Math.round(archiveScoreSliderPercent)}% of current range ${formatArchiveScoreValue(archiveScoreRange.min)}-${formatArchiveScoreValue(archiveScoreRange.max)}).`;
            }
        }
        if (!archiveRenderedResults.length) return;
        const activeVisible = activeArchiveInspectorIndex >= 0
            && archiveResultPassesThreshold(archiveRenderedResults[activeArchiveInspectorIndex]);
        if ((selectFirstVisible || !activeVisible) && firstVisibleIndex >= 0) {
            showArchiveInspector(firstVisibleIndex);
        } else if (firstVisibleIndex < 0) {
            renderArchiveInspectorEmpty('No results match the current score threshold.');
            highlightActiveArchiveResultCard(-1);
            activeArchiveInspectorIndex = -1;
        }
    }

    function setArchiveScoreThresholdFromInput(value) {
        const percent = normalizeArchiveThresholdPercent(value);
        archiveScoreSliderPercent = percent;
        archiveScoreThreshold = archiveThresholdFromSlider(percent);
        applyArchiveScoreThreshold();
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
            const sourceRaw = String(result.source || '').trim().toLowerCase();
            const logicalSource = archiveLogicalSource(sourceRaw);
            const showDiagnostics = canUseProbeDiagnostics();
            const sourceLabel = archiveSourceLabel(sourceRaw, result.source_label);
            const origin = String(result.origin || result.runtime_source || result?.payload?.origin || result?.payload?.source || '').trim();
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
                `<div class="metric-line"><span class="metric-label">Source:</span> ${escapeHtml(sourceLabel)}</div>`,
                `<div class="metric-line"><span class="metric-label">Time:</span> ${escapeHtml(ts)}</div>`,
                `<div class="metric-line metric-line-wrap"><span class="metric-label">Stream:</span> <span class="metric-value metric-stream-name" title="${safeChannelName}">${safeChannelName}</span></div>`,
                `<div class="metric-line"><span class="metric-label">Severity:</span> <span class="metric-value">${sev}</span></div>`,
            ];
            if (origin && origin !== sourceRaw && origin !== logicalSource) {
                lines.push(`<div class="metric-line"><span class="metric-label">Origin:</span> ${escapeHtml(origin)}</div>`);
            }
            if (showDiagnostics && logicalSource === 'probe') {
                lines.push(`<div class="metric-line metric-line-scores"><span class="metric-label">Scores:</span> <span class="metric-score metric-score-pos">P ${escapeHtml(pos)}</span> <span class="metric-score metric-score-neg">N ${escapeHtml(neg)}</span> <span class="metric-score metric-score-margin">M ${escapeHtml(margin)}</span></div>`);
            }
            if (similarity) {
                const modeHint = mode ? ` <span class="metric-note">${escapeHtml(mode)}</span>` : '';
                lines.push(`<div class="metric-line"><span class="metric-label">Match:</span> ${escapeHtml(similarity)}${modeHint}</div>`);
            }
            if (showDiagnostics && (clipSearch || dinoSearch)) {
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

    function archiveFrameRoleLabel(roleValue) {
        const role = String(roleValue || '').trim().toLowerCase();
        if (role === 'burst_apex') return 'burst apex';
        if (role === 'burst_companion') return 'sharper companion (burst)';
        return role ? role.replace(/_/g, ' ') : '';
    }

    function isBurstArchiveFrameRole(roleValue) {
        const role = String(roleValue || '').trim().toLowerCase();
        return role === 'burst_apex' || role === 'burst_companion';
    }

    function buildResultBadges(result) {
        if (!result || typeof result !== 'object') return '';
        const badges = [];
        if (result.is_detection) {
            const source = archiveLogicalSource(result.source);
            if (source === 'vlm_summary') {
                badges.push({ label: 'Video description', classes: 'mode-clip' });
            } else if (source === 'vlm_alert') {
                badges.push({ label: 'VLM alert', classes: 'warning' });
            } else if (source === 'probe') {
                badges.push({
                    label: canUseProbeDiagnostics() ? 'Secondary CLIP probe' : 'Archive evidence',
                    classes: '',
                });
            } else {
                badges.push({ label: 'Archive frame', classes: '' });
            }
        }

        const payload = archiveResultPayload(result);
        const anchorRole = String(payload.anchor_role || payload.anchor_source_role || '').trim();
        if (isBurstArchiveFrameRole(anchorRole)) {
            badges.push({ label: `⚡ ${archiveFrameRoleLabel(anchorRole)}`, classes: 'attention' });
        }

        const modeRaw = String(result.search_mode || '').trim().toLowerCase();
        if (modeRaw && canUseProbeDiagnostics()) {
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
            if (!item.archive_query) {
                item.archive_query = archiveLastQueryText;
            }
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

    function archiveLogicalSource(source) {
        const normalized = String(source || '').trim().toLowerCase();
        if (['probe', 'probes_run', 'probes_query', 'probe_daemon'].includes(normalized)) return 'probe';
        if (['vlm_summary', 'video_description', 'video_descriptions'].includes(normalized)) return 'vlm_summary';
        if (['vlm_alert', 'alert', 'alerts'].includes(normalized)) return 'vlm_alert';
        return normalized;
    }

    function archiveSourceLabel(source, fallbackLabel = '') {
        const normalized = archiveLogicalSource(source);
        if (normalized === 'probe') return canUseProbeDiagnostics() ? 'Secondary CLIP probe' : 'Archive evidence';
        if (normalized === 'vlm_summary') return 'Video description';
        if (normalized === 'vlm_alert') return 'VLM alert';
        if (fallbackLabel) return String(fallbackLabel);
        return normalized || 'Archive frame';
    }

    function archiveSourcePluralLabel(source) {
        const normalized = archiveLogicalSource(source);
        if (normalized === 'probe') return canUseProbeDiagnostics() ? 'CLIP probe hits' : 'archive evidence';
        if (normalized === 'vlm_summary') return 'video descriptions';
        if (normalized === 'vlm_alert') return 'VLM alerts';
        return 'archive items';
    }

    function archiveProbeFilterActive() {
        const source = archiveSourceFilter ? archiveSourceFilter.value.trim() : '';
        return archiveLogicalSource(source) === 'probe';
    }

    function syncArchiveProbeFilterVisibility() {
        const active = archiveProbeFilterActive();
        setElementHidden(archiveProbeFilterGroup, !active);
        if (archiveProbeFilter) {
            archiveProbeFilter.disabled = !active;
            if (!active) {
                archiveProbeFilter.value = '';
            }
        }
        return active;
    }

    function syncArchiveDiagnosticSourceVisibility() {
        if (!archiveSourceFilter) return;
        const canDiagnostics = canUseProbeDiagnostics();
        const probeOption = archiveSourceFilter.querySelector('option[value="probe"]');
        if (probeOption) {
            setElementHidden(probeOption, !canDiagnostics);
            probeOption.disabled = !canDiagnostics;
        }
        if (!canDiagnostics && archiveLogicalSource(archiveSourceFilter.value) === 'probe') {
            archiveSourceFilter.value = '';
        }
        syncArchiveProbeFilterVisibility();
    }

    function archiveResultPayload(result) {
        return result && typeof result.payload === 'object' && result.payload !== null
            ? result.payload
            : {};
    }

    function isVideoArchiveResult(result) {
        const source = archiveLogicalSource(result && result.source);
        return source === 'vlm_summary' || source === 'vlm_alert';
    }

    function archiveResultHasImage(result) {
        return Boolean(result && (String(result.path || '').trim() || String(result.thumbnail || '').trim()));
    }

    function archiveResultImageSrc(result) {
        if (!result) return '';
        const thumb = String(result.thumbnail || '').trim();
        if (thumb) {
            return thumb.startsWith('data:') ? thumb : `data:image/jpeg;base64,${thumb}`;
        }
        const path = String(result.path || '').trim();
        return path ? buildImageFetchUrl(path, result) : '';
    }

    function abortUiRequest(controller) {
        if (!controller) return;
        try {
            controller.abort();
        } catch (_) {
            // Generation checks still prevent stale DOM writes.
        }
    }

    function cancelArchiveEvidenceRequest() {
        archiveEvidenceRequestGeneration += 1;
        abortUiRequest(archiveEvidenceAbortController);
        archiveEvidenceAbortController = null;
        if (archiveEvidenceBusyButton) {
            setButtonBusy(archiveEvidenceBusyButton, false);
            archiveEvidenceBusyButton = null;
        }
    }

    function beginArchiveEvidenceRequest(busyButton = null) {
        cancelArchiveEvidenceRequest();
        if (archiveReviewModal && archiveReviewModal.style.display === 'block') {
            closeArchiveReviewModal();
        } else {
            cancelArchiveReviewRequest();
        }
        const controller = new AbortController();
        archiveEvidenceAbortController = controller;
        archiveEvidenceBusyButton = busyButton || null;
        if (archiveEvidenceBusyButton) setButtonBusy(archiveEvidenceBusyButton, true);
        return {
            generation: archiveEvidenceRequestGeneration,
            controller,
        };
    }

    function isCurrentArchiveEvidenceRequest(requestContext) {
        return Boolean(
            requestContext
            && requestContext.generation === archiveEvidenceRequestGeneration
            && archiveEvidenceAbortController === requestContext.controller
            && !requestContext.controller.signal.aborted
            && currentMode === 'archive'
        );
    }

    function finishArchiveEvidenceRequest(requestContext) {
        if (!isCurrentArchiveEvidenceRequest(requestContext)) return;
        archiveEvidenceAbortController = null;
        if (archiveEvidenceBusyButton) {
            setButtonBusy(archiveEvidenceBusyButton, false);
            archiveEvidenceBusyButton = null;
        }
    }

    function cancelArchiveFilterRequest() {
        archiveFilterRequestGeneration += 1;
        abortUiRequest(archiveFilterAbortController);
        archiveFilterAbortController = null;
    }

    function beginArchiveFilterRequest() {
        cancelArchiveFilterRequest();
        const controller = new AbortController();
        archiveFilterAbortController = controller;
        return {
            generation: archiveFilterRequestGeneration,
            controller,
        };
    }

    function isCurrentArchiveFilterRequest(requestContext) {
        return Boolean(
            requestContext
            && requestContext.generation === archiveFilterRequestGeneration
            && archiveFilterAbortController === requestContext.controller
            && !requestContext.controller.signal.aborted
            && currentMode === 'archive'
        );
    }

    function cancelArchiveReviewRequest() {
        archiveReviewRequestGeneration += 1;
        abortUiRequest(archiveReviewAbortController);
        archiveReviewAbortController = null;
    }

    function beginArchiveReviewRequest(context) {
        cancelArchiveReviewRequest();
        const controller = new AbortController();
        archiveReviewAbortController = controller;
        return {
            generation: archiveReviewRequestGeneration,
            controller,
            context,
        };
    }

    function isCurrentArchiveReviewRequest(requestContext) {
        return Boolean(
            requestContext
            && requestContext.generation === archiveReviewRequestGeneration
            && archiveReviewAbortController === requestContext.controller
            && !requestContext.controller.signal.aborted
            && archiveReviewContext === requestContext.context
            && currentMode === 'archive'
        );
    }

    function clearArchiveMediaVideo() {
        if (!archiveMediaVideo) return;
        archiveMediaVideo.onloadedmetadata = null;
        archiveMediaVideo.oncanplay = null;
        archiveMediaVideo.onplaying = null;
        archiveMediaVideo.onwaiting = null;
        archiveMediaVideo.onstalled = null;
        archiveMediaVideo.onended = null;
        archiveMediaVideo.onerror = null;
        try {
            archiveMediaVideo.pause();
        } catch (_) {
            // Best-effort media cleanup.
        }
        archiveMediaVideo.removeAttribute('src');
        try {
            archiveMediaVideo.load();
        } catch (_) {
            // Best-effort media cleanup.
        }
        if (archiveMediaObjectUrl) {
            URL.revokeObjectURL(archiveMediaObjectUrl);
            archiveMediaObjectUrl = null;
        }
        archiveMediaVideo.style.display = 'none';
    }

    function cancelArchiveMediaRequest(clearUi = false) {
        archiveMediaRequestGeneration += 1;
        abortUiRequest(archiveMediaAbortController);
        archiveMediaAbortController = null;
        if (archiveMediaLoadTimer) {
            clearTimeout(archiveMediaLoadTimer);
            archiveMediaLoadTimer = null;
        }
        if (archiveMediaLoopTimer) {
            clearTimeout(archiveMediaLoopTimer);
            archiveMediaLoopTimer = null;
        }
        clearArchiveMediaVideo();
        if (archiveReviewImg) {
            archiveReviewImg.onload = null;
            archiveReviewImg.onerror = null;
        }
        if (archiveMediaRetryBtn) archiveMediaRetryBtn.hidden = true;
        if (clearUi) {
            if (archiveMediaStatus) archiveMediaStatus.textContent = '';
            if (archiveReviewFrameContainer) delete archiveReviewFrameContainer.dataset.mediaState;
        }
    }

    function ensureArchiveMediaUi() {
        if (!archiveReviewFrameContainer) return {};
        if (!archiveMediaVideo || !archiveMediaVideo.isConnected) {
            archiveMediaVideo = document.createElement('video');
            archiveMediaVideo.autoplay = true;
            archiveMediaVideo.muted = true;
            archiveMediaVideo.controls = true;
            archiveMediaVideo.playsInline = true;
            archiveMediaVideo.preload = 'metadata';
            archiveMediaVideo.setAttribute('aria-label', 'Luxriot archive video playback');
            Object.assign(archiveMediaVideo.style, {
                width: '100%',
                height: '100%',
                objectFit: 'contain',
                background: '#000',
                display: 'none',
            });
            archiveReviewFrameContainer.insertBefore(archiveMediaVideo, archiveReviewFrameEmpty || null);
        }
        if (!archiveMediaStatus || !archiveMediaStatus.isConnected) {
            archiveMediaStatus = document.createElement('div');
            archiveMediaStatus.setAttribute('role', 'status');
            Object.assign(archiveMediaStatus.style, {
                position: 'absolute',
                left: '12px',
                bottom: '12px',
                zIndex: '6',
                padding: '6px 9px',
                borderRadius: '6px',
                background: 'rgba(7, 10, 14, 0.78)',
                color: '#d5dee8',
                fontSize: '12px',
            });
            archiveReviewFrameContainer.appendChild(archiveMediaStatus);
        }
        if (!archiveMediaRetryBtn || !archiveMediaRetryBtn.isConnected) {
            archiveMediaRetryBtn = document.createElement('button');
            archiveMediaRetryBtn.type = 'button';
            archiveMediaRetryBtn.className = 'feature-btn';
            archiveMediaRetryBtn.textContent = 'Retry playback';
            archiveMediaRetryBtn.hidden = true;
            archiveMediaRetryBtn.addEventListener('click', () => {
                const context = archiveReviewContext;
                if (context && context.result) startArchiveMediaPlayback(context.result, context, true);
            });
            Object.assign(archiveMediaRetryBtn.style, {
                position: 'absolute',
                right: '12px',
                bottom: '12px',
                zIndex: '7',
            });
            archiveReviewFrameContainer.appendChild(archiveMediaRetryBtn);
        }
        return { video: archiveMediaVideo, status: archiveMediaStatus, retry: archiveMediaRetryBtn };
    }

    function setArchiveMediaState(state, detail) {
        const normalized = ['loading', 'playing', 'degraded', 'error'].includes(state) ? state : 'error';
        const ui = ensureArchiveMediaUi();
        if (archiveReviewFrameContainer) archiveReviewFrameContainer.dataset.mediaState = normalized;
        if (ui.status) {
            ui.status.hidden = false;
            ui.status.textContent = detail || ({
                loading: 'Loading archive video…',
                playing: 'Archive video playing',
                degraded: 'Static archive frame — not video',
                error: 'Archive video unavailable',
            }[normalized]);
            ui.status.style.color = normalized === 'error'
                ? '#ff8da1'
                : normalized === 'degraded'
                    ? '#ffd27a'
                    : '#d5dee8';
        }
        if (ui.retry) {
            ui.retry.textContent = 'Retry playback';
            ui.retry.hidden = !['degraded', 'error'].includes(normalized);
        }
    }

    function archiveStaticFallbackUrl(result, timeMsOverride = null) {
        const channelId = Number(result && result.channel_id);
        const requestedOverride = Number(timeMsOverride);
        const timeMs = Number.isSafeInteger(requestedOverride) && requestedOverride > 0
            ? requestedOverride
            : Number(archiveFrameTimestampMs(result));
        if (!Number.isFinite(channelId) || channelId <= 0 || !Number.isFinite(timeMs) || timeMs <= 0) return '';
        const params = new URLSearchParams({
            time_ms: String(Math.trunc(timeMs)),
            stream: 'mainStream',
        });
        return `/luxriot/archive_snapshot/${encodeURIComponent(String(channelId))}?${params.toString()}`;
    }

    function isCurrentArchiveMediaRequest(requestContext) {
        return Boolean(
            requestContext
            && requestContext.generation === archiveMediaRequestGeneration
            && archiveMediaAbortController === requestContext.controller
            && archiveReviewContext === requestContext.context
            && archiveReviewContext?.result === requestContext.result
            && archiveReviewFrameIdentity(requestContext.result) === requestContext.identity
            && currentMode === 'archive'
            && archiveReviewModal?.style.display === 'block'
        );
    }

    function showArchiveStaticFallback(requestContext, reason) {
        if (!isCurrentArchiveMediaRequest(requestContext) || !archiveReviewImg) return;
        clearArchiveMediaVideo();
        const storedImage = archiveResultImageSrc(requestContext.result);
        const recorderSnapshotUrl = archiveStaticFallbackUrl(requestContext.result, requestContext.timeMs);
        let triedRecorderSnapshot = false;
        const markDegraded = (detail) => {
            if (!isCurrentArchiveMediaRequest(requestContext)) return;
            archiveReviewImg.classList.remove('is-hidden');
            if (archiveReviewFrameEmpty) archiveReviewFrameEmpty.classList.add('is-hidden');
            setArchiveMediaState('degraded', detail);
        };
        const markUnavailable = () => {
            if (!isCurrentArchiveMediaRequest(requestContext)) return;
            archiveReviewImg.classList.add('is-hidden');
            if (archiveReviewFrameEmpty) archiveReviewFrameEmpty.classList.remove('is-hidden');
            setArchiveMediaState('error', 'Archive video and fallback frames are unavailable.');
        };
        const storedEvidenceDetail = `${reason || 'Archive video is unavailable.'} Stored evidence frame shown — not video.`;
        const showRecorderSnapshot = () => {
            if (!isCurrentArchiveMediaRequest(requestContext)) return;
            if (triedRecorderSnapshot || !recorderSnapshotUrl) {
                markUnavailable();
                return;
            }
            triedRecorderSnapshot = true;
            archiveReviewImg.onload = () => markDegraded(
                `${reason || 'Archive video is unavailable.'} Static recorder snapshot — not video.`
            );
            archiveReviewImg.onerror = markUnavailable;
            archiveReviewImg.src = `${recorderSnapshotUrl}&request=${Date.now()}`;
        };
        // The stored frame is the exact evidence that fed VLM/CLIP/archive; a
        // recorder snapshot resolved at a shifted time must never replace it.
        if (storedImage) {
            if (archiveReviewImg.getAttribute('src') === storedImage) {
                markDegraded(storedEvidenceDetail);
                return;
            }
            archiveReviewImg.onload = () => markDegraded(storedEvidenceDetail);
            archiveReviewImg.onerror = showRecorderSnapshot;
            archiveReviewImg.src = storedImage;
            return;
        }
        showRecorderSnapshot();
    }

    function archivePlaybackWindow(result) {
        const payload = archiveResultPayload(result);
        const frameTimeMs = Number(archiveFrameTimestampMs(result));
        const rawStartMs = Number(payload.batch_start_ms ?? result?.batch_start_ms);
        const rawEndMs = Number(payload.batch_end_ms ?? result?.batch_end_ms);
        const hasBatchWindow = Number.isFinite(rawStartMs) && rawStartMs > 0
            && Number.isFinite(rawEndMs) && rawEndMs > 0;
        const startMs = hasBatchWindow
            ? Math.trunc(Math.min(rawStartMs, rawEndMs))
            : Math.trunc(frameTimeMs);
        const batchSpanMs = hasBatchWindow ? Math.abs(rawEndMs - rawStartMs) : 0;
        // Batch timestamps describe sampled instants, so include the final
        // second instead of clipping the last evidence frame from playback.
        const durationSec = hasBatchWindow
            ? Math.max(1, Math.min(15, Math.ceil(batchSpanMs / 1000) + 1))
            : 15;
        return {
            startMs,
            endMs: startMs + durationSec * 1000,
            durationSec,
            hasBatchWindow,
        };
    }

    function startArchiveMediaPlayback(result, context, force = false) {
        if (!result || !context || archiveReviewContext !== context) return;
        const identity = archiveReviewFrameIdentity(result);
        if (!force && context.mediaIdentity === identity && ['loading', 'playing', 'degraded'].includes(context.mediaState || '')) return;
        cancelArchiveMediaRequest(false);
        context.mediaIdentity = identity;
        context.mediaState = 'loading';
        const channelId = Number(result.channel_id);
        const playbackWindow = archivePlaybackWindow(result);
        const timeMs = Number(playbackWindow.startMs);
        const durationSec = Number(playbackWindow.durationSec);
        if (!isVideoArchiveResult(result) || !Number.isFinite(channelId) || channelId <= 0 || !Number.isFinite(timeMs) || timeMs <= 0) {
            context.mediaState = 'degraded';
            setArchiveMediaState('degraded', 'This result has a static evidence frame, not playable archive video.');
            return;
        }
        const controller = new AbortController();
        archiveMediaAbortController = controller;
        const requestContext = {
            generation: archiveMediaRequestGeneration,
            controller,
            context,
            result,
            identity,
            timeMs,
            durationSec,
        };
        const mediaUrl = luxriotMediaBrokerUrl('archive', channelId, {
            stream: 'mainStream',
            timeMs,
            durationSec,
        });
        setArchiveMediaState('loading', `Preparing ${durationSec}s batch loop…`);
        const negotiationTimeoutMs = Math.max(30000, Math.min(60000, durationSec * 3000 + 15000));
        void fetchLuxriotMediaBlob(mediaUrl, controller, negotiationTimeoutMs)
            .then((negotiated) => {
                if (!isCurrentArchiveMediaRequest(requestContext)) return;
                context.mediaSegmentTimeMs = timeMs;
                context.mediaLastSampleTimestampMs = negotiated.lastSampleTimestampMs;
                context.mediaDurationSec = Number(negotiated.durationSec) || durationSec;
                const capabilityDetail = [];
                if (negotiated.resolvedTimeMs && negotiated.resolvedTimeMs !== timeMs) {
                    capabilityDetail.push(`aligned to ${new Date(negotiated.resolvedTimeMs).toLocaleTimeString()}`);
                } else if (negotiated.frameAlignment === 'next_frame_time_unavailable') {
                    capabilityDetail.push('exact next-frame seek is unavailable on this Evo variant');
                }
                if (negotiated.html5Compatibility === 'unsupported_fallback') {
                    capabilityDetail.push('Evo rejected html5compatible; using its legacy stream response');
                }
                capabilityDetail.unshift(`${context.mediaDurationSec}s batch loop`);
                const capabilitySuffix = capabilityDetail.length ? ` (${capabilityDetail.join('; ')})` : '';
                archiveMediaObjectUrl = URL.createObjectURL(negotiated.blob);
                const failToStatic = (reason) => {
                    if (archiveMediaLoadTimer) {
                        clearTimeout(archiveMediaLoadTimer);
                        archiveMediaLoadTimer = null;
                    }
                    context.mediaState = 'degraded';
                    showArchiveStaticFallback(requestContext, reason);
                };
                archiveMediaLoadTimer = window.setTimeout(
                    () => failToStatic('Archive playback load timed out.'),
                    12000,
                );
                if (negotiated.mediaKind === 'mjpeg') {
                    archiveReviewImg.onload = () => {
                        if (!isCurrentArchiveMediaRequest(requestContext)) return;
                        clearTimeout(archiveMediaLoadTimer);
                        archiveMediaLoadTimer = null;
                        context.mediaState = 'playing';
                        archiveReviewImg.classList.remove('is-hidden');
                        if (archiveReviewFrameEmpty) archiveReviewFrameEmpty.classList.add('is-hidden');
                        setArchiveMediaState('playing', `Archive MJPEG playing${capabilitySuffix}`);
                        if (!archiveMediaLoopTimer) {
                            archiveMediaLoopTimer = window.setTimeout(() => {
                                archiveMediaLoopTimer = null;
                                if (!isCurrentArchiveMediaRequest(requestContext)) return;
                                startArchiveMediaPlayback(result, context, true);
                            }, Math.max(1000, context.mediaDurationSec * 1000 + 500));
                        }
                    };
                    archiveReviewImg.onerror = () => failToStatic('The archive MJPEG stream could not be decoded.');
                    archiveReviewImg.src = archiveMediaObjectUrl;
                    return;
                }
                const ui = ensureArchiveMediaUi();
                const video = ui.video;
                if (!video) {
                    failToStatic('The browser archive video element could not be initialized.');
                    return;
                }
                archiveReviewImg.classList.add('is-hidden');
                if (archiveReviewFrameEmpty) archiveReviewFrameEmpty.classList.add('is-hidden');
                video.style.display = 'block';
                video.loop = true;
                const markPlayable = (detail) => {
                    if (!isCurrentArchiveMediaRequest(requestContext)) return;
                    clearTimeout(archiveMediaLoadTimer);
                    archiveMediaLoadTimer = null;
                    context.mediaState = 'playing';
                    setArchiveMediaState('playing', detail);
                };
                video.onloadedmetadata = () => {
                    if (!isCurrentArchiveMediaRequest(requestContext)) return;
                    setArchiveMediaState('loading', `Archive video metadata loaded…${capabilitySuffix}`);
                };
                video.oncanplay = () => {
                    if (!isCurrentArchiveMediaRequest(requestContext)) return;
                    markPlayable(`Archive video ready${capabilitySuffix}`);
                    const playPromise = video.play();
                    if (playPromise && typeof playPromise.catch === 'function') playPromise.catch(() => {});
                };
                video.onplaying = () => markPlayable(`Archive video playing${capabilitySuffix}`);
                video.onwaiting = () => {
                    if (!isCurrentArchiveMediaRequest(requestContext)) return;
                    setArchiveMediaState('loading', 'Archive video buffering…');
                };
                video.onstalled = () => {
                    if (!isCurrentArchiveMediaRequest(requestContext)) return;
                    setArchiveMediaState('loading', 'Archive video transport stalled…');
                };
                video.onerror = () => {
                    if (!isCurrentArchiveMediaRequest(requestContext)) return;
                    failToStatic('The browser rejected the archive video container or codec.');
                };
                video.onended = () => {
                    if (!isCurrentArchiveMediaRequest(requestContext)) return;
                    video.currentTime = 0;
                    const playPromise = video.play();
                    if (playPromise && typeof playPromise.catch === 'function') playPromise.catch(() => {});
                };
                video.src = archiveMediaObjectUrl;
                video.load();
            })
            .catch((error) => {
                if (!isCurrentArchiveMediaRequest(requestContext)) return;
                context.mediaState = 'degraded';
                showArchiveStaticFallback(
                    requestContext,
                    controller.signal.aborted ? 'Archive batch preparation timed out.' : (error.message || 'Archive video is unavailable.'),
                );
            });
    }

    function invalidateArchiveResultContext() {
        cancelArchiveEvidenceRequest();
        if (archiveReviewModal && archiveReviewModal.style.display === 'block') {
            closeArchiveReviewModal();
        } else {
            cancelArchiveReviewRequest();
        }
    }

    function base64ToBlob(base64, mimeType = 'image/jpeg') {
        const clean = stripBase64Payload(base64);
        if (!clean) return null;
        const binary = atob(clean);
        const len = binary.length;
        const bytes = new Uint8Array(len);
        for (let idx = 0; idx < len; idx += 1) {
            bytes[idx] = binary.charCodeAt(idx);
        }
        return new Blob([bytes], { type: mimeType });
    }

    async function archiveResultImageBlob(result, signal = null) {
        if (!result) throw new Error('No archive result selected.');
        const thumb = String(result.thumbnail || '').trim();
        if (thumb) {
            const blob = base64ToBlob(thumb, 'image/jpeg');
            if (!blob) throw new Error('Archived frame thumbnail is empty.');
            return blob;
        }
        const path = String(result.path || '').trim();
        if (!path) throw new Error('No image is available for this archive result.');
        const imageResponse = await fetch(buildImageFetchUrl(path, result), { signal });
        if (!imageResponse.ok) throw new Error('Failed to load archived image.');
        return imageResponse.blob();
    }

    function archiveFrameTimestampMs(result) {
        const payload = archiveResultPayload(result);
        const candidates = [
            payload.frame_timestamp_ms,
            payload.anchor_frame_timestamp_ms,
            result && result.timestamp_ms,
            payload.batch_start_ms,
        ];
        for (const value of candidates) {
            const numeric = Number(value);
            if (Number.isFinite(numeric) && numeric > 0) return numeric;
        }
        return null;
    }

    function archiveResultCanOpenVlmFeed(result) {
        const channelId = Number(result && result.channel_id);
        const timestampMs = Number(archiveFrameTimestampMs(result));
        return Number.isFinite(channelId) && channelId > 0
            && Number.isFinite(timestampMs) && timestampMs > 0;
    }

    function archiveResultSummaryWindow(result) {
        const payload = archiveResultPayload(result);
        const targetMs = Number(archiveFrameTimestampMs(result));
        const batchStartMs = Number(payload.batch_start_ms ?? result?.batch_start_ms);
        const batchEndMs = Number(payload.batch_end_ms ?? result?.batch_end_ms);
        const fallbackStartMs = Number.isFinite(targetMs) && targetMs > 0 ? targetMs : Date.now();
        const baseStartMs = Number.isFinite(batchStartMs) && batchStartMs > 0 ? batchStartMs : fallbackStartMs;
        const baseEndMs = Number.isFinite(batchEndMs) && batchEndMs > 0 ? batchEndMs : baseStartMs;
        const startMs = Math.max(0, Math.min(baseStartMs, baseEndMs) - 60000);
        const endMs = Math.max(startMs + 1000, Math.max(baseStartMs, baseEndMs) + 60000);
        return {
            targetMs: Number.isFinite(targetMs) && targetMs > 0 ? targetMs : baseStartMs,
            startMs,
            endMs,
        };
    }

    function formatArchiveTimestamp(ms) {
        const numeric = Number(ms);
        if (!Number.isFinite(numeric) || numeric <= 0) return 'n/a';
        return new Date(numeric).toLocaleString();
    }

    function archiveChannelLabel(result) {
        const channelId = Number(result && result.channel_id);
        const channelText = Number.isFinite(channelId) ? `#${channelId}` : '#?';
        const name = Number.isFinite(channelId)
            ? (luxriotChannelNameById[String(channelId)] || '')
            : '';
        return name ? `${channelText} | ${name}` : channelText;
    }

    function archiveReviewMatchText(result) {
        const source = archiveSourceLabel(result && result.source, result && result.source_label);
        const clip = Number(result && (result.clip_similarity ?? result.similarity));
        const dino = Number(result && result.dino_similarity);
        const pos = Number(result && result.pos_score);
        const neg = Number(result && result.neg_score);
        const margin = Number(result && result.margin);
        const parts = [];
        if (Number.isFinite(clip)) parts.push(`CLIP ${formatPercent(clip)}`);
        if (Number.isFinite(dino) && dino > 0) parts.push(`DINO ${formatPercent(dino)}`);
        const hasMeaningfulProbeScores = !isVideoArchiveResult(result)
            || [pos, neg, margin].some((value) => Number.isFinite(value) && Math.abs(value) > 0.000001);
        if (hasMeaningfulProbeScores && (Number.isFinite(pos) || Number.isFinite(neg) || Number.isFinite(margin))) {
            parts.push(`P/N/M ${formatPercent(pos || 0)}/${formatPercent(neg || 0)}/${formatPercent(margin || 0)}`);
        }
        const payload = archiveResultPayload(result);
        const role = String(payload.anchor_role || payload.anchor_source_role || '').trim();
        parts.push(`Source: ${role ? `${source}/${role}` : source}`);
        return parts.join(' · ');
    }

    function archiveReviewQueryText(result) {
        const query = String(result?.archive_query || archiveLastQueryText || '').trim();
        return query || 'Archive result';
    }

    function archiveSummaryText(result) {
        const payload = archiveResultPayload(result);
        const text = String(payload.summary || result?.summary || '').trim();
        if (text) return text;
        if (isVideoArchiveResult(result)) return 'No L0 summary excerpt is stored for this frame.';
        return 'No summary is associated with this archive result.';
    }

    function archiveFrameRoleText(result) {
        const payload = archiveResultPayload(result);
        const role = String(payload.anchor_role || payload.anchor_source_role || '').trim();
        const frameIndex = Number(payload.frame_index ?? payload.anchor_frame_index);
        const parts = [];
        const roleLabel = archiveFrameRoleLabel(role);
        if (roleLabel) parts.push(roleLabel);
        if (Number.isFinite(frameIndex)) parts.push(`frame ${frameIndex}`);
        return parts.length ? parts.join(' · ') : 'Frame';
    }

    function applySelectOptions(selectEl, options, selected = '') {
        if (!selectEl) return;
        const before = String(selectEl.value || '');
        const previous = selected || selectEl.value || '';
        selectEl.innerHTML = options.map((opt) => `<option value="${escapeHtml(String(opt.value))}">${escapeHtml(String(opt.label))}</option>`).join('');
        const hasPrevious = options.some((opt) => String(opt.value) === String(previous));
        selectEl.value = hasPrevious ? String(previous) : String(options[0]?.value || '');
        if (String(selectEl.value || '') !== before) {
            invalidateArchiveResultContext();
        }
    }

    async function refreshArchiveChannelFilter(requestContext = null) {
        if (!archiveChannelFilter) return;
        const ownedRequest = !requestContext;
        const context = requestContext || beginArchiveFilterRequest();
        try {
            const response = await fetch('/luxriot/channels', {
                signal: context.controller.signal,
            });
            const data = await parseApiJson(response, 'Failed to load channels');
            if (!isCurrentArchiveFilterRequest(context)) return false;
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
            return true;
        } catch (error) {
            if ((error && error.name === 'AbortError') || !isCurrentArchiveFilterRequest(context)) return false;
            applySelectOptions(archiveChannelFilter, [{ value: '', label: 'All streams' }], '');
            return false;
        } finally {
            if (ownedRequest && archiveFilterAbortController === context.controller) {
                archiveFilterAbortController = null;
            }
        }
    }

    async function refreshArchiveProbeFilter(requestContext = null) {
        if (!archiveProbeFilter) return;
        const ownedRequest = !requestContext;
        const context = requestContext || beginArchiveFilterRequest();
        if (!syncArchiveProbeFilterVisibility()) {
            applySelectOptions(archiveProbeFilter, [{ value: '', label: 'Probe filter available for CLIP probe hits' }], '');
            if (ownedRequest && archiveFilterAbortController === context.controller) {
                archiveFilterAbortController = null;
            }
            return true;
        }
        try {
            const params = new URLSearchParams({ hours: '168', limit: '300' });
            applyArchiveTimeFilters(params);
            const channelId = archiveChannelFilter ? archiveChannelFilter.value.trim() : '';
            const source = archiveSourceFilter ? archiveSourceFilter.value.trim() : '';
            if (channelId) {
                params.set('channel_id', channelId);
            }
            if (source) {
                params.set('source', source);
            }
            const response = await fetch(`/detections/summary?${params.toString()}`, {
                signal: context.controller.signal,
            });
            const data = await parseApiJson(response, 'Failed to load archive items');
            if (!isCurrentArchiveFilterRequest(context)) return false;
            const summary = Array.isArray(data.summary) ? data.summary : [];
            const options = [{ value: '', label: `All ${archiveSourcePluralLabel(source)}` }];
            summary.forEach((item) => {
                const id = String(item.probe_id || '').trim();
                if (!id) return;
                const labelBase = item.probe_name ? String(item.probe_name) : id;
                const label = `${labelBase} (${item.hit_count || 0})`;
                options.push({ value: id, label });
            });
            applySelectOptions(archiveProbeFilter, options, archiveProbeFilter.value);
            return true;
        } catch (error) {
            if ((error && error.name === 'AbortError') || !isCurrentArchiveFilterRequest(context)) return false;
            applySelectOptions(archiveProbeFilter, [{ value: '', label: 'All CLIP probes' }], '');
            return false;
        } finally {
            if (ownedRequest && archiveFilterAbortController === context.controller) {
                archiveFilterAbortController = null;
            }
        }
    }

    async function refreshArchiveFilters() {
        const context = beginArchiveFilterRequest();
        archiveDetectionsOffset = 0;
        archiveDetectionsHasMore = false;
        updateArchiveDetectionsNav();
        try {
            await refreshArchiveChannelFilter(context);
            if (!isCurrentArchiveFilterRequest(context)) return false;
            await refreshArchiveProbeFilter(context);
            return isCurrentArchiveFilterRequest(context);
        } finally {
            if (archiveFilterAbortController === context.controller) {
                archiveFilterAbortController = null;
            }
        }
    }

    function normalizeDetectionResults(detections) {
        return (detections || []).map((det, idx) => {
            const ts = Number.isFinite(det?.timestamp_ms) ? det.timestamp_ms : null;
            const source = archiveLogicalSource(det?.source || det?.payload?.source || '');
            const payload = det?.payload || null;
            const channelId = det?.channel_id;
            const probeLabel = det?.probe_name || det?.probe_id || 'probe';
            const frameRole = payload && typeof payload === 'object'
                ? String(payload.anchor_role || payload.anchor_source_role || '').trim()
                : '';
            const frameIndex = payload && typeof payload === 'object'
                ? Number(payload.frame_index ?? payload.anchor_frame_index)
                : NaN;
            let filename = String(probeLabel);
            if (source === 'vlm_summary' || source === 'vlm_alert') {
                const timeLabel = ts ? formatArchiveTimestamp(ts) : 'n/a';
                const roleLabel = frameRole ? ` · ${frameRole.replace(/_/g, ' ')}` : '';
                const indexLabel = Number.isFinite(frameIndex) ? ` · frame ${frameIndex}` : '';
                filename = `${archiveSourceLabel(source)} ch #${channelId ?? '?'}${indexLabel}${roleLabel} · ${timeLabel}`;
            }
            return {
                filename,
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
                source_label: det?.source_label || '',
                archive_item_type: det?.archive_item_type || '',
                origin: det?.origin || det?.payload?.origin || det?.payload?.source || '',
                payload,
                archive_query: archiveLastQueryText,
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
        const source = archiveSourceFilter ? archiveSourceFilter.value.trim() : '';
        const probeId = archiveProbeFilterActive() && archiveProbeFilter ? archiveProbeFilter.value.trim() : '';
        const limitRaw = archiveDetectionsLimit ? archiveDetectionsLimit.value : '24';
        const params = new URLSearchParams();
        archiveLastQueryText = 'Loaded archive frames';
        cancelArchiveEvidenceRequest();
        let timeWindow;
        try {
            timeWindow = applyArchiveTimeFilters(params);
        } catch (error) {
            resultsContainer.innerHTML = `<div class="loading">Error: ${escapeHtml(error.message)}</div>`;
            setArchiveDetectionsMeta(error.message, true);
            renderArchiveInspectorEmpty(error.message);
            updateArchiveDetectionsNav();
            return;
        }
        if (channelId) params.set('channel_id', channelId);
        if (probeId) params.set('probe_id', probeId);
        if (source) params.set('source', source);
        const limit = Number.parseInt(limitRaw, 10);
        params.set('limit', String(Number.isFinite(limit) ? limit : 24));
        const requestedOffset = Math.max(0, archiveDetectionsOffset);
        params.set('offset', String(requestedOffset));

        const requestContext = beginArchiveEvidenceRequest(loadDetectionsBtn);
        resultsContainer.innerHTML = '<div class="loading"><div class="spinner"></div> Loading frame archive...</div>';
        setArchiveDetectionsMeta('Loading archive frames...');
        renderArchiveInspectorEmpty('Loading frame archive...');
        try {
            const response = await fetch(`/detections/list?${params.toString()}`, {
                signal: requestContext.controller.signal,
            });
            const data = await parseApiJson(response, 'Failed to load frame archive');
            if (!isCurrentArchiveEvidenceRequest(requestContext)) return false;
            const detections = Array.isArray(data.detections) ? data.detections : [];
            archiveDetectionsTotal = Number.isFinite(data.total) ? data.total : detections.length;
            archiveDetectionsHasMore = Boolean(data.has_more);
            const mapped = normalizeDetectionResults(detections);
            if (!mapped.length) {
                archiveRenderedResults = [];
                refreshArchiveScoreScale(archiveRenderedResults);
                applyArchiveScoreThreshold();
                resultsContainer.innerHTML = '<div class="loading">No archive frames found for selected filters.</div>';
                setArchiveDetectionsMeta('No archive frames found for selected filters.');
                renderArchiveInspectorEmpty('No archive frames found for the selected filters.');
                updateArchiveDetectionsNav();
                return;
            }
            displayResults(mapped);
            const shownFrom = requestedOffset + 1;
            const shownTo = requestedOffset + mapped.length;
            const windowNote = timeWindow.absolute ? ' in selected time window' : '';
            setArchiveDetectionsMeta(`Showing archive frames ${shownFrom}-${shownTo} of ${archiveDetectionsTotal}${windowNote}.`);
            updateArchiveDetectionsNav();
        } catch (err) {
            if ((err && err.name === 'AbortError') || !isCurrentArchiveEvidenceRequest(requestContext)) return false;
            const payload = err && err.payload ? err.payload : {};
            const message = payload.not_ready === 'archive_store'
                ? `Archive storage is not migrated yet. Apply database migration ${payload.required_revision || '20260612_0005'} and reload.`
                : (err.message || String(err));
            resultsContainer.innerHTML = `<div class="loading">Error: ${escapeHtml(message)}</div>`;
            setArchiveDetectionsMeta(`Error loading archive frames: ${message}`, true);
            renderArchiveInspectorEmpty(`Frame archive error: ${message}`);
            archiveDetectionsHasMore = false;
            updateArchiveDetectionsNav();
            return false;
        } finally {
            finishArchiveEvidenceRequest(requestContext);
        }
    }

    function buildDetectionSearchFilters() {
        const payload = {};
        const channelId = archiveChannelFilter ? archiveChannelFilter.value.trim() : '';
        const source = archiveSourceFilter ? archiveSourceFilter.value.trim() : '';
        const probeId = archiveProbeFilterActive() && archiveProbeFilter ? archiveProbeFilter.value.trim() : '';
        if (channelId) payload.channel_id = channelId;
        if (probeId) payload.probe_id = probeId;
        if (source) payload.source = source;
        applyArchiveTimeFilters(payload);
        return payload;
    }

    function isDetectionsScope() {
        return !searchScopeSelect || searchScopeSelect.value === 'detections';
    }

    function updateSearchScopeUI() {
        if (searchScopeSelect && searchScopeSelect.value !== 'detections') {
            searchScopeSelect.value = 'detections';
        }
        if (isDetectionsScope()) {
            if (searchInput) {
                searchInput.placeholder = 'Describe archived scene (filtered by stream/source/time)...';
            }
            setArchiveDetectionsMeta('Archive active: text/image search runs over video descriptions, VLM alerts, and evidence frames.');
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
                    syncUiAccess();
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
    syncUiAccess();

    // Settings modal functionality
    settingsNavButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
            if (btn.classList.contains('is-hidden')) return;
            const targetId = btn.dataset.settingsTarget;
            if (!targetId) return;
            scrollSettingsSectionIntoView(targetId);
        });
    });

    if (settingsBtn) settingsBtn.addEventListener('click', () => {
        if (!syncSettingsAccess()) return;
        settingsModal.style.display = 'block';
        if (settingsStatus) {
            settingsStatus.textContent = '';
            settingsStatus.className = 'settings-status';
            settingsStatus.style.display = 'none';
        }
        if (userHasAnyPermission(['settings:view', 'settings:manage'])) {
            loadSettings();
        }
        if (userHasPermission('settings:manage')) {
            loadEnvEditor();
        }
        if (syncAdminUsersAccess()) {
            loadAdminConsole();
        }
        if (syncAuditAccess()) {
            loadAuditEvents();
        }
        if (settingsScrollArea) {
            settingsScrollArea.scrollTop = 0;
        }
        const firstTarget = firstVisibleSettingsTarget();
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
    if (closeArchiveReviewBtn) {
        closeArchiveReviewBtn.addEventListener('click', () => {
            closeArchiveReviewModal();
        });
    }
    if (archiveReviewModal) {
        archiveReviewModal.addEventListener('click', (e) => {
            if (e.target === archiveReviewModal) {
                closeArchiveReviewModal();
            }
        });
    }
    if (archiveReviewPrevFrameBtn) {
        archiveReviewPrevFrameBtn.addEventListener('click', () => {
            archiveReviewStepFrame(-1);
        });
    }
    if (archiveReviewNextFrameBtn) {
        archiveReviewNextFrameBtn.addEventListener('click', () => {
            archiveReviewStepFrame(1);
        });
    }
    if (archiveReviewFilmstrip) {
        archiveReviewFilmstrip.addEventListener('click', (e) => {
            const btn = e.target.closest('[data-archive-review-frame-index]');
            if (!btn) return;
            const frameIndex = Number.parseInt(btn.dataset.archiveReviewFrameIndex || '-1', 10);
            if (Number.isFinite(frameIndex)) {
                archiveReviewSetFrame(frameIndex);
            }
        });
    }
    if (archiveReviewDescribeBtn) {
        archiveReviewDescribeBtn.addEventListener('click', () => {
            const context = archiveReviewContext;
            if (!context) return;
            closeArchiveReviewModal();
            describeImageWithLM(context.index, context.result.path || '', null, context.result);
        });
    }
    if (archiveReviewSimilarBtn) {
        archiveReviewSimilarBtn.addEventListener('click', () => {
            const context = archiveReviewContext;
            if (!context) return;
            closeArchiveReviewModal();
            findSimilarImages(context.result.path || '', context.result);
        });
    }
    if (archiveReviewJumpBtn) {
        archiveReviewJumpBtn.addEventListener('click', () => {
            const context = archiveReviewContext;
            if (!context) return;
            void jumpToVideoSummaryFromArchive(context.result);
        });
    }
    if (archiveReviewCopyBtn) {
        archiveReviewCopyBtn.addEventListener('click', () => {
            copyArchiveReviewSummary();
        });
    }
    
    // Close modal when clicking outside
    settingsModal.addEventListener('click', (e) => {
        if (e.target === settingsModal) {
            settingsModal.style.display = 'none';
        }
    });

    document.addEventListener('keydown', (e) => {
        if (archiveReviewModal && archiveReviewModal.style.display === 'block') {
            if (e.key === 'ArrowLeft') {
                e.preventDefault();
                archiveReviewStepFrame(-1);
                return;
            }
            if (e.key === 'ArrowRight') {
                e.preventDefault();
                archiveReviewStepFrame(1);
                return;
            }
        }
        if (e.key !== 'Escape') return;
        if (imageLightboxModal && imageLightboxModal.style.display === 'block') {
            closeImageLightbox();
            return;
        }
        if (archiveReviewModal && archiveReviewModal.style.display === 'block') {
            closeArchiveReviewModal();
            return;
        }
        if (probeSnapModal && probeSnapModal.style.display === 'block') {
            setProbeSnapModalVisibility(false);
            return;
        }
        if (probeCastModal && probeCastModal.style.display === 'block') {
            setProbeCastModalVisibility(false);
            return;
        }
        if (probeEditorModal && probeEditorModal.style.display === 'block') {
            setProbeEditorModalVisibility(false);
            return;
        }
        if (luxriotPromptModal && luxriotPromptModal.style.display === 'block') {
            closeLuxriotPromptModal();
            return;
        }
        if (agentSkillModal && agentSkillModal.style.display === 'block') {
            agentSkillModal.style.display = 'none';
            return;
        }
        if (settingsModal && settingsModal.style.display === 'block') {
            settingsModal.style.display = 'none';
        }
    });

    window.addEventListener('pagehide', () => {
        stopLuxriotPreview(true);
        stopLuxriotSummaryPoll();
        stopProbePreview();
        stopProbeRunLoop();
        stopProbeStatusPoll();
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
    if (adminAllowedChannelsInput) {
        adminAllowedChannelsInput.addEventListener('change', () => {
            syncAdminChannelPickerFromText();
        });
    }
    if (adminChannelList) {
        adminChannelList.addEventListener('change', (event) => {
            const checkbox = event.target.closest('input[type="checkbox"]');
            if (!checkbox) return;
            syncAdminChannelTextFromPicker();
        });
    }
    if (adminChannelsAllBtn) {
        adminChannelsAllBtn.addEventListener('click', () => {
            if (adminAllowedChannelsInput) adminAllowedChannelsInput.value = '*';
            renderAdminChannelPicker(['*']);
        });
    }
    if (adminChannelsNoneBtn) {
        adminChannelsNoneBtn.addEventListener('click', () => {
            if (adminAllowedChannelsInput) adminAllowedChannelsInput.value = '';
            renderAdminChannelPicker([]);
        });
    }
    if (adminChannelsRefreshBtn) {
        adminChannelsRefreshBtn.addEventListener('click', async () => {
            setButtonBusy(adminChannelsRefreshBtn, true);
            try {
                await loadAdminChannelCatalog(true);
            } finally {
                setButtonBusy(adminChannelsRefreshBtn, false);
            }
        });
    }
    if (adminUserSaveBtn) {
        adminUserSaveBtn.addEventListener('click', () => {
            saveAdminUser();
        });
    }
    if (adminUserResetPasswordBtn) {
        adminUserResetPasswordBtn.addEventListener('click', () => {
            resetSelectedAdminUserPassword();
        });
    }
    if (adminUserToggleActiveBtn) {
        adminUserToggleActiveBtn.addEventListener('click', () => {
            toggleSelectedAdminUserActive();
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
            void openProbeSnapModalFromPreview();
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

    [
        luxriotBatchSizeSelect,
        luxriotSnapshotIntervalInput,
        luxriotSummaryRetentionDaysInput,
        luxriotSummaryHistoryLimitInput,
        archiveRetentionEnabledInput,
        archiveRowRetentionDaysInput,
        archiveThumbnailRetentionDaysInput,
        archiveMaxRecordsInput,
        archiveEstimateChannelsInput,
        archiveEstimateFramesPerBatchInput,
        archiveEstimateAvgJpegKbInput,
        archiveEstimateProbeRowsInput
    ].forEach((control) => {
        if (!control) return;
        control.addEventListener('input', scheduleArchiveCapacityEstimate);
        control.addEventListener('change', scheduleArchiveCapacityEstimate);
    });

    async function loadEnvEditor() {
        if (!envEditorInput) return;
        try {
            const response = await fetch('/settings/env');
            const data = await response.json();
            if (data.success) {
                envEditorInput.value = String(data.envText || '');
                const precedence = data.precedence && typeof data.precedence === 'object'
                    ? data.precedence
                    : null;
                const different = precedence && Array.isArray(precedence.different_process_and_file_keys)
                    ? precedence.different_process_and_file_keys
                    : [];
                if (different.length > 0) {
                    const sourceKnown = Boolean(precedence?.declared_file_matches_project);
                    showSettingsStatus(
                        sourceKnown
                            ? `${different.length} setting change(s) are pending restart: ${different.join(', ')}.`
                            : `${different.length} .env value(s) differ from the running process: ${different.join(', ')}. Restart may apply them, or an external service override may win; check the service environment.`,
                        'warning'
                    );
                }
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
                const pending = Array.isArray(data.pendingOrOverriddenKeys) ? data.pendingOrOverriddenKeys : [];
                showSettingsStatus(
                    data.message || 'Environment variables saved.',
                    pending.length > 0 ? 'warning' : 'success'
                );
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

    function formatArchiveBytes(bytes) {
        const value = Number(bytes);
        if (!Number.isFinite(value) || value <= 0) return '0 B';
        const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
        let scaled = value;
        let unitIndex = 0;
        while (scaled >= 1024 && unitIndex < units.length - 1) {
            scaled /= 1024;
            unitIndex += 1;
        }
        const digits = scaled >= 100 || unitIndex === 0 ? 0 : (scaled >= 10 ? 1 : 2);
        return `${scaled.toFixed(digits)} ${units[unitIndex]}`;
    }

    function formatArchiveCount(value) {
        const num = Number(value);
        if (!Number.isFinite(num)) return '0';
        return Math.round(num).toLocaleString();
    }

    function archiveCapacityParams() {
        const params = new URLSearchParams();
        const setParam = (key, input) => {
            if (!input) return;
            const value = String(input.value || '').trim();
            if (value) params.set(key, value);
        };
        setParam('channels', archiveEstimateChannelsInput);
        setParam('batch_size', luxriotBatchSizeSelect);
        setParam('snapshot_interval_sec', luxriotSnapshotIntervalInput);
        setParam('frames_per_batch', archiveEstimateFramesPerBatchInput);
        setParam('avg_jpeg_kb', archiveEstimateAvgJpegKbInput);
        setParam('probe_records_per_channel_day', archiveEstimateProbeRowsInput);
        setParam('summary_retention_days', luxriotSummaryRetentionDaysInput);
        setParam('summary_history_limit', luxriotSummaryHistoryLimitInput);
        setParam('frame_retention_days', archiveRowRetentionDaysInput);
        setParam('thumbnail_retention_days', archiveThumbnailRetentionDaysInput);
        setParam('max_records', archiveMaxRecordsInput);
        return params;
    }

    function renderArchiveCapacity(data) {
        if (!archiveCapacitySummary) return;
        const estimate = data?.estimate || data?.archiveCapacityEstimate || data;
        if (!estimate || !estimate.bytes || !estimate.daily || !estimate.retained) {
            archiveCapacitySummary.textContent = 'Capacity estimate unavailable.';
            return;
        }
        const bytes = estimate.bytes;
        const daily = estimate.daily;
        const retained = estimate.retained;
        const current = data?.current || data?.archiveStorageSummary || null;
        const capped = retained.capped_by_max_records ? ' · capped' : '';
        const currentBits = current && current.available
            ? `<div>Current DB rows: <strong>${formatArchiveCount(current.row_count)}</strong> · DB previews: <strong>${formatArchiveBytes(current.thumbnail_bytes)}</strong></div>`
            : '';
        archiveCapacitySummary.innerHTML = `
            <div>Daily writes: <strong>${formatArchiveCount(daily.summary_rows)}</strong> descriptions · <strong>${formatArchiveCount(daily.frame_rows)}</strong> frame/probe rows</div>
            <div>Retained: <strong>${formatArchiveCount(retained.summary_rows)}</strong> descriptions · <strong>${formatArchiveCount(retained.frame_rows)}</strong> frame rows${capped}</div>
            <div>Database: <strong>${formatArchiveBytes(bytes.database)}</strong> · files: <strong>${formatArchiveBytes(bytes.archive_files)}</strong> · total: <strong>${formatArchiveBytes(bytes.total)}</strong></div>
            ${currentBits}
        `;
    }

    let archiveCapacityTimer = null;
    async function refreshArchiveCapacityEstimate() {
        if (!archiveCapacitySummary) return;
        try {
            const params = archiveCapacityParams();
            const suffix = params.toString() ? `?${params.toString()}` : '';
            const response = await fetch(`/settings/archive_capacity${suffix}`);
            const data = await response.json();
            if (data.success) {
                renderArchiveCapacity(data);
            } else {
                archiveCapacitySummary.textContent = data.error || 'Capacity estimate unavailable.';
            }
        } catch (error) {
            archiveCapacitySummary.textContent = error.message || 'Capacity estimate unavailable.';
        }
    }

    function scheduleArchiveCapacityEstimate() {
        if (!archiveCapacitySummary) return;
        window.clearTimeout(archiveCapacityTimer);
        archiveCapacityTimer = window.setTimeout(() => {
            refreshArchiveCapacityEstimate();
        }, 250);
    }

    // Load current settings
    async function loadSettings() {
        try {
            const response = await fetch('/settings');
            const data = await response.json();

            if (data.success) {
                const settings = data.settings;
                experimentalEmbeddersEnabled = toBool(settings.experimentalEmbeddersEnabled, false);
                productionClipModel = settings.productionClipModel || 'ViT-B/32';
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
                applyEmbeddingPolicyUI();
                rerankEnabledInput.checked = toBool(settings.rerankEnabled, false);
                const parsedRerankTopK = parseInt(settings.rerankTopK, 10);
                rerankTopKInput.value = Number.isFinite(parsedRerankTopK) ? parsedRerankTopK : 50;
                updateRerankUI(rerankEnabledInput.checked);
                clipModelSelect.value = settings.clipModel || productionClipModel;
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
                if (luxriotSummaryRetentionDaysInput) luxriotSummaryRetentionDaysInput.value = settings.luxriotSummaryRetentionDays ?? 7;
                if (luxriotSummaryHistoryLimitInput) luxriotSummaryHistoryLimitInput.value = settings.luxriotSummaryHistoryLimit ?? 10080;
                if (archiveRetentionEnabledInput) archiveRetentionEnabledInput.checked = toBool(settings.archiveRetentionEnabled, true);
                if (archiveRowRetentionDaysInput) archiveRowRetentionDaysInput.value = settings.archiveRowRetentionDays ?? 90;
                if (archiveThumbnailRetentionDaysInput) archiveThumbnailRetentionDaysInput.value = settings.archiveThumbnailRetentionDays ?? 14;
                if (archiveMaxRecordsInput) archiveMaxRecordsInput.value = settings.archiveMaxRecords ?? 5000000;
                if (archiveEstimateChannelsInput) archiveEstimateChannelsInput.value = settings.archiveEstimateChannels ?? 50;
                if (archiveEstimateFramesPerBatchInput) archiveEstimateFramesPerBatchInput.value = settings.archiveEstimateFramesPerBatch ?? 4;
                if (archiveEstimateAvgJpegKbInput) archiveEstimateAvgJpegKbInput.value = settings.archiveEstimateAvgJpegKb ?? 100;
                if (archiveEstimateProbeRowsInput) archiveEstimateProbeRowsInput.value = settings.archiveEstimateProbeRecordsPerChannelDay ?? 250;
                renderArchiveCapacity({
                    estimate: settings.archiveCapacityEstimate,
                    current: settings.archiveStorageSummary
                });
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
                applyEmbeddingPolicyUI();
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
                clipModel: clipModelSelect.value,
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
                luxriotSummaryRetentionDays: parseFloat(luxriotSummaryRetentionDaysInput ? luxriotSummaryRetentionDaysInput.value : '7'),
                luxriotSummaryHistoryLimit: parseInt(luxriotSummaryHistoryLimitInput ? luxriotSummaryHistoryLimitInput.value : '10080'),
                archiveRetentionEnabled: archiveRetentionEnabledInput ? archiveRetentionEnabledInput.checked : true,
                archiveRowRetentionDays: parseFloat(archiveRowRetentionDaysInput ? archiveRowRetentionDaysInput.value : '90'),
                archiveThumbnailRetentionDays: parseFloat(archiveThumbnailRetentionDaysInput ? archiveThumbnailRetentionDaysInput.value : '14'),
                archiveMaxRecords: parseInt(archiveMaxRecordsInput ? archiveMaxRecordsInput.value : '5000000'),
                archiveEstimateChannels: parseInt(archiveEstimateChannelsInput ? archiveEstimateChannelsInput.value : '50'),
                archiveEstimateFramesPerBatch: parseFloat(archiveEstimateFramesPerBatchInput ? archiveEstimateFramesPerBatchInput.value : '4'),
                archiveEstimateAvgJpegKb: parseFloat(archiveEstimateAvgJpegKbInput ? archiveEstimateAvgJpegKbInput.value : '100'),
                archiveEstimateProbeRecordsPerChannelDay: parseFloat(archiveEstimateProbeRowsInput ? archiveEstimateProbeRowsInput.value : '250'),
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

            if (!experimentalEmbeddersEnabled) {
                settings.embedder = 'clip';
                settings.fusionEnabled = false;
                settings.clipModel = productionClipModel || 'ViT-B/32';
                settings.indexMode = 'clip';
                settings.segmentsEnabled = false;
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
            if (!Number.isFinite(settings.luxriotSummaryRetentionDays) || settings.luxriotSummaryRetentionDays < 0) {
                settings.luxriotSummaryRetentionDays = 7;
            }
            if (!Number.isFinite(settings.luxriotSummaryHistoryLimit) || settings.luxriotSummaryHistoryLimit < 40) {
                settings.luxriotSummaryHistoryLimit = 10080;
            }
            if (!Number.isFinite(settings.archiveRowRetentionDays) || settings.archiveRowRetentionDays < 0) {
                settings.archiveRowRetentionDays = 90;
            }
            if (!Number.isFinite(settings.archiveThumbnailRetentionDays) || settings.archiveThumbnailRetentionDays < 0) {
                settings.archiveThumbnailRetentionDays = 14;
            }
            if (!Number.isFinite(settings.archiveMaxRecords) || settings.archiveMaxRecords < 1000) {
                settings.archiveMaxRecords = 5000000;
            }
            if (!Number.isFinite(settings.archiveEstimateChannels) || settings.archiveEstimateChannels < 1) {
                settings.archiveEstimateChannels = 50;
            }
            if (!Number.isFinite(settings.archiveEstimateFramesPerBatch) || settings.archiveEstimateFramesPerBatch < 0) {
                settings.archiveEstimateFramesPerBatch = 4;
            }
            if (!Number.isFinite(settings.archiveEstimateAvgJpegKb) || settings.archiveEstimateAvgJpegKb < 1) {
                settings.archiveEstimateAvgJpegKb = 100;
            }
            if (!Number.isFinite(settings.archiveEstimateProbeRecordsPerChannelDay) || settings.archiveEstimateProbeRecordsPerChannelDay < 0) {
                settings.archiveEstimateProbeRecordsPerChannelDay = 250;
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
                const pending = Array.isArray(data.pendingOrOverriddenKeys)
                    ? data.pendingOrOverriddenKeys
                    : [];
                const sourceKnown = Boolean(data.precedence?.declared_file_matches_project);
                showSettingsStatus(
                    data.message,
                    pending.length > 0 && !sourceKnown ? 'warning' : 'success'
                );
                scheduleArchiveCapacityEstimate();
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
            clipModelSelect.value = productionClipModel || 'ViT-B/32';
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
            if (luxriotSummaryRetentionDaysInput) luxriotSummaryRetentionDaysInput.value = '7';
            if (luxriotSummaryHistoryLimitInput) luxriotSummaryHistoryLimitInput.value = '10080';
            if (archiveRetentionEnabledInput) archiveRetentionEnabledInput.checked = true;
            if (archiveRowRetentionDaysInput) archiveRowRetentionDaysInput.value = '90';
            if (archiveThumbnailRetentionDaysInput) archiveThumbnailRetentionDaysInput.value = '14';
            if (archiveMaxRecordsInput) archiveMaxRecordsInput.value = '5000000';
            if (archiveEstimateChannelsInput) archiveEstimateChannelsInput.value = '50';
            if (archiveEstimateFramesPerBatchInput) archiveEstimateFramesPerBatchInput.value = '4';
            if (archiveEstimateAvgJpegKbInput) archiveEstimateAvgJpegKbInput.value = '100';
            if (archiveEstimateProbeRowsInput) archiveEstimateProbeRowsInput.value = '250';
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
            applyEmbeddingPolicyUI();
            updateFusionUI(false);
            updateRerankUI(false);
            updateSegmentsUI(false);
            refreshSegmentsPanels();
            applyEmbedderUI(embedderSelect.value);
            scheduleArchiveCapacityEstimate();
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
            settingsStatus.textContent = '';
            settingsStatus.style.display = 'none';
        }, 5000);
    }

    function setSettingRowDisabled(control, disabled) {
        if (!control) return;
        control.disabled = disabled;
        control.classList.toggle('disabled', disabled);
        const row = control.closest('.settings-row');
        if (row) row.classList.toggle('is-disabled', disabled);
    }

    function applyEmbeddingPolicyUI() {
        const productionModel = productionClipModel || 'ViT-B/32';
        Array.from(embedderSelect.options).forEach(option => {
            const locked = !experimentalEmbeddersEnabled && option.value !== 'clip';
            option.disabled = locked;
            option.title = locked ? 'Experimental backend disabled for this deployment.' : '';
        });
        if (!experimentalEmbeddersEnabled && embedderSelect.value !== 'clip') {
            embedderSelect.value = 'clip';
        }

        Array.from(clipModelSelect.options).forEach(option => {
            const locked = !experimentalEmbeddersEnabled && option.value !== productionModel;
            option.disabled = locked;
            option.title = locked ? 'Experimental CLIP/SigLIP model disabled for this deployment.' : '';
        });
        if (!experimentalEmbeddersEnabled && clipModelSelect.value !== productionModel) {
            clipModelSelect.value = productionModel;
        }

        if (!experimentalEmbeddersEnabled) {
            fusionEnabledInput.checked = false;
            indexModeSelect.value = 'clip';
            segmentsEnabledInput.checked = false;
        }
        setSettingRowDisabled(fusionEnabledInput, !experimentalEmbeddersEnabled);
        setSettingRowDisabled(indexModeSelect, !experimentalEmbeddersEnabled);
        setSettingRowDisabled(segmentsEnabledInput, !experimentalEmbeddersEnabled);

        [dinoModelInput, dinoEmbedDimInput, dinoWeightsInput].forEach(input => {
            setSettingRowDisabled(input, !experimentalEmbeddersEnabled);
        });

        updateFusionUI(fusionEnabledInput.checked);
        updateSegmentsUI(segmentsEnabledInput.checked);
    }

    function updateFusionUI(enabled) {
        const available = experimentalEmbeddersEnabled && enabled;
        fusionAlphaInput.disabled = !available;
        fusionAlphaValue.textContent = Number(fusionAlphaInput.value).toFixed(2);
        fusionAlphaValue.classList.toggle('disabled', !available);
        const fusionAlphaRow = fusionAlphaInput.closest('.settings-row');
        if (fusionAlphaRow) fusionAlphaRow.classList.toggle('is-disabled', !available);
        const fusionOption = embedderSelect.querySelector('option[value="fusion"]');
        if (fusionOption) {
            fusionOption.disabled = !available;
        }
        if (!available && embedderSelect.value === 'fusion') {
            embedderSelect.value = 'clip';
            applyEmbedderUI('clip');
        }
    }

    function updateRerankUI(enabled) {
        rerankTopKInput.disabled = !enabled;
        rerankTopKInput.classList.toggle('disabled', !enabled);
    }

    applyEmbeddingPolicyUI();
    updateRerankUI(rerankEnabledInput.checked);
    
    function updateSegmentsUI(enabled) {
        const available = experimentalEmbeddersEnabled && enabled;
        segmentMinPatchesInput.disabled = !available;
        segmentMinPatchesInput.classList.toggle('disabled', !available);
        const row = segmentMinPatchesInput.closest('.settings-row');
        if (row) row.classList.toggle('is-disabled', !available);
        updateSegmentControlsUI(available);
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
            const channelId = getSelectedLuxriotChannel();
            clearLuxriotLiveIntervalDirty(channelId);
            fetchLuxriotChannels(true).then(() => {
                syncProbeChannelSelect();
                syncLuxriotLiveIntervalInput(channelId, { force: true });
                void refreshLuxriotPromptSettings(false, channelId);
                refreshLuxriotStreams();
            });
        });
    }
    if (luxriotToggleCaptureBtn) {
        luxriotToggleCaptureBtn.addEventListener('click', toggleLuxriotCapture);
    }
    if (luxriotFlushCaptureBtn) {
        luxriotFlushCaptureBtn.addEventListener('click', flushLuxriotCapture);
    }
    if (luxriotContextToggleCaptureBtn) {
        luxriotContextToggleCaptureBtn.addEventListener('click', toggleLuxriotCapture);
    }
    if (luxriotContextFlushCaptureBtn) {
        luxriotContextFlushCaptureBtn.addEventListener('click', flushLuxriotCapture);
    }
    roadSceneGroundingBtns.forEach((button) => {
        button.addEventListener('click', refreshRoadSceneGrounding);
    });
    if (luxriotBatchSizeSelect) {
        luxriotBatchSizeSelect.addEventListener('change', updateLuxriotBatchInfo);
    }
    if (luxriotLiveIntervalInput) {
        luxriotLiveIntervalInput.addEventListener('input', () => {
            markLuxriotLiveIntervalDirty();
            updateLuxriotBatchInfo();
        });
        luxriotLiveIntervalInput.addEventListener('change', () => {
            markLuxriotLiveIntervalDirty();
            const intervalSec = getLuxriotLiveIntervalInputValue();
            luxriotLiveIntervalInput.value = formatLuxriotLiveIntervalInput(intervalSec);
            storeLuxriotLiveInterval(getSelectedLuxriotChannel(), intervalSec);
            updateLuxriotBatchInfo();
        });
    }
    if (luxriotLiveModelInput) {
        luxriotLiveModelInput.addEventListener('change', updateLuxriotStreamContext);
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
    if (luxriotPromptResetBtn) {
        luxriotPromptResetBtn.addEventListener('click', async () => {
            try {
                await resetLuxriotPromptOverrides();
            } catch (err) {
                setLuxriotStatus(err.message || 'Failed to reset prompt settings', true);
            }
        });
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
            resetSummaryArchivePaging();
            applySummaryResolutionMode();
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
            if (luxriotSummaryRangePreset === 'custom') {
                if (!Number.isFinite(luxriotSummaryFromTs) || !Number.isFinite(luxriotSummaryToTs)) {
                    const initial = summaryLocalDayBounds(-1);
                    luxriotSummaryFromTs = initial.fromTs;
                    luxriotSummaryToTs = initial.toTs;
                    if (luxriotSummaryFromInput) luxriotSummaryFromInput.value = formatSummaryDatetimeInput(initial.fromTs);
                    if (luxriotSummaryToInput) luxriotSummaryToInput.value = formatSummaryDatetimeInput(initial.toTs);
                }
                luxriotSummaryRunFilter = 'all';
                luxriotSummaryFollowLive = false;
                resetSummaryArchivePaging();
                applySummaryResolutionMode();
                syncSummaryRangeUI();
                updateSummaryControlsUI();
                return;
            }
            applySummaryFiltersFromInputs();
            syncSummaryRangeUI();
            setSummaryUnread(0);
            refreshLuxriotSummaryView(getSelectedSummaryChannel(), true);
        });
    }
    if (luxriotSummaryLevelSelect) {
        luxriotSummaryLevelSelect.addEventListener('change', () => {
            luxriotSummaryResolutionMode = normalizeSummaryResolutionMode(luxriotSummaryLevelSelect.value);
            applySummaryResolutionMode();
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
            if (!isLiveSummaryPeriod() || isRollupViewActive()) {
                luxriotSummaryRangePreset = 'live';
                luxriotSummaryRunFilter = 'live';
                luxriotSummaryFollowLive = true;
                luxriotSummaryAutoRefresh = true;
                if (luxriotSummaryRangeSelect) luxriotSummaryRangeSelect.value = 'live';
                resetSummaryArchivePaging();
                applySummaryResolutionMode();
                syncSummaryRangeUI();
                setSummaryUnread(0);
                updateSummaryControlsUI();
                refreshLuxriotSummaryView(getSelectedSummaryChannel(), true);
                return;
            }
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
            if (!isLiveSummaryPeriod() || isRollupViewActive()) {
                luxriotSummaryRangePreset = 'live';
                luxriotSummaryRunFilter = 'live';
                luxriotSummaryAutoRefresh = true;
                if (luxriotSummaryRangeSelect) luxriotSummaryRangeSelect.value = 'live';
                resetSummaryArchivePaging();
                applySummaryResolutionMode();
                syncSummaryRangeUI();
            }
            luxriotSummaryFollowLive = true;
            setSummaryUnread(0);
            updateSummaryControlsUI();
            refreshLuxriotSummaryView(getSelectedSummaryChannel(), true);
        });
    }
    if (luxriotSummaryPreviousPeriodBtn) {
        luxriotSummaryPreviousPeriodBtn.addEventListener('click', () => {
            if (shiftSelectedSummaryPeriod(-1)) {
                refreshLuxriotSummaryView(getSelectedSummaryChannel(), true);
            }
        });
    }
    if (luxriotSummaryNextPeriodBtn) {
        luxriotSummaryNextPeriodBtn.addEventListener('click', () => {
            if (shiftSelectedSummaryPeriod(1)) {
                refreshLuxriotSummaryView(getSelectedSummaryChannel(), true);
            }
        });
    }
    if (luxriotSummaryLoadEarlierBtn) {
        luxriotSummaryLoadEarlierBtn.addEventListener('click', () => {
            void refreshLuxriotArchivedSummaries(getSelectedSummaryChannel(), null, true);
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
            const rollupGenerateBtn = target.closest('[data-luxriot-rollup-generate]');
            if (rollupGenerateBtn instanceof HTMLButtonElement) {
                const idx = parseInt(rollupGenerateBtn.dataset.luxriotRollupGenerate || '', 10);
                if (!Number.isFinite(idx)) return;
                event.preventDefault();
                void generateLuxriotSemanticRollup(idx, rollupGenerateBtn);
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
                    luxriotSummaryRangePreset = 'live';
                    luxriotSummaryRunFilter = 'live';
                    if (luxriotSummaryRangeSelect) luxriotSummaryRangeSelect.value = 'live';
                    resetSummaryArchivePaging();
                    applySummaryResolutionMode();
                    syncSummaryRangeUI();
                    setSummaryUnread(0);
                    luxriotSummaryFollowLive = true;
                    luxriotSummaryAutoRefresh = true;
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
            clearLuxriotLiveIntervalDirty();
            luxriotActiveChannel = getSelectedLuxriotChannel();
            syncProbeChannelSelect();
            syncLuxriotSummaryChannelSelect();
            resetSummaryArchivePaging();
            applySummaryResolutionMode();
            syncLuxriotLiveIntervalInput(luxriotActiveChannel, { force: true });
            updateLuxriotCaptureToggleButton(luxriotActiveChannel);
            updateLuxriotStreamContext();
            void refreshLuxriotPromptSettings(false, luxriotActiveChannel);
            resetRoadSceneGrounding();
            if (currentMode === 'video') {
                startLuxriotPreview();
                refreshLuxriotSummaryView(getSelectedSummaryChannel(), true);
            }
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

    function setProbeCastStatus(message, isError = false) {
        if (!probeCastStatus) return;
        probeCastStatus.textContent = message;
        probeCastStatus.classList.toggle('error', Boolean(isError));
    }

    function getProbeCastChannelCatalog() {
        const select = probeChannelSelect && probeChannelSelect.options?.length
            ? probeChannelSelect
            : luxriotChannelSelect;
        const seen = new Set();
        return Array.from(select?.options || [])
            .map((option) => {
                const id = Number.parseInt(String(option.value || ''), 10);
                if (!Number.isFinite(id) || id <= 0 || seen.has(id)) return null;
                seen.add(id);
                return {
                    id,
                    label: String(option.textContent || '').trim() || getLuxriotChannelLabel(id),
                };
            })
            .filter((channel) => Boolean(channel))
            .sort((left, right) => left.id - right.id);
    }

    function updateProbeCastSelectionMeta() {
        if (!probeCastSelectedMeta) return;
        const count = probeCastSelectedChannels.size;
        probeCastSelectedMeta.textContent = `${count} selected`;
    }

    function renderProbeCastChannels() {
        if (!probeCastChannelList) return;
        const channels = getProbeCastChannelCatalog();
        if (!channels.length) {
            probeCastChannelList.innerHTML = '<div class="admin-empty">Channel list is not loaded yet.</div>';
            updateProbeCastSelectionMeta();
            return;
        }
        probeCastChannelList.innerHTML = channels.map((channel) => {
            const checked = probeCastSelectedChannels.has(channel.id) ? ' checked' : '';
            return `
                <label class="admin-channel-item" title="${escapeHtml(channel.label)}">
                    <input type="checkbox" value="${escapeHtml(String(channel.id))}"${checked} />
                    <span class="admin-channel-id">#${escapeHtml(String(channel.id))}</span>
                    <span class="admin-channel-name">${escapeHtml(channel.label)}</span>
                </label>
            `;
        }).join('');
        updateProbeCastSelectionMeta();
    }

    function setProbeCastSelection(channelIds) {
        const available = new Set(getProbeCastChannelCatalog().map((channel) => channel.id));
        probeCastSelectedChannels = new Set(
            (Array.isArray(channelIds) ? channelIds : [])
                .map((value) => Number.parseInt(String(value), 10))
                .filter((id) => Number.isFinite(id) && id > 0 && available.has(id))
        );
        renderProbeCastChannels();
    }

    async function openProbeCastModal() {
        const payload = collectProbeForm();
        const hasPos = payload.positives.length > 0 || (payload.image_probe?.enabled && payload.image_probe?.data);
        if (!hasPos) {
            setProbeStatus('Add a text positive or enable an image probe before casting.', true);
            return;
        }
        if (!getProbeCastChannelCatalog().length) {
            await fetchLuxriotChannels(true);
            syncProbeChannelSelect();
        }
        const currentChannel = getSelectedProbeChannelId();
        if (probeCastEnableInput) {
            probeCastEnableInput.checked = probeEnableToggle ? probeEnableToggle.checked !== false : true;
        }
        if (probeCastCopyRoiInput) {
            probeCastCopyRoiInput.checked = false;
            probeCastCopyRoiInput.disabled = !Boolean(probeRoiEnabled && normalizeProbeRoiNorm(probeRoiNorm));
        }
        if (probeCastStartStreamsInput) probeCastStartStreamsInput.checked = false;
        if (probeCastConflictInput) probeCastConflictInput.value = 'skip';
        setProbeCastSelection([currentChannel]);
        setProbeCastStatus('Ready.');
        setProbeCastModalVisibility(true);
    }

    function selectedProbeCastChannelIds() {
        if (!probeCastChannelList) return [];
        return Array.from(probeCastChannelList.querySelectorAll('input[type="checkbox"]:checked'))
            .map((input) => Number.parseInt(String(input.value || ''), 10))
            .filter((id) => Number.isFinite(id) && id > 0);
    }

    async function applyProbeCast() {
        const channelIds = selectedProbeCastChannelIds();
        if (!channelIds.length) {
            setProbeCastStatus('Select at least one channel.', true);
            return;
        }
        const payload = collectProbeForm();
        const hasPos = payload.positives.length > 0 || (payload.image_probe?.enabled && payload.image_probe?.data);
        if (!hasPos) {
            setProbeCastStatus('Add a text positive or enable an image probe.', true);
            return;
        }
        delete payload.id;
        payload.channel_ids = channelIds;
        payload.enabled = probeCastEnableInput ? probeCastEnableInput.checked : payload.enabled;
        payload.conflict = probeCastConflictInput ? probeCastConflictInput.value : 'skip';
        payload.copy_roi = probeCastCopyRoiInput ? probeCastCopyRoiInput.checked : false;
        if (!payload.copy_roi) {
            payload.roi_enabled = false;
            payload.roi_norm = null;
        }
        setProbeCastStatus(`Casting to ${channelIds.length} channels...`);
        setControlDisabled(probeCastApplyBtn, true);
        try {
            const resp = await fetch('/probes/cast', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await resp.json();
            if (!resp.ok && !data.counts) {
                throw new Error(data.error || 'Cast failed');
            }
            const counts = data.counts || {};
            const summary = `Cast: ${counts.created || 0} created, ${counts.updated || 0} updated, ${counts.skipped || 0} skipped, ${counts.failed || 0} failed.`;
            await loadProbeList();
            if (probeCastStartStreamsInput?.checked) {
                setProbeCastStatus(`${summary} Starting streams...`);
                for (const channelId of channelIds) {
                    await ensureProbeCapture(channelId, true, { forceStart: true });
                }
            }
            const hasFailures = (counts.failed || 0) > 0;
            setProbeStatus(summary, hasFailures);
            setProbeCastStatus(summary, hasFailures);
            if (!hasFailures) {
                setProbeCastModalVisibility(false);
            }
        } catch (err) {
            setProbeCastStatus(err.message, true);
        } finally {
            setControlDisabled(probeCastApplyBtn, false);
        }
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
        const videoActive = probePreviewVideo
            && probePreviewVideo.style.display !== 'none'
            && Number(probePreviewVideo.videoWidth) > 0;
        const naturalWidth = videoActive ? Number(probePreviewVideo.videoWidth) : (probePreviewImg.naturalWidth || 0);
        const naturalHeight = videoActive ? Number(probePreviewVideo.videoHeight) : (probePreviewImg.naturalHeight || 0);
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

    function ensureProbePreviewVideo() {
        if (probePreviewVideo && probePreviewVideo.isConnected) return probePreviewVideo;
        if (!probePreviewViewport) return null;
        const video = document.createElement('video');
        video.className = 'probe-operator-video';
        video.autoplay = true;
        video.muted = true;
        video.controls = true;
        video.playsInline = true;
        video.preload = 'metadata';
        video.setAttribute('aria-label', 'Luxriot operator video for probe monitoring');
        Object.assign(video.style, {
            position: 'absolute',
            inset: '0',
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            background: '#000',
            display: 'none',
        });
        probePreviewViewport.insertBefore(video, probePreviewOverlay || probeRoiLayer || null);
        probePreviewVideo = video;
        return video;
    }

    function ensureProbePreviewRetryButton() {
        if (probePreviewRetryBtn && probePreviewRetryBtn.isConnected) return probePreviewRetryBtn;
        if (!probePreviewViewport) return null;
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'feature-btn';
        button.textContent = 'Retry video';
        button.hidden = true;
        button.addEventListener('click', () => startProbePreview(getSelectedProbeChannelId()));
        Object.assign(button.style, {
            position: 'absolute',
            right: '10px',
            bottom: '10px',
            zIndex: '8',
        });
        probePreviewViewport.appendChild(button);
        probePreviewRetryBtn = button;
        return button;
    }

    function clearProbePreviewVideo() {
        if (!probePreviewVideo) return;
        probePreviewVideo.onloadedmetadata = null;
        probePreviewVideo.oncanplay = null;
        probePreviewVideo.onplaying = null;
        probePreviewVideo.onwaiting = null;
        probePreviewVideo.onstalled = null;
        probePreviewVideo.onprogress = null;
        probePreviewVideo.ontimeupdate = null;
        probePreviewVideo.onended = null;
        probePreviewVideo.onerror = null;
        try {
            probePreviewVideo.pause();
        } catch (_) {
            // Best-effort media cleanup.
        }
        probePreviewVideo.removeAttribute('src');
        try {
            probePreviewVideo.load();
        } catch (_) {
            // Best-effort media cleanup.
        }
        probePreviewVideo.style.display = 'none';
    }

    function replaceProbePreviewImageElement() {
        if (!probePreviewImg || !probePreviewImg.parentNode) return;
        const previous = probePreviewImg;
        previous.onload = null;
        previous.onerror = null;
        const replacement = previous.cloneNode(false);
        replacement.removeAttribute('src');
        replacement.style.display = 'none';
        previous.replaceWith(replacement);
        probePreviewImg = replacement;
    }

    function setProbeMediaState(state, detail = '') {
        probePreviewMediaState = ['idle', 'loading', 'playing', 'degraded', 'error'].includes(state) ? state : 'error';
        const overlayText = {
            idle: '',
            loading: detail || 'Loading live video…',
            playing: '',
            degraded: 'Static frame fallback — not video',
            error: detail || 'Video unavailable',
        }[probePreviewMediaState];
        if (probePreviewViewport) probePreviewViewport.dataset.mediaState = probePreviewMediaState;
        setPreviewState(overlayText);
        const retry = ensureProbePreviewRetryButton();
        if (retry) retry.hidden = !['degraded', 'error'].includes(probePreviewMediaState);
    }

    function isCurrentProbeMediaRequest(generation, channelId) {
        return generation === probePreviewGeneration
            && probePreviewChannelId === channelId
            && currentMode === 'monitor'
            && probeEditorModal
            && probeEditorModal.style.display === 'block'
            && getSelectedProbeChannelId() === channelId;
    }

    function scheduleProbePreviewRenewal(generation, channelId, delayMs, detail) {
        if (probePreviewRenewTimer) clearTimeout(probePreviewRenewTimer);
        const parsedDelay = Number(delayMs);
        const safeDelay = delayMs !== null && delayMs !== undefined && Number.isFinite(parsedDelay) && parsedDelay > 0
            ? Math.max(750, Math.min(120000, Math.trunc(parsedDelay)))
            : 20000;
        probePreviewRenewTimer = window.setTimeout(() => {
            probePreviewRenewTimer = null;
            if (!isCurrentProbeMediaRequest(generation, channelId)) return;
            setProbeMediaState('loading', detail || 'Renewing bounded operator video…');
            startProbePreview(channelId, true);
        }, safeDelay);
    }

    function clearProbePreviewStallWatchdog() {
        if (!probePreviewStallTimer) return;
        clearTimeout(probePreviewStallTimer);
        probePreviewStallTimer = null;
    }

    function armProbePreviewStallWatchdog(generation, channelId) {
        clearProbePreviewStallWatchdog();
        probePreviewStallTimer = window.setTimeout(() => {
            probePreviewStallTimer = null;
            if (!isCurrentProbeMediaRequest(generation, channelId)) return;
            scheduleProbePreviewRenewal(
                generation,
                channelId,
                750,
                'Operator video stalled; reconnecting without touching probe capture…',
            );
        }, 5000);
    }

    function showProbeStaticFallback(generation, channelId, reason) {
        if (!isCurrentProbeMediaRequest(generation, channelId) || !probePreviewImg) return;
        clearProbePreviewVideo();
        probePreviewImg.onload = () => {
            if (!isCurrentProbeMediaRequest(generation, channelId)) return;
            probePreviewImg.style.display = 'block';
            setProbeMediaState('degraded', reason);
            renderProbeRoiBox();
        };
        probePreviewImg.onerror = () => {
            if (!isCurrentProbeMediaRequest(generation, channelId)) return;
            setProbeMediaState('error', 'Video and static fallback are unavailable.');
        };
        probePreviewImg.style.display = 'block';
        probePreviewImg.src = `/luxriot/snapshot/${encodeURIComponent(String(channelId))}?stream=mainStream&t=${Date.now()}`;
    }

    function stopProbePreview() {
        if (probePreviewTimer) {
            clearTimeout(probePreviewTimer);
            probePreviewTimer = null;
        }
        if (probePreviewRenewTimer) {
            clearTimeout(probePreviewRenewTimer);
            probePreviewRenewTimer = null;
        }
        clearProbePreviewStallWatchdog();
        probePreviewGeneration += 1;
        abortUiRequest(probePreviewAbortController);
        probePreviewAbortController = null;
        clearProbePreviewVideo();
        if (probePreviewImg) {
            probePreviewImg.onload = null;
            probePreviewImg.onerror = null;
            probePreviewImg.removeAttribute('src');
            probePreviewImg.style.display = 'none';
        }
        if (probePreviewRetryBtn) probePreviewRetryBtn.hidden = true;
        probePreviewChannelId = null;
        probePreviewMediaState = 'idle';
    }

    function startProbePreview(channelId, force = false) {
        if (!probePreviewImg || !probePreviewViewport) return;
        if (currentMode !== 'monitor') return;
        if (
            !force
            && probePreviewChannelId === channelId
            && ['loading', 'playing'].includes(probePreviewMediaState)
        ) return;
        stopProbePreview();
        replaceProbePreviewImageElement();
        if (!channelId && channelId !== 0) {
            setPreviewState('No channel', true);
            return;
        }
        probePreviewChannelId = channelId;
        const generation = ++probePreviewGeneration;
        const controller = new AbortController();
        const cachedNegotiation = force
            && Number(probePreviewNegotiation?.channelId) === Number(channelId)
            ? probePreviewNegotiation.value
            : null;
        probePreviewAbortController = cachedNegotiation ? null : controller;
        const sharedVideoStream = selectedLuxriotStream(channelId, 'video');
        const useAttentionPreview = Boolean(sharedVideoStream?.running);
        const mediaUrl = luxriotMediaBrokerUrl(
            useAttentionPreview ? 'attention' : 'live',
            channelId,
            { stream: 'mainStream' },
        );
        setProbeMediaState(
            'loading',
            useAttentionPreview ? 'Loading shared EVA attention frames…' : 'Loading operator video…',
        );
        const negotiationRequest = cachedNegotiation
            ? Promise.resolve(cachedNegotiation)
            : negotiateLuxriotMedia(mediaUrl, controller);
        void negotiationRequest
            .then((negotiated) => {
                if (probePreviewAbortController === controller) probePreviewAbortController = null;
                if (!isCurrentProbeMediaRequest(generation, channelId)) return;
                probePreviewNegotiation = { channelId, value: negotiated };
                const failToStatic = (reason) => {
                    if (probePreviewTimer) {
                        clearTimeout(probePreviewTimer);
                        probePreviewTimer = null;
                    }
                    if (probePreviewRenewTimer) {
                        clearTimeout(probePreviewRenewTimer);
                        probePreviewRenewTimer = null;
                    }
                    clearProbePreviewStallWatchdog();
                    showProbeStaticFallback(generation, channelId, reason);
                };
                probePreviewTimer = window.setTimeout(
                    () => failToStatic('Operator video load timed out.'),
                    12000,
                );
                if (negotiated.mediaKind === 'mjpeg') {
                    probePreviewImg.onload = () => {
                        if (!isCurrentProbeMediaRequest(generation, channelId)) return;
                        clearTimeout(probePreviewTimer);
                        probePreviewTimer = null;
                        probePreviewImg.style.display = 'block';
                        setProbeMediaState('playing');
                        renderProbeRoiBox();
                    };
                    probePreviewImg.onerror = () => failToStatic('The MJPEG stream could not be decoded.');
                    probePreviewImg.style.display = 'block';
                    scheduleProbePreviewRenewal(
                        generation,
                        channelId,
                        negotiated.renewAfterMs,
                        negotiated.attentionPreview
                            ? 'Renewing shared EVA attention frames…'
                            : 'Renewing bounded MJPEG operator video…',
                    );
                    probePreviewImg.src = `${mediaUrl}&request=${Date.now()}`;
                    return;
                }
                const video = ensureProbePreviewVideo();
                if (!video) {
                    failToStatic('The browser video element could not be initialized.');
                    return;
                }
                video.style.display = 'block';
                const markPlayable = () => {
                    if (!isCurrentProbeMediaRequest(generation, channelId)) return;
                    clearTimeout(probePreviewTimer);
                    probePreviewTimer = null;
                    clearProbePreviewStallWatchdog();
                    setProbeMediaState('playing');
                    renderProbeRoiBox();
                };
                video.onloadedmetadata = () => {
                    if (!isCurrentProbeMediaRequest(generation, channelId)) return;
                    renderProbeRoiBox();
                };
                video.oncanplay = () => {
                    if (!isCurrentProbeMediaRequest(generation, channelId)) return;
                    markPlayable();
                    const playPromise = video.play();
                    if (playPromise && typeof playPromise.catch === 'function') playPromise.catch(() => {});
                };
                video.onplaying = markPlayable;
                video.onwaiting = () => {
                    if (!isCurrentProbeMediaRequest(generation, channelId)) return;
                    setProbeMediaState('loading', 'Buffering operator video…');
                    armProbePreviewStallWatchdog(generation, channelId);
                };
                video.onstalled = () => {
                    if (!isCurrentProbeMediaRequest(generation, channelId)) return;
                    setProbeMediaState('loading', 'Operator video transport stalled…');
                    armProbePreviewStallWatchdog(generation, channelId);
                };
                video.onprogress = clearProbePreviewStallWatchdog;
                video.ontimeupdate = clearProbePreviewStallWatchdog;
                video.onerror = () => failToStatic('The browser rejected the Luxriot video container or codec.');
                video.onended = () => {
                    if (!isCurrentProbeMediaRequest(generation, channelId)) return;
                    setProbeMediaState('loading', 'Reconnecting operator video…');
                    scheduleProbePreviewRenewal(
                        generation,
                        channelId,
                        750,
                        'The bounded operator video segment ended; reconnecting…',
                    );
                };
                scheduleProbePreviewRenewal(
                    generation,
                    channelId,
                    negotiated.renewAfterMs,
                    'Renewing bounded operator video before its server lease expires…',
                );
                video.src = `${mediaUrl}&request=${Date.now()}`;
                video.load();
            })
            .catch((error) => {
                if (probePreviewAbortController === controller) probePreviewAbortController = null;
                if (!isCurrentProbeMediaRequest(generation, channelId)) return;
                showProbeStaticFallback(
                    generation,
                    channelId,
                    controller.signal.aborted ? 'Operator media negotiation timed out.' : (error.message || 'Operator video is unavailable.'),
                );
            });
    }

    function syncProbePreview(channelIdOverride = null) {
        if (currentMode !== 'monitor') {
            stopProbePreview();
            return;
        }
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
        // Operator media is independent from probe/VLM capture lifecycle.
        startProbePreview(channelId);
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

    async function captureProbeSnapshotFromServer() {
        const channelId = getSelectedProbeChannelId();
        if (!Number.isFinite(channelId) || channelId <= 0) {
            throw new Error('Select a channel before snapping.');
        }
        const roiNorm = probeRoiEnabled ? normalizeProbeRoiNorm(probeRoiNorm) : null;
        if (probeRoiEnabled && !roiNorm) {
            throw new Error('ROI is enabled. Draw ROI before snapping.');
        }
        const payload = {
            roi_enabled: Boolean(roiNorm),
            quality: 92,
        };
        if (roiNorm) {
            payload.roi_norm = roiNorm;
        }
        const response = await fetch(`/luxriot/snapshot/${channelId}/capture`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await parseApiJson(response, 'Snapshot capture failed');
        const base64 = String(data.snapshot_b64 || '').trim();
        if (!base64) {
            throw new Error('Snapshot capture returned no image.');
        }
        const meta = data.meta || {};
        const timestampMs = Number(meta.captured_at_ms) || Date.now();
        const width = Number(meta.width) || 0;
        const height = Number(meta.height) || 0;
        return {
            dataUrl: `data:image/jpeg;base64,${base64}`,
            base64,
            width,
            height,
            timestampMs,
            channelId,
            roi: Boolean(roiNorm),
            sha1: meta.sha1 || '',
            filename: data.filename || _buildProbeSnapFilename(channelId, timestampMs, Boolean(roiNorm)),
        };
    }

    async function openProbeSnapModalFromPreview() {
        try {
            setProbeStatus('Capturing fresh snapshot...');
            const snap = await captureProbeSnapshotFromServer();
            probeSnapState = snap;
            if (probeSnapImg) {
                probeSnapImg.src = snap.dataUrl;
            }
            if (probeSnapMeta) {
                const mode = snap.roi ? 'ROI snapshot' : 'Full-frame snapshot';
                const digest = snap.sha1 ? ` · ${String(snap.sha1).slice(0, 8)}` : '';
                probeSnapMeta.textContent = `${mode} · ${snap.width}×${snap.height} · Channel #${snap.channelId}${digest}`;
            }
            if (probeSnapActualSizeInput) {
                probeSnapActualSizeInput.checked = false;
            }
            updateProbeSnapScaleMode();
            setProbeSnapModalVisibility(true);
            setProbeStatus('Fresh snapshot captured.');
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
        const payload = {
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
        if (userHasPermission('bookmarks:create')) {
            payload.bookmark = probeBookmarkToggle ? probeBookmarkToggle.checked : true;
            payload.bookmark_cooldown_sec = probeBookmarkCooldownLocalInput ? (parseFloat(probeBookmarkCooldownLocalInput.value) || 0) : 8;
            payload.bookmark_dedupe_window_sec = probeBookmarkDedupeWindowLocalInput ? (parseFloat(probeBookmarkDedupeWindowLocalInput.value) || 0.5) : 20;
        }
        return payload;
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
            enable: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="m380-300 280-180-280-180v360Z"/></svg>',
            disable: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M320-320v-320h320v320H320Z"/></svg>',
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
            const emptyCardsHtml = `
                <div class="probe-mini-card new-probe-card">
                    <button class="probe-new-btn" data-action="new" aria-label="Create probe" title="Create probe">
                        ${probeActionIcon('new')}
                        <span>New Probe</span>
                    </button>
                </div>`;
            if (probeCardsRenderKey !== emptyCardsHtml) {
                probeCards.innerHTML = emptyCardsHtml;
                probeCardsRenderKey = emptyCardsHtml;
            }
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
            const toggleTitle = status === 'disabled' ? 'Start probe' : 'Stop probe';
            const scores = `P: ${Number.isFinite(last?.pos_score) ? last.pos_score.toFixed(3) : '—'} · N: ${Number.isFinite(last?.neg_score) ? last.neg_score.toFixed(3) : '—'} · M: ${Number.isFinite(last?.margin) ? last.margin.toFixed(3) : '—'}`;
            const gateView = describeProbeBookmarkGate(p.bookmark_gate, p.bookmark !== false);
            return `
                <div class="probe-mini-card ${activeProbeId === p.id ? 'active' : ''}" data-probe-id="${p.id}">
                    <div class="probe-mini-card-head">
                        <div class="probe-status-pill ${pillClass}">${status}</div>
                        <div class="probe-mini-actions probe-mini-primary-actions">
                            <button class="probe-action-btn" data-action="${toggleAction}" data-id="${p.id}" title="${toggleTitle}" aria-label="${toggleTitle}">${probeActionIcon(toggleAction)}</button>
                            <button class="probe-action-btn" data-action="expand" data-id="${p.id}" title="Edit probe" aria-label="Edit probe">${probeActionIcon('expand')}</button>
                        </div>
                    </div>
                    <div class="probe-mini-thumb ${thumbSrc ? '' : 'is-empty'}">
                        ${thumbSrc ? `<img src="data:image/jpeg;base64,${thumbSrc}" alt="${escapeHtml(p.name || 'probe preview')}" />` : ''}
                    </div>
                    <div class="probe-mini-content">
                        <div class="probe-mini-name" title="${escapeHtml(p.name || 'unnamed')}">${escapeHtml(p.name || 'unnamed')}</div>
                        <div class="probe-mini-meta">Ch ${p.channel_id || luxriotActiveChannel} · Last ${last ? ts : 'n/a'}</div>
                        <div class="probe-mini-score">${scores}</div>
                        <div class="probe-mini-card-foot">
                            <div class="probe-mini-gate" title="${escapeHtml(gateView.title)}">${escapeHtml(gateView.text)}</div>
                            <div class="probe-mini-actions probe-mini-danger-actions">
                                <button class="probe-action-btn delete" data-action="delete" data-id="${p.id}" title="Delete probe" aria-label="Delete probe">${probeActionIcon('delete')}</button>
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
        const cardsHtml = cards.join('');
        if (probeCardsRenderKey !== cardsHtml) {
            probeCards.innerHTML = cardsHtml;
            probeCardsRenderKey = cardsHtml;
        }
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
    if (probeCastBtn) {
        probeCastBtn.addEventListener('click', () => {
            openProbeCastModal();
        });
    }
    if (closeProbeCastBtn) {
        closeProbeCastBtn.addEventListener('click', () => {
            setProbeCastModalVisibility(false);
        });
    }
    if (probeCastCloseBtn) {
        probeCastCloseBtn.addEventListener('click', () => {
            setProbeCastModalVisibility(false);
        });
    }
    if (probeCastApplyBtn) {
        probeCastApplyBtn.addEventListener('click', () => {
            applyProbeCast();
        });
    }
    if (probeCastAllBtn) {
        probeCastAllBtn.addEventListener('click', () => {
            setProbeCastSelection(getProbeCastChannelCatalog().map((channel) => channel.id));
        });
    }
    if (probeCastNoneBtn) {
        probeCastNoneBtn.addEventListener('click', () => {
            setProbeCastSelection([]);
        });
    }
    if (probeCastCurrentBtn) {
        probeCastCurrentBtn.addEventListener('click', () => {
            setProbeCastSelection([getSelectedProbeChannelId()]);
        });
    }
    if (probeCastChannelList) {
        probeCastChannelList.addEventListener('change', () => {
            probeCastSelectedChannels = new Set(selectedProbeCastChannelIds());
            updateProbeCastSelectionMeta();
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
            invalidateArchiveResultContext();
            archiveDetectionsOffset = 0;
            archiveDetectionsHasMore = false;
            updateArchiveDetectionsNav();
            refreshArchiveProbeFilter();
        });
    }
    if (archiveSourceFilter) {
        archiveSourceFilter.addEventListener('change', () => {
            invalidateArchiveResultContext();
            archiveDetectionsOffset = 0;
            archiveDetectionsHasMore = false;
            syncArchiveDiagnosticSourceVisibility();
            if (archiveProbeFilter) archiveProbeFilter.value = '';
            updateArchiveDetectionsNav();
            refreshArchiveProbeFilter();
        });
    }
    if (archiveProbeFilter) {
        archiveProbeFilter.addEventListener('change', () => {
            invalidateArchiveResultContext();
            archiveDetectionsOffset = 0;
            archiveDetectionsHasMore = false;
            updateArchiveDetectionsNav();
        });
    }
    if (archiveTimeFilter) {
        archiveTimeFilter.addEventListener('change', () => {
            invalidateArchiveResultContext();
            archiveDetectionsOffset = 0;
            archiveDetectionsHasMore = false;
            updateArchiveDetectionsNav();
        });
    }
    [archiveFromTimeInput, archiveToTimeInput].forEach((input) => {
        if (!input) return;
        input.addEventListener('change', () => {
            invalidateArchiveResultContext();
            archiveDetectionsOffset = 0;
            archiveDetectionsHasMore = false;
            updateArchiveDetectionsNav();
        });
    });
    if (archiveDetectionsLimit) {
        archiveDetectionsLimit.addEventListener('change', () => {
            invalidateArchiveResultContext();
            archiveDetectionsOffset = 0;
            archiveDetectionsHasMore = false;
            updateArchiveDetectionsNav();
        });
    }
    if (archiveScoreThresholdInput) {
        archiveScoreThresholdInput.addEventListener('input', () => {
            setArchiveScoreThresholdFromInput(archiveScoreThresholdInput.value);
        });
        setArchiveScoreThresholdFromInput(archiveScoreThresholdInput.value);
    } else {
        updateArchiveThresholdUi();
    }
    if (searchScopeSelect) {
        searchScopeSelect.value = 'detections';
        searchScopeSelect.addEventListener('change', () => {
            invalidateArchiveResultContext();
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
            const error = new Error(message);
            error.status = response.status;
            error.payload = data;
            throw error;
        }
        return data;
    }

    function activeFolderPath() {
        return folderInput ? folderInput.value.trim() : '';
    }
    
    // Index folder
    if (indexBtn) indexBtn.addEventListener('click', async () => {
        const folder = activeFolderPath();
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
        const folder = activeFolderPath();
        const limit = resultLimitSelect.value;
        const sortBy = sortBySelect.value;
        const detectionsScope = isDetectionsScope();
        
        if (!query || (!detectionsScope && !folder)) return;
        archiveLastQueryText = query;
        
        const requestContext = beginArchiveEvidenceRequest(searchBtn);
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
                    signal: requestContext.controller.signal,
                });
            } else {
                response = await fetch('/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ folder, query, limit, sort_by: sortBy }),
                    signal: requestContext.controller.signal,
                });
            }
            
            const data = await parseApiJson(response, 'Text search failed');
            if (!isCurrentArchiveEvidenceRequest(requestContext)) return;
            
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
                archiveRenderedResults = [];
                refreshArchiveScoreScale(archiveRenderedResults);
                applyArchiveScoreThreshold();
                resultsContainer.innerHTML = '<div class="loading">No results found</div>';
                renderArchiveInspectorEmpty('No results found for this query.');
            }
        } catch (error) {
            if ((error && error.name === 'AbortError') || !isCurrentArchiveEvidenceRequest(requestContext)) return;
            resultsContainer.innerHTML = '<div class="loading">Error: ' + error.message + '</div>';
            renderArchiveInspectorEmpty(`Search error: ${error.message}`);
        } finally {
            finishArchiveEvidenceRequest(requestContext);
        }
    });
    
    // Image search
    imageSearchBtn.addEventListener('click', async () => {
        setMode('archive');
        const folder = activeFolderPath();
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
        archiveLastQueryText = `Image query: ${file.name || 'uploaded reference image'}`;
        
        const requestContext = beginArchiveEvidenceRequest(imageSearchBtn);
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
                body: formData,
                signal: requestContext.controller.signal,
            });
            
            const data = await parseApiJson(response, 'Image search failed');
            if (!isCurrentArchiveEvidenceRequest(requestContext)) return;
            
            if (data.results && data.results.length > 0) {
                const renderedResults = detectionsScope
                    ? decorateDetectionSearchResults(data.results, data.mode_used, data.mode_requested)
                    : data.results;
                displayResults(renderedResults);
            } else {
                archiveRenderedResults = [];
                refreshArchiveScoreScale(archiveRenderedResults);
                applyArchiveScoreThreshold();
                resultsContainer.innerHTML = '<div class="loading">No results found</div>';
                renderArchiveInspectorEmpty('No visual matches found for this reference image.');
            }
        } catch (error) {
            if ((error && error.name === 'AbortError') || !isCurrentArchiveEvidenceRequest(requestContext)) return;
            resultsContainer.innerHTML = '<div class="loading">Error: ' + error.message + '</div>';
            renderArchiveInspectorEmpty(`Image search error: ${error.message}`);
        } finally {
            finishArchiveEvidenceRequest(requestContext);
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

    function selectedOfflineMediaFile() {
        if (!videoUploadInput || !videoUploadInput.files || !videoUploadInput.files.length) return null;
        return videoUploadInput.files[0] || null;
    }

    function offlineMediaKind(file) {
        if (!file) return 'path';
        const type = String(file.type || '').toLowerCase();
        const name = String(file.name || '').toLowerCase();
        if (type.startsWith('image/') || /\.(jpe?g|png|bmp|webp)$/.test(name)) return 'image';
        if (type.startsWith('video/') || /\.(mp4|mov|m4v|avi|mkv|webm)$/.test(name)) return 'video';
        return 'unknown';
    }

    function setVideoUploadName(file) {
        if (!videoUploadName) return;
        if (!file) {
            videoUploadName.textContent = 'No file selected';
            videoUploadName.classList.add('is-hidden');
            return;
        }
        videoUploadName.textContent = file.name || 'Selected file';
        videoUploadName.classList.remove('is-hidden');
    }

    function hideOfflineSummaryOutput() {
        if (!videoOutput) return;
        videoOutput.classList.add('is-hidden');
        videoOutput.style.display = '';
        videoOutput.innerHTML = '';
    }

    function showOfflineSummaryOutput(content, plainText = false) {
        if (!videoOutput) return;
        if (plainText) {
            videoOutput.textContent = content;
        } else {
            videoOutput.innerHTML = content;
        }
        videoOutput.style.display = '';
        videoOutput.classList.remove('is-hidden');
    }

    function renderOfflineDiagnostics(diag) {
        if (!diag || typeof diag !== 'object') return '';
        const rows = Object.entries(diag)
            .filter(([, value]) => value !== undefined && value !== null && value !== '')
            .map(([key, value]) => `<div><span>${escapeHtml(key)}</span>: <code>${escapeHtml(String(value))}</code></div>`);
        if (!rows.length) return '';
        return `<div class="offline-diagnostics"><h4>Diagnostics</h4>${rows.join('')}</div>`;
    }

    if (videoUploadInput) {
        videoUploadInput.addEventListener('change', () => {
            const file = selectedOfflineMediaFile();
            setVideoUploadName(file);
            if (file && videoPathInput) {
                videoPathInput.value = '';
            }
        });
    }

    if (videoPathInput && videoUploadInput) {
        videoPathInput.addEventListener('input', () => {
            if (!videoPathInput.value.trim()) return;
            if (videoUploadInput.value) {
                videoUploadInput.value = '';
                setVideoUploadName(null);
            }
        });
    }

    async function runVideoUnderstanding() {
        const videoPath = videoPathInput.value.trim();
        const uploadFile = selectedOfflineMediaFile();
        const uploadKind = offlineMediaKind(uploadFile);
        const frameCount = parseInt(videoFrameCount.value, 10) || 16;
        const sampleFpsValue = Number.parseFloat(videoSampleFpsInput.value);
        const prompt = videoPromptInput.value.trim();
        const modelId = videoModelInput ? videoModelInput.value.trim() : '';

        if (!uploadFile && !videoPath) {
            videoStatus.textContent = 'Choose a media file or provide a server path.';
            videoStatus.className = 'video-status error';
            return;
        }

        if (uploadFile && uploadKind === 'unknown') {
            videoStatus.textContent = 'Unsupported media file type.';
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
        videoStatus.dataset.base = uploadFile
            ? `Uploading ${uploadKind === 'image' ? 'image' : 'video'} and querying the model...`
            : 'Sampling frames and querying the model...';
        videoStatus.textContent = videoStatus.dataset.base;
        videoStatus.className = 'video-status';
        hideOfflineSummaryOutput();
        renderVideoFrames([]);
        startVideoTimer();

        try {
            let response;
            if (uploadFile) {
                const formData = new FormData();
                formData.append(uploadKind === 'image' ? 'image' : 'video', uploadFile);
                formData.append('prompt', prompt);
                formData.append('frame_count', String(frameCount));
                if (modelId) formData.append('model', modelId);
                if (Number.isFinite(sampleFpsValue) && sampleFpsValue > 0) {
                    formData.append('sample_fps', String(sampleFpsValue));
                }
                response = await fetch(uploadKind === 'image' ? '/describe_image' : '/video_understanding', {
                    method: 'POST',
                    body: formData,
                });
            } else {
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
                response = await fetch('/video_understanding', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
            }
            const data = await response.json();
            if (!response.ok || data.error) {
                videoStatus.dataset.base = data.error || 'Video understanding request failed.';
                videoStatus.textContent = videoStatus.dataset.base;
                videoStatus.className = 'video-status error';
                const diagHtml = renderOfflineDiagnostics(data.diagnostics);
                if (diagHtml) {
                    showOfflineSummaryOutput(diagHtml);
                }
                stopVideoTimer();
                return;
            }
            const durationLabel = typeof data.duration_sec === 'number' ? ` · Duration: ${formatDuration(data.duration_sec)}` : '';
            const framesSent = uploadKind === 'image' ? 1 : ((data.frames || []).length || frameCount);
            const sourceLabel = uploadFile ? ` · Uploaded: ${data.filename || uploadFile.name || 'file'}` : '';
            videoStatus.dataset.base = `Model: ${data.model || modelId || 'LM Studio'} · Frames sent: ${framesSent}${durationLabel}${sourceLabel}`;
            videoStatus.textContent = videoStatus.dataset.base;
            if (data.summary) {
                showOfflineSummaryOutput(`${renderMarkdown(data.summary)}${renderOfflineDiagnostics(data.diagnostics)}`);
                lastSummaryText = data.summary;
                lastSummaryTarget = null;
                saveSummaryBtn.style.display = 'none';
            } else {
                showOfflineSummaryOutput(`(No summary returned)${renderOfflineDiagnostics(data.diagnostics)}`);
                lastSummaryText = '';
                lastSummaryTarget = null;
                saveSummaryBtn.style.display = 'none';
            }
            if (uploadKind === 'image' && data.thumbnail) {
                renderVideoFrames([{ index: 0, time_sec: 0, thumbnail: data.thumbnail }]);
            } else {
                renderVideoFrames(data.frames || []);
            }
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
        const folder = activeFolderPath();
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
    if (showCommentedBtn) showCommentedBtn.addEventListener('click', async () => {
        const folder = activeFolderPath();
        
        if (!folder) {
            alert('Please enter a folder path first');
            return;
        }
        setMode('archive');
        
        const requestContext = beginArchiveEvidenceRequest(showCommentedBtn);
        resultsContainer.innerHTML = '<div class="loading"><div class="spinner"></div> Loading commented images...</div>';
        renderArchiveInspectorEmpty('Loading commented images...');
        
        try {
            const response = await fetch('/commented_images', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ folder }),
                signal: requestContext.controller.signal,
            });
            
            const data = await parseApiJson(response, 'Loading commented images failed');
            if (!isCurrentArchiveEvidenceRequest(requestContext)) return;
            
            if (data.results && data.results.length > 0) {
                displayCommentedResults(data.results);
            } else {
                archiveRenderedResults = [];
                refreshArchiveScoreScale(archiveRenderedResults);
                applyArchiveScoreThreshold();
                resultsContainer.innerHTML = '<div class="loading">No commented images found</div>';
                renderArchiveInspectorEmpty('No commented images found for the current archive.');
            }
        } catch (error) {
            if ((error && error.name === 'AbortError') || !isCurrentArchiveEvidenceRequest(requestContext)) return;
            resultsContainer.innerHTML = '<div class="loading">Error: ' + error.message + '</div>';
            renderArchiveInspectorEmpty(`Commented image load failed: ${error.message}`);
        } finally {
            finishArchiveEvidenceRequest(requestContext);
        }
    });
    
    function renderArchiveInspectorEmpty(message = 'Select a result to inspect the full image, metrics, comments, and segmentation tools.') {
        if (resultsContainer) {
            resultsContainer.classList.remove('results-grid--detections');
        }
        activeArchiveInspectorIndex = -1;
        archiveInspectorRenderKey = '';
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

    function archiveResultIdentityKey(result, index) {
        if (!result) return `empty:${index}`;
        const payload = archiveResultPayload(result);
        return [
            archiveRenderedCommented ? 'commented' : 'result',
            index,
            result.id,
            result.detection_id,
            result.path,
            result.filename,
            result.timestamp_ms,
            result.score,
            result.is_detection ? 'detection' : 'image',
            payload && payload.frame_id,
            payload && payload.batch_start_ms,
            payload && payload.batch_end_ms,
        ].map((value) => String(value ?? '')).join('|');
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

    function archiveReviewFrameNumber(result) {
        const payload = archiveResultPayload(result);
        const value = Number(payload.frame_index ?? payload.anchor_frame_index);
        return Number.isFinite(value) ? value : null;
    }

    function archiveReviewBatchKey(result) {
        if (!result || !isVideoArchiveResult(result)) return '';
        const payload = archiveResultPayload(result);
        const channelId = Number(result.channel_id);
        const startMs = Number(payload.batch_start_ms ?? result.batch_start_ms);
        const endMs = Number(payload.batch_end_ms ?? result.batch_end_ms);
        const runId = String(payload.run_id || '').trim();
        if (!Number.isFinite(channelId) || !Number.isFinite(startMs) || !Number.isFinite(endMs)) return '';
        return `${channelId}:${runId}:${Math.trunc(startMs)}:${Math.trunc(endMs)}`;
    }

    function archiveReviewFrameIdentity(result) {
        if (!result) return '';
        const payload = archiveResultPayload(result);
        const batchKey = archiveReviewBatchKey(result);
        const frameNo = archiveReviewFrameNumber(result);
        const ts = Number(archiveFrameTimestampMs(result));
        if (batchKey && (frameNo !== null || Number.isFinite(ts))) {
            return [
                batchKey,
                frameNo !== null ? `f:${frameNo}` : '',
                Number.isFinite(ts) ? `t:${Math.trunc(ts)}` : '',
            ].filter(Boolean).join('|');
        }
        const id = result.detection_id ?? result.id;
        if (id !== undefined && id !== null && String(id).trim()) {
            return `id:${String(id)}`;
        }
        const role = String(payload.anchor_role || payload.anchor_source_role || result.source || '').trim();
        return [
            batchKey,
            Number.isFinite(frameNo) ? `f:${frameNo}` : '',
            Number.isFinite(ts) ? `t:${Math.trunc(ts)}` : '',
            role,
        ].filter(Boolean).join('|');
    }

    function archiveReviewSortFrames(frames) {
        return [...frames].sort((a, b) => {
            const aFrame = archiveReviewFrameNumber(a);
            const bFrame = archiveReviewFrameNumber(b);
            if (aFrame !== null && bFrame !== null && aFrame !== bFrame) return aFrame - bFrame;
            const aTs = Number(archiveFrameTimestampMs(a));
            const bTs = Number(archiveFrameTimestampMs(b));
            if (Number.isFinite(aTs) && Number.isFinite(bTs) && aTs !== bTs) return aTs - bTs;
            return String(archiveReviewFrameIdentity(a)).localeCompare(String(archiveReviewFrameIdentity(b)));
        });
    }

    function archiveReviewMergeFrames(frames) {
        const unique = new Map();
        (frames || []).forEach((frame) => {
            if (!frame || !archiveResultHasImage(frame)) return;
            const key = archiveReviewFrameIdentity(frame);
            if (!key || unique.has(key)) return;
            unique.set(key, frame);
        });
        return archiveReviewSortFrames(Array.from(unique.values()));
    }

    function archiveReviewLocalBatchFrames(result) {
        const frames = [];
        if (result && archiveResultHasImage(result)) frames.push(result);
        const sourceKey = archiveReviewBatchKey(result);
        if (sourceKey && Array.isArray(archiveRenderedResults)) {
            archiveRenderedResults.forEach((candidate) => {
                if (!candidate || !archiveResultHasImage(candidate)) return;
                if (!isVideoArchiveResult(candidate)) return;
                if (archiveReviewBatchKey(candidate) === sourceKey) {
                    frames.push(candidate);
                }
            });
        }
        return archiveReviewMergeFrames(frames);
    }

    function archiveReviewActiveFrameIndex(frames, result) {
        const targetIdentity = archiveReviewFrameIdentity(result);
        const exact = frames.findIndex((frame) => archiveReviewFrameIdentity(frame) === targetIdentity);
        if (exact >= 0) return exact;
        const targetFrame = archiveReviewFrameNumber(result);
        if (targetFrame !== null) {
            const byFrame = frames.findIndex((frame) => archiveReviewFrameNumber(frame) === targetFrame);
            if (byFrame >= 0) return byFrame;
        }
        const targetTs = Number(archiveFrameTimestampMs(result));
        if (Number.isFinite(targetTs) && frames.length) {
            let bestIndex = 0;
            let bestDistance = Infinity;
            frames.forEach((frame, idx) => {
                const ts = Number(archiveFrameTimestampMs(frame));
                if (!Number.isFinite(ts)) return;
                const distance = Math.abs(ts - targetTs);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestIndex = idx;
                }
            });
            return bestIndex;
        }
        return 0;
    }

    function archiveReviewRenderFrameNav() {
        const context = archiveReviewContext;
        const frames = Array.isArray(context?.frames) ? context.frames : [];
        const activeIndex = Number.isFinite(context?.activeFrameIndex) ? context.activeFrameIndex : 0;
        const activeResult = frames[activeIndex] || context?.result || null;
        if (archiveReviewFrameRole) {
            archiveReviewFrameRole.textContent = archiveFrameRoleText(activeResult);
        }
        if (archiveReviewTimestamp) {
            archiveReviewTimestamp.textContent = `Timestamp: ${formatArchiveTimestamp(archiveFrameTimestampMs(activeResult))}`;
        }
        const previousFrame = activeIndex > 0 ? frames[activeIndex - 1] : null;
        const nextFrame = activeIndex < frames.length - 1 ? frames[activeIndex + 1] : null;
        if (archiveReviewPrevFrameBtn) {
            const previousNumber = previousFrame ? archiveReviewFrameNumber(previousFrame) : null;
            archiveReviewPrevFrameBtn.disabled = !previousFrame;
            archiveReviewPrevFrameBtn.innerHTML = previousNumber !== null
                ? `<span>${escapeHtml(String(previousNumber))}</span><span>&lsaquo;</span>`
                : '&lsaquo;';
        }
        if (archiveReviewNextFrameBtn) {
            const nextNumber = nextFrame ? archiveReviewFrameNumber(nextFrame) : null;
            archiveReviewNextFrameBtn.disabled = !nextFrame;
            archiveReviewNextFrameBtn.innerHTML = nextNumber !== null
                ? `<span>&rsaquo;</span><span>${escapeHtml(String(nextNumber))}</span>`
                : '&rsaquo;';
        }
    }

    function archiveReviewRenderFilmstrip() {
        if (!archiveReviewFilmstrip) return;
        const context = archiveReviewContext;
        const frames = Array.isArray(context?.frames) ? context.frames : [];
        const activeIndex = Number.isFinite(context?.activeFrameIndex) ? context.activeFrameIndex : 0;
        if (context?.framesLoading) {
            archiveReviewFilmstrip.innerHTML = '<div class="archive-review-filmstrip-status">Loading batch frames...</div>';
            return;
        }
        if (context?.framesError && frames.length <= 1) {
            archiveReviewFilmstrip.innerHTML = `<div class="archive-review-filmstrip-status is-error">${escapeHtml(context.framesError)}</div>`;
            return;
        }
        if (frames.length <= 1) {
            archiveReviewFilmstrip.innerHTML = '<div class="archive-review-filmstrip-status">No neighboring batch frames returned.</div>';
            return;
        }
        archiveReviewFilmstrip.innerHTML = frames.map((frame, idx) => {
            const imageSrc = archiveResultImageSrc(frame);
            const frameNo = archiveReviewFrameNumber(frame);
            const label = frameNo !== null ? `Frame ${frameNo}` : `Frame ${idx + 1}`;
            const roleText = archiveFrameRoleText(frame).replace(/\s+/g, ' ');
            const framePayload = archiveResultPayload(frame);
            const frameRole = String(framePayload.anchor_role || framePayload.anchor_source_role || '').trim();
            const attentionMarker = isBurstArchiveFrameRole(frameRole)
                ? '<span class="archive-review-strip-attention" aria-label="Burst attention frame">⚡</span>'
                : '';
            const activeClass = idx === activeIndex ? ' is-active' : '';
            return `
                <button class="archive-review-strip-frame${activeClass}" type="button" data-archive-review-frame-index="${idx}" title="${escapeHtml(roleText)}">
                    <img src="${escapeHtml(imageSrc)}" alt="${escapeHtml(label)}" loading="lazy">
                    ${attentionMarker}
                    <span>${escapeHtml(label)}</span>
                </button>
            `;
        }).join('');
        const activeButton = archiveReviewFilmstrip.querySelector('.archive-review-strip-frame.is-active');
        if (activeButton) {
            activeButton.scrollIntoView({ inline: 'center', block: 'nearest', behavior: 'smooth' });
        }
    }

    function archiveReviewRenderActiveFrame() {
        const context = archiveReviewContext;
        if (!context) return;
        const frames = Array.isArray(context.frames) && context.frames.length ? context.frames : [context.baseResult || context.result];
        const activeIndex = Math.max(0, Math.min(frames.length - 1, Number(context.activeFrameIndex) || 0));
        const activeResult = frames[activeIndex] || context.baseResult || context.result;
        context.frames = frames;
        context.activeFrameIndex = activeIndex;
        context.result = activeResult;
        const identity = archiveReviewFrameIdentity(activeResult);
        if (context.mediaIdentity && context.mediaIdentity !== identity) {
            // Stepping to another frame returns to the stored evidence view;
            // archive playback stays operator-initiated per frame.
            cancelArchiveMediaRequest(true);
            context.mediaIdentity = '';
            context.mediaState = 'idle';
        }
        // While media for this exact frame is negotiating or playing, a
        // re-render (for example the batch filmstrip finishing its load) must
        // not clobber the live element with the static thumbnail.
        const mediaActive = ['loading', 'playing'].includes(context.mediaState || '');
        const imageSrc = archiveResultImageSrc(activeResult);
        if (archiveReviewImg && !mediaActive) {
            archiveReviewImg.src = imageSrc || '';
            archiveReviewImg.classList.toggle('is-hidden', !imageSrc);
        }
        if (archiveReviewFrameEmpty && !mediaActive) {
            archiveReviewFrameEmpty.classList.toggle('is-hidden', Boolean(imageSrc));
        }
        archiveReviewRenderFrameNav();
        archiveReviewRenderFilmstrip();
        if (archiveReviewJumpBtn) {
            archiveReviewJumpBtn.disabled = !archiveResultCanOpenVlmFeed(activeResult);
        }
        if (archiveReviewDescribeBtn) {
            archiveReviewDescribeBtn.disabled = !archiveResultHasImage(activeResult);
        }
        if (archiveReviewSimilarBtn) {
            archiveReviewSimilarBtn.disabled = !archiveResultHasImage(activeResult);
        }
        syncArchiveReviewPlayButton(context);
    }

    function archiveResultCanPlayArchiveVideo(result) {
        return isVideoArchiveResult(result) && archiveResultCanOpenVlmFeed(result);
    }

    function syncArchiveReviewPlayButton(context) {
        const ui = ensureArchiveMediaUi();
        if (!ui.retry || !context) return;
        if ((context.mediaState || 'idle') === 'idle') {
            // Archive playback is opt-in: the modal reviews the exact stored
            // evidence frames; recorder video never starts on its own.
            if (ui.status) ui.status.hidden = true;
            ui.retry.textContent = 'Play archive video';
            ui.retry.hidden = !archiveResultCanPlayArchiveVideo(context.result);
        }
    }

    function archiveReviewSetFrame(frameIndex) {
        const context = archiveReviewContext;
        if (!context || !Array.isArray(context.frames) || !context.frames.length) return;
        const nextIndex = Math.max(0, Math.min(context.frames.length - 1, Number(frameIndex) || 0));
        if (nextIndex === context.activeFrameIndex) return;
        context.activeFrameIndex = nextIndex;
        archiveReviewRenderActiveFrame();
    }

    function archiveReviewStepFrame(delta) {
        const context = archiveReviewContext;
        if (!context || !Array.isArray(context.frames) || !context.frames.length) return;
        archiveReviewSetFrame((Number(context.activeFrameIndex) || 0) + delta);
    }

    async function archiveReviewLoadBatchFrames(context) {
        if (!context || !context.baseResult || !isVideoArchiveResult(context.baseResult)) return;
        const payload = archiveResultPayload(context.baseResult);
        const channelId = Number(context.baseResult.channel_id);
        const batchStart = Number(payload.batch_start_ms ?? context.baseResult.batch_start_ms);
        const batchEnd = Number(payload.batch_end_ms ?? context.baseResult.batch_end_ms);
        if (!Number.isFinite(channelId) || channelId <= 0 || !Number.isFinite(batchStart) || !Number.isFinite(batchEnd)) return;
        const requestContext = beginArchiveReviewRequest(context);
        context.framesLoading = true;
        archiveReviewRenderFilmstrip();
        const params = new URLSearchParams();
        params.set('channel_id', String(Math.trunc(channelId)));
        params.set('source', 'vlm_summary');
        params.set('since_ms', String(Math.trunc(Math.min(batchStart, batchEnd))));
        params.set('until_ms', String(Math.trunc(Math.max(batchStart, batchEnd))));
        params.set('limit', '120');
        params.set('offset', '0');
        try {
            const response = await fetch(`/detections/list?${params.toString()}`, {
                signal: requestContext.controller.signal,
            });
            const data = await parseApiJson(response, 'Failed to load batch frames');
            if (!isCurrentArchiveReviewRequest(requestContext)) return;
            const loaded = normalizeDetectionResults(Array.isArray(data.detections) ? data.detections : [])
                .filter((item) => isVideoArchiveResult(item) && archiveResultHasImage(item))
                .filter((item) => {
                    const itemPayload = archiveResultPayload(item);
                    const itemRun = String(itemPayload.run_id || '').trim();
                    const targetRun = String(payload.run_id || '').trim();
                    if (targetRun && itemRun && itemRun !== targetRun) return false;
                    return true;
                });
            const merged = archiveReviewMergeFrames([context.baseResult, ...loaded]);
            context.frames = merged.length ? merged : archiveReviewLocalBatchFrames(context.baseResult);
            context.activeFrameIndex = archiveReviewActiveFrameIndex(context.frames, context.baseResult);
            context.framesError = '';
        } catch (error) {
            if ((error && error.name === 'AbortError') || !isCurrentArchiveReviewRequest(requestContext)) return;
            context.frames = archiveReviewLocalBatchFrames(context.baseResult);
            context.activeFrameIndex = archiveReviewActiveFrameIndex(context.frames, context.baseResult);
            context.framesError = error.message || 'Batch frames unavailable.';
        } finally {
            if (isCurrentArchiveReviewRequest(requestContext)) {
                context.framesLoading = false;
                archiveReviewRenderActiveFrame();
                archiveReviewAbortController = null;
            }
        }
    }

    function closeArchiveReviewModal() {
        if (!archiveReviewModal) return;
        cancelArchiveReviewRequest();
        cancelArchiveMediaRequest(true);
        archiveReviewModal.style.display = 'none';
        archiveReviewContext = null;
        if (archiveReviewImg) archiveReviewImg.src = '';
        if (archiveReviewFilmstrip) archiveReviewFilmstrip.innerHTML = '';
    }

    function openArchiveReviewModal(index, result) {
        if (!archiveReviewModal || !result) return;
        cancelArchiveReviewRequest();
        cancelArchiveMediaRequest(true);
        const frames = archiveReviewLocalBatchFrames(result);
        archiveReviewContext = {
            index,
            baseResult: result,
            result,
            frames,
            activeFrameIndex: archiveReviewActiveFrameIndex(frames, result),
            framesLoading: false,
            framesError: '',
            mediaIdentity: '',
            mediaState: 'idle',
        };
        if (archiveReviewTitle) {
            archiveReviewTitle.textContent = 'Archive research review for video description streams';
        }
        if (archiveReviewQuery) {
            archiveReviewQuery.textContent = `User query: ${archiveReviewQueryText(result)}`;
        }
        if (archiveReviewMatch) {
            archiveReviewMatch.textContent = archiveReviewMatchText(result);
        }
        if (archiveReviewChannel) {
            archiveReviewChannel.textContent = `Channel: ID: ${archiveChannelLabel(result)}`;
        }
        if (archiveReviewSummary) {
            const summary = archiveSummaryText(result);
            const truncated = archiveResultPayload(result).summary_truncated;
            archiveReviewSummary.innerHTML = `${renderMarkdown(summary)}${truncated ? '<div class="archive-review-note">Summary was truncated for archive storage.</div>' : ''}`;
        }
        archiveReviewRenderActiveFrame();
        archiveReviewModal.style.display = 'block';
        void archiveReviewLoadBatchFrames(archiveReviewContext);
    }

    async function copyArchiveReviewSummary() {
        const result = archiveReviewContext && archiveReviewContext.result;
        if (!result) return;
        const text = archiveSummaryText(result);
        try {
            if (navigator.clipboard && window.isSecureContext) {
                await navigator.clipboard.writeText(text);
            } else {
                const textArea = document.createElement('textarea');
                textArea.value = text;
                textArea.style.position = 'fixed';
                textArea.style.left = '-999999px';
                textArea.style.top = '-999999px';
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();
                document.execCommand('copy');
                textArea.remove();
            }
        } catch (error) {
            console.error('Failed to copy summary:', error);
        }
    }

    async function jumpToVideoSummaryFromArchive(result) {
        if (!result || !archiveResultCanOpenVlmFeed(result)) return;
        const channelId = Number(result.channel_id);
        if (!Number.isFinite(channelId)) return;
        const payload = archiveResultPayload(result);
        const summaryWindow = archiveResultSummaryWindow(result);
        closeArchiveReviewModal();
        setMode('video');
        if (luxriotChannelSelect) {
            luxriotChannelSelect.value = String(channelId);
        }
        luxriotActiveChannel = channelId;
        syncLuxriotLiveIntervalInput(channelId);
        updateLuxriotCaptureToggleButton(channelId);
        updateLuxriotStreamContext();
        resetRoadSceneGrounding();
        luxriotSummaryChannel = channelId;
        if (luxriotSummaryChannelSelect) {
            syncLuxriotSummaryChannelSelect();
            luxriotSummaryChannelSelect.value = String(channelId);
        }
        setSummaryBaseLevel('L0');
        luxriotSummaryFollowLive = false;
        luxriotSummaryAutoRefresh = false;
        luxriotSummaryRunFilter = String(payload.run_id || '').trim() || 'all';
        if (luxriotSummaryRunSelect) {
            const hasRunValue = Array.from(luxriotSummaryRunSelect.options || [])
                .some((opt) => String(opt.value) === luxriotSummaryRunFilter);
            luxriotSummaryRunSelect.value = hasRunValue ? luxriotSummaryRunFilter : 'all';
        }
        luxriotSummaryRangePreset = 'custom';
        luxriotSummaryFromTs = summaryWindow.startMs / 1000;
        luxriotSummaryToTs = summaryWindow.endMs / 1000;
        if (luxriotSummaryRangeSelect) {
            luxriotSummaryRangeSelect.value = 'custom';
        }
        if (luxriotSummaryFromInput) {
            luxriotSummaryFromInput.value = formatSummaryDatetimeInput(luxriotSummaryFromTs);
        }
        if (luxriotSummaryToInput) {
            luxriotSummaryToInput.value = formatSummaryDatetimeInput(luxriotSummaryToTs);
        }
        resetSummaryArchivePaging();
        syncSummaryRangeUI();
        setSummaryUnread(0);
        setLuxriotStatus(`Opening VLM feed around ${formatArchiveTimestamp(summaryWindow.targetMs)}...`);
        const refreshed = await refreshLuxriotSummaryView(channelId, true, false);
        const scrolled = scrollLuxriotSummaryToTimestamp(summaryWindow.targetMs);
        if (!scrolled && refreshed === false) {
            window.setTimeout(() => {
                scrollLuxriotSummaryToTimestamp(summaryWindow.targetMs);
            }, 700);
        } else if (!scrolled) {
            setLuxriotStatus(`VLM feed opened near ${formatArchiveTimestamp(summaryWindow.targetMs)}; no matching summary card was returned.`, true);
        }
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

        const nextRenderKey = archiveResultIdentityKey(result, index);
        if (
            activeArchiveInspectorIndex === index
            && archiveInspectorRenderKey === nextRenderKey
            && !archiveInspectorBody.classList.contains('is-hidden')
            && archiveInspectorBody.querySelector('.result-item')
        ) {
            highlightActiveArchiveResultCard(index);
            return;
        }

        activeArchiveInspectorIndex = index;
        archiveInspectorRenderKey = nextRenderKey;
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
        const activeFolder = activeFolderPath();
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
                commentsContainer.innerHTML = '<div class="no-comments">This detection can be described; inline comments are unavailable for archive-only results.</div>';
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
        const hasImage = archiveResultHasImage(result);
        const isDetectionResult = Boolean(result && result.is_detection);
        const isVideoResult = isVideoArchiveResult(result);
        const showFilenameRow = !isDetectionResult;
        const activeFolder = activeFolderPath();
        const canUseFolderComments = Boolean(hasPath && activeFolder && rawPath.startsWith(activeFolder));
        const safePath = escapeHtml(rawPath);
        const fallbackSvg = encodeURIComponent(
            '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="260">' +
            '<rect width="100%" height="100%" fill="#1f2026"/>' +
            '<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#9aa0ad" font-size="18">No thumbnail</text>' +
            '</svg>'
        );
        const imageSrc = archiveResultImageSrc(result);
        const thumbnailSrc = imageSrc || `data:image/svg+xml;charset=utf-8,${fallbackSvg}`;
        const detailImageSrc = imageSrc || thumbnailSrc;
        const overlayIcon = variant === 'detail'
            ? '<path d="M240-240v-200h80v120h120v80H240Zm400-400v-80h80v200H520v-80h120v-40Z"/>'
            : '<path d="M240-240v-240h72v168h168v72H240Zm408-240v-168H480v-72h240v240h-72Z"/>';
        const reviewButtonMarkup = isVideoResult ? `
            <button class="action-icon archive-review-icon" data-index="${index}" title="Review VLM frame">
                <svg xmlns="http://www.w3.org/2000/svg" height="18px" viewBox="0 -960 960 960" width="18px" fill="#e3e3e3">
                    <path d="M480-320q75 0 127.5-52.5T660-500q0-75-52.5-127.5T480-680q-75 0-127.5 52.5T300-500q0 75 52.5 127.5T480-320Zm0-72q-45 0-76.5-31.5T372-500q0-45 31.5-76.5T480-608q45 0 76.5 31.5T588-500q0 45-31.5 76.5T480-392Zm0 192q-146 0-266-81.5T40-500q54-137 174-218.5T480-800q146 0 266 81.5T920-500q-54 137-174 218.5T480-200Zm0-300Zm0 220q113 0 207.5-59.5T832-500q-50-101-144.5-160.5T480-720q-113 0-207.5 59.5T128-500q50 101 144.5 160.5T480-280Z"/>
                </svg>
            </button>
        ` : '';

        if (variant === 'card') {
            return `
                <div class="image-container">
                    <img src="${thumbnailSrc}" class="thumbnail" alt="" />
                    <div class="image-overlay">
                        ${hasImage ? `
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
                        ${reviewButtonMarkup}
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
        const commentsPanelMarkup = hasImage ? `
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
                        <div class="no-comments">Comments are unavailable for archive-only results.</div>
                    </div>
                `}
            </div>
        ` : '';

        return `
            <div class="image-container">
                <img src="${detailImageSrc}" class="thumbnail" alt="" />
                <div class="image-overlay">
                    ${hasImage ? `
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
                    ${reviewButtonMarkup}
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
        const hasImage = archiveResultHasImage(result);

        const expandCollapseIcon = item.querySelector('.expand-collapse-icon');
        if (expandCollapseIcon && hasImage) {
            expandCollapseIcon.addEventListener('click', (e) => {
                e.stopPropagation();
                if (variant === 'detail') {
                    openImageLightbox(archiveResultImageSrc(result), result.filename || result.path || 'Preview');
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
            if (hasImage) {
                findSimilarIcon.addEventListener('click', (e) => {
                    e.stopPropagation();
                    findSimilarImages(result.path || '', result);
                });
            } else {
                findSimilarIcon.style.display = 'none';
            }
        }

        const describeIcon = item.querySelector('.describe-icon');
        if (describeIcon) {
            if (hasImage) {
                describeIcon.addEventListener('click', (e) => {
                    e.stopPropagation();
                    describeImageWithLM(index, result.path || '', item, result);
                });
            } else {
                describeIcon.style.display = 'none';
            }
        }

        const archiveReviewIcon = item.querySelector('.archive-review-icon');
        if (archiveReviewIcon) {
            if (isVideoArchiveResult(result)) {
                archiveReviewIcon.addEventListener('click', (e) => {
                    e.stopPropagation();
                    openArchiveReviewModal(index, result);
                });
            } else {
                archiveReviewIcon.style.display = 'none';
            }
        }

        const saveBtn = item.querySelector(`#save-btn-${index}`);
        const commentInput = item.querySelector(`#comment-input-${index}`);
        const activeFolder = activeFolderPath();
        const canUseFolderComments = Boolean(result.path && activeFolder && String(result.path).startsWith(activeFolder));

        if (saveBtn) {
            if (canUseFolderComments || (result.path && !result.is_detection)) {
                saveBtn.addEventListener('click', () => {
                    saveComment(index, result.path, activeFolderPath(), commentInput.value.trim());
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
        if (img && hasImage && variant === 'detail') {
            if (!result.is_detection && result.path) {
                img.addEventListener('click', (e) => {
                    if (segmentsEnabledInput.checked) {
                        handleSegmentClick(e, result, index, item);
                        return;
                    }
                    openImageLightbox(archiveResultImageSrc(result), result.filename || result.path || 'Preview');
                });
            } else {
                img.addEventListener('click', () => {
                    openImageLightbox(archiveResultImageSrc(result), result.filename || result.path || 'Preview');
                });
            }
        }
    }

    function replaceChildrenStable(container, fragment) {
        if (!container) return;
        if (typeof container.replaceChildren === 'function') {
            container.replaceChildren(fragment);
            return;
        }
        container.innerHTML = '';
        container.appendChild(fragment);
    }

    // Display results
    function displayResults(results) {
        segmentContextByIndex = {};
        archiveRenderedResults = Array.isArray(results) ? results : [];
        archiveRenderedCommented = false;
        archiveInspectorRenderKey = '';
        refreshArchiveScoreScale(archiveRenderedResults);
        syncArchiveResultsLayout(archiveRenderedResults);

        const fragment = document.createDocumentFragment();
        archiveRenderedResults.forEach((result, index) => {
            const item = document.createElement('div');
            item.className = 'result-item';
            if (result && result.is_detection) {
                item.classList.add('result-item--detection-card');
            }
            item.dataset.resultIndex = index;
            item.innerHTML = generateResultItemHTML(result, index, false, 'card');
            
            setupResultItemEventHandlers(item, result, index, { variant: 'card' });
            fragment.appendChild(item);
        });
        replaceChildrenStable(resultsContainer, fragment);

        if (archiveRenderedResults.length) {
            showArchiveInspector(0);
        } else {
            renderArchiveInspectorEmpty('Run a text search, image search, or load detections to populate the inspector.');
        }
        applyArchiveScoreThreshold({ selectFirstVisible: true });
    }
    
    // Display commented results (similar to displayResults but with comment info)
    function displayCommentedResults(results) {
        segmentContextByIndex = {};
        archiveRenderedResults = Array.isArray(results) ? results : [];
        archiveRenderedCommented = true;
        archiveInspectorRenderKey = '';
        refreshArchiveScoreScale(archiveRenderedResults);
        syncArchiveResultsLayout(archiveRenderedResults);

        const fragment = document.createDocumentFragment();
        archiveRenderedResults.forEach((result, index) => {
            const item = document.createElement('div');
            item.className = 'result-item';
            if (result && result.is_detection) {
                item.classList.add('result-item--detection-card');
            }
            item.dataset.resultIndex = index;
            item.innerHTML = generateResultItemHTML(result, index, true, 'card');
            
            setupResultItemEventHandlers(item, result, index, { variant: 'card' });
            fragment.appendChild(item);
        });
        replaceChildrenStable(resultsContainer, fragment);

        if (archiveRenderedResults.length) {
            showArchiveInspector(0);
        } else {
            renderArchiveInspectorEmpty('No commented images found for the current archive.');
        }
        applyArchiveScoreThreshold({ selectFirstVisible: true });
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
        const folder = activeFolderPath();
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
        const folder = activeFolderPath();
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

        const folder = activeFolderPath();
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
            const activeFolder = activeFolderPath();
            if (activeFolder && String(imagePath || '').startsWith(activeFolder)) {
                params.set('folder', activeFolder);
                return `/image?${params.toString()}`;
            }
            return `/detections/image?${params.toString()}`;
        }
        const activeFolder = activeFolderPath();
        if (activeFolder) {
            params.set('folder', activeFolder);
        }
        return `/image?${params.toString()}`;
    }

    async function findSimilarImages(imagePath, result = null) {
        const folder = activeFolderPath();
        const limit = resultLimitSelect.value;
        const sortBy = sortBySelect.value;
        const detectionResult = Boolean(result && result.is_detection);
        const hasArchiveImage = archiveResultHasImage(result);
        
        if (!detectionResult && !folder) {
            alert('Please enter a folder path first');
            return;
        }
        if (!hasArchiveImage && !imagePath) {
            alert('No image is available for similarity search.');
            return;
        }
        
        indexStatus.textContent = 'Finding similar images...';
        indexStatus.className = 'status';
        const requestContext = beginArchiveEvidenceRequest();
        
        try {
            const shouldUseStoredBlob = hasArchiveImage && (!imagePath || isVideoArchiveResult(result));
            const imageBlob = shouldUseStoredBlob
                ? await archiveResultImageBlob(result, requestContext.controller.signal)
                : await (async () => {
                    const imageResponse = await fetch(buildImageFetchUrl(imagePath, result), {
                        signal: requestContext.controller.signal,
                    });
                    if (!imageResponse.ok) {
                        if (hasArchiveImage) {
                            return archiveResultImageBlob(result, requestContext.controller.signal);
                        }
                        throw new Error('Failed to load image file');
                    }
                    return imageResponse.blob();
                })();
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
                    body: formData,
                    signal: requestContext.controller.signal,
                });
                const data = await parseApiJson(response, 'Detection image search failed');
                if (!isCurrentArchiveEvidenceRequest(requestContext)) return;
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
                body: formData,
                signal: requestContext.controller.signal,
            });
            const data = await parseApiJson(response, 'Image search failed');
            if (!isCurrentArchiveEvidenceRequest(requestContext)) return;
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
            if ((error && error.name === 'AbortError') || !isCurrentArchiveEvidenceRequest(requestContext)) return;
            console.error('Find similar error:', error);
            indexStatus.textContent = 'Error finding similar images: ' + error.message;
            indexStatus.className = 'status error';
        } finally {
            finishArchiveEvidenceRequest(requestContext);
        }
    }

    async function describeImageWithLM(index, imagePath, item = null, result = null) {
        const hasArchiveImage = archiveResultHasImage(result);
        if (!imagePath && !hasArchiveImage) {
            alert('No image is available for this result.');
            return;
        }
        const detectionResult = Boolean(result && result.is_detection);
        const folder = activeFolderPath();
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
            let response;
            const shouldUploadArchiveBlob = hasArchiveImage && (!imagePath || isVideoArchiveResult(result));
            if (shouldUploadArchiveBlob) {
                const formData = new FormData();
                formData.append('image', await archiveResultImageBlob(result), 'archive_frame.jpg');
                formData.append('prompt', prompt);
                if (modelId) formData.append('model', modelId);
                response = await fetch('/describe_image', {
                    method: 'POST',
                    body: formData,
                });
            } else {
                response = await fetch('/describe_image', {
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
            }
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
        setArchiveDetectionsMeta('Archive filters unavailable. Start video descriptions or CLIP probes to populate archive frames.');
    });
    
    // Enter key support
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') searchBtn.click();
    });
    
    if (folderInput && indexBtn) {
        folderInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') indexBtn.click();
        });
    }
    
    // Check index on folder change
    if (folderInput) {
        folderInput.addEventListener('blur', async () => {
            const folder = activeFolderPath();
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
    }

    // =====================================================================
    // AGENT TAB
    // =====================================================================
    (function() {
        const AGENT_LS_SESSION = 'evs_agent_session_id';
        const AGENT_EVIDENCE_ID_LIMIT = 12;
        let _agentInitDone = false;
        let _agentCurrentSession = null;    // session_id string or null
        let _agentStreaming = false;
        let _agentPendingBubble = null;     // { el, bodyEl } for the current streaming bubble
        let _agentPendingImageB64 = null;   // base64 string of attached image
        let _agentContextActive = false;
        let _agentContextTimer = null;
        let _agentContextRequestGeneration = 0;
        let _agentContextAbortController = null;

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
                <div class="agent-msg-welcome-sub">Ask me about video summaries, VLM alerts, coverage gaps, archive evidence, and live frames. I use probes as secondary semantic signals when the task needs them.</div>
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
            const bubble = { el: div, bodyEl, textEl, traceEl: null, actionsEl: null, actionCount: 0, text: text || '' };
            syncAgentEvidenceIdCard(bubble);
            scrollToBottom(true);
            return bubble;
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
            traceEl.open = false;
            traceEl.hidden = true;
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
            const bubble = { el: div, bodyEl, textEl, traceEl, actionsEl, actionCount: 0, text: '', currentToolName: '' };
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
            syncAgentEvidenceIdCard(bubble);
            scrollToBottom(stickToBottom);
        }

        function appendActionCard(bubble, name, result) {
            const stickToBottom = isAgentNearBottom();
            const card = isStandaloneProbeApprovalResult(name, result)
                ? buildAgentProbeApprovalCard(name, result)
                : buildActionCard(name, result);
            if (!card) return;
            const standaloneApproval = Boolean(
                (card.dataset && card.dataset.agentStandaloneApproval === 'true')
                || card.querySelector('.agent-approval-footer')
            );
            if (standaloneApproval && bubble.bodyEl) {
                card.dataset.agentStandaloneApproval = 'true';
                if (card.classList.contains('agent-action-card')) {
                    card.classList.add('agent-approval-card', 'agent-approval-card-legacy');
                }
                const before = bubble.traceEl && bubble.traceEl.parentNode === bubble.bodyEl ? bubble.traceEl : null;
                bubble.bodyEl.insertBefore(card, before);
            } else if (bubble.actionsEl) {
                if (bubble.traceEl) bubble.traceEl.hidden = false;
                bubble.actionsEl.appendChild(card);
                bubble.actionCount = (bubble.actionCount || 0) + 1;
                updateAgentTraceSummary(bubble);
            } else {
                bubble.bodyEl.appendChild(card);
            }
            promoteStandaloneAgentApprovalCards(bubble);
            scrollToBottom(stickToBottom);
        }

        function appendProgressNote(bubble, evt) {
            if (!bubble || !bubble.actionsEl) return;
            const stickToBottom = isAgentNearBottom();
            if (bubble.traceEl) bubble.traceEl.hidden = false;
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

        function isLegacyProbeApprovalCard(card) {
            if (!card || !card.classList || !card.classList.contains('agent-action-card')) return false;
            const head = card.querySelector('.agent-action-card-head');
            const text = String(head && head.textContent || '').toUpperCase();
            return /\bPROBE(S)?\s+(CREATE|UPDATE|UPSERT|DELETE|DELETED|UPDATED|CREATED)\b/.test(text)
                && (/\bPREVIEW\b/.test(text) || card.querySelector('.agent-approval-footer'));
        }

        function promoteStandaloneAgentApprovalCards(bubble) {
            if (!bubble || !bubble.bodyEl || !bubble.actionsEl || !bubble.traceEl) return;
            const candidates = Array.from(bubble.actionsEl.children).filter((node) => {
                if (!(node instanceof HTMLElement)) return false;
                return node.dataset.agentStandaloneApproval === 'true'
                    || node.classList.contains('agent-approval-card')
                    || Boolean(node.querySelector('.agent-approval-footer'))
                    || isLegacyProbeApprovalCard(node);
            });
            if (!candidates.length) return;
            const before = bubble.traceEl.parentNode === bubble.bodyEl ? bubble.traceEl : null;
            candidates.forEach((node) => {
                node.dataset.agentStandaloneApproval = 'true';
                if (
                    isLegacyProbeApprovalCard(node)
                    || (node.classList.contains('agent-action-card') && node.querySelector('.agent-approval-footer'))
                ) {
                    node.classList.add('agent-approval-card', 'agent-approval-card-legacy');
                }
                bubble.bodyEl.insertBefore(node, before);
                bubble.actionCount = Math.max(0, Number(bubble.actionCount || 0) - 1);
            });
            updateAgentTraceSummary(bubble);
            bubble.traceEl.hidden = Number(bubble.actionCount || 0) <= 0;
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
        function agentImageUrlForItem(item) {
            if (!item || typeof item !== 'object') return '';
            const direct = item.image_url || item.imageUrl || item.url || null;
            if (direct) return sanitizeUrl(direct);
            const imagePath = item.image_path || item.path || null;
            if (imagePath && String(imagePath).startsWith('/')) {
                return `/detections/image?image_path=${encodeURIComponent(String(imagePath))}`;
            }
            const thumbnail = String(item.thumbnail || item.thumbnail_b64 || '').trim();
            if (thumbnail) {
                return sanitizeUrl(/^data:image\//i.test(thumbnail) ? thumbnail : `data:image/jpeg;base64,${thumbnail}`);
            }
            return '';
        }

        function agentThumbTitle(item) {
            if (!item || typeof item !== 'object') return 'Preview';
            const ts = item.timestamp_ms || item.event_timestamp_ms || item.recorded_at_ms || null;
            const tsText = ts ? fmtDate(ts) : '';
            const source = item.source_label || item.source || item.archive_item_type || '';
            const id = item.id || item.detection_id || item.probe_id || item.path || item.image_path || '';
            return [source, id ? `#${id}` : '', tsText].filter(Boolean).join(' · ') || 'Preview';
        }

        // Helper: build a thumbnail element using image_url (backend-provided) or fallback.
        // The tile opens the in-app lightbox; the "open" link is a real href for copying/new tab.
        function _makeThumb(item, cls, scoreVal) {
            const div = document.createElement('div');
            div.className = cls;
            const url = agentImageUrlForItem(item);
            const score = scoreVal != null ? String(scoreVal) : '';
            const previewTitle = item.filename || item.name || agentThumbTitle(item);
            const addScore = () => {
                if (!score) return;
                const badge = document.createElement('div');
                badge.className = cls === 'agent-det-thumb' ? 'agent-det-score' : 'agent-search-score';
                badge.textContent = score;
                div.appendChild(badge);
            };
            const showMissingImage = () => {
                delete div.dataset.previewImage;
                delete div.dataset.previewTitle;
                div.classList.add('agent-thumb-missing-image');
                div.textContent = item && (item.id || item.detection_id)
                    ? `No image #${item.id || item.detection_id}`
                    : 'No image';
                addScore();
            };
            if (url) {
                div.dataset.previewImage = String(url);
                div.dataset.previewTitle = String(previewTitle);
                div.title = String(previewTitle);
                const img = document.createElement('img');
                img.src = String(url);
                img.alt = String(previewTitle);
                img.loading = 'lazy';
                img.addEventListener('error', showMissingImage, { once: true });
                div.appendChild(img);
                addScore();
                const link = document.createElement('a');
                link.className = 'agent-thumb-open-link';
                link.href = String(url);
                link.target = '_blank';
                link.rel = 'noopener noreferrer';
                link.title = 'Open image';
                link.dataset.openImageLink = '';
                link.textContent = 'open';
                div.appendChild(link);
            } else {
                showMissingImage();
            }
            return div;
        }

        function appendAgentThumbGrid(body, items, options = {}) {
            const rows = Array.isArray(items) ? items.filter(Boolean) : [];
            if (!rows.length) return false;
            const grid = document.createElement('div');
            grid.className = options.gridClass || 'agent-det-grid';
            rows.slice(0, options.limit || 8).forEach((item) => {
                let score = null;
                if (options.scoreFormatter) {
                    score = options.scoreFormatter(item);
                } else if (item && item.score != null) {
                    score = Number(item.score).toFixed(3);
                } else if (item && item.margin != null) {
                    score = Number(item.margin).toFixed(3);
                } else if (item && item.severity) {
                    score = String(item.severity);
                }
                grid.appendChild(_makeThumb(item, options.thumbClass || 'agent-det-thumb', score));
            });
            body.appendChild(grid);
            if (rows.length > (options.limit || 8)) {
                const more = document.createElement('div');
                more.className = 'agent-card-muted-note';
                more.textContent = `+${rows.length - (options.limit || 8)} more`;
                body.appendChild(more);
            }
            return true;
        }

        function extractAgentEvidenceIds(text) {
            const source = String(text || '');
            if (!source) return [];
            const ids = [];
            const seen = new Set();
            const mdWrap = '[`*_~]*';
            const idAtom = `${mdWrap}\\d{2,9}${mdWrap}`;
            const idList = `(?:${idAtom}(?:\\s*(?:,|and|&|/)\\s*)?){1,12}`;
            const deniedIdContext = (absoluteIndex) => {
                if (!Number.isFinite(absoluteIndex) || absoluteIndex < 0) return false;
                const before = source.slice(Math.max(0, absoluteIndex - 56), absoluteIndex);
                return /\b(?:channel|ch|probe|plan|approval|session|job|batch|task|run|tenant|user|account|model)\s*(?:id|#)?\s*$/i.test(before);
            };
            const addId = (raw, absoluteIndex = -1) => {
                const value = String(raw || '').trim();
                const match = value.match(/\b\d{2,9}\b/);
                if (!match) return;
                if (deniedIdContext(absoluteIndex)) return;
                const id = Number(match[0]);
                if (!Number.isSafeInteger(id) || id <= 0 || seen.has(id)) return;
                seen.add(id);
                ids.push(id);
            };
            const parseList = (raw, baseIndex = -1) => {
                const rawText = String(raw || '');
                const matcher = /\b\d{2,9}\b/g;
                let numberMatch;
                while ((numberMatch = matcher.exec(rawText)) && ids.length < AGENT_EVIDENCE_ID_LIMIT) {
                    const absoluteIndex = baseIndex >= 0 ? baseIndex + numberMatch.index : -1;
                    addId(numberMatch[0], absoluteIndex);
                }
            };
            const parseMatchList = (match) => {
                const raw = match && match[1] ? String(match[1]) : '';
                if (!raw) return;
                const localIndex = String(match[0] || '').indexOf(raw);
                const baseIndex = localIndex >= 0 ? match.index + localIndex : -1;
                parseList(raw, baseIndex);
            };
            const directPatterns = [
                new RegExp(`\\bevidence\\s+ids?\\s*[:#-]?\\s*(${idList})`, 'gi'),
                new RegExp(`\\b(?:detections?|frames?|candidates?|images?|snapshots?)\\s+ids?\\s*[:#-]?\\s*(${idList})`, 'gi'),
                new RegExp(`\\b(?:detection|frame|image|snapshot)\\s+(?:id|#)\\s*[:#-]?\\s*(${idAtom})(?=\\W|$)`, 'gi'),
                new RegExp(`\\b(?:candidate)\\s+\\d+\\s*[:#-]\\s*(?:detection\\s+id\\s*)?(${idAtom})(?=\\W|$)`, 'gi'),
                new RegExp(`\\b(?:detection_id|frame_id|snapshot_id|image_id)\\s*[:=]\\s*(${idAtom})(?=\\W|$)`, 'gi'),
                /\b(?:detections?|frames?|candidates?|images?|snapshots?)\s*#?\s*(\d{2,9})\b/gi,
            ];
            directPatterns.forEach((pattern) => {
                let match;
                while ((match = pattern.exec(source)) && ids.length < AGENT_EVIDENCE_ID_LIMIT) {
                    parseMatchList(match);
                }
            });

            const contextualIdList = new RegExp(`\\bids?\\s*[:#-]?\\s*(${idList})`, 'gi');
            let match;
            while ((match = contextualIdList.exec(source)) && ids.length < AGENT_EVIDENCE_ID_LIMIT) {
                const start = Math.max(0, match.index - 90);
                const end = Math.min(source.length, match.index + match[0].length + 60);
                const context = source.slice(start, end);
                if (/\b(?:detections?|frames?|candidates?|evidence|visual|snapshots?|thumbnails?|images?)\b/i.test(context)) {
                    parseMatchList(match);
                }
            }
            return ids.slice(0, AGENT_EVIDENCE_ID_LIMIT);
        }

        function buildAgentEvidenceIdCard(ids) {
            const cleanIds = Array.isArray(ids) ? ids.filter(Boolean) : [];
            if (!cleanIds.length) return null;
            const card = document.createElement('div');
            card.className = 'agent-action-card agent-evidence-id-card';
            card.innerHTML = `<div class="agent-action-card-head">&#9670; EVIDENCE LINKS — ${cleanIds.length} ID${cleanIds.length === 1 ? '' : 'S'}</div>`;
            const body = document.createElement('div');
            body.className = 'agent-action-card-body';
            appendAgentThumbGrid(body, cleanIds.map((id) => ({
                id,
                detection_id: id,
                image_url: `/detections/thumbnail/${id}`,
                filename: `Detection #${id}`,
                source_label: 'Evidence',
            })), {
                gridClass: 'agent-search-results-grid',
                thumbClass: 'agent-search-thumb',
                limit: AGENT_EVIDENCE_ID_LIMIT,
                scoreFormatter: (item) => item && item.detection_id ? `#${item.detection_id}` : null,
            });
            const note = document.createElement('div');
            note.className = 'agent-card-muted-note';
            note.textContent = 'Click a preview to inspect the frame; open uses the archive thumbnail link with your current access rights.';
            body.appendChild(note);
            card.appendChild(body);
            return card;
        }

        function syncAgentEvidenceIdCard(bubble) {
            if (!bubble || !bubble.bodyEl) return;
            const ids = extractAgentEvidenceIds(bubble.text || '');
            const key = ids.join(',');
            const existing = bubble.bodyEl.querySelector('.agent-evidence-id-card');
            if (bubble.evidenceIdsKey === key && existing) return;
            bubble.evidenceIdsKey = key;
            bubble.bodyEl.querySelectorAll('.agent-evidence-id-card').forEach((node) => node.remove());
            if (!ids.length) return;
            const card = buildAgentEvidenceIdCard(ids);
            if (!card) return;
            const before = bubble.traceEl && bubble.traceEl.parentNode === bubble.bodyEl ? bubble.traceEl : null;
            bubble.bodyEl.insertBefore(card, before);
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

        function isProbeMutationTool(toolName) {
            return ['create_probe', 'update_probe', 'delete_probes'].includes(String(toolName || ''));
        }

        function isStandaloneProbeApprovalResult(toolName, result) {
            if (!isProbeMutationTool(toolName) || !result || typeof result !== 'object') return false;
            const status = String(result.status || '').toLowerCase();
            return status === 'preview' || status === 'applied' || Boolean(result.approval);
        }

        function agentProbeMutationTitle(toolName, result) {
            const status = String(result && result.status || '').toLowerCase();
            const preview = status === 'preview' || Boolean(result && result.approval);
            const applied = status === 'applied';
            const action = String(result && result.action || '');
            if (toolName === 'delete_probes') {
                return preview ? 'Probe delete preview' : (applied ? 'Probes deleted' : 'Probe delete result');
            }
            if (toolName === 'update_probe') {
                return preview ? 'Probe update preview' : (applied ? 'Probe updated' : 'Probe update result');
            }
            if (action === 'update_existing') {
                return preview ? 'Probe upsert preview' : (applied ? 'Probe updated via create' : 'Probe upsert result');
            }
            return preview ? 'Probe create preview' : (applied ? 'Probe created' : 'Probe create result');
        }

        function appendAgentApprovalField(parent, key, value) {
            if (value === undefined || value === null || String(value).trim() === '') return;
            const row = document.createElement('div');
            row.className = 'agent-approval-field';
            const keyEl = document.createElement('span');
            keyEl.className = 'agent-approval-key';
            keyEl.textContent = key;
            const valEl = document.createElement('span');
            valEl.className = 'agent-approval-val';
            valEl.textContent = agentCompactValue(value);
            row.appendChild(keyEl);
            row.appendChild(valEl);
            parent.appendChild(row);
        }

        function buildAgentProbeApprovalCard(toolName, result) {
            const status = String(result && result.status || '').toLowerCase();
            const isPreview = status === 'preview' || Boolean(result && result.approval);
            const isApplied = status === 'applied';
            const card = document.createElement('div');
            card.className = `agent-approval-card${isPreview ? ' is-preview' : ''}${isApplied ? ' is-applied-card' : ''}`;
            card.dataset.agentStandaloneApproval = 'true';

            const head = document.createElement('div');
            head.className = 'agent-approval-card-head';
            const titleWrap = document.createElement('div');
            titleWrap.className = 'agent-approval-title-wrap';
            const kicker = document.createElement('div');
            kicker.className = 'agent-approval-kicker';
            kicker.textContent = isPreview ? 'Operator approval required' : 'Probe action receipt';
            const title = document.createElement('div');
            title.className = 'agent-approval-title';
            title.textContent = agentProbeMutationTitle(toolName, result);
            titleWrap.appendChild(kicker);
            titleWrap.appendChild(title);
            const statusBadge = document.createElement('span');
            statusBadge.className = `agent-approval-status${isApplied ? ' is-applied' : ''}`;
            statusBadge.textContent = isPreview ? 'Preview' : (isApplied ? 'Applied' : 'Result');
            head.appendChild(titleWrap);
            head.appendChild(statusBadge);
            card.appendChild(head);

            const body = document.createElement('div');
            body.className = 'agent-approval-card-body';
            const fields = document.createElement('div');
            fields.className = 'agent-approval-fields';
            const probe = (result && (result.proposed || result.probe)) || {};
            const probeName = probe.name || (result && result.probe_name);
            appendAgentApprovalField(fields, 'Probe', probeName || 'unknown');
            appendAgentApprovalField(fields, 'Channel', probe.channel_id ?? (result && result.channel_id));
            appendAgentApprovalField(fields, 'Action', (result && result.action) || toolName);
            appendAgentApprovalField(fields, 'Status', (result && result.status) || (isPreview ? 'preview' : 'result'));

            const targets = Array.isArray(result && result.targets) ? result.targets : [];
            if (targets.length) {
                appendAgentApprovalField(fields, 'Targets', targets.map((item) => item && (item.name || item.id || item.probe_id || '?')).join(', '));
            }

            const diff = (result && result.diff) || {};
            Object.entries(diff).forEach(([key, value]) => appendAgentApprovalField(fields, key, value));

            const proposedKeys = [
                'positive_prompt', 'negative_prompt', 'pos_floor', 'margin',
                'threshold', 'threshold_margin', 'cooldown_sec', 'bookmark_enabled',
            ];
            proposedKeys.forEach((key) => {
                if (probe && probe[key] !== undefined && diff[key] === undefined) {
                    appendAgentApprovalField(fields, key, probe[key]);
                }
            });
            body.appendChild(fields);

            const conflicts = Array.isArray(result && result.conflicts) ? result.conflicts : [];
            if (conflicts.length) {
                const warning = document.createElement('div');
                warning.className = 'agent-approval-warning';
                warning.textContent = `Potential conflicts: ${conflicts.map((item) => item && (item.name || item.id || '?')).join(', ')}`;
                body.appendChild(warning);
            }
            if (isPreview) {
                const note = document.createElement('div');
                note.className = 'agent-approval-note';
                note.textContent = 'Preview only. Apply commits this probe change through the server-side approval plan.';
                body.appendChild(note);
            }
            card.appendChild(body);
            appendApprovalControl(card, toolName, result);
            return card;
        }

        function agentResultList(value) {
            if (Array.isArray(value)) return value.filter(Boolean);
            if (value && typeof value === 'object') {
                const itemKeys = [
                    'type', 'transition_type', 'state', 'visual_state', 'summary', 'description',
                    'time', 'time_range', 'start_time', 'end_time', 'thumbnail', 'thumbnail_b64',
                    'image_url', 'image_path', 'path', 'before', 'after', 'frame', 'boundary_frame',
                    'evidence_frame',
                ];
                if (itemKeys.some((key) => value[key] !== undefined && value[key] !== null)) return [value];
                return Object.values(value).filter(Boolean);
            }
            if (typeof value === 'string' && value.trim()) return [value];
            return [];
        }

        function agentFirstValue(item, keys) {
            if (!item || typeof item !== 'object') return '';
            for (const key of keys) {
                const value = item[key];
                if (value !== undefined && value !== null && String(value).trim() !== '') return value;
            }
            return '';
        }

        function agentHumanizeKey(key) {
            return String(key || '')
                .replace(/[_-]+/g, ' ')
                .replace(/\b\w/g, (ch) => ch.toUpperCase());
        }

        function agentFormatScalar(value) {
            if (value === undefined || value === null || value === '') return '';
            if (typeof value === 'number') {
                if (!Number.isFinite(value)) return String(value);
                if (Number.isInteger(value)) return String(value);
                const abs = Math.abs(value);
                return abs > 0 && abs < 1 ? value.toFixed(3) : value.toFixed(2);
            }
            if (typeof value === 'boolean') return value ? 'yes' : 'no';
            return String(value);
        }

        function agentCompactValue(value) {
            if (value === undefined || value === null || value === '') return '';
            if (Array.isArray(value)) {
                return value.map((item) => agentCompactValue(item)).filter(Boolean).join(' · ');
            }
            if (typeof value === 'object') return JSON.stringify(value);
            return agentFormatScalar(value);
        }

        function agentTimeLabel(item) {
            if (!item || typeof item !== 'object') return '';
            const direct = agentFirstValue(item, [
                'time_range', 'range', 'window', 'time', 'timestamp',
                'boundary_time', 'boundary_time_text', 'at',
            ]);
            if (direct !== '') return agentCompactValue(direct);
            const start = agentFirstValue(item, ['start_time', 'start', 'from_time', 'start_ts', 'start_timestamp']);
            const end = agentFirstValue(item, ['end_time', 'end', 'to_time', 'end_ts', 'end_timestamp']);
            if (start !== '' || end !== '') {
                return `${agentCompactValue(start || '?')} - ${agentCompactValue(end || '?')}`;
            }
            const ms = agentFirstValue(item, ['timestamp_ms', 'event_timestamp_ms', 'boundary_timestamp_ms', 'recorded_at_ms']);
            if (ms !== '') return fmtDate(ms);
            return '';
        }

        function agentStateChangeText(item) {
            const from = agentFirstValue(item, ['from_state', 'from', 'previous_state', 'before_state', 'negative_state']);
            const to = agentFirstValue(item, ['to_state', 'to', 'next_state', 'after_state', 'positive_state']);
            if (from !== '' || to !== '') return `${agentCompactValue(from || '?')} -> ${agentCompactValue(to || '?')}`;
            return '';
        }

        function agentTransitionTypeLabel(item) {
            const type = agentFirstValue(item, ['transition_type', 'type', 'kind']);
            if (type !== '') return agentHumanizeKey(type);
            const stateChange = agentStateChangeText(item);
            return stateChange || 'Transition';
        }

        function agentScoreSummary(item) {
            if (!item || typeof item !== 'object') return '';
            return ['score', 'confidence', 'positive_score', 'negative_score', 'margin']
                .filter((key) => item[key] !== undefined && item[key] !== null && String(item[key]).trim() !== '')
                .map((key) => `${agentHumanizeKey(key)}: ${agentCompactValue(item[key])}`)
                .join(' · ');
        }

        function agentCoverageText(coverage) {
            if (!coverage) return '';
            if (typeof coverage === 'string') return coverage;
            if (typeof coverage !== 'object') return String(coverage);
            if (coverage.note) {
                const status = String(coverage.status || '').trim();
                return `${status ? `Coverage ${status}: ` : 'Coverage: '}${String(coverage.note)}`;
            }
            const parts = ['status', 'start_time', 'end_time', 'covered_entries', 'total_entries', 'sampled_frames']
                .filter((key) => coverage[key] !== undefined && coverage[key] !== null && String(coverage[key]).trim() !== '')
                .map((key) => `${agentHumanizeKey(key)}: ${agentCompactValue(coverage[key])}`);
            return parts.length ? `Coverage: ${parts.join(' · ')}` : '';
        }

        function agentCoverageClass(coverage) {
            const status = coverage && typeof coverage === 'object' ? String(coverage.status || 'unknown') : 'unknown';
            return status.toLowerCase().replace(/[^a-z0-9_-]+/g, '_') || 'unknown';
        }

        function agentCompletenessSources(result) {
            const root = result && typeof result === 'object' ? result : {};
            const scope = root.scope && typeof root.scope === 'object' ? root.scope : null;
            const inventory = root.inventory && typeof root.inventory === 'object' ? root.inventory : null;
            return [
                root,
                root.coverage && typeof root.coverage === 'object' ? root.coverage : null,
                scope,
                inventory,
                inventory && inventory.coverage && typeof inventory.coverage === 'object' ? inventory.coverage : null,
            ].filter(Boolean);
        }

        function agentFirstCompletenessValue(sources, key) {
            for (const source of sources) {
                if (Object.prototype.hasOwnProperty.call(source, key) && source[key] !== undefined && source[key] !== null) {
                    return source[key];
                }
            }
            return undefined;
        }

        function agentCompletenessIds(sources, key) {
            const ids = [];
            const seen = new Set();
            sources.forEach((source) => {
                const values = Array.isArray(source[key]) ? source[key] : [];
                values.forEach((value) => {
                    const normalized = String(value ?? '').trim();
                    if (!normalized || seen.has(normalized)) return;
                    seen.add(normalized);
                    ids.push(normalized);
                });
            });
            return ids;
        }

        function agentCompletenessErrors(sources) {
            const errors = [];
            const seen = new Set();
            const add = (value) => {
                const text = typeof value === 'object' && value !== null
                    ? String(value.error || value.message || value.channel_id || JSON.stringify(value))
                    : String(value ?? '');
                const clean = text.trim();
                if (!clean || seen.has(clean)) return;
                seen.add(clean);
                errors.push(clean);
            };
            sources.forEach((source) => {
                if (Array.isArray(source.errors)) source.errors.forEach(add);
                if (source.channel_inventory_error) add(source.channel_inventory_error);
            });
            return errors;
        }

        function appendAgentCompleteness(body, result, options = {}) {
            if (!body) return;
            const sources = agentCompletenessSources(result);
            const root = result && typeof result === 'object' ? result : {};
            const inventory = root.inventory && typeof root.inventory === 'object' ? root.inventory : null;
            const explicitCoverage = root.coverage ?? (inventory ? inventory.coverage : null);
            const coverage = (explicitCoverage ?? sources.find((source) => (
                Object.prototype.hasOwnProperty.call(source, 'coverage_status')
                || Object.prototype.hasOwnProperty.call(source, 'coverage_note')
                || Object.prototype.hasOwnProperty.call(source, 'scanned_candidates')
                || Object.prototype.hasOwnProperty.call(source, 'channel_inventory_status')
            ))) || null;
            let coverageText = agentCoverageText(coverage);
            if (!coverageText && coverage && coverage.channel_inventory_status) {
                coverageText = `Coverage inventory: ${coverage.channel_inventory_status}${coverage.full_research_note ? ` · ${coverage.full_research_note}` : ''}`;
            }
            if (coverage && (coverage.scanned_candidates !== undefined || coverage.total_candidates !== undefined)) {
                const scanned = coverage.scanned_candidates ?? '?';
                const total = coverage.total_candidates ?? '?';
                coverageText = `${coverageText || 'Coverage'} · scanned ${scanned} of ${total} candidates`;
            }
            if (coverageText || options.alwaysCoverage) {
                const note = document.createElement('div');
                note.className = `agent-coverage-note agent-coverage-${agentCoverageClass(coverage)}`;
                note.textContent = coverageText || 'Coverage: not reported by the backend for this result.';
                body.appendChild(note);
            }

            const hasBackendTruncated = sources.some((source) => Object.prototype.hasOwnProperty.call(source, 'backend_truncated'));
            const backendTruncated = sources.some((source) => Boolean(source.backend_truncated));
            if (hasBackendTruncated || options.alwaysBackendTruncation) {
                const note = document.createElement('div');
                note.className = `agent-coverage-note agent-coverage-${backendTruncated ? 'partial' : 'covered'}`;
                note.textContent = `Backend truncated: ${hasBackendTruncated ? (backendTruncated ? 'yes — older or additional backend rows may be unchecked' : 'no') : 'not reported'}.`;
                body.appendChild(note);
            }

            const hasTruncated = sources.some((source) => (
                Object.prototype.hasOwnProperty.call(source, 'truncated')
                || Object.prototype.hasOwnProperty.call(source, '_truncated')
                || Object.prototype.hasOwnProperty.call(source, 'id_lists_truncated')
            ));
            const truncated = sources.some((source) => Boolean(source.truncated || source._truncated || source.id_lists_truncated));
            if (hasTruncated || options.alwaysTruncation) {
                const note = document.createElement('div');
                note.className = `agent-coverage-note agent-coverage-${truncated ? 'partial' : 'covered'}`;
                note.textContent = `Result truncation: ${hasTruncated ? (truncated ? 'yes — displayed evidence is incomplete' : 'no') : 'not reported'}.`;
                body.appendChild(note);
            }

            const uncheckedIds = agentCompletenessIds(sources, 'unchecked_channel_ids');
            const deferredIds = agentCompletenessIds(sources, 'deferred_channel_ids');
            const uncheckedRaw = agentFirstCompletenessValue(sources, 'unchecked_count');
            const deferredRaw = agentFirstCompletenessValue(sources, 'deferred_count');
            const errorsRaw = agentFirstCompletenessValue(sources, 'error_count');
            const errors = agentCompletenessErrors(sources);
            const uncheckedCount = Number.isFinite(Number(uncheckedRaw)) ? Number(uncheckedRaw) : uncheckedIds.length;
            const deferredCount = Number.isFinite(Number(deferredRaw)) ? Number(deferredRaw) : deferredIds.length;
            const errorCount = Number.isFinite(Number(errorsRaw)) ? Number(errorsRaw) : errors.length;
            const hasScopeSignals = options.alwaysScope
                || uncheckedRaw !== undefined || deferredRaw !== undefined || errorsRaw !== undefined
                || uncheckedIds.length || deferredIds.length || errors.length;
            if (hasScopeSignals) {
                const scope = document.createElement('div');
                const incomplete = uncheckedCount > 0 || deferredCount > 0 || errorCount > 0 || errors.length > 0;
                scope.className = `agent-coverage-note agent-coverage-${incomplete ? 'partial' : 'covered'}`;
                const parts = [
                    `Unchecked: ${uncheckedCount}${uncheckedIds.length ? ` (${uncheckedIds.join(', ')})` : ''}`,
                    `Deferred: ${deferredCount}${deferredIds.length ? ` (${deferredIds.join(', ')})` : ''}`,
                    `Errors: ${Math.max(errorCount, errors.length)}${errors.length ? ` (${errors.slice(0, 3).join(' · ')})` : ''}`,
                ];
                scope.textContent = parts.join(' · ');
                body.appendChild(scope);
            }
        }

        function agentTransitionCountEntries(counts) {
            if (!counts || typeof counts !== 'object' || Array.isArray(counts)) return [];
            const nested = ['transition_types', 'by_transition_type', 'by_type', 'types']
                .map((key) => counts[key])
                .find((value) => value && typeof value === 'object' && !Array.isArray(value));
            const source = nested || counts;
            return Object.entries(source)
                .filter(([, value]) => value !== undefined && value !== null && String(agentCompactValue(value)).trim() !== '');
        }

        function agentBoundaryFrameItem(frame, index, role, parent) {
            const item = {
                ...(parent && typeof parent === 'object' ? parent : {}),
                ...(frame && typeof frame === 'object' ? frame : {}),
            };
            const time = agentTimeLabel(item);
            const type = agentTransitionTypeLabel(item);
            const title = [type !== 'Transition' ? type : '', role, time].filter(Boolean).join(' · ') || `Boundary frame ${index + 1}`;
            return {
                ...item,
                filename: item.filename || item.name || title,
                source_label: item.source_label || 'Boundary',
                role: item.role || role,
            };
        }

        function agentBoundaryFrames(boundaryFrames) {
            const rows = agentResultList(boundaryFrames);
            const out = [];
            rows.forEach((frame, index) => {
                if (!frame || typeof frame !== 'object') return;
                const nested = [
                    ['Before', frame.before || frame.pre || frame.previous || frame.from_frame],
                    ['After', frame.after || frame.post || frame.next || frame.to_frame],
                    ['Boundary', frame.frame || frame.boundary_frame || frame.evidence_frame],
                ].filter(([, value]) => value && typeof value === 'object');
                if (nested.length) {
                    nested.forEach(([role, nestedFrame]) => {
                        out.push(agentBoundaryFrameItem(nestedFrame, out.length, role, frame));
                    });
                } else {
                    out.push(agentBoundaryFrameItem(frame, out.length, 'Boundary', null));
                }
            });
            return out;
        }

        function buildActionCard(toolName, result) {
            const card = document.createElement('div');
            card.className = 'agent-action-card';
            if (result && result.error) {
                card.classList.add('agent-action-card-error');
                card.innerHTML = `<div class="agent-action-card-head">&#9670; ${escapeHtml(String(toolName || 'TOOL').toUpperCase())} ERROR</div>`;
                const body = document.createElement('div');
                body.className = 'agent-action-card-body';
                body.innerHTML = `<div class="agent-inline-msg error">${escapeHtml(String(result.error))}</div>`;
                card.appendChild(body);
                return card;
            }
            if (isStandaloneProbeApprovalResult(toolName, result)) {
                return buildAgentProbeApprovalCard(toolName, result);
            }

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
                    appendAgentThumbGrid(body, hits.map((h) => ({
                        ...h,
                        score: h.score != null ? h.score : null,
                    })), {
                        gridClass: 'agent-search-results-grid',
                        thumbClass: 'agent-search-thumb',
                        limit: 8,
                        scoreFormatter: (h) => h && h.score != null ? `${(Number(h.score) * 100).toFixed(0)}%` : null,
                    });
                } else {
                    body.innerHTML = '<div style="font-size:13px;color:var(--muted)">No results found.</div>';
                }
                appendAgentCompleteness(body, result, {
                    alwaysCoverage: true,
                    alwaysTruncation: true,
                });
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
                    appendAgentThumbGrid(body, detections, {
                        gridClass: 'agent-det-grid',
                        thumbClass: 'agent-det-thumb',
                        limit: 8,
                    });
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

            } else if (toolName === 'list_video_summary_channels') {
                const channels = (result && result.candidate_channels) || [];
                const returned = result && result.returned != null ? Number(result.returned) : channels.length;
                const checked = result && result.total_channels_checked != null ? Number(result.total_channels_checked) : channels.length;
                const label = `VIDEO SUMMARY INVENTORY — ${returned} returned · ${checked} checked`;
                card.innerHTML = `<div class="agent-action-card-head">&#9670; ${escapeHtml(label)}</div>`;
                const body = document.createElement('div');
                body.className = 'agent-action-card-body';
                appendAgentCompleteness(body, result, {
                    alwaysCoverage: true,
                    alwaysBackendTruncation: true,
                    alwaysTruncation: true,
                    alwaysScope: true,
                });
                if (channels.length) {
                    const rows = document.createElement('div');
                    rows.innerHTML = channels.slice(0, 12).map((channel) => {
                        const channelId = channel.channel_id ?? channel.id ?? '?';
                        const title = channel.title || channel.name || `Channel ${channelId}`;
                        const status = channel.coverage_status || (channel.quiet ? 'quiet' : 'summary data');
                        const count = channel.summary_count != null ? ` · ${channel.summary_count} summaries` : '';
                        return `<div class="agent-summary-entry"><span class="agent-summary-ts">CH ${escapeHtml(String(channelId))}</span><span class="agent-summary-text">${escapeHtml(String(title))} · ${escapeHtml(String(status))}${escapeHtml(count)}</span></div>`;
                    }).join('');
                    body.appendChild(rows);
                } else {
                    const empty = document.createElement('div');
                    empty.className = 'agent-card-muted-note';
                    empty.textContent = 'No video-summary channels were returned for this window.';
                    body.appendChild(empty);
                }
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
                appendAgentCompleteness(body, result, { alwaysScope: true });
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
                const resultImageUrl = result && result.image_url;
                const b64 = result && result.snapshot_b64;
                const imgPath = result && result.image_path;
                let imgSrc = null;
                if (resultImageUrl) {
                    imgSrc = resultImageUrl;
                } else if (b64) {
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
                const evidenceFrames = (result && result.evidence_frames) || [];
                const coverage = (result && result.coverage) || {};
                const depth = (result && result.depth) || '';
                const ch = (result && result.channel_id) || '';
                const label = `VIDEO SUMMARIES — CH ${ch} · ${depth} · ${entries.length} entries`;
                card.innerHTML = `<div class="agent-action-card-head">&#9670; ${escapeHtml(label)}</div>`;
                const body = document.createElement('div');
                body.className = 'agent-action-card-body';
                appendAgentCompleteness(body, result, {
                    alwaysCoverage: true,
                    alwaysBackendTruncation: true,
                    alwaysTruncation: true,
                    alwaysScope: true,
                });
                if (evidenceFrames.length) {
                    const head = document.createElement('div');
                    head.className = 'agent-evidence-title';
                    head.textContent = `Archive snaps · ${evidenceFrames.length}`;
                    body.appendChild(head);
                    appendAgentThumbGrid(body, evidenceFrames, {
                        gridClass: 'agent-det-grid agent-evidence-grid',
                        thumbClass: 'agent-det-thumb agent-evidence-thumb',
                        limit: 8,
                    });
                }
                if (entries.length) {
                    const summaryWrap = document.createElement('div');
                    summaryWrap.innerHTML = entries.map(e => {
                        const t = escapeHtml(e.time || '');
                        const s = escapeHtml(e.summary || '');
                        return `<div class="agent-summary-entry"><span class="agent-summary-ts">${t}</span><span class="agent-summary-text">${s}</span></div>`;
                    }).join('');
                    body.appendChild(summaryWrap);
                } else {
                    const empty = document.createElement('div');
                    empty.className = 'agent-card-muted-note';
                    const sourceWindows = Number(result && result.total_in_window) || 0;
                    const semanticStatus = String((result && result.semantic_status) || '').toLowerCase();
                    const semanticPending = Number(result && result.semantic_pending_count) || 0;
                    if (sourceWindows > 0 && (semanticStatus === 'pending' || semanticPending > 0)) {
                        empty.textContent = `${sourceWindows} source windows retained; semantic summaries are being generated.`;
                    } else if (sourceWindows > 0) {
                        empty.textContent = `${sourceWindows} source windows retained; no completed semantic narrative yet. Drill into L0 observations.`;
                    } else {
                        empty.textContent = 'No video-description data in this time range.';
                    }
                    body.appendChild(empty);
                }
                card.appendChild(body);

            } else if (toolName === 'count_video_summary_events') {
                const counts = (result && result.counts) || {};
                const coverage = (result && result.coverage) || {};
                const events = (result && result.transition_events) || [];
                const entity = result && result.entity_query ? ` · ${result.entity_query}` : '';
                const ch = result && result.channel_id != null ? `CH ${result.channel_id}` : 'CH ?';
                const label = `EVENT COUNT — ${ch}${entity}`;
                card.innerHTML = `<div class="agent-action-card-head">&#9670; ${escapeHtml(label)}</div>`;
                const body = document.createElement('div');
                body.className = 'agent-action-card-body';
                const countRows = [
                    ['Appearances', counts.appearance_count],
                    ['Disappearances', counts.disappearance_count],
                    ['Explicit', `${counts.explicit_appearance_count || 0} in / ${counts.explicit_disappearance_count || 0} out`],
                    ['Inferred', `${counts.inferred_appearance_count || 0} in / ${counts.inferred_disappearance_count || 0} out`],
                ];
                body.innerHTML = `<div class="agent-probe-update-row">${
                    countRows.map(([key, value]) => `<div class="agent-probe-update-field"><span class="agent-probe-update-key">${escapeHtml(key)}:</span><span class="agent-probe-update-val">${escapeHtml(String(value ?? 0))}</span></div>`).join('')
                }</div>`;
                appendAgentCompleteness(body, result, {
                    alwaysCoverage: true,
                    alwaysBackendTruncation: true,
                    alwaysTruncation: true,
                    alwaysScope: true,
                });
                if (events.length) {
                    const wrap = document.createElement('div');
                    wrap.innerHTML = events.slice(0, 8).map((event) => {
                        const type = escapeHtml(event.type || 'event');
                        const basis = escapeHtml(event.basis || '');
                        const time = escapeHtml(event.time || '');
                        const summary = escapeHtml(event.summary || '');
                        return `<div class="agent-summary-entry"><span class="agent-summary-ts">${time} · ${type}</span><span class="agent-summary-text">${basis}${basis ? ' · ' : ''}${summary}</span></div>`;
                    }).join('');
                    body.appendChild(wrap);
                    if (events.length > 8) {
                        const more = document.createElement('div');
                        more.className = 'agent-card-muted-note';
                        more.textContent = `+${events.length - 8} more transition events`;
                        body.appendChild(more);
                    }
                } else {
                    const empty = document.createElement('div');
                    empty.className = 'agent-card-muted-note';
                    empty.textContent = 'No transition events counted in returned summaries.';
                    body.appendChild(empty);
                }
                card.appendChild(body);

            } else if (toolName === 'track_visual_state_transitions') {
                const counts = (result && result.counts) || {};
                const coverage = (result && result.coverage) || {};
                const transitions = agentResultList(result && result.transitions);
                const segments = agentResultList(result && result.segments);
                const boundaryFrames = agentBoundaryFrames(result && result.boundary_frames);
                const candidateFrames = agentBoundaryFrames(result && result.candidate_frames);
                const warnings = agentResultList(result && result.warnings);
                const ch = result && result.channel_id != null ? `CH ${result.channel_id}` : 'CH ?';
                const transitionTotal = counts.transition_count ?? counts.total_transitions ?? counts.transition_total ?? counts.total ?? transitions.length;
                const label = `VISUAL STATE TRANSITIONS — ${ch} · ${transitionTotal} transition${Number(transitionTotal) === 1 ? '' : 's'}`;
                card.innerHTML = `<div class="agent-action-card-head">&#9670; ${escapeHtml(label)}</div>`;
                const body = document.createElement('div');
                body.className = 'agent-action-card-body';
                const addTitle = (text) => {
                    const title = document.createElement('div');
                    title.className = 'agent-evidence-title';
                    title.textContent = text;
                    body.appendChild(title);
                };

                const queryRows = [
                    ['Subject', result && result.subject_query],
                    ['Positive state', result && result.positive_state_query],
                    ['Negative state', result && result.negative_state_query],
                    ['Effective negative', result && result.negative_state_query_effective],
                ].filter(([, value]) => value !== undefined && value !== null && String(value).trim() !== '');
                if (queryRows.length) {
                    const queryWrap = document.createElement('div');
                    queryWrap.className = 'agent-probe-update-row';
                    queryWrap.innerHTML = queryRows.map(([key, value]) => (
                        `<div class="agent-probe-update-field"><span class="agent-probe-update-key">${escapeHtml(key)}:</span><span class="agent-probe-update-val">${escapeHtml(agentCompactValue(value))}</span></div>`
                    )).join('');
                    body.appendChild(queryWrap);
                }

                addTitle('Transition counts');
                const countEntries = agentTransitionCountEntries(counts);
                if (countEntries.length) {
                    const countWrap = document.createElement('div');
                    countWrap.className = 'agent-probe-update-row';
                    countWrap.innerHTML = countEntries.map(([key, value]) => (
                        `<div class="agent-probe-update-field"><span class="agent-probe-update-key">${escapeHtml(agentHumanizeKey(key))}:</span><span class="agent-probe-update-val">${escapeHtml(agentCompactValue(value))}</span></div>`
                    )).join('');
                    body.appendChild(countWrap);
                } else {
                    const emptyCounts = document.createElement('div');
                    emptyCounts.className = 'agent-card-muted-note';
                    emptyCounts.textContent = 'No transition counts returned.';
                    body.appendChild(emptyCounts);
                }

                appendAgentCompleteness(body, result, {
                    alwaysCoverage: true,
                    alwaysBackendTruncation: true,
                    alwaysTruncation: true,
                    alwaysScope: true,
                });

                if (boundaryFrames.length) {
                    addTitle(`Boundary frames · ${boundaryFrames.length}`);
                    appendAgentThumbGrid(body, boundaryFrames, {
                        gridClass: 'agent-det-grid agent-evidence-grid',
                        thumbClass: 'agent-det-thumb agent-evidence-thumb',
                        limit: 8,
                        scoreFormatter: (item) => {
                            const role = agentFirstValue(item, ['role', 'state', 'transition_type', 'type']);
                            return role !== '' ? agentHumanizeKey(role) : null;
                        },
                    });
                }
                if (candidateFrames.length) {
                    addTitle(`Top candidate frames · ${candidateFrames.length}`);
                    appendAgentThumbGrid(body, candidateFrames, {
                        gridClass: 'agent-det-grid agent-evidence-grid',
                        thumbClass: 'agent-det-thumb agent-evidence-thumb',
                        limit: 8,
                        scoreFormatter: (item) => {
                            const p = Number(item && item.positive_score);
                            if (Number.isFinite(p)) return `P ${p.toFixed(3)}`;
                            const state = agentFirstValue(item, ['state', 'role']);
                            return state !== '' ? agentHumanizeKey(state) : null;
                        },
                    });
                }

                if (transitions.length) {
                    const transitionLimit = 8;
                    addTitle(`Transitions · ${Math.min(transitions.length, transitionLimit)} of ${transitions.length}`);
                    const wrap = document.createElement('div');
                    wrap.innerHTML = transitions.slice(0, transitionLimit).map((transition, idx) => {
                        const time = agentTimeLabel(transition);
                        const type = agentTransitionTypeLabel(transition);
                        const head = [time, type].filter(Boolean).join(' · ') || `#${idx + 1}`;
                        const stateChange = agentStateChangeText(transition);
                        const basis = agentFirstValue(transition, ['basis', 'evidence', 'method', 'source']);
                        const summary = agentFirstValue(transition, ['summary', 'description', 'note', 'reason']);
                        const score = agentScoreSummary(transition);
                        const text = [stateChange, basis, summary, score]
                            .map((value) => agentCompactValue(value))
                            .filter(Boolean)
                            .join(' · ') || 'No transition details.';
                        return `<div class="agent-summary-entry"><span class="agent-summary-ts">${escapeHtml(head)}</span><span class="agent-summary-text">${escapeHtml(text)}</span></div>`;
                    }).join('');
                    body.appendChild(wrap);
                    if (transitions.length > transitionLimit) {
                        const more = document.createElement('div');
                        more.className = 'agent-card-muted-note';
                        more.textContent = `+${transitions.length - transitionLimit} more transitions`;
                        body.appendChild(more);
                    }
                } else {
                    const emptyTransitions = document.createElement('div');
                    emptyTransitions.className = 'agent-card-muted-note';
                    emptyTransitions.textContent = 'No transitions returned.';
                    body.appendChild(emptyTransitions);
                }

                if (segments.length) {
                    const segmentLimit = 6;
                    addTitle(`Segments · ${Math.min(segments.length, segmentLimit)} of ${segments.length}`);
                    const wrap = document.createElement('div');
                    wrap.innerHTML = segments.slice(0, segmentLimit).map((segment, idx) => {
                        const time = agentTimeLabel(segment) || `#${idx + 1}`;
                        const state = agentFirstValue(segment, ['state', 'visual_state', 'label', 'classification']);
                        const head = [time, state !== '' ? agentCompactValue(state) : ''].filter(Boolean).join(' · ');
                        const summary = agentFirstValue(segment, ['summary', 'description', 'note']);
                        const frameCount = agentFirstValue(segment, ['frame_count', 'frames', 'sample_count', 'entries']);
                        const frameCountText = Array.isArray(frameCount) ? String(frameCount.length) : agentCompactValue(frameCount);
                        const score = agentScoreSummary(segment);
                        const details = [
                            summary,
                            frameCount !== '' ? `Frames: ${frameCountText}` : '',
                            score,
                        ].map((value) => agentCompactValue(value)).filter(Boolean).join(' · ') || 'No segment details.';
                        return `<div class="agent-summary-entry"><span class="agent-summary-ts">${escapeHtml(head)}</span><span class="agent-summary-text">${escapeHtml(details)}</span></div>`;
                    }).join('');
                    body.appendChild(wrap);
                    if (segments.length > segmentLimit) {
                        const more = document.createElement('div');
                        more.className = 'agent-card-muted-note';
                        more.textContent = `+${segments.length - segmentLimit} more segments`;
                        body.appendChild(more);
                    }
                }

                if (result && result.score_semantics) {
                    const scoreNote = document.createElement('div');
                    scoreNote.className = 'agent-card-muted-note';
                    scoreNote.textContent = `Score semantics: ${result.score_semantics}`;
                    body.appendChild(scoreNote);
                }

                if (warnings.length) {
                    const warn = document.createElement('div');
                    warn.className = 'agent-coverage-note agent-coverage-partial';
                    warn.textContent = `Warnings: ${warnings.map((item) => agentCompactValue(item)).filter(Boolean).join(' · ')}`;
                    body.appendChild(warn);
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
                appendAgentCompleteness(body, result, {
                    alwaysCoverage: true,
                    alwaysBackendTruncation: true,
                    alwaysTruncation: true,
                    alwaysScope: true,
                });
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
                    if (evt.name) {
                        bubble.currentToolName = String(evt.name);
                        setStreamingStatus(bubble, `Running ${evt.name}...`, 'working');
                    }
                    break;
                case 'tool_result':
                    appendActionCard(bubble, evt.name, evt.result);
                    if (bubble.currentToolName === String(evt.name || '')) bubble.currentToolName = '';
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
                    setStreamingStatus(
                        bubble,
                        bubble.currentToolName ? `Running ${bubble.currentToolName}...` : 'Still working...',
                        bubble.currentToolName ? 'working' : 'thinking'
                    );
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
                bubble.traceEl.hidden = !hasActions;
                bubble.traceEl.open = false;
            }
            promoteStandaloneAgentApprovalCards(bubble);
            syncAgentEvidenceIdCard(bubble);
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

        // ---- Analytics runtime list (context sidebar; not the camera inventory) ----
        function agentLabelAnalyticsStreams() {
            const el = elProbeList();
            const title = el && el.closest('.agent-ctx-section')
                ? el.closest('.agent-ctx-section').querySelector('.agent-ctx-title')
                : null;
            if (title) {
                title.textContent = 'Analytics Streams';
                title.title = 'Runtime VLM/probe analytics streams; this is not the full Luxriot camera inventory.';
            }
        }

        function agentAdmissionRow(admission) {
            if (!admission || typeof admission !== 'object') {
                return `<div class="agent-probe-mini">
                    <div class="agent-probe-dot warn"></div>
                    <span class="agent-probe-name">LM admission</span>
                    <span class="agent-probe-score">unavailable</span>
                </div>`;
            }
            const active = Number(admission.active || 0);
            const queued = Number(admission.queued || 0);
            const resources = Array.isArray(admission.resources) ? admission.resources : [];
            const oldest = resources.reduce((maxAge, row) => {
                const age = Number(row && row.oldest_queue_age_sec);
                return Number.isFinite(age) ? Math.max(maxAge, age) : maxAge;
            }, 0);
            const dotCls = queued > 0 ? 'warn' : (active > 0 ? 'on' : 'off');
            const score = `${active} active · ${queued} queued · oldest ${oldest.toFixed(1)}s`;
            const profiles = Array.isArray(admission.profiles) ? admission.profiles : [];
            const modelChecks = profiles.map((profile) => {
                if (profile && profile.model_match === false) {
                    const configured = String(profile.configured_model || 'unknown');
                    const servedModels = Array.isArray(profile.served_models)
                        ? profile.served_models.map((model) => String(model || '')).filter(Boolean)
                        : [];
                    const served = servedModels.length ? servedModels.join(', ') : 'none';
                    const message = `LM model mismatch: configured ${configured}, serving ${served}`;
                    return `<div class="agent-lm-model-badge mismatch" title="${escapeHtml(message)}">${escapeHtml(message)}</div>`;
                }
                if (profile && profile.model_match === 'unknown') {
                    return '<div class="agent-lm-model-badge unknown">model check unavailable</div>';
                }
                return '';
            }).filter(Boolean).join('');
            return `<div class="agent-probe-mini">
                <div class="agent-probe-dot ${dotCls}"></div>
                <span class="agent-probe-name">LM admission</span>
                <span class="agent-probe-score" title="Shared model admission queue">${escapeHtml(score)}</span>
            </div>${modelChecks}`;
        }

        function renderAgentAnalyticsStreams(data, admission) {
            const el = elProbeList();
            if (!el) return;
            const videoStreams = Array.isArray(data.video_streams) ? data.video_streams : [];
            const analyticsStreams = Array.isArray(data.analytics_streams) ? data.analytics_streams : [];
            const missing = Array.isArray(data.desired_video_missing) ? data.desired_video_missing : [];
            const intro = '<div class="agent-probe-empty">Runtime analytics only — not the full camera inventory.</div>';
            const videoRows = videoStreams.map((stream) => {
                const running = Boolean(stream.running);
                const lastError = String(stream.last_error || stream.last_restore_error || '').trim();
                const dotCls = lastError ? 'warn' : (running ? 'on' : 'off');
                const channel = stream.channel_id ?? '?';
                const model = String(stream.model || 'default');
                const pending = Number(stream.pending_frames || 0);
                const dropped = Number(stream.dropped_frames || 0) + Number(stream.queue_dropped_batches || 0);
                const logs = Number(stream.log_count || 0);
                const score = lastError ? 'error' : `${pending}q · ${logs}s${dropped ? ` · ${dropped}d` : ''}`;
                return `<div class="agent-probe-mini">
                    <div class="agent-probe-dot ${dotCls}"></div>
                    <span class="agent-probe-name">VLM CH ${escapeHtml(String(channel))} · ${escapeHtml(model)}</span>
                    <span class="agent-probe-score" title="${escapeHtml(lastError || 'pending · summaries · dropped')}">${escapeHtml(score)}</span>
                </div>`;
            });
            const probeRows = analyticsStreams.map((stream) => {
                const running = Boolean(stream.running);
                const lastError = String(stream.last_error || '').trim();
                const dotCls = lastError ? 'warn' : (running ? 'on' : 'off');
                const channel = stream.channel_id ?? '?';
                const buffered = Number(stream.pending_frames || 0);
                const shared = Boolean(stream.shared_capture);
                const score = lastError ? 'error' : `${buffered} buffered${shared ? ' · shared' : ''}`;
                return `<div class="agent-probe-mini">
                    <div class="agent-probe-dot ${dotCls}"></div>
                    <span class="agent-probe-name">Probe analytics CH ${escapeHtml(String(channel))}</span>
                    <span class="agent-probe-score" title="${escapeHtml(lastError || 'probe analytics capture runtime')}">${escapeHtml(score)}</span>
                </div>`;
            });
            const missingRows = missing.map((row) => {
                const channel = row.channel_id ?? '?';
                const error = String(row.last_restore_error || '').trim();
                return `<div class="agent-probe-mini">
                    <div class="agent-probe-dot warn"></div>
                    <span class="agent-probe-name">VLM CH ${escapeHtml(String(channel))} · desired</span>
                    <span class="agent-probe-score" title="${escapeHtml(error || 'desired analytics stream is not running')}">missing</span>
                </div>`;
            });
            const rows = [agentAdmissionRow(admission), ...videoRows, ...probeRows, ...missingRows];
            if (rows.length === 1 && !videoStreams.length && !analyticsStreams.length && !missing.length) {
                rows.push('<div class="agent-probe-empty">No VLM or probe analytics streams running.</div>');
            }
            el.innerHTML = intro + rows.join('');
        }

        async function agentLoadAnalyticsStreams() {
            const el = elProbeList();
            if (!el || !_agentContextActive || currentMode !== 'agent') return false;
            _agentContextRequestGeneration += 1;
            abortUiRequest(_agentContextAbortController);
            const generation = _agentContextRequestGeneration;
            const controller = new AbortController();
            _agentContextAbortController = controller;
            try {
                const [streamsResponse, admissionResponse] = await Promise.all([
                    fetch(`/luxriot/streams?t=${Date.now()}`, {
                        cache: 'no-store',
                        signal: controller.signal,
                    }),
                    fetch(`/lm/admission?t=${Date.now()}`, {
                        cache: 'no-store',
                        signal: controller.signal,
                    }),
                ]);
                const streamsData = await streamsResponse.json().catch(() => ({}));
                const admissionData = await admissionResponse.json().catch(() => ({}));
                if (!streamsResponse.ok || streamsData.error) {
                    throw new Error(streamsData.error || 'Failed to load analytics streams');
                }
                if (
                    generation !== _agentContextRequestGeneration
                    || controller.signal.aborted
                    || !_agentContextActive
                    || currentMode !== 'agent'
                ) return false;
                renderAgentAnalyticsStreams(
                    streamsData,
                    admissionResponse.ok && !admissionData.error ? admissionData : null,
                );
                return true;
            } catch (error) {
                if (error && error.name === 'AbortError') return false;
                if (generation !== _agentContextRequestGeneration || !_agentContextActive || currentMode !== 'agent') return false;
                el.innerHTML = '<div class="agent-probe-empty">Analytics stream runtime unavailable</div>';
                return false;
            } finally {
                if (_agentContextAbortController === controller) {
                    _agentContextAbortController = null;
                }
            }
        }

        function agentSetContextActive(active) {
            _agentContextActive = Boolean(active);
            if (_agentContextTimer) {
                clearInterval(_agentContextTimer);
                _agentContextTimer = null;
            }
            if (!_agentContextActive) {
                _agentContextRequestGeneration += 1;
                abortUiRequest(_agentContextAbortController);
                _agentContextAbortController = null;
                return;
            }
            agentLabelAnalyticsStreams();
            void agentLoadAnalyticsStreams();
            _agentContextTimer = window.setInterval(() => {
                if (_agentContextActive && currentMode === 'agent') {
                    void agentLoadAnalyticsStreams();
                }
            }, 8000);
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
                    if (target.closest('[data-open-image-link]')) return;
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
            if (agentSkillModal) {
                agentSkillModal.addEventListener('click', (event) => {
                    if (event.target === agentSkillModal) {
                        closeAgentSkillModal();
                    }
                });
            }
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

            agentSetContextActive(currentMode === 'agent');
        }

        // Expose agentInit to outer scope
        window._agentInit = agentInit;
        window._agentSetActive = agentSetContextActive;
    })();

    function agentInit() {
        if (window._agentInit) window._agentInit();
    }
