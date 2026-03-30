/**
 * WWTP Lab Report Dashboard — App Logic
 * Loads JSON data and renders all charts using Chart.js.
 */

// ─── Constants ───────────────────────────────────────────────────────────────

const MONTHS = ['july', 'august', 'september', 'october', 'november', 'december'];

const STAGES_DEFAULT = ['Inlet', 'Primary', 'Secondary', 'Sec. Sed.', 'Effluent'];
const STAGES_TSS = ['Inlet', 'Grit', 'Primary', 'Secondary', 'Sec. Sed.', 'Effluent'];

const STAGE_FIELDS = {
    ph: {
        stages: STAGES_DEFAULT,
        fields: ['inlet_ph', 'primary_ph', 'secondary_ph', 'sec_sed_ph', 'effluent_ph'],
    },
    bod: {
        stages: STAGES_DEFAULT,
        fields: ['inlet_bod', 'primary_bod', 'secondary_bod', 'sec_sed_bod', 'effluent_bod'],
    },
    cod: {
        stages: STAGES_DEFAULT,
        fields: ['inlet_cod', 'primary_cod', 'secondary_cod', 'sec_sed_cod', 'effluent_cod'],
    },
    tss: {
        stages: STAGES_TSS,
        fields: ['inlet_tss', 'grit_tss', 'primary_tss', 'secondary_tss', 'sec_sed_tss', 'effluent_tss'],
    },
};

const CONTROL_LIMITS_DAILY = {
    ph: [
        { min: 6.0, max: 9.0, stageIdx: 0, label: 'Inlet: 6.0–9.0' },
        { min: 6.5, max: 8.0, stageIdx: 4, label: 'Effluent: 6.5–8.0' },
    ],
    bod: [
        { value: 300, stageIdx: 0, label: 'Inlet limit: 300' },
        { value: 10, stageIdx: 4, label: 'Effluent limit: <10' },
    ],
    cod: [
        { value: 800, stageIdx: 0, label: 'Inlet limit: 800' },
        { value: 250, stageIdx: 4, label: 'Effluent limit: <250' },
    ],
    tss: [
        { value: 400, stageIdx: 0, label: 'Inlet limit: 400' },
        { value: 10, stageIdx: 5, label: 'Effluent limit: <10' },
    ],
};

const TRACKED_FIELDS = [
    'inlet_ph', 'inlet_bod', 'inlet_cod', 'inlet_tss',
    'grit_tss',
    'primary_ph', 'primary_tss', 'primary_bod', 'primary_cod',
    'secondary_ph', 'secondary_tss', 'secondary_bod', 'secondary_cod',
    'sec_sed_ph', 'sec_sed_tss', 'sec_sed_bod', 'sec_sed_cod',
    'effluent_ph', 'effluent_bod', 'effluent_cod', 'effluent_tss',
    'effluent_frc',
];

// Attribute groups for the monthly missing-values breakdown chart
const MISSING_GROUPS = [
    { label: 'pH',    color: '#2563eb', fields: ['inlet_ph', 'primary_ph', 'secondary_ph', 'sec_sed_ph', 'effluent_ph'] },
    { label: 'BOD',   color: '#9333ea', fields: ['inlet_bod', 'primary_bod', 'secondary_bod', 'sec_sed_bod', 'effluent_bod'] },
    { label: 'COD',   color: '#16a34a', fields: ['inlet_cod', 'primary_cod', 'secondary_cod', 'sec_sed_cod', 'effluent_cod'] },
    { label: 'TSS',   color: '#d97706', fields: ['inlet_tss', 'grit_tss', 'primary_tss', 'secondary_tss', 'sec_sed_tss', 'effluent_tss'] },
    { label: 'FRC',   color: '#0891b2', fields: ['effluent_frc'] },
    { label: 'Flow',  color: '#6b7280', fields: ['flow'] },
    { label: 'Power', color: '#374151', fields: ['power_nea', 'power_ge', 'power_total'] },
];

const COMPLIANCE_PARAMS = [
    { key: 'effluent_ph', label: 'pH', type: 'range', min: 6.5, max: 8.0 },
    { key: 'effluent_bod', label: 'BOD₅', type: 'max', limit: 10 },
    { key: 'effluent_cod', label: 'COD', type: 'max', limit: 250 },
    { key: 'effluent_tss', label: 'TSS', type: 'max', limit: 10 },
];

// Stage series for monthly parameter trend charts
const MONTHLY_STAGE_SERIES = {
    ph: [
        { key: 'inlet_ph',     label: 'Inlet',     color: '#2563eb' },
        { key: 'primary_ph',   label: 'Primary',   color: '#7c3aed' },
        { key: 'secondary_ph', label: 'Secondary', color: '#16a34a' },
        { key: 'sec_sed_ph',   label: 'Sec. Sed.', color: '#d97706' },
        { key: 'effluent_ph',  label: 'Effluent',  color: '#dc2626' },
    ],
    bod: [
        { key: 'inlet_bod',     label: 'Inlet',     color: '#2563eb' },
        { key: 'primary_bod',   label: 'Primary',   color: '#7c3aed' },
        { key: 'secondary_bod', label: 'Secondary', color: '#16a34a' },
        { key: 'sec_sed_bod',   label: 'Sec. Sed.', color: '#d97706' },
        { key: 'effluent_bod',  label: 'Effluent',  color: '#dc2626' },
    ],
    cod: [
        { key: 'inlet_cod',     label: 'Inlet',     color: '#2563eb' },
        { key: 'primary_cod',   label: 'Primary',   color: '#7c3aed' },
        { key: 'secondary_cod', label: 'Secondary', color: '#16a34a' },
        { key: 'sec_sed_cod',   label: 'Sec. Sed.', color: '#d97706' },
        { key: 'effluent_cod',  label: 'Effluent',  color: '#dc2626' },
    ],
    tss: [
        { key: 'inlet_tss',     label: 'Inlet',     color: '#2563eb' },
        { key: 'grit_tss',      label: 'Grit',      color: '#0891b2' },
        { key: 'primary_tss',   label: 'Primary',   color: '#7c3aed' },
        { key: 'secondary_tss', label: 'Secondary', color: '#16a34a' },
        { key: 'sec_sed_tss',   label: 'Sec. Sed.', color: '#d97706' },
        { key: 'effluent_tss',  label: 'Effluent',  color: '#dc2626' },
    ],
};

// Limit line colors — amber for inlet, purple for effluent.
// Both are distinct from the blue-first day palette and from each other.
const LIMIT_INLET_COLOR    = '#f59e0b'; // amber
const LIMIT_EFFLUENT_COLOR = '#7c3aed'; // purple

const LIMIT_LINE_DATA = {
    ph:  { inlet: { value: null, label: 'Inlet pH: 6.0–9.0' },       effluent: { value: null, label: 'Effluent pH: 6.5–8.0' } },
    bod: { inlet: { value: 300,  label: 'Inlet limit: 300 mg/L' },    effluent: { value: 10,   label: 'Effluent limit: 10 mg/L' } },
    cod: { inlet: { value: 800,  label: 'Inlet limit: 800 mg/L' },    effluent: { value: 250,  label: 'Effluent limit: 250 mg/L' } },
    tss: { inlet: { value: 400,  label: 'Inlet limit: 400 mg/L' },    effluent: { value: 10,   label: 'Effluent limit: 10 mg/L' } },
};

// Blue-first palette — first 8 entries are distinct blue shades, then expands
// to teals, purples, greens, ambers, and pinks. Red/orange tones are excluded
// so day lines never clash with limit annotation colors.
const DAY_COLORS = [
    '#1d4ed8', '#2563eb', '#0ea5e9', '#3b82f6',  // deep → sky blue
    '#1e3a8a', '#0284c7', '#0369a1', '#60a5fa',  // navy → pale blue
    '#0891b2', '#06b6d4', '#0e7490', '#14b8a6',  // teal / cyan
    '#0f766e', '#7c3aed', '#6d28d9', '#9333ea',  // teal-green + purples
    '#4f46e5', '#4338ca', '#a855f7', '#16a34a',  // indigo + green
    '#059669', '#15803d', '#65a30d', '#ca8a04',  // greens + amber
    '#b45309', '#d97706', '#db2777', '#be185d',  // amber + pinks
    '#9d174d', '#ec4899', '#c026d3',             // dark pink + magenta
];

// ─── State ───────────────────────────────────────────────────────────────────

let allData = {};
let charts = {};
let stagePickerMeta = {}; // canvasId -> { paramKey, seriesCount, effluentIdx }

// ─── Init ────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
    // Close any open picker when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.day-picker')) {
            document.querySelectorAll('details.day-picker').forEach(d => d.open = false);
        }
    });

    const yearSelect  = document.getElementById('year-select');
    const monthSelect = document.getElementById('month-select');

    await loadAllData(yearSelect.value);

    yearSelect.addEventListener('change', async () => {
        await loadAllData(yearSelect.value);
        renderAll(monthSelect.value);
    });
    monthSelect.addEventListener('change', () => renderAll(monthSelect.value));
    renderAll(monthSelect.value);
});

async function loadAllData(year) {
    const path = year == '2021'
        ? '../data/all_months.json'
        : `../../${year}/data/all_months.json`;
    try {
        const resp = await fetch(path);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        allData = await resp.json();
    } catch {
        allData = {};
    }
}

function renderAll(month) {
    const data = allData[month];
    if (!data) return;

    // Daily trend charts + day pickers
    renderDailyTrend('chart-ph', 'ph', data, 'pH');
    initDayPicker('chart-ph', data.days, data.month);
    renderDailyTrend('chart-bod', 'bod', data, 'mg/L');
    initDayPicker('chart-bod', data.days, data.month);
    renderDailyTrend('chart-cod', 'cod', data, 'mg/L');
    initDayPicker('chart-cod', data.days, data.month);
    renderDailyTrend('chart-tss', 'tss', data, 'mg/L');
    initDayPicker('chart-tss', data.days, data.month);

    // Monthly overview charts
    renderFlowChart(data);
    renderPowerChart(data);
    renderPowerFlowChart(data);
    renderMissingChart(data);


    // Monthly parameter trend charts (one line per stage, x = days)
    renderMonthlyParam('chart-monthly-ph', 'ph', 'pH', data);
    initStagePicker('chart-monthly-ph', MONTHLY_STAGE_SERIES['ph'].map(s => s.label), 'ph');
    renderMonthlyParam('chart-monthly-bod', 'bod', 'mg/L', data);
    initStagePicker('chart-monthly-bod', MONTHLY_STAGE_SERIES['bod'].map(s => s.label), 'bod');
    renderMonthlyParam('chart-monthly-cod', 'cod', 'mg/L', data);
    initStagePicker('chart-monthly-cod', MONTHLY_STAGE_SERIES['cod'].map(s => s.label), 'cod');
    renderMonthlyParam('chart-monthly-tss', 'tss', 'mg/L', data);
    initStagePicker('chart-monthly-tss', MONTHLY_STAGE_SERIES['tss'].map(s => s.label), 'tss');

    renderComplianceGrid(data);
    renderEfficiencyChart(data);
}

// ─── Daily Trend Charts ──────────────────────────────────────────────────────

function renderDailyTrend(canvasId, paramKey, monthData, yLabel) {
    const config = STAGE_FIELDS[paramKey];
    const days = monthData.days;
    const limitInfo = LIMIT_LINE_DATA[paramKey];

    if (charts[canvasId]) charts[canvasId].destroy();

    // Day datasets (one line per day)
    const datasets = days.map((day, i) => {
        const values = config.fields.map(f => day[f]);
        return {
            label: formatDateLong(day.date, monthData.month),
            data: values,
            borderColor: DAY_COLORS[i % DAY_COLORS.length],
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            pointRadius: values.map(v => v === null ? 6 : 3),
            pointStyle: values.map(v => v === null ? 'triangle' : 'circle'),
            pointBackgroundColor: values.map(v => v === null ? '#d97706' : DAY_COLORS[i % DAY_COLORS.length]),
            pointBorderColor: values.map(v => v === null ? '#d97706' : DAY_COLORS[i % DAY_COLORS.length]),
            tension: 0.2,
            spanGaps: false,
            hidden: i !== 0,
            isLimit: false,
        };
    });

    // Legend-only datasets — empty data so they don't pin the y-axis range,
    // but they appear in the legend with the correct color and dash style.
    const limitDatasets = [
        {
            label: limitInfo.inlet.label,
            data: [],
            borderColor: LIMIT_INLET_COLOR,
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            borderDash: [6, 4],
            pointRadius: 0,
            isLimit: true,
        },
        {
            label: limitInfo.effluent.label,
            data: [],
            borderColor: LIMIT_EFFLUENT_COLOR,
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            borderDash: [6, 4],
            pointRadius: 0,
            isLimit: true,
        },
    ];

    // Annotations: colored lines/boxes matching legend colors — no text labels
    const annotations = {};
    if (paramKey === 'ph') {
        annotations['inlet_band'] = {
            type: 'box',
            yMin: 6.0,
            yMax: 9.0,
            backgroundColor: 'rgba(245, 158, 11, 0.08)',
            borderColor: 'rgba(245, 158, 11, 0.55)',
            borderWidth: 1.5,
        };
        annotations['effluent_band'] = {
            type: 'box',
            yMin: 6.5,
            yMax: 8.0,
            backgroundColor: 'rgba(124, 58, 237, 0.08)',
            borderColor: 'rgba(124, 58, 237, 0.6)',
            borderWidth: 1.5,
            borderDash: [5, 3],
        };
    } else {
        if (limitInfo.inlet.value !== null) {
            annotations['inlet_limit'] = {
                type: 'line',
                yMin: limitInfo.inlet.value,
                yMax: limitInfo.inlet.value,
                borderColor: LIMIT_INLET_COLOR,
                borderWidth: 1.5,
                borderDash: [6, 4],
            };
        }
        if (limitInfo.effluent.value !== null) {
            annotations['effluent_limit'] = {
                type: 'line',
                yMin: limitInfo.effluent.value,
                yMax: limitInfo.effluent.value,
                borderColor: LIMIT_EFFLUENT_COLOR,
                borderWidth: 1.5,
                borderDash: [6, 4],
            };
        }
    }

    const ctx = document.getElementById(canvasId).getContext('2d');
    charts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: { labels: config.stages, datasets: [...datasets, ...limitDatasets] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'nearest', intersect: false },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        filter: (item, data) => data.datasets[item.datasetIndex]?.isLimit === true,
                        boxWidth: 28,
                        boxHeight: 12,
                        padding: 10,
                        font: { size: 11 },
                    },
                },
                tooltip: {
                    callbacks: {
                        title: (items) => {
                            const ds = items[0]?.dataset;
                            return ds?.isLimit ? null : (ds?.label || '');
                        },
                        label: (item) => {
                            if (item.dataset.isLimit) return null;
                            const val = item.parsed.y;
                            if (val === null || isNaN(val)) return `${config.stages[item.dataIndex]}: ⚠ No data`;
                            return `${config.stages[item.dataIndex]}: ${val}`;
                        },
                    },
                },
                annotation: { annotations },
            },
            scales: {
                x: {
                    title: { display: true, text: 'Treatment Stage', font: { size: 11 } },
                    grid: { display: false },
                },
                y: {
                    title: { display: true, text: yLabel, font: { size: 11 } },
                    grace: '5%',
                    grid: { color: '#f0f0f0' },
                },
            },
        },
    });
}

// ─── Day Picker ──────────────────────────────────────────────────────────────

function initDayPicker(canvasId, days, monthName) {
    const container = document.getElementById(`picker-${canvasId}`);
    if (!container) return;

    const picker = document.createElement('details');
    picker.className = 'day-picker';
    picker.id = `details-${canvasId}`;

    const summary = document.createElement('summary');
    summary.id = `summary-${canvasId}`;
    summary.textContent = `All ${days.length} days ▾`;
    picker.appendChild(summary);

    const panel = document.createElement('div');
    panel.className = 'day-picker-panel';

    // Select All / None buttons
    const actions = document.createElement('div');
    actions.className = 'day-picker-actions';
    const allBtn = document.createElement('button');
    allBtn.textContent = 'All';
    allBtn.type = 'button';
    allBtn.addEventListener('click', (e) => { e.stopPropagation(); setAllDays(canvasId, true); });
    const noneBtn = document.createElement('button');
    noneBtn.textContent = 'None';
    noneBtn.type = 'button';
    noneBtn.addEventListener('click', (e) => { e.stopPropagation(); setAllDays(canvasId, false); });
    actions.appendChild(allBtn);
    actions.appendChild(noneBtn);
    panel.appendChild(actions);

    // Day checkboxes
    const list = document.createElement('div');
    list.className = 'day-picker-list';
    list.id = `day-list-${canvasId}`;

    const chart = charts[canvasId];
    days.forEach((day, i) => {
        const isVisible = chart ? chart.isDatasetVisible(i) : true;
        const label = document.createElement('label');
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = isVisible;
        cb.addEventListener('change', () => applyDayVisibility(canvasId));
        label.appendChild(cb);
        label.appendChild(document.createTextNode(` ${formatDateLong(day.date, monthName)}`));
        list.appendChild(label);
    });

    panel.appendChild(list);
    picker.appendChild(panel);
    container.innerHTML = '';
    container.appendChild(picker);
    updatePickerSummary(canvasId);
}

function setAllDays(canvasId, checked) {
    const list = document.getElementById(`day-list-${canvasId}`);
    if (!list) return;
    list.querySelectorAll('input[type=checkbox]').forEach(cb => { cb.checked = checked; });
    applyDayVisibility(canvasId);
}

function applyDayVisibility(canvasId) {
    const chart = charts[canvasId];
    if (!chart) return;
    const list = document.getElementById(`day-list-${canvasId}`);
    if (!list) return;
    list.querySelectorAll('input[type=checkbox]').forEach((cb, i) => {
        chart.setDatasetVisibility(i, cb.checked);
    });
    chart.update();
    updatePickerSummary(canvasId);
}

function updatePickerSummary(canvasId) {
    const list = document.getElementById(`day-list-${canvasId}`);
    if (!list) return;
    const checkboxes = list.querySelectorAll('input[type=checkbox]');
    const total = checkboxes.length;
    const checked = [...checkboxes].filter(cb => cb.checked).length;
    const summary = document.getElementById(`summary-${canvasId}`);
    if (summary) {
        summary.textContent = checked === total
            ? `All ${total} days ▾`
            : `${checked} of ${total} days ▾`;
    }
}

// ─── Monthly Overview Charts ─────────────────────────────────────────────────

function renderFlowChart(monthData) {
    const id = 'chart-flow';
    if (charts[id]) charts[id].destroy();

    const days = monthData.days;
    const labels = days.map(d => formatDate(d.date));
    const values = days.map(d => d.flow);

    const ctx = document.getElementById(id).getContext('2d');
    charts[id] = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Raw Sewage Flow (MLD)',
                    data: values,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37,99,235,0.08)',
                    fill: true,
                    borderWidth: 2,
                    pointRadius: values.map(v => v === null ? 6 : 2),
                    pointStyle: values.map(v => v === null ? 'triangle' : 'circle'),
                    pointBackgroundColor: values.map(v => v === null ? '#d97706' : '#2563eb'),
                    pointBorderColor: values.map(v => v === null ? '#d97706' : '#2563eb'),
                    tension: 0.3,
                    spanGaps: true,
                    isLimit: false,
                },
                {
                    label: 'Design capacity: 32.4 MLD',
                    data: [],
                    borderColor: '#dc2626',
                    backgroundColor: 'transparent',
                    borderWidth: 1.5,
                    borderDash: [6, 4],
                    pointRadius: 0,
                    isLimit: true,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        filter: (item, data) => data.datasets[item.datasetIndex]?.isLimit === true,
                        boxWidth: 28,
                        boxHeight: 12,
                        padding: 10,
                        font: { size: 11 },
                    },
                },
                annotation: {
                    annotations: {
                        flowLimit: {
                            type: 'line',
                            yMin: 32.4,
                            yMax: 32.4,
                            borderColor: '#dc2626',
                            borderWidth: 1.5,
                            borderDash: [6, 4],
                        },
                    },
                },
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 10 } } },
                y: {
                    title: { display: true, text: 'MLD', font: { size: 11 } },
                    grace: '5%',
                    grid: { color: '#f0f0f0' },
                },
            },
        },
    });
}

function renderPowerChart(monthData) {
    const id = 'chart-power';
    if (charts[id]) charts[id].destroy();

    const days = monthData.days;
    const labels = days.map(d => formatDate(d.date));

    const ctx = document.getElementById(id).getContext('2d');
    charts[id] = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Total (NEA + GE)',
                    data: days.map(d => d.power_total),
                    borderColor: '#1a1a1a',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: days.map(d => d.power_total === null ? 6 : 2),
                    pointStyle: days.map(d => d.power_total === null ? 'triangle' : 'circle'),
                    pointBackgroundColor: days.map(d => d.power_total === null ? '#d97706' : '#1a1a1a'),
                    pointBorderColor: days.map(d => d.power_total === null ? '#d97706' : '#1a1a1a'),
                    tension: 0.3,
                    spanGaps: true,
                },
                {
                    label: 'NEA (Grid)',
                    data: days.map(d => d.power_nea),
                    borderColor: '#6b7280',
                    backgroundColor: 'transparent',
                    borderWidth: 1.5,
                    borderDash: [5, 3],
                    pointRadius: days.map(d => d.power_nea === null ? 6 : 1.5),
                    pointStyle: days.map(d => d.power_nea === null ? 'triangle' : 'circle'),
                    pointBackgroundColor: days.map(d => d.power_nea === null ? '#d97706' : '#6b7280'),
                    pointBorderColor: days.map(d => d.power_nea === null ? '#d97706' : '#6b7280'),
                    tension: 0.3,
                    spanGaps: true,
                },
                {
                    label: 'Gas Engine',
                    data: days.map(d => d.power_ge),
                    borderColor: '#9ca3af',
                    backgroundColor: 'transparent',
                    borderWidth: 1.5,
                    borderDash: [2, 2],
                    pointRadius: days.map(d => d.power_ge === null ? 6 : 1.5),
                    pointStyle: days.map(d => d.power_ge === null ? 'triangle' : 'circle'),
                    pointBackgroundColor: days.map(d => d.power_ge === null ? '#d97706' : '#9ca3af'),
                    pointBorderColor: days.map(d => d.power_ge === null ? '#d97706' : '#9ca3af'),
                    tension: 0.3,
                    spanGaps: true,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: { font: { size: 11 }, usePointStyle: false, boxWidth: 20 },
                },
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 10 } } },
                y: {
                    title: { display: true, text: 'MW', font: { size: 11 } },
                    grace: '5%',
                    grid: { color: '#f0f0f0' },
                },
            },
        },
    });
}

function renderPowerFlowChart(monthData) {
    const id = 'chart-power-flow';
    if (charts[id]) charts[id].destroy();

    const days = monthData.days;
    const labels = days.map(d => formatDate(d.date));
    const values = days.map(d => d.power_per_flow);

    const ctx = document.getElementById(id).getContext('2d');
    charts[id] = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'KWh / ML',
                    data: values,
                    borderColor: '#2563eb',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: values.map(v => v === null ? 6 : 2),
                    pointStyle: values.map(v => v === null ? 'triangle' : 'circle'),
                    pointBackgroundColor: values.map(v => v === null ? '#d97706' : '#2563eb'),
                    pointBorderColor: values.map(v => v === null ? '#d97706' : '#2563eb'),
                    tension: 0.3,
                    spanGaps: true,
                    isLimit: false,
                },
                {
                    label: 'Baseline: 482.02 KWh/ML',
                    data: [],
                    borderColor: '#dc2626',
                    backgroundColor: 'transparent',
                    borderWidth: 1.5,
                    borderDash: [6, 4],
                    pointRadius: 0,
                    isLimit: true,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        filter: (item, data) => data.datasets[item.datasetIndex]?.isLimit === true,
                        boxWidth: 28,
                        boxHeight: 12,
                        padding: 10,
                        font: { size: 11 },
                    },
                },
                annotation: {
                    annotations: {
                        baseline: {
                            type: 'line',
                            yMin: 482.02,
                            yMax: 482.02,
                            borderColor: '#dc2626',
                            borderWidth: 1.5,
                            borderDash: [6, 4],
                        },
                    },
                },
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 10 } } },
                y: {
                    title: { display: true, text: 'KWh / ML', font: { size: 11 } },
                    beginAtZero: false,
                    grid: { color: '#f0f0f0' },
                },
            },
        },
    });
}

function renderMissingChart(monthData) {
    const id = 'chart-missing';
    if (charts[id]) charts[id].destroy();

    const days = monthData.days;
    const labels = days.map(d => formatDate(d.date));

    // One dataset per attribute group — grouped (side-by-side) bars per day
    const datasets = MISSING_GROUPS.map(group => ({
        label: group.label,
        data: days.map(day =>
            group.fields.filter(f => day[f] === null || day[f] === undefined).length
        ),
        backgroundColor: group.color,
        borderRadius: 2,
        borderSkipped: false,
    }));

    const ctx = document.getElementById(id).getContext('2d');
    charts[id] = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        font: { size: 11 }, boxWidth: 12, boxHeight: 12, padding: 10,
                        filter: (item, data) => data.datasets[item.datasetIndex].data.some(v => v > 0),
                    },
                },
                tooltip: {
                    callbacks: {
                        label: (item) => `${item.dataset.label}: ${item.parsed.y} missing`,
                        footer: (items) => {
                            const total = items.reduce((s, i) => s + i.parsed.y, 0);
                            return total > 0 ? `Total: ${total} missing` : 'No missing values';
                        },
                    },
                },
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 10 }, maxTicksLimit: 16, autoSkip: true } },
                y: {
                    title: { display: true, text: 'Missing values', font: { size: 11 } },
                    beginAtZero: true,
                    grace: '10%',
                    grid: { color: '#f0f0f0' },
                    ticks: { stepSize: 1 },
                },
            },
        },
    });
}

// ─── Monthly Parameter Trend Charts ──────────────────────────────────────────

function renderMonthlyParam(canvasId, paramKey, yLabel, monthData) {
    if (charts[canvasId]) charts[canvasId].destroy();

    const series = MONTHLY_STAGE_SERIES[paramKey];
    const limitInfo = LIMIT_LINE_DATA[paramKey];
    const days = monthData.days;
    const labels = days.map(d => formatDate(d.date));

    // Stage datasets — default only Effluent visible
    const stageDatasets = series.map(s => {
        const values = days.map(d => (d[s.key] === null || d[s.key] === undefined) ? null : d[s.key]);
        return {
            label: s.label,
            data: values,
            borderColor: s.color,
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: values.map(v => v === null ? 5 : 2),
            pointStyle: values.map(v => v === null ? 'triangle' : 'circle'),
            pointBackgroundColor: values.map(v => v === null ? '#d97706' : s.color),
            pointBorderColor: values.map(v => v === null ? '#d97706' : s.color),
            tension: 0.3,
            spanGaps: true,
            hidden: s.label !== 'Effluent',
            isLimit: false,
        };
    });

    // Legend-only limit datasets (empty data → don't affect y-axis scale)
    // Inlet hidden by default since only Effluent stage is selected on load
    const limitDatasets = [
        {
            label: limitInfo.inlet.label,
            data: [],
            borderColor: LIMIT_INLET_COLOR,
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            borderDash: [6, 4],
            pointRadius: 0,
            hidden: true,
            isLimit: true,
        },
        {
            label: limitInfo.effluent.label,
            data: [],
            borderColor: LIMIT_EFFLUENT_COLOR,
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            borderDash: [6, 4],
            pointRadius: 0,
            hidden: false,
            isLimit: true,
        },
    ];

    // Annotations matching limit colors — no text labels
    // Inlet annotation hidden by default (Inlet stage not selected on load)
    const annotations = {};
    if (paramKey === 'ph') {
        annotations['inlet_band'] = {
            type: 'box',
            yMin: 6.0, yMax: 9.0,
            backgroundColor: 'rgba(245, 158, 11, 0.08)',
            borderColor: 'rgba(245, 158, 11, 0.55)',
            borderWidth: 1.5,
            display: false,
        };
        annotations['effluent_band'] = {
            type: 'box',
            yMin: 6.5, yMax: 8.0,
            backgroundColor: 'rgba(124, 58, 237, 0.08)',
            borderColor: 'rgba(124, 58, 237, 0.6)',
            borderWidth: 1.5,
            borderDash: [5, 3],
            display: true,
        };
    } else {
        if (limitInfo.inlet.value !== null) {
            annotations['inlet_limit'] = {
                type: 'line',
                yMin: limitInfo.inlet.value, yMax: limitInfo.inlet.value,
                borderColor: LIMIT_INLET_COLOR,
                borderWidth: 1.5,
                borderDash: [6, 4],
                display: false,
            };
        }
        if (limitInfo.effluent.value !== null) {
            annotations['effluent_limit'] = {
                type: 'line',
                yMin: limitInfo.effluent.value, yMax: limitInfo.effluent.value,
                borderColor: LIMIT_EFFLUENT_COLOR,
                borderWidth: 1.5,
                borderDash: [6, 4],
                display: true,
            };
        }
    }

    const ctx = document.getElementById(canvasId).getContext('2d');
    charts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets: [...stageDatasets, ...limitDatasets] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        // Only show limit entries that are currently visible (not hidden by stage picker)
                        filter: (item, data) => data.datasets[item.datasetIndex]?.isLimit === true && !item.hidden,
                        boxWidth: 28,
                        boxHeight: 12,
                        padding: 10,
                        font: { size: 11 },
                    },
                },
                tooltip: {
                    callbacks: {
                        label: (item) => {
                            if (item.dataset.isLimit) return null;
                            const val = item.parsed.y;
                            if (val === null || isNaN(val)) return `${item.dataset.label}: ⚠ No data`;
                            return `${item.dataset.label}: ${val}`;
                        },
                    },
                },
                annotation: { annotations },
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 10 } } },
                y: {
                    title: { display: true, text: yLabel, font: { size: 11 } },
                    grace: '5%',
                    grid: { color: '#f0f0f0' },
                },
            },
        },
    });
}

// ─── Stage Picker (for monthly param charts) ──────────────────────────────────

function initStagePicker(canvasId, seriesLabels, paramKey) {
    const container = document.getElementById(`picker-${canvasId}`);
    if (!container) return;

    // Store metadata so applyStageVisibility can sync limit lines
    stagePickerMeta[canvasId] = {
        paramKey,
        seriesCount: seriesLabels.length,
        effluentIdx: seriesLabels.length - 1,
    };

    const picker = document.createElement('details');
    picker.className = 'day-picker';
    picker.id = `stage-details-${canvasId}`;

    const summary = document.createElement('summary');
    summary.id = `stage-summary-${canvasId}`;
    picker.appendChild(summary);

    const panel = document.createElement('div');
    panel.className = 'day-picker-panel';

    const actions = document.createElement('div');
    actions.className = 'day-picker-actions';
    const allBtn = document.createElement('button');
    allBtn.textContent = 'All';
    allBtn.type = 'button';
    allBtn.addEventListener('click', (e) => { e.stopPropagation(); setAllStages(canvasId, true); });
    const noneBtn = document.createElement('button');
    noneBtn.textContent = 'None';
    noneBtn.type = 'button';
    noneBtn.addEventListener('click', (e) => { e.stopPropagation(); setAllStages(canvasId, false); });
    actions.appendChild(allBtn);
    actions.appendChild(noneBtn);
    panel.appendChild(actions);

    const list = document.createElement('div');
    list.className = 'day-picker-list';
    list.id = `stage-list-${canvasId}`;

    const chart = charts[canvasId];
    seriesLabels.forEach((label, i) => {
        const isVisible = chart ? chart.isDatasetVisible(i) : (label === 'Effluent');
        const labelEl = document.createElement('label');
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = isVisible;
        cb.addEventListener('change', () => applyStageVisibility(canvasId));
        labelEl.appendChild(cb);
        labelEl.appendChild(document.createTextNode(` ${label}`));
        list.appendChild(labelEl);
    });

    panel.appendChild(list);
    picker.appendChild(panel);
    container.innerHTML = '';
    container.appendChild(picker);
    updateStageSummary(canvasId);
}

function applyStageVisibility(canvasId) {
    const chart = charts[canvasId];
    if (!chart) return;
    const list = document.getElementById(`stage-list-${canvasId}`);
    if (!list) return;
    const checkboxes = [...list.querySelectorAll('input[type=checkbox]')];

    // Toggle stage datasets
    checkboxes.forEach((cb, i) => {
        chart.setDatasetVisibility(i, cb.checked);
    });

    // Sync limit lines with inlet / effluent stage selection
    const meta = stagePickerMeta[canvasId];
    if (meta) {
        const { paramKey, seriesCount, effluentIdx } = meta;
        const inletVisible = checkboxes[0]?.checked || false;
        const effluentVisible = checkboxes[effluentIdx]?.checked || false;

        // Toggle legend-only limit datasets (inlet = seriesCount, effluent = seriesCount+1)
        chart.setDatasetVisibility(seriesCount, inletVisible);
        chart.setDatasetVisibility(seriesCount + 1, effluentVisible);

        // Toggle annotations
        const annotations = chart.options.plugins.annotation.annotations;
        if (paramKey === 'ph') {
            if (annotations.inlet_band)    annotations.inlet_band.display    = inletVisible;
            if (annotations.effluent_band) annotations.effluent_band.display = effluentVisible;
        } else {
            if (annotations.inlet_limit)    annotations.inlet_limit.display    = inletVisible;
            if (annotations.effluent_limit) annotations.effluent_limit.display = effluentVisible;
        }
    }

    chart.update();
    updateStageSummary(canvasId);
}

function setAllStages(canvasId, checked) {
    const list = document.getElementById(`stage-list-${canvasId}`);
    if (!list) return;
    list.querySelectorAll('input[type=checkbox]').forEach(cb => { cb.checked = checked; });
    applyStageVisibility(canvasId);
}

function updateStageSummary(canvasId) {
    const list = document.getElementById(`stage-list-${canvasId}`);
    if (!list) return;
    const checkboxes = list.querySelectorAll('input[type=checkbox]');
    const total = checkboxes.length;
    const checked = [...checkboxes].filter(cb => cb.checked).length;
    const summary = document.getElementById(`stage-summary-${canvasId}`);
    if (summary) {
        summary.textContent = checked === total
            ? `All stages ▾`
            : `${checked} of ${total} stages ▾`;
    }
}

// ─── Compliance Grid ─────────────────────────────────────────────────────────

function renderComplianceGrid(monthData) {
    const container = document.getElementById('compliance-grid');
    const days = monthData.days;

    let html = '<table class="compliance-table"><thead><tr><th>Day</th>';
    COMPLIANCE_PARAMS.forEach(p => { html += `<th>${p.label}</th>`; });
    html += '</tr></thead><tbody>';

    days.forEach(day => {
        html += `<tr><td class="row-label">${formatDateLong(day.date, monthData.month)}</td>`;
        COMPLIANCE_PARAMS.forEach(param => {
            const val = day[param.key];
            if (val === null || val === undefined) {
                html += '<td class="no-data">—</td>';
            } else {
                let pass;
                if (param.type === 'range') {
                    pass = val >= param.min && val <= param.max;
                } else {
                    pass = val <= param.limit;
                }
                html += `<td class="${pass ? 'pass' : 'fail'}">${typeof val === 'number' ? val.toFixed(1) : val}</td>`;
            }
        });
        html += '</tr>';
    });

    html += '</tbody></table>';
    container.innerHTML = html;

    // ── Compliance summary ────────────────────────────────────────────────────
    const summaryEl = document.getElementById('compliance-summary');
    if (!summaryEl) return;

    let summaryHtml = '<div class="compliance-summary">';
    COMPLIANCE_PARAMS.forEach(param => {
        const results = days.map(day => {
            const val = day[param.key];
            if (val === null || val === undefined) return null; // no data
            if (param.type === 'range') return val >= param.min && val <= param.max;
            return val <= param.limit;
        });

        const withData  = results.filter(r => r !== null);
        const passCount = withData.filter(Boolean).length;
        const pct = withData.length > 0 ? Math.round(passCount / withData.length * 100) : null;

        const exceeded = days
            .filter((day, i) => results[i] === false)
            .map(day => `${formatDateLong(day.date, monthData.month)}, ${monthData.year}`);

        const pctClass = pct === null ? 'cs-nodata' : pct === 100 ? 'cs-pass' : 'cs-fail';
        const pctStr   = pct !== null ? `${pct}%` : '—';

        summaryHtml += `<div class="compliance-summary-row">`;
        summaryHtml += `<span class="cs-label">${param.label}</span>`;
        summaryHtml += `<span class="cs-pct ${pctClass}">${pctStr}</span>`;
        if (exceeded.length > 0) {
            summaryHtml += `<span class="cs-exceeded">Exceeded: ${exceeded.join(' · ')}</span>`;
        }
        summaryHtml += `</div>`;
    });
    summaryHtml += '</div>';
    summaryEl.innerHTML = summaryHtml;
}

// ─── Removal Efficiency Chart ────────────────────────────────────────────────

function renderEfficiencyChart(monthData) {
    const id = 'chart-efficiency';
    if (charts[id]) charts[id].destroy();

    const days = monthData.days;
    const labels = days.map(d => formatDate(d.date));

    const calcEfficiency = (inlet, effluent) => {
        if (inlet === null || effluent === null || inlet === 0) return null;
        return ((inlet - effluent) / inlet) * 100;
    };

    const params = [
        { label: 'BOD₅', inKey: 'inlet_bod', outKey: 'effluent_bod', color: '#2563eb' },
        { label: 'COD', inKey: 'inlet_cod', outKey: 'effluent_cod', color: '#6b7280' },
        { label: 'TSS', inKey: 'inlet_tss', outKey: 'effluent_tss', color: '#1a1a1a' },
    ];

    const datasets = params.map(p => {
        const values = days.map(d => calcEfficiency(d[p.inKey], d[p.outKey]));
        return {
            label: p.label,
            data: values,
            borderColor: p.color,
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: values.map(v => v === null ? 6 : 2),
            pointStyle: values.map(v => v === null ? 'triangle' : 'circle'),
            pointBackgroundColor: values.map(v => v === null ? '#d97706' : p.color),
            pointBorderColor: values.map(v => v === null ? '#d97706' : p.color),
            tension: 0.3,
            spanGaps: true,
        };
    });

    const ctx = document.getElementById(id).getContext('2d');
    charts[id] = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: { font: { size: 11 }, usePointStyle: false, boxWidth: 20 },
                },
                tooltip: {
                    callbacks: {
                        label: (item) => {
                            const val = item.parsed.y;
                            if (val === null || isNaN(val)) return `${item.dataset.label}: ⚠ No data`;
                            return `${item.dataset.label}: ${val.toFixed(1)}%`;
                        },
                    },
                },
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 10 } } },
                y: {
                    title: { display: true, text: 'Removal %', font: { size: 11 } },
                    grace: '5%',
                    grid: { color: '#f0f0f0' },
                },
            },
        },
    });
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function formatDate(dateStr) {
    if (!dateStr) return '?';
    // Slice directly from "YYYY-MM-DD" to avoid UTC→local timezone shift
    return String(parseInt(dateStr.slice(8, 10), 10));
}

function formatDateLong(dateStr, monthName) {
    if (!dateStr) return '?';
    return `${monthName} ${parseInt(dateStr.slice(8, 10), 10)}`;
}

