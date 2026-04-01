/**
 * WWTP Lab Report Dashboard — App Logic
 * Multi-year (2021–2025), dynamic month dropdown, composite toggle.
 */

// ─── Constants ───────────────────────────────────────────────────────────────

const MONTH_ORDER = [
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december',
];

const MONTH_LABELS = {
    january: 'January', february: 'February', march: 'March',
    april: 'April', may: 'May', june: 'June',
    july: 'July', august: 'August', september: 'September',
    october: 'October', november: 'November', december: 'December',
};

// Treatment stage configs for each parameter (grab only)
const STAGE_FIELDS = {
    ph:  { stages: ['Inlet', 'Primary', 'Secondary', 'Sec. Sed.', 'Effluent'],
           fields: ['inlet_ph', 'primary_ph', 'secondary_ph', 'sec_sed_ph', 'effluent_ph'],
           inletCompKey: 'inlet_ph_comp', effluentCompKey: 'effluent_ph_comp' },
    bod: { stages: ['Inlet', 'Primary', 'Secondary', 'Sec. Sed.', 'Effluent'],
           fields: ['inlet_bod', 'primary_bod', 'secondary_bod', 'sec_sed_bod', 'effluent_bod'],
           inletCompKey: 'inlet_bod_comp', effluentCompKey: 'effluent_bod_comp' },
    cod: { stages: ['Inlet', 'Primary', 'Secondary', 'Sec. Sed.', 'Effluent'],
           fields: ['inlet_cod', 'primary_cod', 'secondary_cod', 'sec_sed_cod', 'effluent_cod'],
           inletCompKey: 'inlet_cod_comp', effluentCompKey: 'effluent_cod_comp' },
    tss: { stages: ['Inlet', 'Grit', 'Primary', 'Secondary', 'Sec. Sed.', 'Effluent'],
           fields: ['inlet_tss', 'grit_tss', 'primary_tss', 'secondary_tss', 'sec_sed_tss', 'effluent_tss'],
           inletCompKey: 'inlet_tss_comp', effluentCompKey: 'effluent_tss_comp' },
};

// Stage series for monthly parameter trend charts (grab only)
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

// Composite series appended when composite toggle is ON
const MONTHLY_COMP_SERIES = {
    ph:  [
        { key: 'inlet_ph_comp',    label: 'Inlet (Comp)',    baseColor: '#2563eb' },
        { key: 'effluent_ph_comp', label: 'Effluent (Comp)', baseColor: '#dc2626' },
    ],
    bod: [
        { key: 'inlet_bod_comp',    label: 'Inlet (Comp)',    baseColor: '#2563eb' },
        { key: 'effluent_bod_comp', label: 'Effluent (Comp)', baseColor: '#dc2626' },
    ],
    cod: [
        { key: 'inlet_cod_comp',    label: 'Inlet (Comp)',    baseColor: '#2563eb' },
        { key: 'effluent_cod_comp', label: 'Effluent (Comp)', baseColor: '#dc2626' },
    ],
    tss: [
        { key: 'inlet_tss_comp',    label: 'Inlet (Comp)',    baseColor: '#2563eb' },
        { key: 'effluent_tss_comp', label: 'Effluent (Comp)', baseColor: '#dc2626' },
    ],
};

const LIMIT_INLET_COLOR    = '#f59e0b';
const LIMIT_EFFLUENT_COLOR = '#7c3aed';

const LIMIT_LINE_DATA = {
    ph:  { inlet: { value: null, label: 'Inlet pH: 6.0–9.0' },       effluent: { value: null, label: 'Effluent pH: 6.5–8.0' } },
    bod: { inlet: { value: 300,  label: 'Inlet limit: 300 mg/L' },    effluent: { value: 10,   label: 'Effluent limit: 10 mg/L' } },
    cod: { inlet: { value: 800,  label: 'Inlet limit: 800 mg/L' },    effluent: { value: 250,  label: 'Effluent limit: 250 mg/L' } },
    tss: { inlet: { value: 400,  label: 'Inlet limit: 400 mg/L' },    effluent: { value: 10,   label: 'Effluent limit: 10 mg/L' } },
};

// Missing-value groups (grab)
const MISSING_GROUPS_GRAB = [
    { label: 'pH',    color: '#2563eb', fields: ['inlet_ph', 'primary_ph', 'secondary_ph', 'sec_sed_ph', 'effluent_ph'] },
    { label: 'BOD',   color: '#9333ea', fields: ['inlet_bod', 'primary_bod', 'secondary_bod', 'sec_sed_bod', 'effluent_bod'] },
    { label: 'COD',   color: '#16a34a', fields: ['inlet_cod', 'primary_cod', 'secondary_cod', 'sec_sed_cod', 'effluent_cod'] },
    { label: 'TSS',   color: '#d97706', fields: ['inlet_tss', 'grit_tss', 'primary_tss', 'secondary_tss', 'sec_sed_tss', 'effluent_tss'] },
    { label: 'FRC',   color: '#0891b2', fields: ['effluent_frc'] },
    { label: 'Flow',  color: '#6b7280', fields: ['flow'] },
    { label: 'Power', color: '#374151', fields: ['power_nea', 'power_ge', 'power_total'] },
];

// Missing-value groups (composite) — shown only when has_composite is true
const MISSING_GROUPS_COMP = [
    { label: 'pH (Comp)',  color: '#93c5fd', fields: ['inlet_ph_comp',  'effluent_ph_comp']  },
    { label: 'BOD (Comp)', color: '#c4b5fd', fields: ['inlet_bod_comp', 'effluent_bod_comp'] },
    { label: 'COD (Comp)', color: '#86efac', fields: ['inlet_cod_comp', 'effluent_cod_comp'] },
    { label: 'TSS (Comp)', color: '#fcd34d', fields: ['inlet_tss_comp', 'effluent_tss_comp'] },
];

const COMPLIANCE_PARAMS = [
    { key: 'effluent_ph',  label: 'pH',   type: 'range', min: 6.5, max: 8.0 },
    { key: 'effluent_bod', label: 'BOD₅', type: 'max',   limit: 10 },
    { key: 'effluent_cod', label: 'COD',  type: 'max',   limit: 250 },
    { key: 'effluent_tss', label: 'TSS',  type: 'max',   limit: 10 },
];

const DAY_COLORS = [
    '#1d4ed8', '#2563eb', '#0ea5e9', '#3b82f6',
    '#1e3a8a', '#0284c7', '#0369a1', '#60a5fa',
    '#0891b2', '#06b6d4', '#0e7490', '#14b8a6',
    '#0f766e', '#7c3aed', '#6d28d9', '#9333ea',
    '#4f46e5', '#4338ca', '#a855f7', '#16a34a',
    '#059669', '#15803d', '#65a30d', '#ca8a04',
    '#b45309', '#d97706', '#db2777', '#be185d',
    '#9d174d', '#ec4899', '#c026d3',
];

// ─── State ───────────────────────────────────────────────────────────────────

let allData = {};
let charts = {};
let stagePickerMeta = {};
let compositeState = {}; // canvasId → boolean

// ─── Init ────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.day-picker')) {
            document.querySelectorAll('details.day-picker').forEach(d => d.open = false);
        }
    });

    // Composite toggle buttons
    document.querySelectorAll('.comp-toggle').forEach(btn => {
        const canvasId = btn.dataset.canvas;
        compositeState[canvasId] = false;
        btn.addEventListener('click', () => toggleComposite(canvasId));
    });

    // ── View toggle ──────────────────────────────────────────────────────────
    document.getElementById('view-toggle-btn').addEventListener('click', () => {
        if (currentView === 'eda') {
            switchToOperational();
        } else {
            switchToEda();
        }
    });

    // ── Start in EDA view ────────────────────────────────────────────────────
    await initEdaView();

    // ── Operational view setup (lazy — only wire selects, don't render yet) ──
    const yearSelect  = document.getElementById('year-select');
    const monthSelect = document.getElementById('month-select');

    yearSelect.addEventListener('change', async () => {
        Object.keys(compositeState).forEach(k => { compositeState[k] = false; });
        document.querySelectorAll('.comp-toggle').forEach(btn => btn.classList.remove('active'));
        await loadAllData(yearSelect.value);
        const firstMonth = monthSelect.options[0]?.value;
        if (firstMonth) renderAll(firstMonth);
    });

    monthSelect.addEventListener('change', () => renderAll(monthSelect.value));
});

async function loadAllData(year) {
    const path = `../${year}/data/all_months.json`;
    try {
        const resp = await fetch(path);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        allData = await resp.json();
    } catch {
        allData = {};
    }
    populateMonthDropdown();
}

function populateMonthDropdown() {
    const sel = document.getElementById('month-select');
    const currentVal = sel.value;

    sel.innerHTML = '';
    const available = MONTH_ORDER.filter(m => m in allData);
    available.forEach(mk => {
        const opt = document.createElement('option');
        opt.value = mk;
        opt.textContent = MONTH_LABELS[mk];
        sel.appendChild(opt);
    });

    // Restore previous selection if still available, otherwise first month
    if (available.includes(currentVal)) {
        sel.value = currentVal;
    } else if (available.length) {
        sel.value = available[0];
    }
}

// ─── Composite toggle ─────────────────────────────────────────────────────────

function toggleComposite(canvasId) {
    const monthKey = document.getElementById('month-select').value;
    const monthData = allData[monthKey];
    if (!monthData) return;

    // Disable composite for months/years with no composite data
    if (!monthData.has_composite) return;

    compositeState[canvasId] = !compositeState[canvasId];
    const btn = document.querySelector(`.comp-toggle[data-canvas="${canvasId}"]`);
    if (btn) btn.classList.toggle('active', compositeState[canvasId]);

    // Re-render the affected chart
    const paramMap = {
        'chart-ph':         () => { renderDailyTrend('chart-ph',  'ph',  monthData, 'pH');    initDayPicker('chart-ph',  monthData.days, monthData.month); },
        'chart-bod':        () => { renderDailyTrend('chart-bod', 'bod', monthData, 'mg/L');  initDayPicker('chart-bod', monthData.days, monthData.month); },
        'chart-cod':        () => { renderDailyTrend('chart-cod', 'cod', monthData, 'mg/L');  initDayPicker('chart-cod', monthData.days, monthData.month); },
        'chart-tss':        () => { renderDailyTrend('chart-tss', 'tss', monthData, 'mg/L');  initDayPicker('chart-tss', monthData.days, monthData.month); },
        'chart-monthly-ph':  () => { renderMonthlyParam('chart-monthly-ph',  'ph',  'pH',    monthData); initStagePicker('chart-monthly-ph',  MONTHLY_STAGE_SERIES['ph'].map(s => s.label),  'ph');  },
        'chart-monthly-bod': () => { renderMonthlyParam('chart-monthly-bod', 'bod', 'mg/L',  monthData); initStagePicker('chart-monthly-bod', MONTHLY_STAGE_SERIES['bod'].map(s => s.label), 'bod'); },
        'chart-monthly-cod': () => { renderMonthlyParam('chart-monthly-cod', 'cod', 'mg/L',  monthData); initStagePicker('chart-monthly-cod', MONTHLY_STAGE_SERIES['cod'].map(s => s.label), 'cod'); },
        'chart-monthly-tss': () => { renderMonthlyParam('chart-monthly-tss', 'tss', 'mg/L',  monthData); initStagePicker('chart-monthly-tss', MONTHLY_STAGE_SERIES['tss'].map(s => s.label), 'tss'); },
    };

    if (paramMap[canvasId]) paramMap[canvasId]();
}

function updateCompToggles(hasComposite) {
    document.querySelectorAll('.comp-toggle').forEach(btn => {
        btn.disabled = !hasComposite;
        btn.title = hasComposite
            ? 'Show/hide composite sampling lines'
            : 'No composite data for this month/year';
        if (!hasComposite) {
            btn.classList.remove('active');
            compositeState[btn.dataset.canvas] = false;
        }
    });
}

// ─── Render all charts ───────────────────────────────────────────────────────

function renderAll(month) {
    const data = allData[month];
    if (!data) return;

    updateCompToggles(!!data.has_composite);

    renderDailyTrend('chart-ph',  'ph',  data, 'pH');    initDayPicker('chart-ph',  data.days, data.month);
    renderDailyTrend('chart-bod', 'bod', data, 'mg/L');  initDayPicker('chart-bod', data.days, data.month);
    renderDailyTrend('chart-cod', 'cod', data, 'mg/L');  initDayPicker('chart-cod', data.days, data.month);
    renderDailyTrend('chart-tss', 'tss', data, 'mg/L');  initDayPicker('chart-tss', data.days, data.month);

    renderFlowChart(data);
    renderPowerChart(data);
    renderPowerFlowChart(data);
    renderMissingChart(data);

    renderMonthlyParam('chart-monthly-ph',  'ph',  'pH',   data); initStagePicker('chart-monthly-ph',  MONTHLY_STAGE_SERIES['ph'].map(s => s.label),  'ph');
    renderMonthlyParam('chart-monthly-bod', 'bod', 'mg/L', data); initStagePicker('chart-monthly-bod', MONTHLY_STAGE_SERIES['bod'].map(s => s.label), 'bod');
    renderMonthlyParam('chart-monthly-cod', 'cod', 'mg/L', data); initStagePicker('chart-monthly-cod', MONTHLY_STAGE_SERIES['cod'].map(s => s.label), 'cod');
    renderMonthlyParam('chart-monthly-tss', 'tss', 'mg/L', data); initStagePicker('chart-monthly-tss', MONTHLY_STAGE_SERIES['tss'].map(s => s.label), 'tss');

    renderComplianceGrid(data);
    renderEfficiencyChart(data);
}

// ─── Daily Trend Charts ──────────────────────────────────────────────────────

function renderDailyTrend(canvasId, paramKey, monthData, yLabel) {
    const config = STAGE_FIELDS[paramKey];
    const days = monthData.days;
    const limitInfo = LIMIT_LINE_DATA[paramKey];
    const showComp = compositeState[canvasId] && monthData.has_composite;

    if (charts[canvasId]) charts[canvasId].destroy();

    // Build stage labels and field lists (optionally with composite positions)
    let stages, fields;
    if (showComp) {
        // Insert Inlet (Comp) right after Inlet, Effluent (Comp) right after Effluent
        stages = [...config.stages];
        fields = [...config.fields];
        // Inlet comp goes at index 1
        stages.splice(1, 0, 'Inlet (Comp)');
        fields.splice(1, 0, config.inletCompKey);
        // Effluent comp goes at end
        stages.push('Effluent (Comp)');
        fields.push(config.effluentCompKey);
    } else {
        stages = config.stages;
        fields = config.fields;
    }

    // Day datasets
    const datasets = days.map((day, i) => {
        const values = fields.map(f => day[f] ?? null);
        const isCompDay = showComp;
        return {
            label: formatDateLong(day.date, monthData.month),
            data: values,
            borderColor: DAY_COLORS[i % DAY_COLORS.length],
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            pointRadius: values.map(v => v === null ? 6 : 3),
            pointStyle: values.map((v, si) => {
                if (v === null) return 'triangle';
                // Mark composite positions with a diamond
                if (isCompDay && (si === 1 || si === fields.length - 1)) return 'rectRot';
                return 'circle';
            }),
            pointBackgroundColor: values.map(v => v === null ? '#d97706' : DAY_COLORS[i % DAY_COLORS.length]),
            pointBorderColor: values.map(v => v === null ? '#d97706' : DAY_COLORS[i % DAY_COLORS.length]),
            tension: 0.2,
            spanGaps: false,
            hidden: i !== 0,
            isLimit: false,
        };
    });

    // Legend-only limit datasets
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

    // Annotations
    const annotations = {};
    if (paramKey === 'ph') {
        annotations['inlet_band'] = {
            type: 'box', yMin: 6.0, yMax: 9.0,
            backgroundColor: 'rgba(245,158,11,0.08)', borderColor: 'rgba(245,158,11,0.55)', borderWidth: 1.5,
        };
        annotations['effluent_band'] = {
            type: 'box', yMin: 6.5, yMax: 8.0,
            backgroundColor: 'rgba(124,58,237,0.08)', borderColor: 'rgba(124,58,237,0.6)', borderWidth: 1.5, borderDash: [5,3],
        };
    } else {
        if (limitInfo.inlet.value !== null) {
            annotations['inlet_limit'] = {
                type: 'line', yMin: limitInfo.inlet.value, yMax: limitInfo.inlet.value,
                borderColor: LIMIT_INLET_COLOR, borderWidth: 1.5, borderDash: [6, 4],
            };
        }
        if (limitInfo.effluent.value !== null) {
            annotations['effluent_limit'] = {
                type: 'line', yMin: limitInfo.effluent.value, yMax: limitInfo.effluent.value,
                borderColor: LIMIT_EFFLUENT_COLOR, borderWidth: 1.5, borderDash: [6, 4],
            };
        }
    }

    const ctx = document.getElementById(canvasId).getContext('2d');
    charts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: { labels: stages, datasets: [...datasets, ...limitDatasets] },
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
                        boxWidth: 28, boxHeight: 12, padding: 10, font: { size: 11 },
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
                            const stageName = stages[item.dataIndex];
                            if (val === null || isNaN(val)) return `${stageName}: ⚠ No data`;
                            return `${stageName}: ${val}`;
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

    const actions = document.createElement('div');
    actions.className = 'day-picker-actions';
    const allBtn = document.createElement('button');
    allBtn.textContent = 'All'; allBtn.type = 'button';
    allBtn.addEventListener('click', (e) => { e.stopPropagation(); setAllDays(canvasId, true); });
    const noneBtn = document.createElement('button');
    noneBtn.textContent = 'None'; noneBtn.type = 'button';
    noneBtn.addEventListener('click', (e) => { e.stopPropagation(); setAllDays(canvasId, false); });
    actions.appendChild(allBtn); actions.appendChild(noneBtn);
    panel.appendChild(actions);

    const list = document.createElement('div');
    list.className = 'day-picker-list';
    list.id = `day-list-${canvasId}`;

    const chart = charts[canvasId];
    days.forEach((day, i) => {
        const isVisible = chart ? chart.isDatasetVisible(i) : true;
        const label = document.createElement('label');
        const cb = document.createElement('input');
        cb.type = 'checkbox'; cb.checked = isVisible;
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
        summary.textContent = checked === total ? `All ${total} days ▾` : `${checked} of ${total} days ▾`;
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
                    label: 'Raw Sewage Flow (MLD)', data: values,
                    borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,0.08)',
                    fill: true, borderWidth: 2,
                    pointRadius: values.map(v => v === null ? 6 : 2),
                    pointStyle: values.map(v => v === null ? 'triangle' : 'circle'),
                    pointBackgroundColor: values.map(v => v === null ? '#d97706' : '#2563eb'),
                    pointBorderColor: values.map(v => v === null ? '#d97706' : '#2563eb'),
                    tension: 0.3, spanGaps: true, isLimit: false,
                },
                {
                    label: 'Design capacity: 32.4 MLD', data: [],
                    borderColor: '#dc2626', backgroundColor: 'transparent',
                    borderWidth: 1.5, borderDash: [6, 4], pointRadius: 0, isLimit: true,
                },
            ],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true, position: 'top',
                    labels: {
                        filter: (item, data) => data.datasets[item.datasetIndex]?.isLimit === true,
                        boxWidth: 28, boxHeight: 12, padding: 10, font: { size: 11 },
                    },
                },
                annotation: { annotations: { flowLimit: { type: 'line', yMin: 32.4, yMax: 32.4, borderColor: '#dc2626', borderWidth: 1.5, borderDash: [6, 4] } } },
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 10 } } },
                y: { title: { display: true, text: 'MLD', font: { size: 11 } }, grace: '5%', grid: { color: '#f0f0f0' } },
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
                    label: 'Total (NEA + GE)', data: days.map(d => d.power_total),
                    borderColor: '#1a1a1a', backgroundColor: 'transparent', borderWidth: 2,
                    pointRadius: days.map(d => d.power_total === null ? 6 : 2),
                    pointStyle: days.map(d => d.power_total === null ? 'triangle' : 'circle'),
                    pointBackgroundColor: days.map(d => d.power_total === null ? '#d97706' : '#1a1a1a'),
                    pointBorderColor: days.map(d => d.power_total === null ? '#d97706' : '#1a1a1a'),
                    tension: 0.3, spanGaps: true,
                },
                {
                    label: 'NEA (Grid)', data: days.map(d => d.power_nea),
                    borderColor: '#6b7280', backgroundColor: 'transparent', borderWidth: 1.5, borderDash: [5, 3],
                    pointRadius: days.map(d => d.power_nea === null ? 6 : 1.5),
                    pointStyle: days.map(d => d.power_nea === null ? 'triangle' : 'circle'),
                    pointBackgroundColor: days.map(d => d.power_nea === null ? '#d97706' : '#6b7280'),
                    pointBorderColor: days.map(d => d.power_nea === null ? '#d97706' : '#6b7280'),
                    tension: 0.3, spanGaps: true,
                },
                {
                    label: 'Gas Engine', data: days.map(d => d.power_ge),
                    borderColor: '#9ca3af', backgroundColor: 'transparent', borderWidth: 1.5, borderDash: [2, 2],
                    pointRadius: days.map(d => d.power_ge === null ? 6 : 1.5),
                    pointStyle: days.map(d => d.power_ge === null ? 'triangle' : 'circle'),
                    pointBackgroundColor: days.map(d => d.power_ge === null ? '#d97706' : '#9ca3af'),
                    pointBorderColor: days.map(d => d.power_ge === null ? '#d97706' : '#9ca3af'),
                    tension: 0.3, spanGaps: true,
                },
            ],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top', labels: { font: { size: 11 }, usePointStyle: false, boxWidth: 20 } },
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 10 } } },
                y: { title: { display: true, text: 'KWh', font: { size: 11 } }, grace: '5%', grid: { color: '#f0f0f0' } },
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
                    label: 'KWh / ML', data: values,
                    borderColor: '#2563eb', backgroundColor: 'transparent', borderWidth: 2,
                    pointRadius: values.map(v => v === null ? 6 : 2),
                    pointStyle: values.map(v => v === null ? 'triangle' : 'circle'),
                    pointBackgroundColor: values.map(v => v === null ? '#d97706' : '#2563eb'),
                    pointBorderColor: values.map(v => v === null ? '#d97706' : '#2563eb'),
                    tension: 0.3, spanGaps: true, isLimit: false,
                },
                {
                    label: 'Baseline: 482.02 KWh/ML', data: [],
                    borderColor: '#dc2626', backgroundColor: 'transparent',
                    borderWidth: 1.5, borderDash: [6, 4], pointRadius: 0, isLimit: true,
                },
            ],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true, position: 'top',
                    labels: {
                        filter: (item, data) => data.datasets[item.datasetIndex]?.isLimit === true,
                        boxWidth: 28, boxHeight: 12, padding: 10, font: { size: 11 },
                    },
                },
                annotation: { annotations: { baseline: { type: 'line', yMin: 482.02, yMax: 482.02, borderColor: '#dc2626', borderWidth: 1.5, borderDash: [6, 4] } } },
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 10 } } },
                y: { title: { display: true, text: 'KWh / ML', font: { size: 11 } }, beginAtZero: false, grid: { color: '#f0f0f0' } },
            },
        },
    });
}

function renderMissingChart(monthData) {
    const id = 'chart-missing';
    if (charts[id]) charts[id].destroy();

    const days = monthData.days;
    const labels = days.map(d => formatDate(d.date));

    // Include composite groups only when the month has composite data
    const groups = monthData.has_composite
        ? [...MISSING_GROUPS_GRAB, ...MISSING_GROUPS_COMP]
        : MISSING_GROUPS_GRAB;

    const datasets = groups.map(group => ({
        label: group.label,
        data: days.map(day => group.fields.filter(f => day[f] === null || day[f] === undefined).length),
        backgroundColor: group.color,
        borderRadius: 2,
        borderSkipped: false,
    }));

    const ctx = document.getElementById(id).getContext('2d');
    charts[id] = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true, position: 'top',
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
                y: { title: { display: true, text: 'Missing values', font: { size: 11 } }, beginAtZero: true, grace: '10%', grid: { color: '#f0f0f0' }, ticks: { stepSize: 1 } },
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
    const showComp = compositeState[canvasId] && monthData.has_composite;

    // Grab stage datasets
    const stageDatasets = series.map(s => {
        const values = days.map(d => d[s.key] ?? null);
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
            tension: 0.3, spanGaps: true,
            hidden: s.label !== 'Effluent',
            isLimit: false, isComp: false,
        };
    });

    // Composite datasets (dashed, same colors as Inlet/Effluent)
    const compDatasets = showComp ? MONTHLY_COMP_SERIES[paramKey].map(cs => {
        const values = days.map(d => d[cs.key] ?? null);
        return {
            label: cs.label,
            data: values,
            borderColor: cs.baseColor,
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            borderDash: [5, 3],
            pointRadius: values.map(v => v === null ? 5 : 3),
            pointStyle: values.map(v => v === null ? 'triangle' : 'rectRot'),
            pointBackgroundColor: values.map(v => v === null ? '#d97706' : cs.baseColor),
            pointBorderColor: values.map(v => v === null ? '#d97706' : cs.baseColor),
            tension: 0.3, spanGaps: true,
            hidden: false,
            isLimit: false, isComp: true,
        };
    }) : [];

    // Legend-only limit datasets
    const limitDatasets = [
        {
            label: limitInfo.inlet.label, data: [],
            borderColor: LIMIT_INLET_COLOR, backgroundColor: 'transparent',
            borderWidth: 1.5, borderDash: [6, 4], pointRadius: 0, hidden: true, isLimit: true,
        },
        {
            label: limitInfo.effluent.label, data: [],
            borderColor: LIMIT_EFFLUENT_COLOR, backgroundColor: 'transparent',
            borderWidth: 1.5, borderDash: [6, 4], pointRadius: 0, hidden: false, isLimit: true,
        },
    ];

    // Annotations
    const annotations = {};
    if (paramKey === 'ph') {
        annotations['inlet_band'] = {
            type: 'box', yMin: 6.0, yMax: 9.0,
            backgroundColor: 'rgba(245,158,11,0.08)', borderColor: 'rgba(245,158,11,0.55)', borderWidth: 1.5, display: false,
        };
        annotations['effluent_band'] = {
            type: 'box', yMin: 6.5, yMax: 8.0,
            backgroundColor: 'rgba(124,58,237,0.08)', borderColor: 'rgba(124,58,237,0.6)', borderWidth: 1.5, borderDash: [5, 3], display: true,
        };
    } else {
        if (limitInfo.inlet.value !== null) {
            annotations['inlet_limit'] = {
                type: 'line', yMin: limitInfo.inlet.value, yMax: limitInfo.inlet.value,
                borderColor: LIMIT_INLET_COLOR, borderWidth: 1.5, borderDash: [6, 4], display: false,
            };
        }
        if (limitInfo.effluent.value !== null) {
            annotations['effluent_limit'] = {
                type: 'line', yMin: limitInfo.effluent.value, yMax: limitInfo.effluent.value,
                borderColor: LIMIT_EFFLUENT_COLOR, borderWidth: 1.5, borderDash: [6, 4], display: true,
            };
        }
    }

    const ctx = document.getElementById(canvasId).getContext('2d');
    charts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets: [...stageDatasets, ...compDatasets, ...limitDatasets] },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true, position: 'top',
                    labels: {
                        filter: (item, data) => {
                            const ds = data.datasets[item.datasetIndex];
                            return ds?.isLimit === true && !item.hidden;
                        },
                        boxWidth: 28, boxHeight: 12, padding: 10, font: { size: 11 },
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
                y: { title: { display: true, text: yLabel, font: { size: 11 } }, grace: '5%', grid: { color: '#f0f0f0' } },
            },
        },
    });
}

// ─── Stage Picker ─────────────────────────────────────────────────────────────

function initStagePicker(canvasId, seriesLabels, paramKey) {
    const container = document.getElementById(`picker-${canvasId}`);
    if (!container) return;

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
    allBtn.textContent = 'All'; allBtn.type = 'button';
    allBtn.addEventListener('click', (e) => { e.stopPropagation(); setAllStages(canvasId, true); });
    const noneBtn = document.createElement('button');
    noneBtn.textContent = 'None'; noneBtn.type = 'button';
    noneBtn.addEventListener('click', (e) => { e.stopPropagation(); setAllStages(canvasId, false); });
    actions.appendChild(allBtn); actions.appendChild(noneBtn);
    panel.appendChild(actions);

    const list = document.createElement('div');
    list.className = 'day-picker-list';
    list.id = `stage-list-${canvasId}`;

    const chart = charts[canvasId];
    seriesLabels.forEach((label, i) => {
        const isVisible = chart ? chart.isDatasetVisible(i) : (label === 'Effluent');
        const labelEl = document.createElement('label');
        const cb = document.createElement('input');
        cb.type = 'checkbox'; cb.checked = isVisible;
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

    checkboxes.forEach((cb, i) => { chart.setDatasetVisibility(i, cb.checked); });

    const meta = stagePickerMeta[canvasId];
    if (meta) {
        const { paramKey, seriesCount, effluentIdx } = meta;
        const inletVisible = checkboxes[0]?.checked || false;
        const effluentVisible = checkboxes[effluentIdx]?.checked || false;

        // Sync limit line datasets (they follow after grab stages + any comp datasets)
        // Count comp datasets
        const monthKey = document.getElementById('month-select').value;
        const hasComp = allData[monthKey]?.has_composite && compositeState[canvasId];
        const compCount = hasComp ? 2 : 0;
        chart.setDatasetVisibility(seriesCount + compCount,     inletVisible);
        chart.setDatasetVisibility(seriesCount + compCount + 1, effluentVisible);

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
        summary.textContent = checked === total ? `All stages ▾` : `${checked} of ${total} stages ▾`;
    }
}

// ─── Compliance Grid ──────────────────────────────────────────────────────────

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
                const pass = param.type === 'range'
                    ? val >= param.min && val <= param.max
                    : val <= param.limit;
                html += `<td class="${pass ? 'pass' : 'fail'}">${typeof val === 'number' ? val.toFixed(1) : val}</td>`;
            }
        });
        html += '</tr>';
    });
    html += '</tbody></table>';
    container.innerHTML = html;

    const summaryEl = document.getElementById('compliance-summary');
    if (!summaryEl) return;

    let summaryHtml = '<div class="compliance-summary">';
    COMPLIANCE_PARAMS.forEach(param => {
        const results = days.map(day => {
            const val = day[param.key];
            if (val === null || val === undefined) return null;
            return param.type === 'range' ? val >= param.min && val <= param.max : val <= param.limit;
        });

        const withData  = results.filter(r => r !== null);
        const passCount = withData.filter(Boolean).length;
        const pct = withData.length > 0 ? Math.round(passCount / withData.length * 100) : null;

        const exceeded = days
            .filter((_, i) => results[i] === false)
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

// ─── Removal Efficiency Chart ─────────────────────────────────────────────────

function renderEfficiencyChart(monthData) {
    const id = 'chart-efficiency';
    if (charts[id]) charts[id].destroy();

    const days = monthData.days;
    const labels = days.map(d => formatDate(d.date));

    const calcEff = (inlet, eff) => (inlet === null || eff === null || inlet === 0) ? null : ((inlet - eff) / inlet) * 100;

    const params = [
        { label: 'BOD₅', inKey: 'inlet_bod', outKey: 'effluent_bod', color: '#2563eb' },
        { label: 'COD',  inKey: 'inlet_cod', outKey: 'effluent_cod', color: '#6b7280' },
        { label: 'TSS',  inKey: 'inlet_tss', outKey: 'effluent_tss', color: '#1a1a1a' },
    ];

    const datasets = params.map(p => {
        const values = days.map(d => calcEff(d[p.inKey], d[p.outKey]));
        return {
            label: p.label, data: values,
            borderColor: p.color, backgroundColor: 'transparent', borderWidth: 2,
            pointRadius: values.map(v => v === null ? 6 : 2),
            pointStyle: values.map(v => v === null ? 'triangle' : 'circle'),
            pointBackgroundColor: values.map(v => v === null ? '#d97706' : p.color),
            pointBorderColor: values.map(v => v === null ? '#d97706' : p.color),
            tension: 0.3, spanGaps: true,
        };
    });

    const ctx = document.getElementById(id).getContext('2d');
    charts[id] = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top', labels: { font: { size: 11 }, usePointStyle: false, boxWidth: 20 } },
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
                y: { title: { display: true, text: 'Removal %', font: { size: 11 } }, grace: '5%', grid: { color: '#f0f0f0' } },
            },
        },
    });
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatDate(dateStr) {
    if (!dateStr) return '?';
    return String(parseInt(dateStr.slice(8, 10), 10));
}

function formatDateLong(dateStr, monthName) {
    if (!dateStr) return '?';
    return `${monthName} ${parseInt(dateStr.slice(8, 10), 10)}`;
}

// ═══════════════════════════════════════════════════════════════════════════════
// EDA VIEW
// ═══════════════════════════════════════════════════════════════════════════════

// ─── State ────────────────────────────────────────────────────────────────────

let currentView = 'eda';
let edaDays     = [];
let edaCharts   = {};

const EDA_SHORT_MONTHS = ['Jan','Feb','Mar','Apr','May','Jun',
                          'Jul','Aug','Sep','Oct','Nov','Dec'];

const EDA_YEAR_COLORS = {
    2021: '#e41a1c', 2022: '#377eb8', 2023: '#4daf4a',
    2024: '#984ea3', 2025: '#ff7f00',
};

// ─── View switching ───────────────────────────────────────────────────────────

function switchToEda() {
    currentView = 'eda';
    document.getElementById('eda-view').style.display          = 'block';
    document.getElementById('operational-view').style.display  = 'none';
    document.getElementById('eda-controls-inline').style.display = '';
    document.getElementById('op-controls-inline').style.display  = 'none';
    document.getElementById('view-toggle-btn').textContent     = 'Operational Dashboard \u2192';
}

async function switchToOperational() {
    currentView = 'operational';
    document.getElementById('eda-view').style.display          = 'none';
    document.getElementById('operational-view').style.display  = 'block';
    document.getElementById('eda-controls-inline').style.display = 'none';
    document.getElementById('op-controls-inline').style.display  = '';
    document.getElementById('view-toggle-btn').textContent     = '\u2190 EDA Overview';

    // Load & render operational view if not done yet
    const yearSelect  = document.getElementById('year-select');
    const monthSelect = document.getElementById('month-select');
    if (Object.keys(allData).length === 0) {
        await loadAllData(yearSelect.value);
    }
    const month = monthSelect.value || monthSelect.options[0]?.value;
    if (month) renderAll(month);
}

// ─── Data loading ──────────────────────────────────────────────────────────────

async function loadEdaData() {
    const years = [2020, 2021, 2022, 2023, 2024, 2025];
    const all = [];
    await Promise.all(years.map(async (year) => {
        try {
            const resp = await fetch(`../${year}/data/all_months.json`);
            if (!resp.ok) return;
            const yearData = await resp.json();
            for (const [, monthData] of Object.entries(yearData)) {
                for (const day of monthData.days) {
                    all.push({ ...day, _year: year });
                }
            }
        } catch { /* skip missing year */ }
    }));
    all.sort((a, b) => (a.date || '').localeCompare(b.date || ''));
    return all;
}

// ─── Stat helpers ──────────────────────────────────────────────────────────────

/** Monthly average for a field: returns { labels: ['2021-07',...], values: [...] } */
function edaMonthlyAvg(days, field) {
    const groups = {};
    for (const d of days) {
        if (!d.date) continue;
        const ym = d.date.slice(0, 7);
        if (!groups[ym]) groups[ym] = [];
        if (d[field] !== null && d[field] !== undefined) groups[ym].push(d[field]);
    }
    const labels = Object.keys(groups).sort();
    return { labels, values: labels.map(ym => {
        const arr = groups[ym];
        return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
    })};
}

/** Seasonal average (by calendar month 1–12) across all years */
function edaSeasonalAvg(days, field) {
    const groups = Array.from({ length: 12 }, () => []);
    for (const d of days) {
        if (!d.date) continue;
        const m = parseInt(d.date.slice(5, 7), 10) - 1;
        if (d[field] !== null && d[field] !== undefined) groups[m].push(d[field]);
    }
    return groups.map(arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null);
}

/** % missing for a field across all days */
function edaMissingPct(days, field) {
    if (!days.length) return 100;
    const missing = days.filter(d => d[field] === null || d[field] === undefined).length;
    return (missing / days.length) * 100;
}

/** Annual average for a specific year */
function edaAnnualAvg(days, field, year) {
    const arr = days.filter(d => d._year === year && d[field] !== null && d[field] !== undefined)
                    .map(d => d[field]);
    return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
}

/** 'YYYY-MM' → 'Jul 2021' */
function edaFmtYM(ym) {
    const [y, m] = ym.split('-');
    return `${EDA_SHORT_MONTHS[parseInt(m, 10) - 1]} ${y}`;
}

// ─── Compliance helper ─────────────────────────────────────────────────────────

const EDA_COMPLIANCE = [
    { key: 'effluent_ph',  label: 'pH',  type: 'range', min: 6.5, max: 8.0  },
    { key: 'effluent_bod', label: 'BOD', type: 'max',   limit: 10            },
    { key: 'effluent_cod', label: 'COD', type: 'max',   limit: 250           },
    { key: 'effluent_tss', label: 'TSS', type: 'max',   limit: 10            },
];

function edaCompliancePct(days, year, param) {
    const subset = days.filter(d => d._year === year);
    const withData = subset.filter(d => d[param.key] !== null && d[param.key] !== undefined);
    if (!withData.length) return null;
    const pass = withData.filter(d => {
        const v = d[param.key];
        return param.type === 'range' ? v >= param.min && v <= param.max : v <= param.limit;
    }).length;
    return (pass / withData.length) * 100;
}

// ─── EDA chart renders ─────────────────────────────────────────────────────────

function edaDestroy(id) {
    if (edaCharts[id]) { edaCharts[id].destroy(); delete edaCharts[id]; }
}

// 1. Missing values horizontal bar
function renderEdaMissing(days) {
    edaDestroy('eda-missing');
    const fields = [
        'flow', 'power_nea', 'power_ge', 'power_total', 'power_per_flow',
        'inlet_ph', 'inlet_bod', 'inlet_cod', 'inlet_tss',
        'inlet_ph_comp', 'inlet_bod_comp', 'inlet_cod_comp', 'inlet_tss_comp',
        'effluent_ph', 'effluent_bod', 'effluent_cod', 'effluent_tss', 'effluent_frc',
        'effluent_ph_comp', 'effluent_bod_comp', 'effluent_cod_comp', 'effluent_tss_comp',
    ];
    const labels = [
        'Flow', 'Power NEA', 'Power GE', 'Power Total', 'Power/Flow',
        'Inlet pH', 'Inlet BOD', 'Inlet COD', 'Inlet TSS',
        'Inlet pH (Comp)', 'Inlet BOD (Comp)', 'Inlet COD (Comp)', 'Inlet TSS (Comp)',
        'Effluent pH', 'Effluent BOD', 'Effluent COD', 'Effluent TSS', 'Effluent FRC',
        'Effluent pH (Comp)', 'Effluent BOD (Comp)', 'Effluent COD (Comp)', 'Effluent TSS (Comp)',
    ];
    const pcts = fields.map(f => edaMissingPct(days, f));

    // Sort descending
    const order = pcts.map((v, i) => i).sort((a, b) => pcts[b] - pcts[a]);
    const sortedLabels = order.map(i => labels[i]);
    const sortedPcts   = order.map(i => pcts[i]);
    const colors = sortedPcts.map(p => {
        if (p >= 50) return '#e11d48';
        if (p >= 20) return '#d97706';
        return '#16a34a';
    });

    const ctx = document.getElementById('eda-missing').getContext('2d');
    edaCharts['eda-missing'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sortedLabels,
            datasets: [{
                label: '% Missing',
                data: sortedPcts,
                backgroundColor: colors,
                borderRadius: 3,
                borderSkipped: false,
            }],
        },
        options: {
            indexAxis: 'y',
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: i => ` ${i.parsed.x.toFixed(1)}% missing` } },
            },
            scales: {
                x: { min: 0, max: 40, title: { display: true, text: '% Missing', font: { size: 11 } },
                     grid: { color: '#f0f0f0' } },
                y: { grid: { display: false }, ticks: { font: { size: 10 } } },
            },
        },
    });
}

const pluginPercentageLabel = {
    id: 'percentageLabel',
    afterDatasetsDraw(chart, args, pluginOptions) {
        const { ctx, data } = chart;
        ctx.save();
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#444';

        chart.data.datasets.forEach((dataset, i) => {
            const meta = chart.getDatasetMeta(i);
            if (!meta.hidden && dataset.pctData) {
                meta.data.forEach((element, index) => {
                    const pct = dataset.pctData[index];
                    const count = dataset.data[index];
                    if (count > 0 && pct !== null) {
                        ctx.save();
                        ctx.translate(element.x, element.y - 4);
                        ctx.rotate(-Math.PI / 2);
                        ctx.fillText(pct.toFixed(1) + ' %', 0, 0);
                        ctx.restore();
                    }
                });
            }
        });
        ctx.restore();
    }
};

function createHatchPattern(color) {
    const canvas = document.createElement('canvas');
    canvas.width = 8;
    canvas.height = 8;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = color + '33';
    ctx.fillRect(0, 0, 8, 8);
    
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, 8);
    ctx.lineTo(8, 0);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(-1, 1);
    ctx.lineTo(1, -1);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(7, 9);
    ctx.lineTo(9, 7);
    ctx.stroke();
    
    return ctx.createPattern(canvas, 'repeat');
}

function renderEdaMissingCounts1(days) {
    edaDestroy('eda-missing-counts-1');
    const years = [2020, 2021, 2022, 2023, 2024, 2025];
    const fields = [
        { key: 'flow', label: 'Flow', color: '#2563eb' },
        { key: 'power_ge', label: 'Power GE', color: '#16a34a' },
        { key: 'power_nea', label: 'Power NEA', color: '#d97706' },
        { key: 'power_total', label: 'Power Total', color: '#9333ea' }
    ];

    const datasets = fields.map(f => {
        const counts = [];
        const pcts = [];
        years.forEach(y => {
            const subset = days.filter(d => d._year === y);
            if (!subset.length) {
                counts.push(0); pcts.push(null);
            } else {
                const missing = subset.filter(d => d[f.key] === null || d[f.key] === undefined).length;
                counts.push(missing);
                pcts.push((missing / subset.length) * 100);
            }
        });
        return {
            label: f.label,
            data: counts,
            pctData: pcts,
            backgroundColor: f.color + 'cc',
            borderColor: f.color,
            borderWidth: 1,
            borderRadius: 2
        };
    });

    const ctx = document.getElementById('eda-missing-counts-1').getContext('2d');
    edaCharts['eda-missing-counts-1'] = new Chart(ctx, {
        type: 'bar',
        data: { labels: years.map(String), datasets },
        plugins: [pluginPercentageLabel],
        options: {
            responsive: true, maintainAspectRatio: false,
            layout: { padding: { top: 15 } },
            plugins: {
                legend: { position: 'top', labels: { font: { size: 10 }, boxWidth: 10 } },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const pct = ctx.dataset.pctData[ctx.dataIndex];
                            if (pct === null) return `${ctx.dataset.label}: No Data for Year`;
                            return `${ctx.dataset.label}: ${ctx.raw} missing (${pct.toFixed(1)}%)`;
                        }
                    }
                }
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 11 } } },
                y: { title: { display: true, text: 'Missing Count', font: { size: 11 } }, grace: '40%' }
            }
        }
    });
}

function renderMissingGroup(days, canvasId, fields) {
    edaDestroy(canvasId);
    const years = [2020, 2021, 2022, 2023, 2024, 2025];
    const datasets = fields.map(f => {
        const counts = [];
        const pcts = [];
        years.forEach(y => {
            const subset = days.filter(d => d._year === y);
            if (!subset.length) {
                counts.push(0); pcts.push(null);
            } else {
                const missing = subset.filter(d => d[f.key] === null || d[f.key] === undefined).length;
                counts.push(missing);
                pcts.push((missing / subset.length) * 100);
            }
        });
        return {
            label: f.label,
            data: counts,
            pctData: pcts,
            backgroundColor: f.isComp ? createHatchPattern(f.color) : f.color + 'cc',
            borderColor: f.color,
            borderWidth: 1,
            borderRadius: 2
        };
    });

    const ctx = document.getElementById(canvasId).getContext('2d');
    edaCharts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: { labels: years.map(String), datasets },
        plugins: [pluginPercentageLabel],
        options: {
            responsive: true, maintainAspectRatio: false,
            layout: { padding: { top: 15 } },
            plugins: {
                legend: { position: 'top', labels: { font: { size: 9 }, boxWidth: 10, padding: 8 } },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const pct = ctx.dataset.pctData[ctx.dataIndex];
                            if (pct === null) return `${ctx.dataset.label}: No Data for Year`;
                            return `${ctx.dataset.label}: ${ctx.raw} missing (${pct.toFixed(1)}%)`;
                        }
                    }
                }
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 10 } } },
                y: { title: { display: true, text: 'Missing', font: { size: 10 } }, grace: '40%' }
            }
        }
    });
}

function renderEdaMissingCountsPh(days) {
    renderMissingGroup(days, 'eda-missing-ph', [
        { key: 'inlet_ph', label: 'Inlet', color: '#1f77b4' },
        { key: 'inlet_ph_comp', label: 'Inlet Comp', color: '#1f77b4', isComp: true },
        { key: 'effluent_ph', label: 'Outlet', color: '#2ca02c' },
        { key: 'effluent_ph_comp', label: 'Outlet Comp', color: '#2ca02c', isComp: true }
    ]);
}

function renderEdaMissingCountsBod(days) {
    renderMissingGroup(days, 'eda-missing-bod', [
        { key: 'inlet_bod', label: 'Inlet', color: '#1f77b4' },
        { key: 'inlet_bod_comp', label: 'Inlet Comp', color: '#1f77b4', isComp: true },
        { key: 'effluent_bod', label: 'Outlet', color: '#2ca02c' },
        { key: 'effluent_bod_comp', label: 'Outlet Comp', color: '#2ca02c', isComp: true }
    ]);
}

function renderEdaMissingCountsCod(days) {
    renderMissingGroup(days, 'eda-missing-cod', [
        { key: 'inlet_cod', label: 'Inlet', color: '#ff7f0e' },
        { key: 'inlet_cod_comp', label: 'Inlet Comp', color: '#ff7f0e', isComp: true },
        { key: 'effluent_cod', label: 'Outlet', color: '#d62728' },
        { key: 'effluent_cod_comp', label: 'Outlet Comp', color: '#d62728', isComp: true }
    ]);
}

function renderEdaMissingCountsTss(days) {
    renderMissingGroup(days, 'eda-missing-tss', [
        { key: 'inlet_tss', label: 'Inlet', color: '#ff7f0e' },
        { key: 'inlet_tss_comp', label: 'Inlet Comp', color: '#ff7f0e', isComp: true },
        { key: 'effluent_tss', label: 'Outlet', color: '#d62728' },
        { key: 'effluent_tss_comp', label: 'Outlet Comp', color: '#d62728', isComp: true }
    ]);
}

// 2. Flow over time (monthly avg)
function renderEdaFlow(days) {
    edaDestroy('eda-flow');
    const { labels, values } = edaMonthlyAvg(days, 'flow');
    const ctx = document.getElementById('eda-flow').getContext('2d');
    edaCharts['eda-flow'] = new Chart(ctx, {
        type: 'line',
        data: { labels: labels.map(edaFmtYM), datasets: [
            {
                label: 'Avg Flow (MLD)', data: values,
                borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,0.07)',
                fill: true, borderWidth: 2,
                pointRadius: 2, tension: 0.3, spanGaps: true,
            },
            {
                label: 'Capacity: 32.4 MLD', data: [],
                borderColor: '#dc2626', borderWidth: 1.5, borderDash: [6, 4],
                pointRadius: 0, isLimit: true,
            },
        ]},
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top',
                    labels: { filter: (item, data) => data.datasets[item.datasetIndex]?.isLimit === true,
                              font: { size: 11 }, boxWidth: 24 } },
                annotation: { annotations: { cap: {
                    type: 'line', yMin: 32.4, yMax: 32.4,
                    borderColor: '#dc2626', borderWidth: 1.5, borderDash: [6, 4],
                }}},
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 9 }, maxTicksLimit: 12, autoSkip: true } },
                y: { title: { display: true, text: 'MLD', font: { size: 11 } }, grace: '5%', grid: { color: '#f0f0f0' } },
            },
        },
    });
}

// 3. Power efficiency over time (monthly avg)
function renderEdaPower(days) {
    edaDestroy('eda-power');
    const { labels, values } = edaMonthlyAvg(days, 'power_per_flow');
    const ctx = document.getElementById('eda-power').getContext('2d');
    edaCharts['eda-power'] = new Chart(ctx, {
        type: 'line',
        data: { labels: labels.map(edaFmtYM), datasets: [
            {
                label: 'Avg Power/Flow (KW/ML)', data: values,
                borderColor: '#7c3aed', backgroundColor: 'rgba(124,58,237,0.06)',
                fill: true, borderWidth: 2,
                pointRadius: 2, tension: 0.3, spanGaps: true,
            },
            {
                label: 'Baseline: 482.02 KW/ML', data: [],
                borderColor: '#dc2626', borderWidth: 1.5, borderDash: [6, 4],
                pointRadius: 0, isLimit: true,
            },
        ]},
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top',
                    labels: { filter: (item, data) => data.datasets[item.datasetIndex]?.isLimit === true,
                              font: { size: 11 }, boxWidth: 24 } },
                annotation: { annotations: { baseline: {
                    type: 'line', yMin: 482.02, yMax: 482.02,
                    borderColor: '#dc2626', borderWidth: 1.5, borderDash: [6, 4],
                }}},
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 9 }, maxTicksLimit: 12, autoSkip: true } },
                y: { title: { display: true, text: 'KW/ML', font: { size: 11 } }, grace: '5%', grid: { color: '#f0f0f0' } },
            },
        },
    });
}

// 4. Inlet quality monthly trends
function renderEdaInlet(days) {
    edaDestroy('eda-inlet');
    const bod = edaMonthlyAvg(days, 'inlet_bod');
    const cod = edaMonthlyAvg(days, 'inlet_cod');
    const tss = edaMonthlyAvg(days, 'inlet_tss');
    const labels = bod.labels.map(edaFmtYM);
    const ctx = document.getElementById('eda-inlet').getContext('2d');
    edaCharts['eda-inlet'] = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets: [
            { label: 'BOD (mg/L)',  data: bod.values, borderColor: '#2563eb', backgroundColor: 'transparent',
              borderWidth: 2, pointRadius: 1.5, tension: 0.3, spanGaps: true },
            { label: 'COD/5 (mg/L)', data: cod.values.map(v => v !== null ? v / 5 : null),
              borderColor: '#9333ea', backgroundColor: 'transparent',
              borderWidth: 1.5, borderDash: [4, 3], pointRadius: 1.5, tension: 0.3, spanGaps: true },
            { label: 'TSS (mg/L)',  data: tss.values, borderColor: '#d97706', backgroundColor: 'transparent',
              borderWidth: 2, pointRadius: 1.5, tension: 0.3, spanGaps: true },
        ]},
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top', labels: { font: { size: 11 }, boxWidth: 20 } },
                tooltip: { callbacks: { label: item => {
                    const label = item.dataset.label;
                    const v = item.parsed.y;
                    if (v === null || isNaN(v)) return `${label}: ⚠ No data`;
                    // COD was divided by 5 for display
                    const raw = label.includes('COD') ? (v * 5).toFixed(1) : v.toFixed(1);
                    return label.includes('COD') ? `COD: ${raw} mg/L (÷5 for scale)` : `${label}: ${raw}`;
                }}},
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 9 }, maxTicksLimit: 12, autoSkip: true } },
                y: { title: { display: true, text: 'mg/L (COD ÷5)', font: { size: 11 } }, grace: '5%', grid: { color: '#f0f0f0' } },
            },
        },
    });
}

// 5. Effluent quality monthly trends
function renderEdaEffluent(days) {
    edaDestroy('eda-effluent');
    const bod = edaMonthlyAvg(days, 'effluent_bod');
    const cod = edaMonthlyAvg(days, 'effluent_cod');
    const tss = edaMonthlyAvg(days, 'effluent_tss');
    const labels = bod.labels.map(edaFmtYM);
    const ctx = document.getElementById('eda-effluent').getContext('2d');
    edaCharts['eda-effluent'] = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets: [
            { label: 'BOD (mg/L)',  data: bod.values, borderColor: '#2563eb', backgroundColor: 'transparent',
              borderWidth: 2, pointRadius: 1.5, tension: 0.3, spanGaps: true },
            { label: 'COD/10 (mg/L)', data: cod.values.map(v => v !== null ? v / 10 : null),
              borderColor: '#9333ea', backgroundColor: 'transparent',
              borderWidth: 1.5, borderDash: [4, 3], pointRadius: 1.5, tension: 0.3, spanGaps: true },
            { label: 'TSS (mg/L)',  data: tss.values, borderColor: '#d97706', backgroundColor: 'transparent',
              borderWidth: 2, pointRadius: 1.5, tension: 0.3, spanGaps: true },
            {
                label: 'BOD/TSS limit: 10 mg/L', data: [],
                borderColor: '#dc2626', borderWidth: 1.5, borderDash: [6, 4],
                pointRadius: 0, isLimit: true,
            },
        ]},
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top',
                    labels: { filter: (item, data) => {
                        const ds = data.datasets[item.datasetIndex];
                        return ds?.isLimit === true || !ds?.isLimit;
                    }, font: { size: 11 }, boxWidth: 20 } },
                tooltip: { callbacks: { label: item => {
                    if (item.dataset.isLimit) return null;
                    const label = item.dataset.label;
                    const v = item.parsed.y;
                    if (v === null || isNaN(v)) return `${label}: ⚠ No data`;
                    const raw = label.includes('COD') ? (v * 10).toFixed(1) : v.toFixed(1);
                    return label.includes('COD') ? `COD: ${raw} mg/L (÷10 for scale)` : `${label}: ${raw}`;
                }}},
                annotation: { annotations: {
                    bod_lim: { type: 'line', yMin: 10, yMax: 10, borderColor: '#dc2626', borderWidth: 1.5, borderDash: [6, 4] },
                }},
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 9 }, maxTicksLimit: 12, autoSkip: true } },
                y: { title: { display: true, text: 'mg/L (COD ÷10)', font: { size: 11 } }, grace: '5%', grid: { color: '#f0f0f0' }, beginAtZero: true },
            },
        },
    });
}

// 6. Removal efficiency monthly avg
function renderEdaRemoval(days) {
    edaDestroy('eda-removal');
    const pairs = [
        { in: 'inlet_bod', out: 'effluent_bod', label: 'BOD', color: '#2563eb' },
        { in: 'inlet_cod', out: 'effluent_cod', label: 'COD', color: '#9333ea' },
        { in: 'inlet_tss', out: 'effluent_tss', label: 'TSS', color: '#d97706' },
    ];

    // Compute monthly removal averages
    const groups = {};
    for (const d of days) {
        if (!d.date) continue;
        const ym = d.date.slice(0, 7);
        if (!groups[ym]) groups[ym] = { bod: [], cod: [], tss: [] };
        const bod = (d.inlet_bod !== null && d.effluent_bod !== null && d.inlet_bod > 0)
            ? (d.inlet_bod - d.effluent_bod) / d.inlet_bod * 100 : null;
        const cod = (d.inlet_cod !== null && d.effluent_cod !== null && d.inlet_cod > 0)
            ? (d.inlet_cod - d.effluent_cod) / d.inlet_cod * 100 : null;
        const tss = (d.inlet_tss !== null && d.effluent_tss !== null && d.inlet_tss > 0)
            ? (d.inlet_tss - d.effluent_tss) / d.inlet_tss * 100 : null;
        // Only keep valid removals (filter obvious errors)
        if (bod !== null && bod >= 0 && bod <= 100) groups[ym].bod.push(bod);
        if (cod !== null && cod >= 0 && cod <= 100) groups[ym].cod.push(cod);
        if (tss !== null && tss >= 0 && tss <= 100) groups[ym].tss.push(tss);
    }
    const labels = Object.keys(groups).sort();
    const avg = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;

    const ctx = document.getElementById('eda-removal').getContext('2d');
    edaCharts['eda-removal'] = new Chart(ctx, {
        type: 'line',
        data: { labels: labels.map(edaFmtYM), datasets: [
            { label: 'BOD %', data: labels.map(ym => avg(groups[ym].bod)),
              borderColor: '#2563eb', backgroundColor: 'transparent', borderWidth: 2,
              pointRadius: 1.5, tension: 0.3, spanGaps: true },
            { label: 'COD %', data: labels.map(ym => avg(groups[ym].cod)),
              borderColor: '#9333ea', backgroundColor: 'transparent', borderWidth: 2,
              pointRadius: 1.5, tension: 0.3, spanGaps: true },
            { label: 'TSS %', data: labels.map(ym => avg(groups[ym].tss)),
              borderColor: '#d97706', backgroundColor: 'transparent', borderWidth: 2,
              pointRadius: 1.5, tension: 0.3, spanGaps: true },
            { label: '95% target', data: [], borderColor: '#16a34a', borderWidth: 1.5,
              borderDash: [6, 4], pointRadius: 0, isLimit: true },
        ]},
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top', labels: { font: { size: 11 }, boxWidth: 20 } },
                tooltip: { callbacks: { label: i => {
                    if (i.dataset.isLimit) return null;
                    const v = i.parsed.y;
                    return v === null || isNaN(v) ? `${i.dataset.label}: ⚠ No data`
                                                  : `${i.dataset.label}: ${v.toFixed(1)}%`;
                }}},
                annotation: { annotations: {
                    t95: { type: 'line', yMin: 95, yMax: 95, borderColor: '#16a34a', borderWidth: 1.5, borderDash: [6, 4] },
                }},
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 9 }, maxTicksLimit: 12, autoSkip: true } },
                y: { title: { display: true, text: 'Removal %', font: { size: 11 } }, min: 60, max: 100, grid: { color: '#f0f0f0' } },
            },
        },
    });
}

// 7. Compliance rate by year (grouped bar)
function renderEdaCompliance(days) {
    edaDestroy('eda-compliance');
    const years = [2020, 2021, 2022, 2023, 2024, 2025];
    const ctx = document.getElementById('eda-compliance').getContext('2d');
    const paramColors = ['#2563eb', '#16a34a', '#9333ea', '#d97706'];

    edaCharts['eda-compliance'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: years.map(String),
            datasets: EDA_COMPLIANCE.map((param, pi) => ({
                label: param.label,
                data: years.map(y => edaCompliancePct(days, y, param)),
                backgroundColor: paramColors[pi] + 'cc',
                borderColor: paramColors[pi],
                borderWidth: 1,
                borderRadius: 3,
            })),
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top', labels: { font: { size: 11 }, boxWidth: 12 } },
                tooltip: { callbacks: { label: i => {
                    const v = i.parsed.y;
                    return v === null ? `${i.dataset.label}: No data` : `${i.dataset.label}: ${v.toFixed(1)}%`;
                }}},
                annotation: { annotations: {
                    full: { type: 'line', yMin: 80, yMax: 80, borderColor: '#6b7280', borderWidth: 1, borderDash: [4, 3] },
                }},
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 11 } } },
                y: { min: 0, max: 100, title: { display: true, text: '% Days Compliant', font: { size: 11 } },
                     grid: { color: '#f0f0f0' } },
            },
        },
    });
}

// 8. Seasonal pattern (flow + power/flow by calendar month)
function renderEdaSeasonal(days) {
    edaDestroy('eda-seasonal');
    const flowSeasonal  = edaSeasonalAvg(days, 'flow');
    const powerSeasonal = edaSeasonalAvg(days, 'power_per_flow');
    const ctx = document.getElementById('eda-seasonal').getContext('2d');
    edaCharts['eda-seasonal'] = new Chart(ctx, {
        type: 'bar',
        data: { labels: EDA_SHORT_MONTHS, datasets: [
            {
                label: 'Avg Flow (MLD)',
                data: flowSeasonal,
                backgroundColor: '#2563eb55',
                borderColor: '#2563eb',
                borderWidth: 1.5,
                borderRadius: 3,
                yAxisID: 'yFlow',
            },
            {
                label: 'Avg Power/Flow (KW/ML)',
                data: powerSeasonal,
                type: 'line',
                borderColor: '#7c3aed',
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointRadius: 3,
                tension: 0.3,
                yAxisID: 'yPower',
            },
        ]},
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top', labels: { font: { size: 11 }, boxWidth: 16 } },
                tooltip: { callbacks: { label: i => {
                    const v = i.parsed.y;
                    const unit = i.dataset.yAxisID === 'yFlow' ? 'MLD' : 'KW/ML';
                    return `${i.dataset.label}: ${v !== null && !isNaN(v) ? v.toFixed(1) : '—'} ${unit}`;
                }}},
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 11 } } },
                yFlow:  { position: 'left',  title: { display: true, text: 'MLD',    font: { size: 11 } }, grace: '5%', grid: { color: '#f0f0f0' } },
                yPower: { position: 'right', title: { display: true, text: 'KW/ML',  font: { size: 11 } }, grace: '5%', grid: { display: false } },
            },
        },
    });
}

// 9. Annual averages as % of control limit
function renderEdaAnnual(days) {
    edaDestroy('eda-annual');
    const years = [2020, 2021, 2022, 2023, 2024, 2025];
    const params = [
        { key: 'inlet_bod',    label: 'Inlet BOD',    limit: 300  },
        { key: 'inlet_tss',    label: 'Inlet TSS',    limit: 400  },
        { key: 'effluent_bod', label: 'Effluent BOD', limit: 10   },
        { key: 'effluent_cod', label: 'Effluent COD', limit: 250  },
        { key: 'effluent_tss', label: 'Effluent TSS', limit: 10   },
    ];
    const paramColors = ['#2563eb', '#0891b2', '#16a34a', '#9333ea', '#d97706'];
    const ctx = document.getElementById('eda-annual').getContext('2d');
    edaCharts['eda-annual'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: years.map(String),
            datasets: params.map((p, pi) => ({
                label: p.label,
                data: years.map(y => {
                    const avg = edaAnnualAvg(days, p.key, y);
                    return avg !== null ? (avg / p.limit) * 100 : null;
                }),
                backgroundColor: paramColors[pi] + 'bb',
                borderColor: paramColors[pi],
                borderWidth: 1,
                borderRadius: 3,
            })),
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top', labels: { font: { size: 11 }, boxWidth: 12 } },
                tooltip: { callbacks: { label: i => {
                    const v = i.parsed.y;
                    const p = params[i.datasetIndex];
                    const raw = v !== null ? (v / 100 * p.limit).toFixed(1) : '—';
                    return v === null ? `${i.dataset.label}: No data`
                                     : `${i.dataset.label}: ${v.toFixed(1)}% of limit (${raw} mg/L)`;
                }}},
                annotation: { annotations: {
                    limit100: { type: 'line', yMin: 100, yMax: 100,
                                borderColor: '#dc2626', borderWidth: 1.5, borderDash: [6, 4],
                                label: { content: 'Control limit', display: true, position: 'end', font: { size: 10 } } },
                }},
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 11 } } },
                y: { min: 0, title: { display: true, text: '% of Control Limit', font: { size: 11 } },
                     grid: { color: '#f0f0f0' } },
            },
        },
    });
}

// ─── Init ──────────────────────────────────────────────────────────────────────

async function initEdaView() {
    edaDays = await loadEdaData();
    renderEdaMissing(edaDays);
    renderEdaMissingCounts1(edaDays);
    renderEdaMissingCountsPh(edaDays);
    renderEdaMissingCountsBod(edaDays);
    renderEdaMissingCountsCod(edaDays);
    renderEdaMissingCountsTss(edaDays);
    renderEdaFlow(edaDays);
    renderEdaPower(edaDays);
    renderEdaInlet(edaDays);
    renderEdaEffluent(edaDays);
    renderEdaRemoval(edaDays);
    renderEdaCompliance(edaDays);
    renderEdaSeasonal(edaDays);
    renderEdaAnnual(edaDays);
}
