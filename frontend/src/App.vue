<script setup>
import { ref, reactive, onMounted, provide, toRaw } from 'vue';
import * as echarts from 'echarts';
import { 
  InitSignalProcessor,
  LoadShpbData, 
  SetWaveRange, 
  SaveSampleDetails, 
  SaveHopkinsonDetails,
  BaselineCorrection, 
  SpectrumFilter,
  ManulAlign,
  TimeAlign,
  StartCalculate,
  ExportData,
  GetLatestRelease
} from '../wailsjs/go/main/App';

import { BrowserOpenURL } from '../wailsjs/runtime'

import MessageContainer from './components/MessageContainer.vue';
import HopkinsonModal from './components/HopkinsonModal.vue';
import SampleModal from './components/SampleModal.vue';
import About from './components/About.vue'; 


// --- 1. 全局状态 ---
// UI 状态
const isLeftCollapsed = ref(false);
const isAboutVisible = ref(false);
const showSerialChooseModal = ref(false);
const showNewHopkinsonModal = ref(false);
const showNewSampleModal = ref(false);
const showUpdateModal = ref(false);

// 图表状态
const chartRef = ref(null);
let myChart = null;
const currentTab = ref('原始波形');
const chartTabs = ref(['原始波形', '频谱图', '时域对齐', '计算结果']);

// 点击固定点
const clickedPoint = ref(null);  // { x, y, seriesName, color, chartX, chartY }

// 数据选择状态
const isCtrlPressed = ref(false);
const selectedType = ref('incident');
const range = reactive({ start: 0, end: 0 });

// 参数配置
const filterParams = reactive({ low: 0, high: 180000 });
const waveParams = reactive({ inc: 0, trans: 0 });
const alignMethod = ref("gb");
const alignOffsetParams = reactive({ inc: 0, trans: 0, ref: 0 });
const calcMethod = ref("incAndTrans");

// 计算与更新状态
const calculating = ref(false);
const progress = ref(0);
const latestInfo = ref({ version: '', url: '', currentVersion: '' });

// 文件状态
let incFile = null, transFile = null;
const incFileName = ref('选择入射波');
const transFileName = ref('选择透射波');

// 数据存储
const chartDataMap = reactive({
  '原始波形': {
    xlabel: '时间 (s)', ylabel: '振幅 (V)',
    curves: [{ name: '入射波', x: [], y: [], color: '#ef4444' }, { name: '透射波', x: [], y: [], color: '#3b82f6' }]
  },
  '频谱图': {
    xlabel: '频率 (Hz)', ylabel: '幅度 (dB)',
    curves: [{ name: '入射波频谱', x: [], y: [], color: '#8b5cf6' }, { name: '透射波频谱', x: [], y: [], color: '#f59e0b' }]
  },
  '时域对齐': {
    xlabel: '对齐时间 (s)', ylabel: '振幅 (V)',
    curves: [
      { name: '入射波', x: [], y: [], color: '#ef4444' },
      { name: '反射波', x: [], y: [], color: '#10b981' },
      { name: '透射波', x: [], y: [], color: '#3b82f6' }
    ]
  },
  '计算结果': {
    xlabel: '应变 (ε)', ylabel: '应力 (σ/MPa)',
    curves: [
      { name: '应力-应变曲线', x: [], y: [], color: '#f59e0b' },
      { name: '应力-时间曲线', x: [], y: [], color: '#3b82f6' },
      { name: '应变-时间曲线', x: [], y: [], color: '#10b981' },
      { name: '应变率-时间曲线', x: [], y: [], color: '#10b981' }
    ]

  }
});

// --- 2. ECharts 核心逻辑 ---
const initChart = () => {
  myChart = echarts.init(chartRef.value);
  myChart.on('brushEnd', handleBrushEnd);
  myChart.on('click', handleChartClick);
  showChart('原始波形');
};

const handleBrushEnd = (params) => {
  clearBrushArea();
  exitBrushMode();

  const area = params.areas[0];
  if (!area || !area.coordRange) return;

  range.start = area.coordRange[0];
  range.end = area.coordRange[1];
  showSerialChooseModal.value = true;
};

// 点击曲线固定点，便于复制 X/Y 值
const handleChartClick = (params) => {
  if (!params || !params.value || !Array.isArray(params.value)) {
    if (clickedPoint.value) clickedPoint.value = null;
    return;
  }
  const pos = myChart.convertToPixel({ seriesIndex: params.seriesIndex }, params.value);
  clickedPoint.value = {
    x: parseFloat(params.value[0]),
    y: parseFloat(params.value[1]),
    seriesName: params.seriesName,
    color: params.color,
    pageX: pos ? pos[0] : 0,
    pageY: pos ? pos[1] : 0,
  };
};

const dismissClickPoint = () => {
  clickedPoint.value = null;
};

const showChart = async (tabName) => {
  currentTab.value = tabName;
  const config = chartDataMap[tabName];
  if (!config || !myChart) return;

  const MAX_POINTS = 50000;
  const series = config.curves.map(curve => {
    let rawX = curve.x || [], rawY = curve.y || [];
    const displayY = rawY.map(y => getDisplayY(tabName, curve.name, y));
    // const shouldNormalize = tabName === '时域对齐';
    const shouldNormalize = false;
    const minY = shouldNormalize ? Math.min(...displayY) : 0;
    const maxY = shouldNormalize ? Math.max(...displayY) : 0;
    const yRange = maxY - minY;
    let sampledData = [];
    
    if (rawX.length > MAX_POINTS) {
      const step = Math.floor(rawX.length / MAX_POINTS);
      for (let i = 0; i < rawX.length; i += step) {
        const y = shouldNormalize ? normalizeDisplayY(displayY[i], minY, yRange) : displayY[i];
        sampledData.push([rawX[i], y]);
      }
    } else {
      sampledData = rawX.map((x, i) => {
        const y = shouldNormalize ? normalizeDisplayY(displayY[i], minY, yRange) : displayY[i];
        return [x, y];
      });
    }

    return {
      name: curve.name, type: 'line',
      symbol: 'circle', showSymbol: false, symbolSize: 8,
      sampling: 'lttb', large: true, smooth: false,
      itemStyle: { color: curve.color },
      lineStyle: { color: curve.color, width: 2 },
      emphasis: {
        itemStyle: { color: curve.color, borderColor: curve.color, borderWidth: 3 },
        lineStyle: { width: 2 },
      },
      data: sampledData
    };
  });

  myChart.setOption({
    title: { text: tabName, left: 'center', top: 10 },
    animation: false,
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross', label: { backgroundColor: '#6a7985' } },
      formatter: function(params) {
        if (!params || params.length === 0) return '';
        let html = '<div style="font-size:13px;line-height:2;max-width:400px">';
        params.forEach(p => {
          if (p.value && Array.isArray(p.value)) {
            const xVal = parseFloat(p.value[0]).toExponential(6);
            const yVal = parseFloat(p.value[1]).toExponential(6);
            html += `<div style="display:flex;align-items:center;gap:4px;white-space:nowrap">`;
            html += `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${p.color};flex-shrink:0"></span>`;
            html += `<b style="flex-shrink:0">${p.seriesName}</b>`;
            html += ` X: <code style="background:#f0f0f0;padding:0 6px;border-radius:3px;user-select:all;cursor:pointer">${xVal}</code>`;
            html += ` Y: <code style="background:#f0f0f0;padding:0 6px;border-radius:3px;user-select:all;cursor:pointer">${yVal}</code>`;
            html += `</div>`;
          }
        });
        html += '</div>';
        return html;
      }
    },
    toolbox:{
      show: true,
      feature: {
        saveAsImage: { title: '保存为图片' }
      },
      right: 40,
      top: 50,
      iconStyle: { borderColor: '#94a3b8' }
    },
    legend: { bottom: 10 },
    grid: { top: '15%', left: '10%', right: '10%', bottom: '15%', containLabel: true },
    xAxis: { type: 'value', name: config.xlabel, nameLocation: 'middle', nameGap: 30 },
    yAxis: { type: 'value', name: config.ylabel },
    brush: { toolbox: ['lineX'], xAxisIndex: 0, brushMode: 'single', brushStyle: {
                            borderWidth: 1,                    // 边框宽度
                            borderColor: 'rgba(255, 0, 0, 0.8)', // 边框颜色 (示例：鲜红色)
                            color: 'rgba(255, 0, 0, 0.3)'       // 选区填充色 (示例：半透明红色)
                        }},
    dataZoom: [
      { type: 'inside', zoomOnMouseWheel: true, moveOnMouseMove: 'shift' },
      { type: 'slider', bottom: 40, height: 20 }
    ],
    legend: { 
      bottom: 10,
      selected: {
        '应力-时间曲线': false,
        '应变-时间曲线': false,
        '应变率-时间曲线': false
      }
    },
    series
  }, true);
  myChart.resize();
};

const getDisplayY = (tabName, curveName, y) => {
  if (tabName === '时域对齐' && curveName === '反射波') {
    return -y;
  }
  return y;
};

const normalizeDisplayY = (y, minY, yRange) => {
  if (!Number.isFinite(y) || !Number.isFinite(minY) || !Number.isFinite(yRange) || yRange === 0) {
    return 0;
  }
  return (y - minY) / yRange;
};

// --- 3. 文件加载与后端通信 ---

const handleFileChange = (e, type) => {
  const file = e.target.files[0];
  if (!file) return;
  if (type === 'inc') {
    incFile = file;
    incFileName.value = `入射波: ${file.name}`;
  } else {
    transFile = file;
    transFileName.value = `透射波: ${file.name}`;
  }
};

const readFileAsText = (file) => new Promise((res, rej) => {
  const r = new FileReader();
  r.onload = () => res(r.result);
  r.onerror = rej;
  r.readAsText(file);
});

const loadData = async () => {
  if (!incFile || !transFile) return notify("请选择两个波形文件", "warning");
  try {
    const [inc, trans] = await Promise.all([readFileAsText(incFile), readFileAsText(transFile)]);
    const result = await LoadShpbData(inc, trans);
    if (result.success) {
      chartDataMap['原始波形'].curves[0].x = result.incWave.X;
      chartDataMap['原始波形'].curves[0].y = result.incWave.Y;
      chartDataMap['原始波形'].curves[1].x = result.transWave.X;
      chartDataMap['原始波形'].curves[1].y = result.transWave.Y;

      const maxFreq = 500000;
      const filterFFT = (x, y) => {
        const newX = [], newY = [];
        for (let i = 0; i < x.length && x[i] <= maxFreq; i++) {
          newX.push(x[i]);
          newY.push(y[i]);
        }
        return { x: newX, y: newY };
      };
      const incFFTFiltered = filterFFT(result.incFFT.X, result.incFFT.Y);
      const transFFTFiltered = filterFFT(result.transFFT.X, result.transFFT.Y);

      chartDataMap['频谱图'].curves = [
        { name: '入射波频谱', x: incFFTFiltered.x, y: incFFTFiltered.y, color: '#8b5cf6' },
        { name: '透射波频谱', x: transFFTFiltered.x, y: transFFTFiltered.y, color: '#ec4899' }
      ];
      showChart('原始波形');
    }
    notify(result.message, result.success ? "success" : "error");
  } catch (err) { notify("加载失败", "error"); }
};

const confirmRange = async () => {
  try {
    const result = await SetWaveRange(selectedType.value, range.start, range.end);
    if (result.success) {
      const idxMap = { incident: 0, reflected: 1, transmitted: 2 };
      const curveIdx = idxMap[selectedType.value];
      chartDataMap['时域对齐'].curves[curveIdx].x = result.Wave.X;
      chartDataMap['时域对齐'].curves[curveIdx].y = result.Wave.Y;
    }
    notify(result.message, result.success ? "success" : "error");
    showSerialChooseModal.value = false;
    myChart.dispatchAction({ type: 'brush', areas: [] });

    if (selectedType.value==="incident"){
        selectedType.value="reflected";
    }else if (selectedType.value==="reflected"){
        selectedType.value="transmitted";
    }else{
        selectedType.value="incident";
    }
  } catch (err) { notify("传递失败", "error"); }
};


//新建样品和杆信息
const handleSampleSave = async(data) => { 
  const result = await SaveSampleDetails(toRaw(data));
  if (result) {
    const display = document.getElementById('sampleDisplay');
    if (display) {
      display.innerText = `样品参数：
      样品号: ${data.Number}
      样品名: ${data.Name}
      样品材质: ${data.Material}
      样品长: ${data.Length}
      样品直径: ${data.Diameter}
      `;
    }
  }
};

const handleBarSave = async(data) => {
  const result = await SaveHopkinsonDetails(toRaw(data));

  if (result) {
    const display = document.getElementById('rodParameters');
    if (display) {
      display.innerText = `杆参数：
      杆型号：${data.Type}
      杆模式：${data.Mode}
      杆材质：${data.Material}
      直径：${data.Diameter} mm
      模量：${data.YoungSPa} N/mm²
      波速：${data.SoundVelocity}m/s
      BridgeType: ${data.BridgeType}
      入射杆标定系数：${data.IncidentCoefficient}
      透射杆标定系数：${data.TransmittedCoefficient}
      `;
    }
  }

};


//数据调整
const syncWaveParams = async () => { 
  const data = toRaw(waveParams);
  const result = await BaselineCorrection(data);

  if (result.Success) {
    const curveIndex = result.WaveType === "inc" ? 0 : 1;

    // 💡 关键修改点 1：使用解构赋值确保响应式触发
    chartDataMap['原始波形'].curves[curveIndex] = {
      ...chartDataMap['原始波形'].curves[curveIndex],
      x: [...result.Wave.X],
      y: [...result.Wave.Y]
    };

    showChart('原始波形');
  }
  notify(result.Message, result.Success ? "success" : "error");
};

const syncFilterParams = async () => {
  const data = toRaw(filterParams);
  const result = await SpectrumFilter(data);
  if (result.Success) { 
    const targetChart = chartDataMap['原始波形'];
     result.Waves.forEach((wave, index) => {
        if (targetChart.curves[index]) {
          targetChart.curves[index] = {
            ...targetChart.curves[index],
            x: wave.X,
            y: wave.Y
          };
        }
      });
    showChart('原始波形');
    notify(result.Message, "success");
  }
};

const syncAlignOffset = async () => { 
  const data = toRaw(alignOffsetParams);
  const result = await ManulAlign(data);
  alignOffsetParams.inc=0;
  alignOffsetParams.trans=0
  alignOffsetParams.ref=0;

  if (result.Success) {
    const targetChart = chartDataMap['时域对齐'];

    result.Waves.forEach((wave, index) => {
      if (targetChart.curves[index]) {
        targetChart.curves[index] = {
          ...targetChart.curves[index],
          x: wave.X,
          y: wave.Y
        };
      }
    });

    showChart('时域对齐');
    notify(result.Message, "success");
  }else {
    notify(result.Message, "error");
  }
};


const handleAlign = async () => {
  try {
    const result = await TimeAlign(alignMethod.value);

    if (result.Success) {
      const targetChart = chartDataMap['时域对齐'];

      result.Waves.forEach((wave, index) => {
        if (targetChart.curves[index]) {
          targetChart.curves[index] = {
            ...targetChart.curves[index],
            x: wave.X,
            y: wave.Y
          };
        }
      });

      showChart('时域对齐');
      notify(result.Message, "success");
    } else {
      notify(result.Message, "error");
    }
  } catch (err) {
    notify("通信异常: " + err, "error");
  }
};

const startCalculation = async () => { 
  try {

    const result = await StartCalculate(calcMethod.value);

    if (result && result.stress_strain) {
      chartDataMap['计算结果'].curves[0].x = result.stress_strain.X;
      chartDataMap['计算结果'].curves[0].y = result.stress_strain.Y;

      chartDataMap['计算结果'].curves[1].x = result.stress_time.X;
      chartDataMap['计算结果'].curves[1].y = result.stress_time.Y;

      chartDataMap['计算结果'].curves[2].x = result.strain_time.X;
      chartDataMap['计算结果'].curves[2].y = result.strain_time.Y;

      chartDataMap['计算结果'].curves[3].x = result.strain_rate_time.X;
      chartDataMap['计算结果'].curves[3].y = result.strain_rate_time.Y;


      showChart('计算结果');
      
      notify("计算完成", "success");
    } else {
      notify("计算失败：返回数据为空", "error");
    }
  } catch (err) {
    console.error(err);
    notify("计算过程发生异常", "error");
  }
};

//保存

const exportResults = async () => {
  try {
    const result = await ExportData();
    
    if (result === "cancelled") {
      // 用户取消了，不做处理
      return;
    }
    
    if (result.startsWith("error:")) {
      notify(result, "error");
    } else {
      notify("数据已成功保存至: " + result, "success");
    }
  } catch (err) {
    notify("导出异常", "error");
  }
};

// --- 4. 图表交互 ---
const enterBrushMode = () => {
  myChart.dispatchAction({ type: 'takeGlobalCursor', key: 'brush', brushOption: { brushType: 'lineX' } });
};

const clearBrushArea = () => {
  myChart.dispatchAction({ type: 'brush', areas: [] });
};

const exitBrushMode = () => {
  myChart.dispatchAction({ type: 'takeGlobalCursor', key: 'brush', brushOption: { brushType: false } });
};


const toggleSidebar = () => {
  isLeftCollapsed.value = !isLeftCollapsed.value;
  setTimeout(() => myChart?.resize(), 310);
};

window.addEventListener('keydown', (e) => {
  if (e.key === 'Control' && !isCtrlPressed.value && myChart) {
    isCtrlPressed.value = true;
    enterBrushMode();
  }
});

window.addEventListener('keyup', (e) => {
  if (e.key === 'Control' && isCtrlPressed.value && myChart) {
    isCtrlPressed.value = false;
  }
});

// --- 5. 更新检查 ---
const checkUpdate = async () => {
  try {
    const res = await GetLatestRelease();
    if (res && res.latestVersion) {
      latestInfo.value = {
        version: res.latestVersion,
        url: res.downloadUrl,
        currentVersion: res.currentVersion
      };
      showUpdateModal.value = true;
    }
  } catch (e) {
    console.error("检查更新失败", e);
  }
};

const doDownload = (source) => {
  let url = latestInfo.value.url;
  if (source === 'accelerate') {
    url = 'https://ghfast.top/https://' + url.replace('https://', '');
  }
  BrowserOpenURL(url);
  showUpdateModal.value = false;
};

// --- 6. 生命周期 ---
onMounted(async () => {
  initChart();
  window.addEventListener('resize', () => myChart?.resize());
  InitSignalProcessor();
  await checkUpdate();
});

// 消息通知
const msgBoxRef = ref(null);
const notify = (content, type = 'info', duration = 3000) => msgBoxRef.value?.addMessage(content, type, duration);
provide('globalNotify', notify);

</script>

<template>
  <div class="app-wrapper">
    <MessageContainer ref="msgBoxRef" />
    <About v-model="isAboutVisible" />

    <aside class="sidebar left-sidebar" :class="{ 'collapsed': isLeftCollapsed }">
      <div class="collapse-trigger" @click="toggleSidebar">
        <span class="arrow-icon">{{ isLeftCollapsed ? '▶' : '◀' }}</span>
      </div>

      <div class="sidebar-content" v-show="!isLeftCollapsed">
        <div class="brand" @click="isAboutVisible = true" style="cursor: pointer;">
          <div class="logo-icon">SHPB</div>
          <h2>Hopkinson</h2>
        </div>
        
        <section class="config-group">
          <h3 class="section-title">📁 数据加载</h3>
          <div class="upload-card">
            <label class="file-btn">
              {{ incFileName }}
              <input type="file" hidden @change="e => handleFileChange(e, 'inc')" />
            </label>

            <label class="file-btn">
              {{ transFileName }}
              <input type="file" hidden @change="e => handleFileChange(e, 'trans')" />
            </label>
            <button class="action-btn primary" @click="loadData">执行加载</button>
          </div>
        </section>

        <!-- 参数设置 -->
        <div class="config-group">
            <h3 class="section-title">⚙️ 参数设置</h3>
            <button class="action-btn" @click="showNewSampleModal = true">新建样品</button>
            <div id="sampleDisplay" class="sample-display">无样品信息</div>
            <button class = "action-btn" @click="showNewHopkinsonModal = true">新建杆参数</button>
            <div id="rodParameters" class="sample-display">杆参数：未设置</div>
        </div>

        <div class="hint-text">收起此栏以进行详细分析 →</div>
      </div>
    </aside>

    <main class="main-content" @click="isLeftCollapsed=true">
      <nav class="chart-tabs">
        <button 
          v-for="t in chartTabs" 
          :key="t" 
          :class="['tab-item', { active: t === currentTab }]"
          @click="showChart(t)" 
        >
          {{ t }}
        </button>
      </nav>

      <div class="chart-wrapper">
        <div ref="chartRef" class="echarts-dom"></div>
        <Transition name="fade">
          <div v-if="clickedPoint" class="pin-tooltip">
            <div class="pin-header">
              <span class="pin-dot" :style="{ background: clickedPoint.color }"></span>
              <b>{{ clickedPoint.seriesName }}</b>
              <span class="pin-close" @click.stop="dismissClickPoint">✕</span>
            </div>
            <div class="pin-row">X = <code class="pin-val">{{ clickedPoint.x.toExponential(6) }}</code></div>
            <div class="pin-row">Y = <code class="pin-val">{{ clickedPoint.y.toExponential(6) }}</code></div>
            <!-- <div class="pin-hint">点击此卡片关闭</div> -->
          </div>
        </Transition>
      </div>
    </main>

    <aside class="sidebar right-sidebar" :class="{ 'auto-hidden': !isLeftCollapsed }">
      <div class="sidebar-content" v-show="isLeftCollapsed">
        
        <section class="config-group">
          <h3 class="section-title">📊 数据调整</h3>
          
          <div class="control-card">
            <label>基线调整</label>
            <div class="input-item">
              <span class="sub-label">入射波调整: {{ waveParams.inc/1000.0 }} V</span>
              <div class="slider-group">
                <input type="range" v-model.number="waveParams.inc" min="-1000" max="1000" @change="syncWaveParams" />
                <input type="number" v-model.number="waveParams.inc" class="mini-input" @keyup.enter="syncWaveParams" />
                <span class="unit-label">mV</span>
              </div>
            </div>
            
            <div class="input-item">
              <span class="sub-label">透射波调整: {{ waveParams.trans/1000.0 }} V</span>
              <div class="slider-group">
                <input type="range" v-model.number="waveParams.trans" min="-1000" max="1000" @change="syncWaveParams" />
                <input type="number" v-model.number="waveParams.trans" class="mini-input" @keyup.enter="syncWaveParams" />
                <span class="unit-label">mV</span>
              </div>
            </div>
          </div>

          <div class="control-card">
            <label>滤波参数</label>
            <div class="input-item">
              <span class="sub-label">低频截止: {{ filterParams.low }} Hz</span>
              <div class="slider-group">
                <input type="range" v-model.number="filterParams.low" min="0" max="1000" @change="syncFilterParams" />
                <input type="number" v-model.number="filterParams.low" class="mini-input" @keyup.enter="syncFilterParams" />
                <span class="unit-label">Hz</span>
              </div>
            </div>

            <div class="input-item">
              <span class="sub-label">高频截止: {{ filterParams.high }} Hz</span>
              <div class="slider-group">
                <input type="range" v-model.number="filterParams.high" min="0" max="500000" @change="syncFilterParams" />
                <input type="number" v-model.number="filterParams.high" class="mini-input" @keyup.enter="syncFilterParams" />
                <span class="unit-label">Hz</span>
              </div>
            </div>
          </div>

          <div class="control-card">
            <label>时域对齐</label>
            <div class="input-item">
              <select class="modern-select" v-model="alignMethod">
                <option value="gb">国标对齐</option>
                <option value="peak">波峰对齐</option>
                <option value="time">时间轴对齐</option>
                <option value="manual">手动调整</option>
              </select>
            </div>
            <div v-if="alignMethod=='manual'" class="sub-panel">

              <div class="input-item">
                <span class="sub-label">入射波:
                  <input type="range" v-model.number="alignOffsetParams.inc" min="-1000" max="1000" @change="syncAlignOffset" />
                  <input type="number" v-model.number="alignOffsetParams.inc" class="mini-input" @keyup.enter="syncAlignOffset" />
                </span>
              </div>
              <div class="input-item">
                <span class="sub-label">反射波:
                  <input type="range" v-model.number="alignOffsetParams.ref" min="-1000" max="1000" @change="syncAlignOffset" />
                  <input type="number" v-model.number="alignOffsetParams.ref" class="mini-input" @keyup.enter="syncAlignOffset" />
                </span>
              </div>
              <div class="input-item">
                <span class="sub-label">透射波:
                  <input type="range" v-model.number="alignOffsetParams.trans" min="-1000" max="1000" @change="syncAlignOffset" />
                  <input type="number" v-model.number="alignOffsetParams.trans" class="mini-input" @keyup.enter="syncAlignOffset" />
                </span>
              </div>
              

            </div>
            <button class="action-btn primary"@click="handleAlign">应用对齐</button>
            <!-- <button class="action-btn primary"@click="showManualAlign = !showManualAlign">手动微调</button>
            <div v-if="showManualAlign" class="sub-panel">
            </div> -->

          </div>
        </section>

        <section class="config-group">
          <h3 class="section-title">💾 数据计算与保存</h3>
          <div class="control-card highlight">
            <label>计算方法</label>
            <div class="input-item">
              <select class="modern-select" v-model="calcMethod">
                <option value="incAndTrans">双波法(入射+透射)</option>
                <option value="refAndTrans">双波法(反射+透射)</option>
                <option value="threeWave">三波法</option>
              </select>
            </div>
            
            <button class="action-btn big-run" @click="startCalculation">
              开始计算
            </button>
            
            <div v-if="calculating" class="progress-container">
              <div class="progress-bar" :style="{ width: progress + '%' }"></div>
            </div>

            <div v-else class="result-container"> 
            </div>

            <button class="action-btn outline-btn" @click="exportResults">导出结果</button>
          </div>
        </section>

      </div>
    </aside>
  </div>

  <div v-if="showSerialChooseModal" class="modal-overlay">
    <div class="modal-content">
      <h3>标记波形区间</h3>
      <p>选定范围: {{ range.start.toFixed(6) }} - {{ range.end.toFixed(6) }}</p>
      
      <div class="form-group">
        <label>选择波形类型:</label>
        <select v-model="selectedType" class="modern-select">
          <option value="incident">入射波 (Incident)</option>
          <option value="reflected">反射波 (Reflected)</option>
          <option value="transmitted">透射波 (Transmitted)</option>
        </select>
      </div>

      <div class="modal-footer">
        <button class="action-btn" @click="showSerialChooseModal = false">取消</button>
        <button class="action-btn primary" @click="confirmRange">确定</button>
      </div>
    </div>
  </div>

  <div class="modal-overlay" v-if="showNewHopkinsonModal">
    <HopkinsonModal 
      @close="showNewHopkinsonModal = false" 
      @save="handleBarSave" 
    />
  </div>

  <div class="modal-overlay" v-if="showNewSampleModal"> 
    <SampleModal 
    @close="showNewSampleModal = false"
    @save="handleSampleSave"
    ></SampleModal>
  </div>

  <Transition name="fade">
    <div v-if="showUpdateModal" class="update-modal-overlay" @click="showUpdateModal = false">
      <div class="update-card">
        <div class="update-icon">🚀</div>
        <h2>发现新版本！</h2>
        <p class="version-text">最新版本: {{ latestInfo.version }}</p>
        <p class="current-text">当前版本: {{ latestInfo.currentVersion }}</p>
        
        <div class="btn-group">
          <button class="btn-cancel" @click="showUpdateModal = false">稍后再说</button>
          <button class="btn-confirm secondary" @click="doDownload('github')">下载（github源）</button>
          <button class="btn-confirm" @click="doDownload('accelerate')">下载（加速源）</button>
        </div>
      </div>
    </div>
  </Transition>
</template>

<style scoped>
.app-wrapper {
  display: flex;
  width: 100vw;
  height: 100vh;
  background-color: #f1f5f9;
  overflow: hidden;
  user-select: none;
}

/* 侧边栏通用样式 */
.sidebar {
  height: 100%;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  flex-shrink: 0;
  z-index: 10;
  display: flex;
}

.sidebar-content {
  width: 240px; /* 保持内容宽度固定 */
  box-sizing: border-box;
  height: 100%;
  display: flex;
  flex-direction: column;
  white-space: nowrap;
  
  /* 允许垂直滚动，隐藏水平滚动 */
  overflow-y: auto;
  overflow-x: hidden;
  
  /* 平滑滚动效果 */
  scroll-behavior: smooth;
  
  /* 增加底部间距，防止最后一个按钮顶到底部 */
  padding-bottom: 40px; 
}

/* 自定义侧边栏滚动条样式 */
.sidebar-content::-webkit-scrollbar {
  width: 5px; /* 滚动条宽度 */
}

.sidebar-content::-webkit-scrollbar-track {
  background: transparent; /* 轨道背景 */
}

.sidebar-content::-webkit-scrollbar-thumb {
  background: rgba(148, 163, 184, 0.3); /* 滑块颜色 */
  border-radius: 10px;
}

.sidebar-content::-webkit-scrollbar-thumb:hover {
  background: rgba(59, 130, 246, 0.5); /* 悬停时变蓝色 */
}

/* 如果是左侧深色侧边栏，微调滑块颜色 */
.left-sidebar .sidebar-content::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.1);
}


/* 左侧栏样式 */
.left-sidebar { 
  width: 240px; 
  background: #1e293b; 
  color: #f8fafc; 
  padding: 20px;
}
.left-sidebar.collapsed { 
  width: 0; 
  padding: 0; 
}

/* 右侧栏样式 */
.right-sidebar { 
  width: 320px;
  min-width: 320px;
  max-width: 320px;
  box-sizing: border-box;
  background: #ffffff; 
  color: #334155; 
  padding: 20px;
  border-left: 1px solid #e2e8f0;
}

.right-sidebar .sidebar-content {
  width: 100%;
}

/* 当左侧未收起时，右侧自动隐藏 */
.right-sidebar.auto-hidden { 
  width: 0;
  min-width: 0;
  max-width: 0;
  padding: 0; 
  border: none;
}

/* 唯一的触发按钮 */
.collapse-trigger {
  position: absolute;
  top: 50%;
  right: -30px;
  transform: translateY(-50%);
  width: 30px;
  height: 150px;
  background: #1e293b;
  color: white;
  display: flex;
  justify-content: center;
  align-items: center;
  cursor: pointer;
  z-index: 100;
  border-radius: 0 8px 8px 0;
  box-shadow: 4px 0 10px rgba(0,0,0,0.1);
}
.collapse-trigger:hover { background: #3b82f6; }

/* 主视图 */
.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 20px;
  min-width: 0;
}

.chart-wrapper {
  flex: 1;
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.05);
  margin-top: 15px;
}

.echarts-dom { width: 100%; height: 100%; }

/* 点击固定点卡片 */
.chart-wrapper { position: relative; }
.pin-tooltip {
  position: absolute;
  top: 16px;
  right: 16px;
  background: #6eb7e8;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 10px 14px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.12);
  font-size: 13px;
  line-height: 1.6;
  z-index: 100;
  cursor: pointer;
  min-width: 180px;
  user-select: text;
}
.pin-header { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
.pin-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.pin-close { margin-left: auto; color: #94a3b8; font-size: 14px; cursor: pointer; }
.pin-row { white-space: nowrap; }
.pin-val {
  background: #6eb7e8;
  padding: 0 6px;
  border-radius: 3px;
  font-family: Consolas, monospace;
  user-select: all;
  cursor: text;
}
.pin-hint { font-size: 11px; color: #94a3b8; margin-top: 4px; text-align: right; }

/* UI 组件 */
.brand { display: flex; align-items: center; gap: 10px; margin-bottom: 30px; }
.logo-icon { 
  background: #3b82f6; 
  padding: 4px 8px; 
  border-radius: 4px; 
  font-weight: bold; 
  font-size: 12px; 
}

.config-group{
  margin-bottom: 10px;
}

.section-title { 
  font-size: 20px; 
  color: #94a3b8; 
  text-transform: uppercase; 
  margin-bottom: 12px; 
  letter-spacing: 1px; 
}
.file-btn { 
  display: block; 
  padding: 12px; 
  border: 2px dashed #475569; 
  border-radius: 8px; 
  text-align: center;
  margin-bottom: 8px; 
  cursor: pointer; 
  font-size: 15px; 
}

.action-btn { 
  width: 80%; 
  padding: 10px;
   border-radius: 6px; 
   border: none; 
   background: #3b82f6; 
   color: white; 
   font-weight: 600; 
   cursor: pointer; 
}

.action-btn:hover {
  background: #1d4ed8;
}

.sample-display {
    margin-top: 10px;
    padding: 10px;
    border-radius: 6px;
    text-align: center;
    font-size: 13px;

}

.action-btn.big-run { 
  background: #10b981; 
}
.action-btn.big-run:hover {
  background: #059669;
}

.control-card { 
  background: #f8fafc; 
  padding: 15px; 
  border-radius: 8px; 
  border: 1px solid #e2e8f0; 
  margin-bottom: 20px;
  width: 100%;
  box-sizing: border-box;
}

.control-card .action-btn {
  margin-top: 4px;
  margin-bottom: 8px;
  font-size: 13px;
  padding: 8px 12px;
}

/* 最后一个元素去掉下边距 */
.control-card *:last-child {
  margin-bottom: 0;
}





.hint-text { 
  margin-top: auto; 
  font-size: 11px; 
  color: #475569; 
  font-style: italic; 
}



/* 右侧特有微调 */
.sub-panel {
  background: #f1f5f9;
  padding: 10px;
  border-radius: 6px;
  margin-top: 10px;
}
.sub-label {
  font-size: 14px;
  color: #64748b;
  display: block;
  margin-bottom: 4px;
  text-align: left;
}
.flex-row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.mini-input {
  width: 60px;
  box-sizing: border-box;
  padding: 4px;
  border: 1px solid #cbd5e1;
  border-radius: 4px;
  font-size: 12px;
}

.slider-group {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 86px auto;
  align-items: center;
  gap: 8px;
}

.slider-group input[type="range"] {
  width: 100%;
  min-width: 0;
}

.slider-group .mini-input {
  width: 86px;
}

.unit-label {
  font-size: 13px;
  color: #64748b;
}
.btn-group {
  display: flex;
  gap: 8px;
  margin-top: 10px;
}
.progress-container {
  height: 6px;
  background: #e2e8f0;
  border-radius: 3px;
  margin: 15px 0;
  overflow: hidden;
}
.progress-bar {
  height: 100%;
  background: #10b981;
  transition: width 0.3s;
}
.highlight {
  border: 1px solid #3b82f644;
  background: #eff6ff;
}

/* 现代下拉框基础样式 */
.modern-select {
  width: 100%;
  padding: 10px 12px;
  font-size: 13px;
  line-height: 1.5;
  color: #334155; /* 默认深灰色文本 */
  background-color: #ffffff;
  background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%2394a3b8' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3e%3c/svg%3e");
  background-repeat: no-repeat;
  background-position: right 0.75rem center;
  background-size: 1.5em 1.5em;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  appearance: none; /* 隐藏原生箭头 */
  -webkit-appearance: none;
  -moz-appearance: none;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  cursor: pointer;
  outline: none;
  margin-bottom: 12px;
}

/* 悬停效果 */
.modern-select:hover {
  border-color: #cbd5e1;
  background-color: #f8fafc;
}

/* 聚焦效果（蓝色光晕） */
.modern-select:focus {
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

/* 当 select 位于深色侧边栏（左侧）时的适配 */
.left-sidebar .modern-select {
  background-color: #334155;
  border-color: #475569;
  color: #f8fafc;
  background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%23f8fafc' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3e%3c/svg%3e");
}

.left-sidebar .modern-select:hover {
  background-color: #475569;
  border-color: #64748b;
}

/* 内部选项样式 (部分浏览器支持有限) */
.modern-select option {
  padding: 10px;
  background-color: #ffffff;
  color: #334155;
}

.left-sidebar .modern-select option {
  background-color: #1e293b;
  color: #f8fafc;
}

/* 选项卡样式 */
.chart-tabs { display: flex; gap: 10px; }
.tab-item { padding: 8px 16px; border: none; background: #e2e8f0; border-radius: 6px; cursor: pointer; color: #64748b; transition: 0.2s; }
.tab-item.active { background: #3b82f6; color: white; }


.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 999;
}

.modal-content {
  background: white;
  padding: 24px;
  border-radius: 12px;
  width: 350px;
  box-shadow: 0 10px 25px rgba(0,0,0,0.2);
  color: #1e293b;
}

.modal-footer {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}


.update-modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.75);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 10000;
  backdrop-filter: blur(4px);
}
.update-card {
  background: #2c3e50;
  padding: 30px;
  border-radius: 16px;
  text-align: center;
  width: 320px;
  border: 1px solid #34495e;
  box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}
.update-icon { font-size: 50px; margin-bottom: 10px; }
.version-text { color: #2ecc71; font-weight: bold; margin: 10px 0 5px; }
.current-text { color: #95a5a6; font-size: 13px; margin-bottom: 20px; }

.update-card .btn-group {
  display: grid;
  grid-template-columns: 1fr 1.25fr 1.25fr;
  gap: 12px;
  width: 100%;
  margin-top: 24px;
}

.update-card .btn-cancel,
.update-card .btn-confirm {
  min-width: 0;
  height: 48px;
  padding: 0 12px;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 700;
  line-height: 1.2;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  white-space: normal;
  word-break: keep-all;
  box-sizing: border-box;
  transition: background-color 0.2s ease, transform 0.2s ease;
}

.update-card .btn-cancel {
  background: #3b5068;
  color: #dbe7f3;
}

.update-card .btn-confirm {
  background: #3498db;
  color: #ffffff;
}

.update-card .btn-cancel:hover,
.update-card .btn-confirm:hover {
  transform: translateY(-1px);
}

.update-card .btn-cancel:hover { background: #46617d; }
.update-card .btn-confirm:hover { background: #2980b9; }

.update-card .btn-confirm.secondary {
  background: #7f8c8d;
}
.update-card .btn-confirm.secondary:hover { background: #6f7d7e; }

.fade-enter-active, .fade-leave-active { transition: opacity 0.3s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>
