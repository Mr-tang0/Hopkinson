<template>
  <div class="sample-form-container">
    <div class="sample-form-header">
      <h2>杆参数设置</h2>
      <p>请输入杆详细信息, 带*号为必填项</p>
    </div>

    <form id="hopkinsonForm" @submit.prevent>
      <div class="sample-form-group">
        <label for="hopkinsonType">杆型号*</label>
        <select id="hopkinsonType" v-model="selectedKey" @change="handleModelChange" required>
          <option v-for="(item, key) in hopkinsonList" :key="key" :value="key">{{ key }}</option>
          <option value="Custom">自定义型号</option>
        </select>
      </div>

      <div class="sample-form-group" v-if="selectedKey === 'Custom'">
        <label for="newName" style="color: #3b82f6;">新配置名称*</label>
        <input type="text" id="newName" v-model="newName" placeholder="保存为新记录的名称" required>
      </div>

      <div class="sample-form-group">
        <label for="rodMode">杆模式*</label>
        <select id="rodMode" v-model="form.Mode" :disabled="!isEditable" required>
          <option value="compression">压杆</option>
          <option value="stretch">拉杆</option>
        </select>
      </div>

      <div class="sample-form-group">
        <label for="rodMaterial">杆材质</label>
        <input type="text" id="rodMaterial" v-model="form.Material" :disabled="!isEditable" placeholder="请输入杆材质" required>
      </div>

      <div class="sample-form-row">
        <div class="sample-form-group">
          <label for="rodDiameter">截面直径ø (mm)*</label>
          <input type="number" id="rodDiameter" v-model.number="form.Diameter" :disabled="!isEditable" placeholder="请输入截面直径" step="0.01" min="0" required>
        </div>
        <div class="sample-form-group">
          <label for="youngsModulus">Young's模量 (GPa)*</label>
          <input type="number" id="youngsModulus" v-model.number="form.YoungSPa" :disabled="!isEditable" placeholder="请输入Young's模量" step="1" min="0" required>
        </div>
      </div>

      <div class="sample-form-group">
        <label for="soundVelocity">声速 (m/s)*</label>
        <input type="number" id="soundVelocity" v-model.number="form.SoundVelocity" :disabled="!isEditable" placeholder="请输入声速" step="1" min="0" required>
      </div>

      <!-- <div class="sample-form-row">
        <div class="sample-form-group">
          <label for="gageFactor">Gage因子*</label>
          <input type="number" id="gageFactor" v-model.number="form.GageFactor" :disabled="!isEditable" placeholder="请输入Gage因子" step="0.1" min="0" required>
        </div>
        <div class="sample-form-group">
          <label for="excitationVoltage">激励电压 (V)*</label>
          <input type="number" id="excitationVoltage" v-model.number="form.BridgeTensionV" :disabled="!isEditable" placeholder="请输入激励电压" step="0.1" min="0" required>
        </div>
      </div> -->

      <div class="sample-form-row">
        <div class="sample-form-group">
          <label for="incidentCalibrationFactor">入射杆标定系数*</label>
          <input type="number" id="incidentCalibrationFactor" v-model.number="form.IncidentCoefficient" :disabled="!isEditable" placeholder="请输入入射杆标定系数" step="0.000001" min="0" required>
        </div>
        <div class="sample-form-group">
          <label for="transmittedCalibrationFactor">透射杆标定系数*</label>
          <input type="number" id="transmittedCalibrationFactor" v-model.number="form.TransmittedCoefficient" :disabled="!isEditable" placeholder="请输入透射杆标定系数" step="0.000001" min="0" required>
        </div>
      </div>

      <div class="sample-form-group">
        <label class="switch-row">
          <span class="switch-label">启用色散矫正</span>
          <input type="checkbox" class="switch-input" v-model="hasDispersion" :disabled="!isEditable">
          <span class="switch-slider"></span>
        </label>
      </div>

      <div v-if="hasDispersion" class="sample-form-grid">
        <div class="sample-form-group">
          <label for="poissonRatio">杆的泊松比</label>
          <input type="number" id="poissonRatio" v-model.number="form.PoissonRatio" :disabled="!isEditable" placeholder="请输入泊松比" step="0.01" min="0" max="0.5">
        </div>

        <div class="sample-form-row">
          <div class="sample-form-group">
            <label for="incidentGaugeToEndDistance">入射应变片到端面距离(mm)</label>
            <input type="number" id="incidentGaugeToEndDistance" v-model.number="form.IncidentGaugeToEndDistance" :disabled="!isEditable" placeholder="请输入距离" step="0.1" min="0">
          </div>
          <div class="sample-form-group">
            <label for="transmitGaugeToEndDistance">透射应变片到端面距离(mm)</label>
            <input type="number" id="transmitGaugeToEndDistance" v-model.number="form.TransmitGaugeToEndDistance" :disabled="!isEditable" placeholder="请输入距离" step="0.1" min="0">
          </div>
        </div>
        
      </div>

      <div class="sample-button-group">
        <button v-if="selectedKey === 'Custom'" type="button" class="btn btn-save" style="background-color: #10b981;" @click="handleFileSave">保存为预设</button>
        <button v-if="selectedKey !== 'Custom'" type="button" class="btn btn-delete" @click="handleDelete">删除预设</button>
        <button type="button" class="btn btn-save" @click="handleSave">确定应用</button>
        <button type="button" class="btn btn-cancel" @click="$emit('close')">取消</button>
      </div>
    </form>
  </div>
</template>

<script setup>
import { reactive, inject, onMounted, ref, computed } from 'vue';
// 导入 Wails 后端方法
import { DeleteBarConfig, LoadConfig, SaveBarConfig } from '../../wailsjs/go/main/App';

const notify = inject('globalNotify');
const emit = defineEmits(['close', 'save']);

const hopkinsonList = ref({}); // 存储配置文件的记录
const selectedKey = ref('Custom'); // 下拉框选中的 key
const newName = ref(''); // 自定义模式下的新 Key 名称

const defaultForm = {
  Type: 'Custom',
  Mode: 'compression',
  Material: '',
  Diameter: 14.5,
  YoungSPa: 210,
  SoundVelocity: 5190,
  IncidentCoefficient: 1.0,
  TransmittedCoefficient: 1.0,
  EnableDispersionCorrection: false,
  PoissonRatio: 0.33,
  IncidentGaugeToEndDistance: 0,
  TransmitGaugeToEndDistance: 0
};

const form = reactive({ ...defaultForm });

const hasDispersion = computed({
  get: () => form.EnableDispersionCorrection,
  set: (value) => {
    form.EnableDispersionCorrection = value;
  }
});

// 计算属性：控制输入框是否可编辑
const isEditable = computed(() => selectedKey.value === 'Custom');

// 挂载时加载配置文件
onMounted(async () => {
  try {
    const res = await LoadConfig();
    if (res && res.hopkinsonList) {
      hopkinsonList.value = res.hopkinsonList;
      // 默认选中第一个已有型号，如果没有则选中自定义
      const keys = Object.keys(res.hopkinsonList);
      if (keys.length > 0) {
        selectedKey.value = keys[0];
        handleModelChange();
      }
    }
  } catch (err) {
    console.error("加载配置失败:", err);
  }
});

const resetForm = () => {
  Object.assign(form, defaultForm);
};

const cloneForm = () => ({ ...form });

// 型号切换逻辑
const handleModelChange = () => {
  if (selectedKey.value === 'Custom') {
    resetForm();
  } else {
    const data = hopkinsonList.value[selectedKey.value];
    // 映射后端数据到表单
    form.Type = selectedKey.value;
    form.Mode = data.Mode;
    form.Material = data.Material;
    form.Diameter = data.Diameter;
    form.YoungSPa = data.YoungSPa;
    form.SoundVelocity = data.SoundVelocity;
    form.IncidentCoefficient = data.IncidentCoefficient;
    form.TransmittedCoefficient = data.TransmittedCoefficient;
    form.EnableDispersionCorrection = !!data.EnableDispersionCorrection;
    form.PoissonRatio = data.PoissonRatio ?? defaultForm.PoissonRatio;
    form.IncidentGaugeToEndDistance = data.IncidentGaugeToEndDistance ?? defaultForm.IncidentGaugeToEndDistance;
    form.TransmitGaugeToEndDistance = data.TransmitGaugeToEndDistance ?? defaultForm.TransmitGaugeToEndDistance;
  }
};

// 物理保存到 JSON 文件
const handleFileSave = async () => {
  if (!newName.value) {
    if (notify) notify("请输入新配置的名称", "warning");
    return;
  }
  try {
    const success = await SaveBarConfig(newName.value, cloneForm());
    if (success) {
      if (notify) notify("配置已保存到本地文件", "success");
      // 更新本地列表，切换选中项
      hopkinsonList.value[newName.value] = { ...form };
      selectedKey.value = newName.value;
      handleModelChange();
    }
  } catch (err) {
    if (notify) notify("保存失败: " + err, "error");
  }
};

// 删除当前预设配置
const handleDelete = async () => {
  if (selectedKey.value === 'Custom') return;

  const deleteKey = selectedKey.value;
  const confirmed = window.confirm(`确定删除预设“${deleteKey}”吗？`);
  if (!confirmed) return;

  try {
    const success = await DeleteBarConfig(deleteKey);
    if (success) {
      if (notify) notify("配置已删除", "success");
      delete hopkinsonList.value[deleteKey];
      const keys = Object.keys(hopkinsonList.value);
      selectedKey.value = keys.length > 0 ? keys[0] : 'Custom';
      handleModelChange();
    }
  } catch (err) {
    if (notify) notify("删除失败: " + err, "error");
  }
};

// 点击“确定”应用到当前内存
const handleSave = async () => {
  try {
    emit('save', { ...form });
    if (notify) notify("参数应用成功", "success");
    emit('close');
  } catch (err) {
    if (notify) notify("应用失败: " + err, "error");
  }
};

</script>

<style scoped>
/* 原有样式全部保留 */
.sample-form-container {
  background: white;
  padding: 25px;
  border-radius: 12px;
  width: 520px;
  max-width: 95vw;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
}


input:disabled, select:disabled {
  background-color: #f8fafc;
  color: #94a3b8;
  cursor: not-allowed;
  border-color: #e2e8f0;
}

/* 拨动开关 */
.switch-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 8px 0;
  cursor: pointer;
}
.switch-label {
  font-weight: 600;
  color: #475569;
  font-size: 0.85rem;
}
.switch-input {
  display: none;
}
.switch-slider {
  position: relative;
  width: 46px;
  height: 26px;
  background-color: #cbd5e1;
  border-radius: 13px;
  transition: background-color 0.2s;
  flex-shrink: 0;
}
.switch-slider::before {
  content: '';
  position: absolute;
  top: 3px;
  left: 3px;
  width: 20px;
  height: 20px;
  background-color: #fff;
  border-radius: 50%;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
  transition: transform 0.2s;
}
.switch-input:checked + .switch-slider {
  background-color: #3b82f6;
}
.switch-input:checked + .switch-slider::before {
  transform: translateX(20px);
}
.switch-input:disabled + .switch-slider {
  opacity: 0.5;
  cursor: not-allowed;
}

/* 其他样式保持原样... */
.sample-form-header { margin-bottom: 20px; border-bottom: 1px solid #f1f5f9; padding-bottom: 10px; }
.sample-form-header h2 { margin: 0; color: #1e293b; font-size: 1.4rem; }
.sample-form-header p { color: #64748b; font-size: 0.85rem; margin: 5px 0 0; }
.sample-form-group { margin-bottom: 15px; display: flex; flex-direction: column; gap: 4px; }
.sample-form-row { display: flex; gap: 15px; }
.sample-form-row .sample-form-group { flex: 1; }
label { font-weight: 600; color: #475569; font-size: 0.85rem; }
input, select { padding: 10px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.95rem; transition: all 0.2s; background-color: #fff; }
input:focus, select:focus { outline: none; border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1); }
.sample-button-group { display: flex; justify-content: flex-end; gap: 12px; margin-top: 20px; padding-top: 15px; border-top: 1px solid #f1f5f9; }
.btn { padding: 10px 24px; border-radius: 6px; cursor: pointer; font-weight: 500; border: none; font-size: 0.95rem; }
.btn-save { background-color: #3b82f6; color: white; }
.btn-save:hover { background-color: #2563eb; }
.btn-delete { background-color: #ef4444; color: white; }
.btn-delete:hover { background-color: #dc2626; }
.btn-cancel { background-color: #f1f5f9; color: #475569; }
.btn-cancel:hover { background-color: #e2e8f0; }
</style>