<template>
  <div class="sample-form-container">
    <div class="sample-form-header">
      <h2>样品参数设置</h2>
      <p>请输入样品详细信息</p>
    </div>

    <form id="sampleForm" @submit.prevent>
      <div class="sample-form-group">
        <label for="sampleNumber">样品编号</label>
        <input 
          type="text" 
          id="sampleNumber" 
          v-model="form.Number" 
          placeholder="请输入样品编号" 
          required
        >
      </div>

      <div class="sample-form-group">
        <label for="sampleName">样品名称</label>
        <input 
          type="text" 
          id="sampleName" 
          v-model="form.Name" 
          placeholder="请输入样品名称" 
          required
        >
      </div>

      <div class="sample-form-group">
        <label for="sampleMaterial">样品材质</label>
        <input 
          type="text" 
          id="sampleMaterial" 
          v-model="form.Material" 
          placeholder="请输入样品材质" 
          required
        >
      </div>

      <div class="sample-form-group">
        <label for="sampleShape">样品形状</label>
        <select id="sampleShape" v-model="form.Shape">
          <option value="cylinder">圆柱</option>
          <option value="cuboid">立方</option>
        </select>
      </div>

      <div class="sample-form-row">
        <div v-if="form.Shape === 'cylinder'" class="sample-form-group">
          <label for="sampleDiameter">截面直径ø (mm)</label>
          <input 
            type="number" 
            id="sampleDiameter" 
            v-model.number="form.Diameter" 
            placeholder="请输入直径" 
            step="0.01" 
            min="0.01"
            required
          >
        </div>

        <template v-else>
          <div class="sample-form-group">
            <label for="sampleWidth">截面宽 (mm)</label>
            <input type="number" id="sampleWidth" v-model.number="form.CrossSectionWidth" step="0.01" min="0.01" required>
          </div>
          <div class="sample-form-group">
            <label for="sampleCrossLength">截面长度 (mm)</label>
            <input type="number" id="sampleCrossLength" v-model.number="form.CrossSectionLength" step="0.01" min="0.01" required>
          </div>
        </template>
      </div>

      <div class="sample-form-row">
        <div class="sample-form-group">
          <label for="sampleLength">样品长度l (mm)</label>
          <input 
            type="number" 
            id="sampleLength" 
            v-model.number="form.Length" 
            placeholder="请输入长度" 
            step="0.01" 
            min="0.01"
            required
          >
        </div>
      </div>

      <div class="info-tag">
        自动计算样品截面积: <strong>{{ calculatedArea.toFixed(4) }}</strong> mm²
      </div>

      <div class="sample-button-group">
        <button type="button" class="btn btn-save" @click="handleSave">保存</button>
        <button type="button" class="btn btn-cancel" @click="$emit('close')">取消</button>
      </div>
    </form>
  </div>
</template>

<script setup>
import { onMounted, reactive, computed, inject, toRaw } from 'vue';
import { LoadConfig, SaveSampleConfig} from '../../wailsjs/go/main/App';

const notify = inject('globalNotify');
const emit = defineEmits(['close', 'save']);

// 响应式表单，字段名首字母大写以匹配 Go 结构体习惯
const form = reactive({
  Number: '1',
  Name: 'Fe',
  Material: '测试',
  Shape: 'cylinder',
  Diameter: 4.5,
  CrossSectionWidth: 4.5,
  CrossSectionLength: 4.5,
  Length: 4.5
});


onMounted(async () => {
  try {
    const res = await LoadConfig();
    if (res && res.sample) {
      form.Number = res.sample.Number;
      form.Name = res.sample.Name;
      form.Material = res.sample.Material;
      form.Shape = res.sample.Shape || 'cylinder';
      form.Diameter = res.sample.Diameter;
      form.CrossSectionWidth = res.sample.CrossSectionWidth || 4.5;
      form.CrossSectionLength = res.sample.CrossSectionLength || 4.5;
      form.Length = res.sample.Length;
    }
  } catch (err) {
    console.error("加载配置失败:", err);
  }
});

const calculatedArea = computed(() => {
  if (form.Shape === 'cuboid') {
    return Number(form.CrossSectionWidth) * Number(form.CrossSectionLength);
  }
  const r = form.Diameter / 2;
  return Math.PI * r * r;
});

const handleSave = async () => {
  const dimensions = form.Shape === 'cuboid'
    ? [form.CrossSectionWidth, form.CrossSectionLength, form.Length]
    : [form.Diameter, form.Length];
  if (!form.Number || !form.Name || !form.Material || dimensions.some(value => !Number.isFinite(value) || value <= 0) || !Number.isFinite(calculatedArea.value) || calculatedArea.value <= 0) {
    if (notify) notify("请填写完整的样品信息", "warning");
    return;
  }

  try {
    const rawData = {
      ...toRaw(form),
      Area: calculatedArea.value
    };

    const success = await SaveSampleConfig({...rawData});
    if (!success) throw new Error('配置写入失败');
    emit('save', rawData);
    if (notify) notify("样品参数保存成功", "success");
    emit('close');
  } catch (err) {
    if (notify) notify("保存失败: " + err, "error");
  }
};

</script>

<style scoped>
/* 样式与 HopkinsonModal 保持高度一致 */
.sample-form-container {
  background: white;
  padding: 30px;
  border-radius: 12px;
  width: 500px;
  max-width: 90vw;
  box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
  flex-shrink: 0;
}

.sample-form-header {
  margin-bottom: 20px;
  border-bottom: 1px solid #f1f5f9;
  padding-bottom: 10px;
}

.sample-form-header h2 {
  margin: 0;
  color: #1e293b;
  font-size: 1.4rem;
}

.sample-form-group {
  margin-bottom: 15px;
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.sample-form-row {
  display: flex;
  gap: 20px;
}

.sample-form-row .sample-form-group {
  flex: 1;
}

label {
  font-weight: 600;
  color: #475569;
  font-size: 0.9rem;
}

input, select {
  padding: 10px;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  font-size: 1rem;
  min-width: 0;
  width: 100%;
  box-sizing: border-box;
}

.info-tag {
  background: #f1f5f9;
  padding: 12px;
  border-radius: 6px;
  font-size: 0.85rem;
  color: #475569;
  margin: 10px 0 20px 0;
}

.sample-button-group {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

.btn {
  padding: 10px 24px;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
  border: none;
}

.btn-save { background-color: #3b82f6; color: white; }
.btn-cancel { background-color: #f1f5f9; color: #475569; }
</style>
