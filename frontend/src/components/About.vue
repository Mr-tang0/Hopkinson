<template>
  <Transition name="modal-fade">
    <div v-if="modelValue" class="about-overlay" @click.self="$emit('update:modelValue', false)">
      <div class="about-card">
        <button class="close-x" @click="$emit('update:modelValue', false)">&times;</button>

        <div class="about-content">
          <div class="about-logo-icon">SHPB</div>
          <h1 class="about-app-name">{{ appInfo.name }}</h1>
          
          <div class="about-details">
            <div class="detail-item">
              <span class="label">作者</span>
              <span class="value">{{ appInfo.author }}</span>
            </div>
            <div class="detail-item">
              <span class="label">版本</span>
              <span class="value">{{ appInfo.version }}</span>
            </div>
            <div class="detail-item">
              <span class="label">最后更新时间</span>
              <span class="value">{{ appInfo.lastEdit }}</span>
            </div>
          </div>

          <button class="baidu-link" @click="visitWebsite">
            <span class="link-icon">🌐</span>
            在线帮助
          </button>
        </div>
      </div>
    </div>
  </Transition>
</template>

<script setup>
import update from '../../../update.json'

// 使用 defineProps 和 defineEmits 实现 v-model 绑定
defineProps(['modelValue']);
defineEmits(['update:modelValue']);


const appInfo = {
  name: update.name,
  author: update.author,
  version: update.version,
  lastEdit: update.lastEdit,
  url: update.html
}


const visitWebsite = () => {
  window.open(appInfo.url, '_blank');
};
</script>

<style scoped>
.about-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(8px);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 9999;
}

.about-card {
  background: #2c3e50;
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  width: 360px;
  padding: 40px 30px;
  position: relative;
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
  text-align: center;
}

.close-x {
  position: absolute;
  top: 15px;
  right: 20px;
  background: none;
  border: none;
  color: #95a5a6;
  font-size: 28px;
  cursor: pointer;
}

.about-logo-icon {
  width: 80px;
  height: 80px;
  background: linear-gradient(135deg, #f39c12, #e67e22);
  margin: 0 auto 20px;
  border-radius: 20px;
  display: flex;
  justify-content: center;
  align-items: center;
  font-weight: bold;
  font-size: 26px;
  color: white;
  box-shadow: 0 8px 20px rgba(243, 156, 18, 0.4);
}

.about-app-name {
  font-size: 22px;
  color: #ecf0f1;
  margin-bottom: 30px;
  letter-spacing: 1px;
}

.about-details {
  background: rgba(0, 0, 0, 0.2);
  border-radius: 12px;
  padding: 15px;
  margin-bottom: 30px;
}

.detail-item {
  display: flex;
  justify-content: space-between;
  margin: 8px 0;
  font-size: 14px;
}

.label { color: #95a5a6; }
.value { color: #ecf0f1; font-weight: 500; }

.baidu-link {
  width: 100%;
  padding: 12px;
  background: #3498db;
  border: none;
  border-radius: 8px;
  color: white;
  font-size: 15px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  transition: background 0.2s;
}

.baidu-link:hover { background: #2980b9; }

.modal-fade-enter-active, .modal-fade-leave-active { transition: all 0.3s ease; }
.modal-fade-enter-from, .modal-fade-leave-to { opacity: 0; transform: scale(0.9); }
</style>