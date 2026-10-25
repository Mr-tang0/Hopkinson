<template>
  <div class="message-manager">
    <transition-group name="msg-list">
      <div 
        v-for="item in msgList" 
        :key="item.id" 
        :class="['msg-box', item.type]"
      >
        <span class="msg-icon">{{ getIcon(item.type) }}</span>
        <span class="msg-text">{{ item.content }}</span>
      </div>
    </transition-group>
  </div>
</template>

<script setup>
import { ref } from 'vue'

// 响应式消息队列
const msgList = ref([])
let count = 0

// 添加消息的方法
const addMessage = (content, type = 'info', duration = 3000) => {
  const id = count++
  const newItem = { id, content, type }
  
  // 从顶部插入
  msgList.value.unshift(newItem)

  // 自动移除
  setTimeout(() => {
    msgList.value = msgList.value.filter(m => m.id !== id)
  }, duration)
}

// 图标映射
const getIcon = (type) => {
  const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' }
  return icons[type] || icons.info
}

// 【关键】必须暴露给父组件调用
defineExpose({ addMessage })
</script>

<style scoped>
.message-manager {
  position: fixed;
  top: 50px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 9999;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  pointer-events: none;
}

.msg-box {
  pointer-events: auto;
  min-width: 280px;
  padding: 10px 16px;
  background: #2c3e50;
  color: white;
  border-radius: 4px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  display: flex;
  align-items: center;
  border-left: 4px solid #95a5a6;
}

/* 不同类型颜色 */
.msg-box.success { border-left-color: #27ae60; }
.msg-box.error { border-left-color: #e74c3c; }
.msg-box.warning { border-left-color: #f1c40f; }

.msg-icon { margin-right: 10px; }
.msg-text { font-size: 14px; }

/* 动画效果 */
.msg-list-enter-active, .msg-list-leave-active {
  transition: all 0.3s ease;
}
.msg-list-enter-from {
  opacity: 0;
  transform: translateY(-30px);
}
.msg-list-leave-to {
  opacity: 0;
  transform: scale(0.9);
}
</style>