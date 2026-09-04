<script setup lang="ts">
/**
 * 破坏性操作二次确认对话框（GB/T 25000.51 易用性-用户差错防御性）。
 * 供删除复核缺陷、取消批次等不可逆动作在执行前确认；
 * 纯前端组件，不依赖原生 dialog（Tauri WebView 无 window.confirm 保障）。
 */
defineProps<{
  open: boolean;
  title: string;
  message: string;
  confirmText?: string;
  /** true 时确认按钮呈危险色（红） */
  danger?: boolean;
}>();

const emit = defineEmits<{ confirm: []; cancel: [] }>();
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="cd-mask"
      @click.self="emit('cancel')"
    >
      <div
        class="cd-panel"
        role="alertdialog"
        aria-modal="true"
        :aria-label="title"
      >
        <p class="cd-title">
          {{ title }}
        </p>
        <p class="cd-msg">
          {{ message }}
        </p>
        <div class="cd-foot">
          <button
            type="button"
            class="cd-btn"
            @click="emit('cancel')"
          >
            取消
          </button>
          <button
            type="button"
            class="cd-btn"
            :class="{ danger }"
            @click="emit('confirm')"
          >
            {{ confirmText ?? "确认" }}
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.cd-mask {
  position: fixed;
  inset: 0;
  z-index: 1000;
  background: rgb(15 23 42 / 45%);
  display: flex;
  align-items: center;
  justify-content: center;
}

.cd-panel {
  width: min(420px, calc(100vw - 48px));
  background: #fff;
  border-radius: 10px;
  padding: 20px 22px 16px;
  box-shadow: 0 12px 40px rgb(15 23 42 / 25%);
}

.cd-title {
  margin: 0 0 8px;
  font-size: 15px;
  font-weight: 600;
  color: #0f172a;
}

.cd-msg {
  margin: 0 0 18px;
  font-size: 13px;
  line-height: 1.7;
  color: #475569;
}

.cd-foot {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}

.cd-btn {
  min-width: 72px;
  padding: 7px 14px;
  border-radius: 6px;
  border: 1px solid #cbd5e1;
  background: #fff;
  color: #0f172a;
  font-size: 13px;
  cursor: pointer;
}

.cd-btn:hover {
  background: #f1f5f9;
}

.cd-btn.danger {
  background: #dc2626;
  border-color: #dc2626;
  color: #fff;
}

.cd-btn.danger:hover {
  background: #b91c1c;
}
</style>
