<script setup lang="ts">
/**
 * 破坏性操作二次确认对话框（GB/T 25000.51 易用性-用户差错防御性）。
 * 供删除复核缺陷、取消批次等不可逆动作在执行前确认；
 * 纯前端组件，不依赖原生 dialog（桌面壳 WebView 无 window.confirm 保障）。
 * 焦点管理：打开即聚焦「取消」（危险操作默认安全侧），ESC 关闭，Tab 在面板内循环，
 * 避免 Enter/Tab 泄漏到触发按钮再次启动危险路径。
 */
import { nextTick, onBeforeUnmount, ref, watch } from "vue";

const props = defineProps<{
  open: boolean;
  title: string;
  message: string;
  confirmText?: string;
  /** true 时确认按钮呈危险色（红） */
  danger?: boolean;
}>();

const emit = defineEmits<{ confirm: []; cancel: [] }>();

const panel = ref<HTMLDivElement | null>(null);
const cancelBtn = ref<HTMLButtonElement | null>(null);

function onKeydown(e: KeyboardEvent): void {
  if (!props.open) return;
  if (e.key === "Escape") {
    e.stopPropagation();
    emit("cancel");
    return;
  }
  if (e.key === "Tab" && panel.value) {
    // 简单焦点陷阱：面板内循环
    const focusables = panel.value.querySelectorAll<HTMLElement>("button");
    if (focusables.length === 0) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }
}

watch(
  () => props.open,
  (open) => {
    if (open) {
      window.addEventListener("keydown", onKeydown, true);
      void nextTick(() => cancelBtn.value?.focus());
    } else {
      window.removeEventListener("keydown", onKeydown, true);
    }
  },
);

onBeforeUnmount(() => window.removeEventListener("keydown", onKeydown, true));
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="cd-mask"
      @click.self="emit('cancel')"
    >
      <div
        ref="panel"
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
            ref="cancelBtn"
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
