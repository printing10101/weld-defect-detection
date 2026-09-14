<script setup lang="ts">
/** 受控导出面板（C-14）：申请 → 审批状态自动轮询 → 领一次性令牌 → 下载。
 *  展示逻辑由 useControlledPdf 组合式函数提供，本组件纯呈现（供报告页/批量页共用）。 */
import type { ControlledPdf } from "../composables/useControlledPdf";

defineProps<{ ctrl: ControlledPdf }>();
</script>

<template>
  <div
    v-if="ctrl.gateOpen.value"
    class="modal-mask"
    @click.self="ctrl.closeGate"
  >
    <div
      class="modal"
      role="dialog"
      aria-modal="true"
      aria-label="受控导出"
    >
      <h3 class="m-title">
        受控导出（C-14）
      </h3>
      <p class="gate-hint">
        报告 PDF 受导出审批管控：需提交申请并由安全保密管理员批准后，
        领取一次性导出令牌完成下载。全程留痕审计。
      </p>
      <label class="gate-lab">申请理由（选填）</label>
      <input
        :model-value="ctrl.gateReason.value"
        class="gate-in"
        type="text"
        maxlength="200"
        placeholder="如：归档移交 / 质量复盘"
        @update:model-value="ctrl.setReason"
      >
      <p
        v-if="ctrl.approval.value === 'pending'"
        class="gate-status"
        role="status"
      >
        <span class="spin" /> 审批状态：待安全保密管理员审批（本面板每 5 秒自动刷新，
        批准后会提示，无需关闭重进）
      </p>
      <p
        v-else-if="ctrl.approval.value === 'approved'"
        class="gate-status ok"
        role="status"
      >
        ✓ 审批已通过，点击「领取令牌并下载」完成导出。
      </p>
      <div
        v-if="ctrl.gateMsg.value"
        class="ok-msg"
      >
        ✓ {{ ctrl.gateMsg.value }}
      </div>
      <div
        v-if="ctrl.gateErr.value"
        class="err show"
      >
        ⚠ {{ ctrl.gateErr.value }}
      </div>
      <div class="gate-acts">
        <button
          class="g-btn"
          type="button"
          :disabled="ctrl.gateBusy.value"
          @click="ctrl.applyExportRequest"
        >
          {{ ctrl.requestId.value ? "重新申请" : "申请导出" }}
        </button>
        <button
          v-if="ctrl.requestId.value"
          class="g-btn primary"
          type="button"
          :disabled="ctrl.gateBusy.value"
          @click="ctrl.fetchTokenAndDownload"
        >
          领取令牌并下载
        </button>
        <button
          class="g-btn ghost"
          type="button"
          @click="ctrl.closeGate"
        >
          关闭
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.modal-mask {
  position: fixed;
  inset: 0;
  background: rgba(8, 12, 22, 0.66);
  display: grid;
  place-items: center;
  z-index: 60;
  padding: 24px;
}
.modal {
  width: min(560px, 100%);
  max-height: 86vh;
  overflow: auto;
  background: #161b2c;
  border: 1px solid rgba(140, 160, 200, 0.22);
  border-radius: 12px;
  padding: 20px 22px;
  box-shadow: 0 18px 48px rgba(0, 0, 0, 0.45);
}
.m-title {
  font-size: 16px;
  margin: 0 0 10px;
  letter-spacing: 0.04em;
  color: #e8edf7;
}
.gate-hint {
  font-size: 13px;
  line-height: 1.7;
  color: #9fb0d0;
  margin: 0 0 12px;
}
.gate-lab {
  display: block;
  font-size: 12px;
  color: #9fb0d0;
  margin-bottom: 4px;
}
.gate-in {
  width: 100%;
  box-sizing: border-box;
  padding: 7px 10px;
  font-size: 13px;
  color: #e8edf7;
  background: #0f1424;
  border: 1px solid rgba(140, 160, 200, 0.3);
  border-radius: 6px;
}
.gate-status {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: #e0a13c;
  margin: 10px 0 0;
}
.gate-status.ok {
  color: #1e9e57;
}
.spin {
  width: 10px;
  height: 10px;
  border: 2px solid rgba(224, 161, 60, 0.4);
  border-top-color: #e0a13c;
  border-radius: 50%;
  animation: spin 0.9s linear infinite;
  flex: none;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
.ok-msg {
  margin-top: 12px;
  padding: 8px 12px;
  border-radius: 8px;
  background: rgba(42, 143, 74, 0.12);
  color: #1e9e57;
  font-size: 13px;
}
.err.show {
  margin-top: 12px;
  padding: 8px 12px;
  border-radius: 8px;
  background: rgba(176, 48, 48, 0.14);
  color: #e08a8a;
  font-size: 13px;
}
.gate-acts {
  display: flex;
  gap: 8px;
  margin-top: 14px;
  flex-wrap: wrap;
}
.g-btn {
  padding: 7px 16px;
  font-size: 13px;
  color: #e8edf7;
  background: rgba(140, 160, 200, 0.12);
  border: 1px solid rgba(140, 160, 200, 0.3);
  border-radius: 6px;
  cursor: pointer;
}
.g-btn.primary {
  background: var(--accent, #2a8f4a);
  border-color: transparent;
}
.g-btn.ghost {
  background: transparent;
}
.g-btn:disabled {
  opacity: 0.5;
  cursor: default;
}
</style>
