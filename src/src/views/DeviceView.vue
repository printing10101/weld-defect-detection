<script setup lang="ts">
/**
 * 设备标定向导（§12.4）：注册设备 → 录入标定（实测像素标定 vs 标定件参考值）→
 * 系统计算相对偏差并判定跨设备一致性（≤5% → 达标 ok，超差 → over）。
 * 数据全部来自真实后端 /devices 系列接口。
 */
import { onMounted, ref } from "vue";
import { addCalibration, getDevice, listDevices, registerDevice } from "../services/api";
import type { CalibrationOut, DeviceDetailOut, DeviceOut } from "../types/api";

const devices = ref<DeviceOut[]>([]);
const selectedId = ref<string | null>(null);
const detail = ref<DeviceDetailOut | null>(null);
const info = ref<string | null>(null);
const error = ref<string | null>(null);

/* 注册表单 */
const regName = ref("");
const regModel = ref("");
const regSerial = ref("");
const regNotes = ref("");

/* 标定表单 */
const calCalibrator = ref("");
const calPixelSpacing = ref("");
const calRefSpacing = ref("");
const calDensity = ref("");
const calNotes = ref("");

async function refreshList(): Promise<void> {
  try {
    devices.value = await listDevices();
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  }
}

async function selectDevice(id: string): Promise<void> {
  selectedId.value = id;
  error.value = null;
  info.value = null;
  try {
    detail.value = await getDevice(id);
  } catch (e) {
    detail.value = null;
    error.value = e instanceof Error ? e.message : String(e);
  }
}

async function doRegister(): Promise<void> {
  error.value = null;
  info.value = null;
  if (!regName.value.trim()) {
    error.value = "设备名称必填。";
    return;
  }
  try {
    const dev = await registerDevice({
      name: regName.value.trim(),
      model: regModel.value.trim() || null,
      serial_no: regSerial.value.trim() || null,
      notes: regNotes.value.trim() || null,
    });
    regName.value = "";
    regModel.value = "";
    regSerial.value = "";
    regNotes.value = "";
    await refreshList();
    await selectDevice(dev.device_id);
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  }
}

async function doCalibrate(): Promise<void> {
  error.value = null;
  info.value = null;
  if (!selectedId.value) return;
  if (!calCalibrator.value.trim()) {
    error.value = "标定员必填。";
    return;
  }
  const spacing = Number(calPixelSpacing.value);
  if (!Number.isFinite(spacing) || spacing <= 0) {
    error.value = "实测像素标定必须为正数。";
    return;
  }
  const refSpacing = calRefSpacing.value.trim() ? Number(calRefSpacing.value) : null;
  if (refSpacing !== null && (!Number.isFinite(refSpacing) || refSpacing <= 0)) {
    error.value = "参考像素标定必须为正数。";
    return;
  }
  const density = calDensity.value.trim() ? Number(calDensity.value) : null;
  try {
    await addCalibration(selectedId.value, {
      calibrator: calCalibrator.value.trim(),
      pixel_spacing_mm: spacing,
      ref_pixel_spacing_mm: refSpacing,
      density_ref: density,
      notes: calNotes.value.trim() || null,
    });
    calCalibrator.value = "";
    calPixelSpacing.value = "";
    calRefSpacing.value = "";
    calDensity.value = "";
    calNotes.value = "";
    await selectDevice(selectedId.value);
    await refreshList();
    info.value = "标定已记录。";
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  }
}

const lastCal = (): CalibrationOut | null => detail.value?.last_calibration ?? null;

onMounted(() => {
  void refreshList();
});
</script>

<template>
  <div>
    <h1
      class="title-zine"
      data-t="设备标定"
    >
      设备标定
    </h1>
    <div class="lede">
      DEVICE · 标定档案 · 跨设备一致率 ≤ 5%
    </div>

    <div class="guide">
      <div class="g">
        <div class="n">
          一 · 注册设备
        </div>
        <div class="t">
          登记检测设备（名称/型号/序列号），作为标定档案主体。
        </div>
      </div>
      <div class="g">
        <div class="n">
          二 · 标定录入
        </div>
        <div class="t">
          实测像素标定与标定件参考值比对，系统计算相对偏差：≤5% 达标，超差标记「over」。
        </div>
      </div>
      <div class="g">
        <div class="n">
          三 · 一致性档案
        </div>
        <div class="t">
          每次标定留档（操作员/时间/偏差），跨设备筛查前可核对设备状态。
        </div>
      </div>
    </div>

    <div
      v-if="error"
      class="err show"
    >
      ⚠ {{ error }}
    </div>
    <div
      v-if="info"
      class="ok show"
    >
      {{ info }}
    </div>

    <div class="row">
      <!-- 左：设备列表 + 注册 -->
      <div class="grow">
        <div class="section-h">
          设备
        </div>
        <div class="dev-list">
          <button
            v-for="d in devices"
            :key="d.device_id"
            type="button"
            class="dev-row"
            :class="{ cur: d.device_id === selectedId }"
            @click="selectDevice(d.device_id)"
          >
            <span class="d-name">{{ d.name }}</span>
            <span class="d-model">{{ d.model || d.serial_no || "—" }}</span>
            <span
              v-if="d.last_calibration"
              class="cal-badge"
              :class="d.last_calibration.status === 'over' ? 'over' : 'ok'"
            >
              {{ d.last_calibration.status === "over" ? "超差" : "达标" }}
            </span>
            <span
              v-else
              class="cal-badge none"
            >未标定</span>
          </button>
          <div
            v-if="devices.length === 0"
            class="hint"
          >
            尚未注册设备。
          </div>
        </div>

        <div
          class="section-h"
          style="margin-top: 18px"
        >
          注册设备
        </div>
        <div class="field">
          <label for="dn">设备名称 <span class="req">*</span></label>
          <input
            id="dn"
            v-model="regName"
            placeholder="如 CR-01"
          >
        </div>
        <div class="field">
          <label for="dm">型号</label>
          <input
            id="dm"
            v-model="regModel"
            placeholder="如 X-Ray 3000"
          >
        </div>
        <div class="field">
          <label for="ds">序列号</label>
          <input
            id="ds"
            v-model="regSerial"
            placeholder="如 SN-001"
          >
        </div>
        <div class="field">
          <label for="dnotes">备注</label>
          <input
            id="dnotes"
            v-model="regNotes"
          >
        </div>
        <button
          class="btn"
          type="button"
          @click="doRegister"
        >
          注册设备 →
        </button>
      </div>

      <!-- 右：选中设备详情 + 标定向导 -->
      <div class="grow">
        <template v-if="detail">
          <div class="section-h">
            {{ detail.name }}
            <span class="sub">{{ detail.model || "" }} {{ detail.serial_no || "" }}</span>
          </div>
          <div
            class="cal-status"
            :class="detail.calibration_count ? 'has' : ''"
          >
            标定 {{ detail.calibration_count }} 次
            <template v-if="lastCal()">
              · 最近 {{ lastCal()!.calibrated_at }} 由 {{ lastCal()!.calibrator }}
              <span :class="lastCal()!.status === 'over' ? 'over' : 'ok'">
                （{{ lastCal()!.status === "over" ? "超差 over" : "达标 ok" }}，
                偏差 {{ lastCal()!.deviation_pct ?? "—" }}%）
              </span>
            </template>
          </div>

          <div
            class="section-h"
            style="margin-top: 16px"
          >
            录入标定
          </div>
          <div class="field">
            <label for="cal1">标定员 <span class="req">*</span></label>
            <input
              id="cal1"
              v-model="calCalibrator"
              placeholder="姓名/工号"
            >
          </div>
          <div class="field">
            <label for="cal2">实测像素标定（mm/px）<span class="req">*</span></label>
            <input
              id="cal2"
              v-model="calPixelSpacing"
              placeholder="如 0.1000"
            >
            <div class="why">
              该设备的实测标定值；检测提交时作为像素标定默认参考。
            </div>
          </div>
          <div class="field">
            <label for="cal3">标定件参考值（mm/px）</label>
            <input
              id="cal3"
              v-model="calRefSpacing"
              placeholder="如 0.1000"
            >
            <div class="why">
              跨设备一致性基准；填后自动计算相对偏差并判定 ≤5%。
            </div>
          </div>
          <div class="field">
            <label for="cal4">黑度校验值（可选）</label>
            <input
              id="cal4"
              v-model="calDensity"
              placeholder="如 2.4"
            >
          </div>
          <div class="field">
            <label for="cal5">备注</label>
            <input
              id="cal5"
              v-model="calNotes"
            >
          </div>
          <button
            class="btn"
            type="button"
            @click="doCalibrate"
          >
            记录标定 →
          </button>

          <div
            class="section-h"
            style="margin-top: 18px"
          >
            标定档案
          </div>
          <div class="hist-list">
            <div
              v-for="c in detail.calibrations"
              :key="c.calibration_id"
              class="cal-row"
            >
              <span class="c-time">{{ c.calibrated_at }}</span>
              <span class="c-who">{{ c.calibrator }}</span>
              <span class="c-val">{{ c.pixel_spacing_mm }} mm/px</span>
              <span
                v-if="c.deviation_pct !== null"
                class="c-dev"
                :class="c.status"
              >
                偏差 {{ c.deviation_pct }}%
              </span>
              <span
                class="c-badge"
                :class="c.status"
              >
                {{ c.status === "over" ? "超差" : "达标" }}
              </span>
            </div>
            <div
              v-if="detail.calibrations.length === 0"
              class="hint"
            >
              尚无标定记录。
            </div>
          </div>
        </template>
        <div
          v-else
          class="hint"
          style="margin-top: 20px"
        >
          从左侧选择设备查看档案，或先注册一台设备。
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.ok {
  color: #1e7a3d;
  font-size: 13px;
  margin: 8px 0;
}
.sub {
  font-size: 12px;
  color: #6a7b99;
  margin-left: 8px;
}
.dev-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-height: 260px;
  overflow-y: auto;
}
.dev-row {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  text-align: left;
  padding: 8px 10px;
  border: 1px solid rgba(120, 140, 180, 0.25);
  border-radius: 8px;
  background: transparent;
  color: #22355c;
  font-size: 13px;
  cursor: pointer;
}
.dev-row:hover,
.dev-row.cur {
  border-color: #2f6bff;
  background: rgba(47, 107, 255, 0.06);
}
.d-name {
  font-weight: 600;
}
.d-model {
  color: #6a7b99;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.cal-badge {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
}
.cal-badge.ok {
  background: rgba(42, 143, 74, 0.15);
  color: #1e7a3d;
}
.cal-badge.over {
  background: rgba(204, 51, 51, 0.13);
  color: #b03030;
}
.cal-badge.none {
  background: rgba(120, 140, 180, 0.15);
  color: #6a7b99;
}
.cal-status {
  font-size: 13px;
  color: #44577a;
  margin: 8px 0;
}
.cal-status .over {
  color: #b03030;
}
.cal-status .ok {
  color: #1e7a3d;
}
.hist-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.cal-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 7px 10px;
  border: 1px solid rgba(120, 140, 180, 0.25);
  border-radius: 8px;
  font-size: 13px;
  color: #22355c;
}
.c-time {
  color: #6a7b99;
  font-size: 12px;
}
.c-who {
  flex: 1;
}
.c-dev.over {
  color: #b03030;
}
.c-dev.ok {
  color: #1e7a3d;
}
.c-badge {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
}
.c-badge.ok {
  background: rgba(42, 143, 74, 0.15);
  color: #1e7a3d;
}
.c-badge.over {
  background: rgba(204, 51, 51, 0.13);
  color: #b03030;
}
</style>
