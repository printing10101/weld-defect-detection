/**
 * LlmView 组件测试：本地大模型面板的显示口径。
 *
 * 重点覆盖两条**有意的设计决策**，防止后续被"优化"掉：
 * 1. 默认列出全部模型（含"跑不动"的），不折叠——筛选会让用户误以为模型不存在；
 * 2. 引擎 `auth_required` 必须与 `ready` 明确区分——这是"进程活着但一个请求都
 *    发不出去"的假绿场景，其中 /health 免鉴权而 /v1/* 返回 401。
 */
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";

import LlmView from "./LlmView.vue";
import { useAuthStore } from "../stores/auth";
import type { LlmEngineOut, LlmModelOut, LlmModelsOut, LlmStatusOut } from "../types/api";

const listLlmModels = vi.fn();
const getLlmStatus = vi.fn();
const selectLlmModel = vi.fn();
const clearLlmSelection = vi.fn();
const addLlmDir = vi.fn();
const removeLlmDir = vi.fn();
const startLlmScan = vi.fn();
const cancelLlmScan = vi.fn();

vi.mock("../services/api", () => ({
  listLlmModels: (...a: unknown[]) => listLlmModels(...a),
  getLlmStatus: (...a: unknown[]) => getLlmStatus(...a),
  selectLlmModel: (...a: unknown[]) => selectLlmModel(...a),
  clearLlmSelection: (...a: unknown[]) => clearLlmSelection(...a),
  addLlmDir: (...a: unknown[]) => addLlmDir(...a),
  removeLlmDir: (...a: unknown[]) => removeLlmDir(...a),
  startLlmScan: (...a: unknown[]) => startLlmScan(...a),
  cancelLlmScan: (...a: unknown[]) => cancelLlmScan(...a),
}));

function model(over: Partial<LlmModelOut> = {}): LlmModelOut {
  return {
    id: "gguf:aaaa",
    name: "Qwen3-4B-Q4_K_M",
    display_name: "Qwen3-4B-Q4_K_M",
    source: "local",
    path: "D:\\扫描检测软件\\models\\llm\\Qwen3-4B-Q4_K_M.gguf",
    endpoint: "",
    size_bytes: 2_500_000_000,
    architecture: "qwen3",
    quant: "Q4_K_M",
    max_ctx: 40960,
    is_embedding: false,
    available: true,
    error: null,
    active: false,
    primary: true,
    duplicate_of: null,
    duplicate_count: 0,
    vram: {
      weights_bytes: 2_500_000_000,
      kv_cache_bytes: null,
      overhead_bytes: 0,
      needed_bytes: 4_200_000_000,
      free_vram_bytes: 12_000_000_000,
      verdict: "full_gpu",
      advice: "可全量上卡。",
      notes: [],
    },
    notes: [],
    ...over,
  };
}

function engine(over: Partial<LlmEngineOut> = {}): LlmEngineOut {
  return {
    enabled: true,
    mode: "managed",
    state: "ready",
    endpoint: "http://127.0.0.1:18780",
    adopted: false,
    error: null,
    advice: null,
    ...over,
  };
}

function statusOf(eng: LlmEngineOut): LlmStatusOut {
  return {
    engine: eng,
    selection: { active_id: null, mode: null, endpoint: "", model_path: "" },
    scan: {
      state: "idle",
      roots: ["C:\\", "D:\\"],
      declared_dirs: [],
      progress: { dirs: 0, found: 0, current: "" },
      found: 0,
      elapsed_sec: null,
      error: null,
    },
    gpu: { name: "RTX 3080 Laptop GPU", total_bytes: 17_179_869_184, free_bytes: 12_000_000_000 },
    model_dirs: [],
    counts: { local: 1, service: 0, available: 1, total: 1 },
  };
}

function modelsOf(models: LlmModelOut[], infeasible = 0): LlmModelsOut {
  return {
    active_id: null,
    selection: { active_id: null, mode: null, endpoint: "", model_path: "" },
    gpu: null,
    scan: statusOf(engine()).scan,
    counts: {
      total: models.length,
      local: models.filter((m) => m.source === "local").length,
      service: models.filter((m) => m.source === "service").length,
      available: models.filter((m) => m.available).length,
      infeasible,
    },
    services: [],
    models,
  };
}

/** 一个"跑不动"的 30B：权重就超可用显存。 */
const HUGE = model({
  id: "gguf:huge",
  display_name: "Qwen3-30B-A3B-Thinking-2507",
  size_bytes: 17_700_000_000,
  max_ctx: 262144,
  vram: {
    weights_bytes: 17_700_000_000,
    kv_cache_bytes: null,
    overhead_bytes: 0,
    needed_bytes: 19_000_000_000,
    free_vram_bytes: 12_000_000_000,
    verdict: "infeasible",
    advice: "权重已超可用显存，无法全量上卡。",
    notes: [],
  },
});

/** 挂载面板：**同一个** pinia 既要预置角色、又要装进组件——用两个实例会导致
 *  组件读到空角色（写操作全被禁用），让"按钮禁用"类断言假过。 */
function mountView(role = "sysadmin") {
  const pinia = createPinia();
  setActivePinia(pinia);
  useAuthStore().role = role;
  return mount(LlmView, { global: { plugins: [pinia] } });
}

describe("LlmView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    selectLlmModel.mockResolvedValue({ ok: true });
    getLlmStatus.mockImplementation(async () => statusOf(engine()));
  });

  it("默认列出全部模型，含判定为跑不动的（不折叠）", async () => {
    listLlmModels.mockResolvedValue(modelsOf([model(), HUGE], 1));
    const w = mountView();
    await flushPromises();

    expect(w.text()).toContain("Qwen3-4B-Q4_K_M");
    expect(w.text()).toContain("Qwen3-30B-A3B-Thinking-2507");
    expect(w.text()).toContain("跑不动");
    // 并且明确告知"仍会列出、不做隐藏"
    expect(w.text()).toContain("不做隐藏");
  });

  it("勾选「只显示当前跑得动的」后过滤掉 infeasible", async () => {
    listLlmModels.mockResolvedValue(modelsOf([model(), HUGE], 1));
    const w = mountView();
    await flushPromises();

    await w.find('input[type="checkbox"]').setValue(true);
    await flushPromises();

    expect(w.text()).toContain("Qwen3-4B-Q4_K_M");
    expect(w.text()).not.toContain("Qwen3-30B-A3B-Thinking-2507");
  });

  it("auth_required 不得显示为就绪，并给出可执行指引", async () => {
    listLlmModels.mockResolvedValue(modelsOf([model()]));
    getLlmStatus.mockImplementation(async () =>
      statusOf(
        engine({
          state: "auth_required",
          error: "llama-server 已加载模型，但 /v1/models 返回 401（要求鉴权）",
          advice: "这是 llama.cpp 从环境变量 LLAMA_API_KEY 自动开启鉴权所致。",
        }),
      ),
    );
    const w = mountView();
    await flushPromises();

    expect(w.text()).toContain("需要 API Key");
    expect(w.text()).not.toContain("就绪");
    expect(w.text()).toContain("LLAMA_API_KEY");
  });

  it("显式标注「上限 ctx」，不与服务实际 -c 混淆", async () => {
    listLlmModels.mockResolvedValue(modelsOf([model({ max_ctx: 262144 })]));
    const w = mountView();
    await flushPromises();

    // 262144 / 1024 = 256 → 显示为 256K，且表头写明是"上限"
    expect(w.text()).toContain("上限 ctx");
    expect(w.text()).toContain("256K");
  });

  it("非系统管理员：选中按钮禁用并说明原因", async () => {
    listLlmModels.mockResolvedValue(modelsOf([model()]));
    const w = mountView("auditor");
    await flushPromises();

    const btn = w.findAll("button").find((b) => b.text() === "选中");
    expect(btn).toBeTruthy();
    expect(btn!.attributes("disabled")).toBeDefined();
    expect(w.text()).toContain("需系统管理员权限");
  });

  it("系统管理员：选中按钮可用", async () => {
    listLlmModels.mockResolvedValue(modelsOf([model()]));
    const w = mountView("sysadmin");
    await flushPromises();

    const btn = w.findAll("button").find((b) => b.text() === "选中");
    expect(btn).toBeTruthy();
    expect(btn!.attributes("disabled")).toBeUndefined();
  });

  it("选中模型调用后端接口", async () => {
    listLlmModels.mockResolvedValue(modelsOf([model()]));
    const w = mountView();
    await flushPromises();

    const btn = w.findAll("button").find((b) => b.text() === "选中")!;
    await btn.trigger("click");
    await flushPromises();

    expect(selectLlmModel).toHaveBeenCalledWith("gguf:aaaa");
  });
});
