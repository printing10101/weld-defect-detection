// BatchProgress 组件测试：纯展示组件的渲染语义与操作事件。
// 数据契约：props.status 为后端 BatchStatusOut 镜像（types/api.ts）。
import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import BatchProgress from "./BatchProgress.vue";
import type { BatchStatusOut } from "../types/api";

function makeStatus(overrides: Partial<BatchStatusOut> = {}): BatchStatusOut {
  return {
    batch_id: "b1",
    status: "running",
    total: 4,
    done: 2,
    failed: 0,
    cancelled: 0,
    estimated_sec: 12,
    progress: 0.5,
    tasks: [
      {
        task_id: "t1",
        image_name: "a.png",
        status: "done",
        error: null,
        image_id: "i1",
        report_id: "r1",
        joint_level: "II",
        need_review: false,
      },
      {
        task_id: "t2",
        image_name: "b.png",
        status: "running",
        error: null,
        image_id: null,
        report_id: null,
        joint_level: null,
        need_review: null,
      },
    ],
    ...overrides,
  };
}

describe("BatchProgress", () => {
  it("渲染进度百分比与计数", () => {
    const w = mount(BatchProgress, { props: { status: makeStatus() } });
    expect(w.text()).toContain("50%");
    expect(w.text()).toContain("完成 2 / 4");
  });

  it("渲染逐任务状态标签与级别", () => {
    const w = mount(BatchProgress, { props: { status: makeStatus() } });
    expect(w.text()).toContain("a.png");
    expect(w.text()).toContain("已完成");
    expect(w.text()).toContain("评定中");
    expect(w.text()).toContain("级别 II");
  });

  it("渲染失败任务的错误提示", () => {
    const status = makeStatus({
      failed: 1,
      tasks: [
        {
          task_id: "t3",
          image_name: "c.png",
          status: "failed",
          error: "底片质量不合格",
          image_id: null,
          report_id: null,
          joint_level: null,
          need_review: null,
        },
      ],
    });
    const w = mount(BatchProgress, { props: { status } });
    expect(w.text()).toContain("失败 1");
    expect(w.text()).toContain("⚠ 底片质量不合格");
  });

  it("running 时显示取消按钮并发出 cancel", async () => {
    const w = mount(BatchProgress, { props: { status: makeStatus() } });
    const cancel = w.findAll("button").find((b) => b.text().includes("取消"));
    expect(cancel).toBeTruthy();
    await cancel!.trigger("click");
    expect(w.emitted("cancel")).toHaveLength(1);
  });

  it("finished 且有失败时显示重试按钮并发出 retry", async () => {
    const status = makeStatus({
      status: "finished",
      failed: 1,
      done: 3,
      progress: 1,
    });
    const w = mount(BatchProgress, { props: { status } });
    const retry = w.findAll("button").find((b) => b.text().includes("重试"));
    expect(retry).toBeTruthy();
    await retry!.trigger("click");
    expect(w.emitted("retry")).toHaveLength(1);
  });

  it("非 running 状态不显示取消按钮", () => {
    const w = mount(BatchProgress, {
      props: { status: makeStatus({ status: "finished", progress: 1 }) },
    });
    const cancel = w.findAll("button").find((b) => b.text().includes("取消批次"));
    expect(cancel).toBeFalsy();
  });

  it("归档入口始终可用并发出 archive", async () => {
    const w = mount(BatchProgress, { props: { status: makeStatus() } });
    const archive = w.findAll("button").find((b) => b.text().includes("检测档案"));
    expect(archive).toBeTruthy();
    await archive!.trigger("click");
    expect(w.emitted("archive")).toHaveLength(1);
  });

  // 批量标注约定：有缺陷的底片红标「缺陷 N 处」，无缺陷底片不标注
  it("检出缺陷的任务红标缺陷数，无缺陷任务不标注", () => {
    const status = makeStatus({
      tasks: [
        {
          task_id: "t1",
          image_name: "a.png",
          status: "done",
          error: null,
          image_id: "i1",
          report_id: "r1",
          joint_level: "III",
          need_review: false,
          defect_count: 3,
        },
        {
          task_id: "t2",
          image_name: "b.png",
          status: "done",
          error: null,
          image_id: "i2",
          report_id: "r2",
          joint_level: "I",
          need_review: false,
          defect_count: 0,
        },
      ],
    });
    const w = mount(BatchProgress, { props: { status } });
    expect(w.text()).toContain("缺陷 3 处");
    expect(w.text()).not.toContain("缺陷 0 处");
  });
});
