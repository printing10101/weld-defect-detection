"""底片数据脱敏（DB50/T 1807-2025 §8.3.2：测试集应采取数据脱敏处理措施）。

覆盖两类隐私载体：
- DICOM/DICONDE：删除患者/机构隐私标签（与 infra/diconde.PHI_TAGS 同一清单），
  像素数据不动；
- JPEG：剥离 EXIF（APP1）、Photoshop 元数据（APP13）、注释段（COM）——
  手机拍摄的现场照片常带 GPS 与设备序列号。

用法：
  审计（只报告不改动）：
    python -m backend.training.anonymize_images --src data/std_test --audit-only
  脱敏到新目录（推荐）：
    python -m backend.training.anonymize_images --src data/std_test --dst data/std_test_anon
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

from pydicom import dcmread
from pydicom.datadict import tag_for_keyword

from backend.infra.diconde import PHI_TAGS

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".dicom")
_DICOM_EXTS = (".dcm", ".dicom")

# JPEG 段标记：可剥离的元数据段（含隐私/设备信息的常见载体）
_JPEG_STRIP_MARKERS = {0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xEC, 0xED, 0xFE}


def strip_jpeg_metadata(data: bytes) -> tuple[bytes, list[str]]:
    """剥离 JPEG 的 APP1/APP2/.../APP13/COM 元数据段，返回 (新字节, 段名清单)。

    只重写 SOI 与各段之间的元数据，压缩数据（SOS 之后）原样保留。
    """
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("非 JPEG 字节流")
    out = bytearray(b"\xff\xd8")
    stripped: list[str] = []
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        if marker == 0xD8:
            i += 2
            continue
        if marker == 0xD9:  # EOI
            out += data[i:]
            return bytes(out), stripped
        if marker == 0xDA:  # SOS：压缩数据开始，其后原样保留
            out += data[i:]
            return bytes(out), stripped
        seg_len = int.from_bytes(data[i + 2 : i + 4], "big")
        if marker in _JPEG_STRIP_MARKERS:
            names = {0xE1: "APP1(Exif/XMP)", 0xED: "APP13(Photoshop)", 0xFE: "COM"}
            stripped.append(names.get(marker, f"APP{marker - 0xE0}"))
            i += 2 + seg_len
            continue
        out += data[i : i + 2 + seg_len]
        i += 2 + seg_len
    # 走到这说明未找到 SOS/EOI（异常文件）：保守原样返回
    return data, stripped


def strip_dicom_phi(data: bytes) -> tuple[bytes, list[str]]:
    """删除 DICOM 患者隐私标签，返回 (新字节, 删除的字段清单)。"""
    ds = dcmread(io.BytesIO(data))
    removed: list[str] = []
    for keyword in PHI_TAGS:
        tag = tag_for_keyword(keyword)
        if tag is not None and tag in ds:
            del ds[tag]
            removed.append(keyword)
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=False)
    return buf.getvalue(), removed


def audit_file(path: Path) -> list[str]:
    """审计单文件，返回隐私残留清单（空 = 干净）。"""
    suffix = path.suffix.lower()
    data = path.read_bytes()
    if suffix in _DICOM_EXTS:
        try:
            from backend.infra.diconde import audit_dicom_phi

            return audit_dicom_phi(path)
        except ValueError:
            return []
    if suffix in (".jpg", ".jpeg"):
        findings = []
        if b"Exif\x00\x00" in data[: 64 * 1024]:
            findings.append("EXIF")
        if b"http://ns.adobe.com/xap" in data[: 64 * 1024]:
            findings.append("XMP")
        return findings
    return []


def _iter_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in _IMG_EXTS)


def audit_directory(root: str | Path) -> list[dict[str, str]]:
    """审计目录下全部影像，返回 [{file, findings}]（只列有隐私残留的）。"""
    out = []
    for p in _iter_images(Path(root)):
        findings = audit_file(p)
        if findings:
            out.append({"file": str(p), "findings": ",".join(findings)})
    return out


def anonymize_tree(src: str | Path, dst: str | Path) -> list[dict[str, str]]:
    """把 src 影像树脱敏复制到 dst，返回 [{file, removed}]（改动清单）。"""
    src, dst = Path(src), Path(dst)
    report: list[dict[str, str]] = []
    for p in _iter_images(src):
        rel = p.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        data = p.read_bytes()
        suffix = p.suffix.lower()
        removed: list[str] = []
        if suffix in _DICOM_EXTS:
            data, removed = strip_dicom_phi(data)
        elif suffix in (".jpg", ".jpeg"):
            data, removed = strip_jpeg_metadata(data)
        target.write_bytes(data)
        report.append({"file": str(rel), "removed": ",".join(removed)})
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="底片数据脱敏（§8.3.2）")
    ap.add_argument("--src", required=True, help="源目录（递归）")
    ap.add_argument("--dst", help="脱敏输出目录（与 --audit-only 二选一）")
    ap.add_argument("--audit-only", action="store_true", help="只审计不脱敏")
    args = ap.parse_args(argv)

    if args.audit_only:
        findings = audit_directory(args.src)
        for row in findings:
            print(f"[残留] {row['file']}: {row['findings']}")
        print(f"审计完成：{len(findings)} 个文件存在隐私残留")
        return 1 if findings else 0

    if not args.dst:
        ap.error("--dst 与 --audit-only 必须二选一")
    report = anonymize_tree(args.src, args.dst)
    changed = [r for r in report if r["removed"]]
    for row in changed:
        print(f"[脱敏] {row['file']}: 移除 {row['removed']}")
    print(f"脱敏完成：复制 {len(report)} 个文件，其中 {len(changed)} 个含隐私元数据已清理")
    leftover = audit_directory(args.dst)
    if leftover:
        print(f"⚠ 脱敏后仍有残留（请检查）：{leftover}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
