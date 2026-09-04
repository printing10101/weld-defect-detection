"""密码应用自评估说明（密评材料）导出（C-24）。

内容四部分：
1. 算法清单——从 crypto provider 运行时内省（provider 可用性、哈希算法名、
   SM2 公钥是否在位），叠加信封格式常量（SDC2/SDC1）；
2. 密钥管理——分层结构（SCAN_CRYPTO_KEY 主密钥 / SM3 域分离 KDF 派生 /
   SCAN_SM2_PRIVATE_KEY 显式覆盖 / Pkcs11Provider 硬件托管边界）；
3. 密码调用链——静态加密 / 审计哈希链 / 报告签名 / 会话令牌 / 登录验签 /
   导出令牌 / 载体销毁证明各环节调用的算法与所在层；
4. 合规差距声明——软实现 vs 商密模块认证、未真机验证的硬件对接、
   遗留 AES 信封、明文本机传输等，诚实列出，供密评机构认定边界。

产物：Markdown + PDF（PDF/A-1b）落 data/compliance/。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.infra.compliance.doc_pdf import build_doc_pdf


def _now_str() -> str:
    return datetime.now(UTC).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def _provider_inventory() -> dict[str, Any]:
    """运行时内省 crypto provider（不可用时不造假，如实记录异常）。"""
    out: dict[str, Any] = {"available": False, "provider": None, "error": None}
    try:
        from backend.infra.crypto import get_provider

        p = get_provider()
        out.update(
            available=True,
            provider=type(p).__name__,
            hash_algo=p.hash_algo,
            has_public_key=bool(getattr(p, "public_key_hex", "")),
            public_key_len=len(getattr(p, "public_key_hex", "") or ""),
        )
    except Exception as exc:  # noqa: BLE001 - 密钥未配置等环境事实如实呈现
        out["error"] = str(exc)[:300]
    return out


def _algorithm_table(inv: dict[str, Any]) -> list[list[str]]:
    rows = [
        [
            "SM4-CTR + HMAC-SM3",
            "静态加密（认证加密，encrypt-then-MAC）",
            "SM4-128（数据密钥）/ HMAC-SM3-256",
            "信封 SDC2：magic||nonce(16B)||ciphertext||mac(32B)，mac 覆盖 aad",
        ],
        [
            "SM3",
            "哈希：审计链、密钥派生 KDF、HMAC、令牌指纹",
            "256 bit（64 hex）",
            "主链/安全链逐条 SM3 链式哈希；KDF 域分离标签（sd-kdf-sm4/mac/sm2）",
        ],
        [
            "SM2（SM3withSM2）",
            "签名：报告防篡改签名、登录挑战-响应验签",
            "256 bit 曲线 sm2p256v1，签名 128 hex r||s",
            "公钥 128 hex（x||y）随签名 sidecar 落档",
        ],
        [
            "AES-256-GCM（遗留，只读）",
            "历史信封 SDC1 解密兼容",
            "256 bit（与主密钥共用）",
            "仅存量数据可解，新写入一律 SDC2 国密信封",
        ],
    ]
    if inv.get("available"):
        rows.append(
            [
                "运行时 provider 内省",
                f"{inv.get('provider')} 已装配",
                f"hash_algo={inv.get('hash_algo')}",
                f"SM2 公钥在位={inv.get('has_public_key')}（{inv.get('public_key_len')} hex）",
            ]
        )
    else:
        rows.append(
            [
                "运行时 provider 内省",
                "未装配（密钥未配置或环境异常）",
                "—",
                f"内省异常: {inv.get('error')}",
            ]
        )
    return rows


def _call_chain_table() -> list[list[str]]:
    return [
        [
            "静态加密（影像副本/门禁留档落盘）",
            "SM4-CTR + HMAC-SM3（SDC2）",
            "app/pipelines.py 经 infra.crypto provider 落盘加密",
        ],
        [
            "审计哈希链（主链 + 安全链）",
            "SM3（逐条链式哈希，含时间戳入哈希）",
            "infra/repository.py / infra/security_store.py（verify_chain / verify_security_chain）",
        ],
        [
            "报告数字签名",
            "SM2（SM3withSM2）+ SM3 指纹",
            "infra/reporting/pdf_reporter.py 签发 sidecar；/verify 端点验签",
        ],
        [
            "会话令牌",
            "SM3（token 只存 SM3 哈希，明文不落库）",
            "app/auth.py get_principal / security_store sessions",
        ],
        [
            "登录挑战-响应",
            "SM2 验签（SM3withSM2，挑战一次一用 60s）",
            "app/auth.py AuthService.login",
        ],
        [
            "导出一次性令牌",
            "SM3（库存哈希、明文仅签发时返回一次）",
            "app/routers/export.py（X-Export-Token）",
        ],
        [
            "账号凭据",
            "SM2 软证书（公钥登记入库，私钥不落系统）",
            "app/auth.py /auth/accounts/{id}/keypair",
        ],
        [
            "载体销毁证明",
            "PDF 归档留档（未做数字签名——见差距声明）",
            "infra/reporting/certificates.py build_destroy_certificate",
        ],
    ]


def _gap_table() -> list[list[str]]:
    return [
        [
            "high",
            (
                "静态加密与签名为纯软件实现（gmssl 3.2.2），未经商用密码产品认证"
                "（GM/T 0028 密码模块安全要求检测认证）"
            ),
            (
                "对接经认证的商密密码卡/加密机/UKey（Pkcs11Provider 预留），"
                "并交由密评机构对软件实现做评估认定"
            ),
        ],
        [
            "high",
            "主密钥 SCAN_CRYPTO_KEY 经环境变量注入软件侧，无硬件保护；数据密钥由其派生",
            "主密钥托管至硬件（密码机/UKey）；轮换与备份流程按密评要求书面化",
        ],
        [
            "medium",
            "Pkcs11Provider 为接口骨架，机制号与信封封装未真机验证",
            "完成厂商硬件对接联调并回归全部加解密/签名测试后再启用",
        ],
        [
            "medium",
            "存量数据存在 AES-256-GCM 历史信封（SDC1），属非国密遗留算法（只读兼容）",
            "对存量数据执行重加密迁移（读出→SDC2 写回）后归档原密文",
        ],
        [
            "medium",
            "本机回环为明文 HTTP（无 TLS），依赖单机部署边界与 IPC 令牌缓解",
            "如需传输加密，挂本机证书启用 TLS 反向代理（当前不在实现范围，已如实声明）",
        ],
        [
            "low",
            "载体销毁证明为 PDF 台账留档，未附加数字签名",
            "销毁证明生成时附加 SM2 签名 sidecar（与报告签名同构）",
        ],
    ]


def build_crypto_materials() -> dict[str, Any]:
    """构造《密码应用自评估说明》内容（markdown 文本 + 结构化数据）。"""
    inv = _provider_inventory()
    now = _now_str()
    alg_rows = _algorithm_table(inv)
    chain_rows = _call_chain_table()
    gap_rows = _gap_table()

    md: list[str] = [
        "# 密码应用自评估说明",
        "",
        f"生成时间：{now}　|　系统：ScanDetection 射线检测系统（单机涉密部署）",
        "",
        "> 本说明由系统自动生成（C-24），算法清单部分来自 crypto provider 运行时内省，",
        "> 供商用密码应用安全性评估（密评）机构作为材料底稿。合规差距如实列出，",
        "> 未经密评机构认定不得视为符合结论。",
        "",
        "## 一、算法清单",
        "",
        "| 算法/组合 | 用途 | 强度 | 封装/实现说明 |",
        "| --- | --- | --- | --- |",
    ]
    for r in alg_rows:
        md.append("| " + " | ".join(r) + " |")
    md += [
        "",
        "## 二、密钥管理",
        "",
        (
            "- **分层结构**：主密钥（SCAN_CRYPTO_KEY，base64/hex 32 字节）→ 数据密钥经"
            ' SM3 域分离 KDF 派生：SM4 密钥 = SM3("sd-kdf-sm4"||master) 前 16 字节；'
            ' HMAC 密钥 = SM3("sd-kdf-mac"||master)；SM2 私钥 = '
            'int(SM3("sd-kdf-sm2"||master)) mod n（可用 SCAN_SM2_PRIVATE_KEY 显式覆盖）。'
        ),
        "- **域分离**：三种数据密钥使用不同标签派生，互不相关，防跨用途重用。",
        (
            "- **硬件模块对接边界**：SCAN_CRYPTO_PROVIDER=pkcs11 时切换 Pkcs11Provider，"
            "密钥在硬件内生成并托管（CKA_TOKEN/CKA_SENSITIVE，禁止导出），软件侧仅持句柄，"
            "主密钥不参与派生；该路径**未真机验证**（需厂商动态库与联调）。"
        ),
        (
            "- **诚实边界**：本环境为软件实现，主密钥经环境变量注入，无硬件保护；"
            '不提供"找不到密钥就随机生成"的行为（密钥缺失显式报错）。'
        ),
        "",
        "## 三、密码调用链",
        "",
        "| 业务环节 | 算法 | 所在层 |",
        "| --- | --- | --- |",
    ]
    for r in chain_rows:
        md.append("| " + " | ".join(r) + " |")
    md += [
        "",
        "## 四、合规差距声明（供密评机构认定）",
        "",
        "| 等级 | 差距 | 处置建议 |",
        "| --- | --- | --- |",
    ]
    for r in gap_rows:
        md.append("| " + " | ".join(r) + " |")
    md += [
        "",
        "---",
        "",
        (
            "声明：本材料自动生成于系统运行时，provider 内省结果见算法清单末行；"
            "以上差距为开发方自评，最终符合性结论以密评机构认定为准。"
        ),
        "",
    ]
    markdown = "\n".join(md)
    structured = {
        "generated_at": now,
        "title": "密码应用自评估说明",
        "provider_inventory": inv,
        "algorithms": alg_rows,
        "call_chain": chain_rows,
        "gaps": gap_rows,
    }
    return {"markdown": markdown, "data": structured}


def crypto_materials_pdf(data: dict[str, Any], out_path: str | Path) -> Path:
    """密评材料 → PDF（PDF/A-1b，与 markdown 同内容）。"""
    inv = data.get("provider_inventory", {})
    meta = [
        ("生成时间", data.get("generated_at", "—")),
        ("系统", "ScanDetection 射线检测系统（单机涉密部署）"),
        (
            "provider 内省",
            (
                f"{inv.get('provider') or '未装配'}（hash_algo={inv.get('hash_algo', '—')}，"
                f"SM2 公钥在位={inv.get('has_public_key', '—')}）"
            ),
        ),
    ]
    sections = [
        {
            "heading": "一、算法清单",
            "table": {
                "head": ["算法/组合", "用途", "强度", "封装/实现说明"],
                "rows": data.get("algorithms", []),
            },
        },
        {
            "heading": "二、密钥管理",
            "paragraphs": [
                (
                    "分层结构：主密钥 SCAN_CRYPTO_KEY（32 字节，环境变量注入）→ 数据密钥经 "
                    "SM3 域分离 KDF 派生（sd-kdf-sm4 / sd-kdf-mac / sd-kdf-sm2 三标签互不相关）；"
                    "SM2 私钥可经 SCAN_SM2_PRIVATE_KEY 显式覆盖。"
                ),
                (
                    "硬件边界：SCAN_CRYPTO_PROVIDER=pkcs11 切换 Pkcs11Provider，密钥硬件托管"
                    "（CKA_TOKEN/CKA_SENSITIVE），软件侧仅持句柄；该路径未真机验证。"
                ),
                (
                    "诚实边界：本环境为软件实现，主密钥无硬件保护；密钥缺失显式报错，"
                    '不存在"随机生成临时密钥"的静默数据丢失路径。'
                ),
            ],
        },
        {
            "heading": "三、密码调用链",
            "table": {
                "head": ["业务环节", "算法", "所在层"],
                "rows": data.get("call_chain", []),
            },
        },
        {
            "heading": "四、合规差距声明（供密评机构认定）",
            "table": {
                "head": ["等级", "差距", "处置建议"],
                "rows": data.get("gaps", []),
            },
        },
        {
            "heading": "声明",
            "paragraphs": [
                (
                    "本材料由系统自动生成，算法清单来自 crypto provider 运行时内省；"
                    "差距为开发方自评，最终符合性结论以密评机构认定为准。"
                ),
            ],
        },
    ]
    return build_doc_pdf("密码应用自评估说明（C-24）", meta, sections, Path(out_path))


def write_crypto_materials(out_dir: str | Path) -> dict[str, str]:
    """材料落盘（Markdown + PDF），返回 {markdown, pdf} 路径。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
    built = build_crypto_materials()
    md_path = out / f"crypto_materials_{ts}.md"
    md_path.write_text(built["markdown"], encoding="utf-8")
    json_path = out / f"crypto_materials_{ts}.json"
    json_path.write_text(json.dumps(built["data"], ensure_ascii=False, indent=2), encoding="utf-8")
    pdf_path = crypto_materials_pdf(built["data"], out / f"crypto_materials_{ts}.pdf")
    return {"markdown": str(md_path), "json": str(json_path), "pdf": str(pdf_path)}
