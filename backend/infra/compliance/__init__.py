"""分级保护合规子包（C-23~C-25）。

- selfcheck       : 分级保护五类自查（活检查，真实查询系统状态）；
- crypto_materials: 密码应用自评估说明（密评材料）导出；
- hardening       : 安全加固自检（口令/端口/接口/TLS/文件权限）；
- doc_pdf         : 三者共用的简易报告 PDF 版式（复用报告字体注册）。

产物统一落 data/compliance/ 目录（JSON + PDF），动作入主审计链。
"""

from backend.infra.compliance.doc_pdf import build_doc_pdf

__all__ = ["build_doc_pdf"]
