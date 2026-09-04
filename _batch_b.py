"""Batch B 剩余补丁（一次性执行后删除）。"""
import io


def patch(path, old, new, count=1):
    s = io.open(path, encoding="utf-8").read()
    assert old in s, f"NOT FOUND in {path}: {old[:80]!r}"
    io.open(path, "w", encoding="utf-8", newline="").write(s.replace(old, new, count))
    print("ok", path)


# classification 内联双写委托
patch(
    "backend/app/routers/classification.py",
    """    reg.repository.append_audit(
        actor=principal.username,
        action="secret_level_change",
        object_type="image",
        object_id=image_id,
        before=snap["before"],
        after=snap["after"],
        note=body.classification_basis,
    )
    reg.security_store.append_security_audit(
        actor=principal.username,
        action="secret_level_change",
        object_type="image",
        object_id=image_id,
        before=snap["before"],
        after=snap["after"],
        note=body.classification_basis,
    )""",
    """    dual_audit(
        reg.repository,
        reg.security_store,
        actor=principal.username,
        action="secret_level_change",
        object_type="image",
        object_id=image_id,
        before=snap["before"],
        after=snap["after"],
        note=body.classification_basis,
        security=True,
    )""",
)
s = io.open("backend/app/routers/classification.py", encoding="utf-8").read()
if "from backend.infra.audit import" not in s:
    s = s.replace(
        "from backend.app.dependencies import",
        "from backend.infra.audit import dual_audit\nfrom backend.app.dependencies import",
        1,
    )
    io.open("backend/app/routers/classification.py", "w", encoding="utf-8", newline="").write(s)
    print("ok classification import")

# auth._on_account_locked 内联双写委托
patch(
    "backend/app/auth.py",
    """            self._store.append_security_audit(
                actor="system",
                action="account_lock",
                object_type="account",
                object_id=account["account_id"],
                before={"status": "active"},
                after={"locked": True, "lockout_min": self._cfg.lockout_min},
                note=msg,
            )
            get_registry().repository.append_audit(
                actor="system",
                action="account_lock",
                object_type="account",
                object_id=account["account_id"],
                before=None,
                after={"username": account["username"]},
                note=msg,
            )""",
    """            dual_audit(
                get_registry().repository,
                self._store,
                actor="system",
                action="account_lock",
                object_type="account",
                object_id=account["account_id"],
                after={"username": account["username"]},
                note=msg,
                security=True,
                sec_before={"status": "active"},
                sec_after={"locked": True, "lockout_min": self._cfg.lockout_min},
            )""",
)
s = io.open("backend/app/auth.py", encoding="utf-8").read()
if "from backend.infra.audit import" not in s:
    anchor = "from backend.app.dependencies import"
    if anchor not in s:
        anchor = "from backend.app.security import"
    assert anchor in s, "auth.py 导入锚点未找到"
    s = s.replace(anchor, "from backend.infra.audit import dual_audit\n" + anchor, 1)
    io.open("backend/app/auth.py", "w", encoding="utf-8", newline="").write(s)
    print("ok auth import")

# export._decision_audit 委托
patch(
    "backend/app/routers/export.py",
    """    reg.repository.append_audit(
        actor=actor,
        action=action,
        object_type="export_request",
        object_id=row["request_id"],
        before={"status": "pending"},
        after={"status": row["status"]},
        note=note,
    )
    reg.security_store.append_security_audit(
        actor=actor,
        action=action,
        object_type="export_request",
        object_id=row["request_id"],
        before={"subject": row["subject"]},
        after={"status": row["status"]},
        note=note,
    )""",
    """    dual_audit(
        reg.repository,
        reg.security_store,
        actor=actor,
        action=action,
        object_type="export_request",
        object_id=row["request_id"],
        before={"status": "pending"},
        after={"status": row["status"]},
        note=note,
        security=True,
        sec_before={"subject": row["subject"]},
    )""",
)
s = io.open("backend/app/routers/export.py", encoding="utf-8").read()
if "from backend.infra.audit import" not in s:
    s = s.replace(
        "from backend.infra.compliance_store import",
        "from backend.infra.audit import dual_audit\nfrom backend.infra.compliance_store import",
        1,
    )
    io.open("backend/app/routers/export.py", "w", encoding="utf-8", newline="").write(s)
    print("ok export import")

print("BATCH B PART1 DONE")
