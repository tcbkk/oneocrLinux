#!/usr/bin/env python3
"""Build the Windows vs Linux install-ID comparison table."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = [
    ("db85017be945856732b32b159c4d1a6df0b766ed024a86498048b6919cd8ae26.jpg", 1),
    ("8ac45dc29ae89476694c1fea429b82001e8e2d08e87dc040d15ad19a6db505b1.jpg", 2),
    ("53995527c3f0d007a430f7a70aced9c3129ba92e43e0afcb2fd227c306a1b17f.jpg", 3),
    ("0e0311b5a1b4bf7c5e18f06c5199fc6bb629cc7f67a52981bf99325103c51354.jpg", 4),
    ("2ebfaabfc75834a6266dc5c8626929c56cd4b8a9c91dee7ccd910b33a27e9844.jpg", 5),
    ("37cc3551854df89ca09104e272018679abdc223c90e0d9b405ef11bff3fcb8cd.jpg", 6),
    ("024a3c7277244e66fd3904c79420bd2a5942057b9f12f37d71796d34c54776a7.jpg", 7),
    ("f841b35077328e62ba052775d40714e8b832a8ffde66c53b223a887bc02e5c4e.jpg", 8),
    ("407e385a04fd3b41e1aaa19a44cd02b0eb05d93d35f0be2f9f86efdcfb6fa67f.jpg", 9),
    ("b75e582fd5deaf779c90f2e18a587e1c1a19cebc58a913978c119b8bfbcbbfea.jpg", 10),
]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    win = load(os.path.join(ROOT, "samples", "batch_windows.json"))
    local = load(os.path.join(ROOT, "samples", "batch_local.json"))
    server = load(os.path.join(ROOT, "samples", "batch_server.json"))

    def get(db, name):
        e = db.get(name)
        return (e or {}).get("windows") or (e or {}).get("linux") or {}

    rows = []
    for raw, idx in ORDER:
        name = "engine-image-" + raw
        w = win.get(name, {}).get("windows", {})
        l = server.get(name, {}).get("linux", {})
        wg, lg = w.get("groups", []), l.get("groups", [])
        same = wg == lg and len(wg) == 9
        rows.append({
            "idx": idx,
            "file": raw[:8],
            "windows": wg,
            "linux": lg,
            "windows_ok": len(wg) == 9,
            "linux_ok": len(lg) == 9,
            "identical": same,
            "w_time": w.get("seconds"),
            "l_time": l.get("seconds"),
        })

    out = {"rows": rows,
           "windows_ok": sum(r["windows_ok"] for r in rows),
           "linux_ok": sum(r["linux_ok"] for r in rows),
           "identical": sum(r["identical"] for r in rows)}
    with open(os.path.join(ROOT, "samples", "compare_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    md = ["| # | 图片 | Windows 版 | Linux 版 | 一致 |", "|---|---|---|---|---|"]
    for r in rows:
        w = " ".join(r["windows"]) if r["windows"] else "—"
        l = " ".join(r["linux"]) if r["linux"] else "—"
        md.append("| %d | `%s` | %s | %s | %s |" % (
            r["idx"], r["file"], w, l, "✅" if r["identical"] else ("≈" if r["windows"] and r["linux"] else "—")))
    md.append("")
    md.append("Windows %d/10，Linux %d/10，逐字一致 %d/10" % (
        out["windows_ok"], out["linux_ok"], out["identical"]))
    text = "\n".join(md)
    with open(os.path.join(ROOT, "samples", "compare_table.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()