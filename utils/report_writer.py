"""
report_writer.py
----------------
把 Markdown 报告写入文件和数据库。
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "logs"


def save_report(
    trade_date: str,
    report_type: str,
    content: str,
    db_path: str | None = None,
) -> str:
    """
    保存 Markdown 报告到文件和数据库。

    Args:
        trade_date: YYYYMMDD
        report_type: 'eod' 或 'analysis'
        content: Markdown 文本
        db_path: 数据库路径（None 时自动获取）

    Returns:
        写入的文件路径
    """
    # 写文件
    sub_dir = LOGS_DIR / report_type
    sub_dir.mkdir(parents=True, exist_ok=True)
    file_path = sub_dir / f"{trade_date}.md"
    file_path.write_text(content, encoding="utf-8")

    # 写数据库
    if db_path is None:
        try:
            from config.loader import ConfigLoader
            db_path = ConfigLoader().get_db_path()
        except Exception:
            db_path = str(ROOT / "data" / "storage" / "trading.db")

    try:
        conn = sqlite3.connect(db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS daily_reports ("
            "trade_date TEXT NOT NULL, report_type TEXT NOT NULL, "
            "content TEXT, created_at TEXT, "
            "PRIMARY KEY (trade_date, report_type))"
        )
        conn.execute(
            "INSERT OR REPLACE INTO daily_reports "
            "(trade_date, report_type, content, created_at) VALUES (?,?,?,?)",
            (trade_date, report_type, content,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # DB 写入失败不影响主流程

    return str(file_path)
