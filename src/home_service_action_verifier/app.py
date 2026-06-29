"""Tkinter desktop app for the current verifier prototype."""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from home_service_action_verifier.analyzer import analyze_tracks
from home_service_action_verifier.io import (
    load_object_tracks_csv,
    load_skeleton_csv,
    load_task_context,
    make_run_dir,
    write_results,
)
from home_service_action_verifier.models import DetectionResult


class VerifierApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("HomeServiceActionVerifier")
        self.geometry("1120x720")
        self.minsize(960, 620)

        self.context_path = tk.StringVar()
        self.skeleton_path = tk.StringVar()
        self.object_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path("outputs/current_app")))
        self.status_text = tk.StringVar(value="入力ファイルを選択してください。")
        self.results: list[DetectionResult] = []

        self._build_ui()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(root, text="骨格・物体局所情報による不審物品操作検知", font=("", 15, "bold"))
        title.pack(anchor=tk.W)

        file_frame = ttk.LabelFrame(root, text="入力", padding=10)
        file_frame.pack(fill=tk.X, pady=(10, 8))
        file_frame.columnconfigure(1, weight=1)

        self._file_row(file_frame, 0, "作業文脈 JSON", self.context_path, self._pick_context)
        self._file_row(file_frame, 1, "骨格 CSV", self.skeleton_path, self._pick_skeleton)
        self._file_row(file_frame, 2, "物体追跡 CSV", self.object_path, self._pick_objects)
        self._file_row(file_frame, 3, "出力フォルダ", self.output_dir, self._pick_output_dir)

        actions = ttk.Frame(root)
        actions.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(actions, text="解析を実行", command=self._run_analysis).pack(side=tk.LEFT)
        ttk.Button(actions, text="結果を書き出し", command=self._export_results).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(actions, textvariable=self.status_text).pack(side=tk.LEFT, padx=(16, 0))

        body = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)

        table_frame = ttk.Frame(body)
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(
            table_frame,
            columns=("object", "role", "label", "score", "touch", "alert", "reason"),
            show="headings",
            height=18,
        )
        for column, text, width in [
            ("object", "対象物", 130),
            ("role", "役割", 90),
            ("label", "判定", 90),
            ("score", "不審度", 80),
            ("touch", "接触時刻", 90),
            ("alert", "警告時刻", 90),
            ("reason", "理由", 360),
        ]:
            self.tree.heading(column, text=text)
            self.tree.column(column, width=width, anchor=tk.W)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)
        body.add(table_frame, weight=3)

        detail_frame = ttk.LabelFrame(body, text="説明", padding=8)
        detail_frame.rowconfigure(0, weight=1)
        detail_frame.columnconfigure(0, weight=1)
        self.detail = tk.Text(detail_frame, wrap=tk.WORD, height=18)
        self.detail.grid(row=0, column=0, sticky="nsew")
        detail_scroll = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=self.detail.yview)
        detail_scroll.grid(row=0, column=1, sticky="ns")
        self.detail.configure(yscrollcommand=detail_scroll.set)
        self._set_detail(
            "このアプリは、作業者の骨格・姿勢系列と対象物体周辺の追跡情報を使い、"
            "作業文脈として自然かどうかを不審度スコアとして出します。\n\n"
            "人物の顔や服装を主入力にせず、部屋全体の画像理解も主手法にしません。"
        )
        body.add(detail_frame, weight=2)

    def _file_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        command,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8, pady=3)
        ttk.Button(parent, text="選択", command=command).grid(row=row, column=2, sticky=tk.E, pady=3)

    def _pick_context(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if path:
            self.context_path.set(path)

    def _pick_skeleton(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if path:
            self.skeleton_path.set(path)

    def _pick_objects(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if path:
            self.object_path.set(path)

    def _pick_output_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    def _run_analysis(self) -> None:
        try:
            context = load_task_context(self._required(self.context_path, "作業文脈 JSON"))
            skeleton = load_skeleton_csv(self._required(self.skeleton_path, "骨格 CSV"))
            objects = load_object_tracks_csv(self._required(self.object_path, "物体追跡 CSV"))
            self.results = analyze_tracks(skeleton, objects, context)
        except Exception as exc:  # noqa: BLE001 - desktop UI should show any user-facing load error
            messagebox.showerror("解析できません", str(exc))
            self.status_text.set("解析に失敗しました。")
            return

        self._populate_results()
        self.status_text.set(f"{len(self.results)} 件の対象物を解析しました。")

    def _required(self, variable: tk.StringVar, label: str) -> str:
        value = variable.get().strip()
        if not value:
            msg = f"{label} を選択してください。"
            raise ValueError(msg)
        return value

    def _populate_results(self) -> None:
        for row in self.tree.get_children():
            self.tree.delete(row)
        for result in self.results:
            reason = "; ".join(result.reasons)
            self.tree.insert(
                "",
                tk.END,
                values=(
                    f"{result.object_id}: {result.object_label}",
                    result.role,
                    result.predicted_label,
                    f"{result.suspicion_score:.2f}",
                    "" if result.first_touch_time is None else f"{result.first_touch_time:.2f}",
                    "" if result.first_alert_time is None else f"{result.first_alert_time:.2f}",
                    reason,
                ),
            )
        self._set_detail(self._make_summary_text())

    def _make_summary_text(self) -> str:
        if not self.results:
            return "解析結果はまだありません。"
        counts: dict[str, int] = {}
        for result in self.results:
            counts[result.predicted_label] = counts.get(result.predicted_label, 0) + 1
        lines = ["解析結果", ""]
        for label in ["normal", "review", "suspicious", "high_risk"]:
            lines.append(f"- {label}: {counts.get(label, 0)}")
        lines.append("")
        lines.append("主な高スコア対象:")
        for result in self.results[:5]:
            lines.append(
                f"- {result.object_id} ({result.object_label}): "
                f"{result.predicted_label} / {result.suspicion_score:.2f}"
            )
            for reason in result.reasons:
                lines.append(f"  - {reason}")
        return "\n".join(lines)

    def _set_detail(self, text: str) -> None:
        self.detail.configure(state=tk.NORMAL)
        self.detail.delete("1.0", tk.END)
        self.detail.insert(tk.END, text)
        self.detail.configure(state=tk.DISABLED)

    def _export_results(self) -> None:
        if not self.results:
            messagebox.showinfo("結果なし", "先に解析を実行してください。")
            return
        try:
            run_dir = make_run_dir(self.output_dir.get().strip() or "outputs/current_app")
            paths = write_results(run_dir, self.results)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("書き出し失敗", str(exc))
            return
        self.status_text.set(f"結果を書き出しました: {run_dir}")
        self._set_detail(
            self._make_summary_text()
            + "\n\n書き出し先:\n"
            + "\n".join(f"- {key}: {path}" for key, path in paths.items())
        )


def _run_cli(args: argparse.Namespace) -> int:
    context = load_task_context(args.context)
    skeleton = load_skeleton_csv(args.skeleton)
    objects = load_object_tracks_csv(args.objects)
    results = analyze_tracks(skeleton, objects, context)
    output_dir = make_run_dir(args.output_dir)
    paths = write_results(output_dir, results)
    print(f"Analyzed {len(results)} object tracks.")
    for result in results:
        print(
            f"{result.object_id}\t{result.object_label}\t{result.predicted_label}\t"
            f"{result.suspicion_score:.2f}\t{'; '.join(result.reasons)}"
        )
    print(f"summary: {paths['summary']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HomeServiceActionVerifier desktop prototype")
    parser.add_argument("--context", type=Path, default=None, help="Task context JSON for CLI analysis")
    parser.add_argument("--skeleton", type=Path, default=None, help="Skeleton CSV for CLI analysis")
    parser.add_argument("--objects", type=Path, default=None, help="Object tracks CSV for CLI analysis")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/current_app"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.context or args.skeleton or args.objects:
        if not (args.context and args.skeleton and args.objects):
            parser.error("--context, --skeleton, and --objects are required together")
        return _run_cli(args)
    app = VerifierApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

