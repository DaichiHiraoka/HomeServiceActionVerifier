from __future__ import annotations

import argparse
import json
import urllib.parse
from pathlib import Path
from typing import NoReturn

from .constants import BBox, DEFAULT_MODEL_PATH
from .models import DetectorConfig

IP_MEMORY_PATH = Path(__file__).resolve().parent.parent / "ip_memory.json"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def prompt_source() -> str:
    """
    CLI引数で映像入力が指定されていない場合、GUIでIP WebcamのURLを作ります。
    """

    try:
        source_text = prompt_ip_webcam_source_gui()
    except RuntimeError:
        raise
    except Exception as error:
        print(f"GUI入力を開けないため、コンソール入力へ切り替えます: {error}")
        source_text = prompt_source_console()

    if not source_text:
        _raise_source_cancelled()

    return source_text


def prompt_source_console() -> str:
    """
    GUIを開けない環境向けのコンソール入力フォールバックです。
    """

    print("映像入力ソースを指定してください。")
    print("  Webカメラ: 0")
    print("  動画ファイル: C:\\path\\input.mp4")
    print("  IP Webcam: 192.168.1.20:8080 または http://192.168.1.20:8080/video")

    try:
        source_text = input("映像入力ソース> ").strip()
    except EOFError as error:
        raise RuntimeError(
            "--source が未指定で、実行後入力も読み取れませんでした。"
        ) from error

    if not source_text:
        print("未入力のため Webカメラ 0 を使います。")
        return "0"

    return source_text


def _raise_source_cancelled() -> NoReturn:
    raise RuntimeError("映像入力ソースの指定がキャンセルされました。")


def ensure_ip_memory_file(path: Path = IP_MEMORY_PATH) -> None:
    """IP Webcam入力履歴用のローカルJSONを、なければ作成します。"""

    if path.exists():
        return

    path.write_text(
        json.dumps({"last_source": ""}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_ip_memory(path: Path = IP_MEMORY_PATH) -> str:
    """保存済みのIP Webcam入力を読み込みます。壊れたJSONは空として扱います。"""

    ensure_ip_memory_file(path)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""

    if not isinstance(data, dict):
        return ""

    source = data.get("last_source", "")
    if not isinstance(source, str):
        return ""

    return normalize_ip_webcam_source_text(source)


def save_ip_memory(source: str, path: Path = IP_MEMORY_PATH) -> None:
    """確定したIP Webcam入力を次回用に保存します。"""

    ensure_ip_memory_file(path)
    normalized = normalize_ip_webcam_source_text(source)
    path.write_text(
        json.dumps({"last_source": normalized}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prompt_ip_webcam_source_gui() -> str:
    """
    IP WebcamのIPアドレスとポートをGUIで入力します。

    Enter:
        3回目までは "." を末尾へ挿入し、4回目は ":" を挿入します。
        ":" の後にポート番号が入っている状態では入力を確定します。

    Esc:
        キャンセルします。
    """

    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("IP Webcam input")
    root.resizable(False, False)

    result: dict[str, str | None] = {"source": None}
    value = tk.StringVar()
    status = tk.StringVar(value="例: 192⏎168⏎1⏎20⏎8080")

    try:
        ensure_ip_memory_file()
    except OSError as error:
        status.set(f"IPmemoryを作成できません: {error}")

    def current_text() -> str:
        return value.get().strip()

    def normalized_current_text() -> str:
        return normalize_ip_webcam_source_text(current_text())

    def cancel() -> None:
        result["source"] = None
        root.destroy()

    def accept() -> None:
        source = normalized_current_text()
        if not source:
            status.set("IPアドレスとポートを入力してください。Escでキャンセルできます。")
            return

        if ":" not in source:
            status.set("ポート番号が未入力です。Enterで ':' を追加してください。")
            return

        host, port = source.rsplit(":", 1)
        if host.count(".") != 3 or not port.isdigit():
            status.set("形式は 192.168.1.20:8080 です。")
            return

        try:
            save_ip_memory(source)
        except OSError as error:
            print(f"IPmemoryを保存できませんでした: {error}")

        result["source"] = source
        root.destroy()

    def insert_next_separator() -> None:
        text = current_text()
        if not text:
            status.set("まずIPアドレスの数字を入力してください。")
            return

        normalized = normalize_ip_webcam_source_text(text)
        if normalized.endswith((".", ":")):
            return

        if ":" in normalized:
            _replace_entry_text(normalized)
            accept()
            return

        separator = "." if normalized.count(".") < 3 else ":"
        _replace_entry_text(f"{normalized}{separator}")

        if separator == ".":
            status.set("次のIP区切りを入力してください。")
        else:
            status.set("ポート番号を入力し、Enterで開始します。")

    def _replace_entry_text(text: str) -> None:
        value.set(text)
        entry.icursor(tk.END)

    def append_digit(digit: str) -> None:
        _replace_entry_text(f"{current_text()}{digit}")
        status.set("数字ボタンまたはキーボードで入力できます。")

    def backspace() -> None:
        _replace_entry_text(current_text()[:-1])
        status.set("1文字削除しました。")

    def clear() -> None:
        _replace_entry_text("")
        status.set("入力をクリアしました。")

    def load_memory() -> None:
        try:
            remembered_source = load_ip_memory()
        except OSError as error:
            status.set(f"IPmemoryを読み込めません: {error}")
            return

        if not remembered_source:
            status.set("IPmemoryは空です。入力を開始すると確定時に保存されます。")
            return

        _replace_entry_text(remembered_source)
        status.set("IPmemoryから入力しました。Enterまたは開始で確定できます。")

    def on_return(_event: object) -> str:
        insert_next_separator()
        return "break"

    def on_escape(_event: object) -> str:
        cancel()
        return "break"

    frame = ttk.Frame(root, padding=16)
    frame.grid(row=0, column=0, sticky="nsew")

    ttk.Label(frame, text="IP Webcam のIPアドレスとポート").grid(
        row=0,
        column=0,
        sticky="w",
    )
    ttk.Label(
        frame,
        text="キーボードまたは数字ボタンで入力し、Enterで '.' と ':' を順に挿入します。",
        foreground="#555555",
    ).grid(row=1, column=0, sticky="w", pady=(4, 10))

    entry = ttk.Entry(frame, textvariable=value, width=34)
    entry.grid(row=2, column=0, sticky="ew")
    entry.bind("<Return>", on_return)
    entry.bind("<Escape>", on_escape)
    root.bind("<Escape>", on_escape)

    ttk.Label(frame, textvariable=status, foreground="#555555").grid(
        row=3,
        column=0,
        sticky="w",
        pady=(8, 12),
    )

    keypad = ttk.Frame(frame)
    keypad.grid(row=4, column=0, sticky="ew")

    for row_index, row_digits in enumerate((("7", "8", "9"), ("4", "5", "6"), ("1", "2", "3"))):
        for column_index, digit in enumerate(row_digits):
            ttk.Button(
                keypad,
                text=digit,
                command=lambda digit=digit: append_digit(digit),
                width=7,
            ).grid(row=row_index, column=column_index, padx=3, pady=3, sticky="nsew")

    ttk.Button(keypad, text="C", command=clear, width=7).grid(
        row=3,
        column=0,
        padx=3,
        pady=3,
        sticky="nsew",
    )
    ttk.Button(keypad, text="0", command=lambda: append_digit("0"), width=7).grid(
        row=3,
        column=1,
        padx=3,
        pady=3,
        sticky="nsew",
    )
    ttk.Button(keypad, text="⌫", command=backspace, width=7).grid(
        row=3,
        column=2,
        padx=3,
        pady=3,
        sticky="nsew",
    )
    ttk.Button(keypad, text="Enter  . / :", command=insert_next_separator).grid(
        row=4,
        column=0,
        columnspan=3,
        padx=3,
        pady=(3, 8),
        sticky="ew",
    )
    ttk.Button(keypad, text="IPmemory", command=load_memory).grid(
        row=5,
        column=0,
        columnspan=3,
        padx=3,
        pady=(0, 8),
        sticky="ew",
    )

    for column_index in range(3):
        keypad.columnconfigure(column_index, weight=1)

    button_row = ttk.Frame(frame)
    button_row.grid(row=5, column=0, sticky="e")
    ttk.Button(button_row, text="キャンセル", command=cancel).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(button_row, text="開始", command=accept).grid(row=0, column=1)

    root.protocol("WM_DELETE_WINDOW", cancel)
    entry.focus_set()
    root.update_idletasks()
    root.minsize(root.winfo_width(), root.winfo_height())
    root.mainloop()

    if result["source"] is None:
        _raise_source_cancelled()

    return result["source"]


def normalize_ip_webcam_source_text(source_text: str) -> str:
    """GUI入力されたIP Webcamの省略表記を host:port へ正規化します。"""

    text = source_text.strip()
    text = text.replace("，", ".").replace(",", ".").replace("：", ":")
    text = text.replace(" ", "").replace("\t", "")
    return text


def normalize_stream_source(source_text: str) -> str:
    """
    IP Webcam向けの省略入力をOpenCVで開けるURLへ補正します。
    """

    stripped = source_text.strip().replace("：", ":")

    if not stripped:
        return "0"

    # Windowsの絶対パス C:\... はURL扱いしません。
    if len(stripped) >= 3 and stripped[1] == ":" and stripped[2] in ("\\", "/"):
        return stripped

    parsed = urllib.parse.urlparse(stripped)

    if parsed.scheme in {"http", "https"}:
        if parsed.path in {"", "/"}:
            return urllib.parse.urlunparse(parsed._replace(path="/video"))
        return stripped

    # IP Webcamでは host:port 形式を入力することが多いので /video を補います。
    if ":" in stripped and "/" not in stripped and "\\" not in stripped:
        return f"http://{stripped}/video"

    return stripped


def parse_source(source_text: str) -> int | str:
    """
    source が整数文字列ならカメラ番号、その他なら動画パスまたはURLとして扱います。
    """

    stripped = normalize_stream_source(source_text)

    if stripped.lstrip("+-").isdigit():
        return int(stripped)

    return stripped


def parse_bbox_argument(bbox_text: str) -> BBox:
    """x,y,w,h 形式の文字列をOpenCV矩形へ変換します。"""

    parts = [part.strip() for part in bbox_text.split(",")]

    if len(parts) != 4:
        raise ValueError("--object-roi は x,y,w,h の4整数で指定してください。")

    try:
        x, y, width, height = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError("--object-roi は x,y,w,h の4整数で指定してください。") from error

    if width <= 0 or height <= 0:
        raise ValueError("--object-roi の width と height は1以上にしてください。")

    return (x, y, width, height)


def parse_csv_tokens(value: str | None) -> tuple[str, ...] | None:
    """カンマ区切りのクラス名またはIDを正規化します。"""

    if value is None:
        return None

    tokens = tuple(
        token.strip().lower()
        for token in value.split(",")
        if token.strip()
    )
    return tokens or None


def build_argument_parser() -> argparse.ArgumentParser:
    """コマンドライン引数を定義します。"""

    parser = argparse.ArgumentParser(
        description=(
            "手の21点骨格と追跡物体から、物体を把持した状態で移動させたかをリアルタイム判定します。"
        )
    )

    parser.add_argument(
        "--source",
        default=None,
        help=(
            "カメラ番号、動画ファイル、IP Webcam URL。"
            "未指定なら起動後にGUIでIP/ポートを入力します。"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=(
            "Hand Landmarkerモデルの保存先。存在しなければ公式配布元から取得します。"
        ),
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        help="同時に検出する最大手数。既定値: 2",
    )
    parser.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="表示だけ左右反転します。推論座標は反転しません。既定値: 有効",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=1280,
        help="カメラ入力時に要求する横解像度。既定値: 1280",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=720,
        help="カメラ入力時に要求する縦解像度。既定値: 720",
    )
    parser.add_argument(
        "--enter-threshold",
        type=float,
        default=0.50,
        help="把持移動開始閾値。既定値: 0.50",
    )
    parser.add_argument(
        "--exit-threshold",
        type=float,
        default=0.36,
        help="把持移動解除閾値。既定値: 0.36",
    )
    parser.add_argument(
        "--enter-delay",
        type=float,
        default=0.18,
        help="把持移動開始に必要な継続秒数。既定値: 0.18",
    )
    parser.add_argument(
        "--exit-delay",
        type=float,
        default=0.15,
        help="把持移動解除に必要な継続秒数。既定値: 0.15",
    )
    parser.add_argument(
        "--ema-time-constant",
        type=float,
        default=0.10,
        help="スコア平滑化EMAの時定数秒。既定値: 0.10",
    )
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=0.55,
        help="MediaPipeの手検出信頼度閾値。既定値: 0.55",
    )
    parser.add_argument(
        "--presence-confidence",
        type=float,
        default=0.55,
        help="MediaPipeの手存在信頼度閾値。既定値: 0.55",
    )
    parser.add_argument(
        "--tracking-confidence",
        type=float,
        default=0.55,
        help="MediaPipeの追跡信頼度閾値。既定値: 0.55",
    )
    parser.add_argument(
        "--object-tracking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="OpenCVによる物体検出・追跡を有効化します。既定値: 有効",
    )
    parser.add_argument(
        "--object-detector",
        choices=("yolo", "motion", "hybrid"),
        default="yolo",
        help=(
            "物体候補の検出方式。yolo: YOLO検出、motion: 従来の背景差分、"
            "hybrid: YOLOと背景差分を併用。既定値: yolo"
        ),
    )
    parser.add_argument(
        "--object-roi",
        default=None,
        help="初期追跡物体の矩形 x,y,w,h。静止物体を追う場合に指定します。",
    )
    parser.add_argument(
        "--select-object",
        action="store_true",
        help="初回フレームで追跡対象をマウス選択します。",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=6,
        help="同時に追跡する最大物体数。既定値: 6",
    )
    parser.add_argument(
        "--object-min-area",
        type=float,
        default=500.0,
        help="自動検出で採用する最小物体面積px。既定値: 500",
    )
    parser.add_argument(
        "--object-contact-threshold",
        type=float,
        default=0.42,
        help="手と物体が接触しているとみなす目安スコア。既定値: 0.42",
    )
    parser.add_argument(
        "--object-motion-speed-threshold",
        type=float,
        default=60.0,
        help="把持中の物体移動とみなす速度px/s。既定値: 60",
    )
    parser.add_argument(
        "--object-motion-displacement-threshold",
        type=float,
        default=35.0,
        help="把持中の物体移動とみなす初期位置からの変位px。既定値: 35",
    )
    parser.add_argument(
        "--object-settle-radius-ratio",
        type=float,
        default=0.15,
        help="物体静定の許容半径をbbox対角長に対する比率で指定します。既定値: 0.15",
    )
    parser.add_argument(
        "--object-settle-radius-min",
        type=float,
        default=8.0,
        help="物体静定の最小許容半径px。既定値: 8",
    )
    parser.add_argument(
        "--object-settle-frames",
        type=int,
        default=15,
        help="物体を静定済みとみなす連続フレーム数。既定値: 15",
    )
    parser.add_argument(
        "--object-rebaseline-hand-distance",
        type=float,
        default=120.0,
        help="手bbox中心がこの距離px以内なら再ベースラインを保留します。既定値: 120",
    )
    parser.add_argument(
        "--object-vacancy-similarity-threshold",
        type=float,
        default=0.72,
        help="元位置に同じ見た目が残っているとみなす類似度閾値。既定値: 0.72",
    )
    parser.add_argument(
        "--object-birth-hand-distance",
        type=float,
        default=90.0,
        help="手bbox近傍で新規トラック生成を抑制する距離px。既定値: 90",
    )
    parser.add_argument(
        "--object-match-threshold",
        type=float,
        default=0.58,
        help="テンプレート追跡の採用閾値。既定値: 0.58",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolo11n.pt",
        help="Ultralytics YOLOモデル。例: yolo11n.pt または custom.pt。既定値: yolo11n.pt",
    )
    parser.add_argument(
        "--yolo-confidence",
        type=float,
        default=0.35,
        help="YOLO検出の信頼度閾値。既定値: 0.35",
    )
    parser.add_argument(
        "--yolo-iou",
        type=float,
        default=0.45,
        help="YOLO NMSのIoU閾値。既定値: 0.45",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=640,
        help="YOLO推論画像サイズ。既定値: 640",
    )
    parser.add_argument(
        "--yolo-classes",
        default=None,
        help="検出対象クラス名またはIDのカンマ区切り。未指定ならperson以外を使います。",
    )
    parser.add_argument(
        "--yolo-ignore-classes",
        default="person",
        help="除外するYOLOクラス名またはIDのカンマ区切り。既定値: person",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="指定した場合、フレームごとの特徴量と判定をCSV保存します。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="指定した場合、描画済み動画を保存します。",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="画面表示を無効化します。動画処理やCSV生成向けです。",
    )

    return parser


def validate_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> DetectorConfig:
    """CLI引数の値域と相互関係を検証します。"""

    probability_arguments = {
        "--enter-threshold": args.enter_threshold,
        "--exit-threshold": args.exit_threshold,
        "--detection-confidence": args.detection_confidence,
        "--presence-confidence": args.presence_confidence,
        "--tracking-confidence": args.tracking_confidence,
        "--object-contact-threshold": args.object_contact_threshold,
        "--object-match-threshold": args.object_match_threshold,
        "--object-settle-radius-ratio": args.object_settle_radius_ratio,
        "--object-vacancy-similarity-threshold": args.object_vacancy_similarity_threshold,
        "--yolo-confidence": args.yolo_confidence,
        "--yolo-iou": args.yolo_iou,
    }

    for argument_name, value in probability_arguments.items():
        if not 0.0 <= value <= 1.0:
            parser.error(f"{argument_name} は 0.0～1.0 で指定してください。")

    if args.enter_threshold <= args.exit_threshold:
        parser.error(
            "--enter-threshold は --exit-threshold より大きくしてください。"
        )

    if args.enter_delay < 0.0 or args.exit_delay < 0.0:
        parser.error("遅延時間は0秒以上で指定してください。")

    if args.ema_time_constant <= 0.0:
        parser.error("--ema-time-constant は0より大きくしてください。")

    if args.max_hands < 1:
        parser.error("--max-hands は1以上で指定してください。")

    if args.camera_width < 1 or args.camera_height < 1:
        parser.error("カメラ解像度は1以上で指定してください。")

    if args.max_objects < 1:
        parser.error("--max-objects は1以上で指定してください。")

    if args.object_min_area < 1.0:
        parser.error("--object-min-area は1以上で指定してください。")

    if args.object_motion_speed_threshold <= 0.0:
        parser.error("--object-motion-speed-threshold は0より大きくしてください。")

    if args.object_motion_displacement_threshold <= 0.0:
        parser.error("--object-motion-displacement-threshold は0より大きくしてください。")

    if args.object_settle_radius_min < 0.0:
        parser.error("--object-settle-radius-min は0以上で指定してください。")

    if args.object_settle_frames < 1:
        parser.error("--object-settle-frames は1以上で指定してください。")

    if args.object_rebaseline_hand_distance < 0.0:
        parser.error("--object-rebaseline-hand-distance は0以上で指定してください。")

    if args.object_birth_hand_distance < 0.0:
        parser.error("--object-birth-hand-distance は0以上で指定してください。")

    if args.yolo_imgsz < 32:
        parser.error("--yolo-imgsz は32以上で指定してください。")

    if args.object_roi is not None:
        try:
            parse_bbox_argument(args.object_roi)
        except ValueError as error:
            parser.error(str(error))

    if args.select_object and args.no_window:
        parser.error("--select-object は --no-window と同時に使えません。")

    if args.no_window and args.output is None and args.csv is None:
        parser.error(
            "--no-window を使う場合は --output または --csv を指定してください。"
        )

    return DetectorConfig(
        enter_threshold=args.enter_threshold,
        exit_threshold=args.exit_threshold,
        enter_delay_sec=args.enter_delay,
        exit_delay_sec=args.exit_delay,
        ema_time_constant_sec=args.ema_time_constant,
        object_tracking_enabled=args.object_tracking,
        object_detector=args.object_detector,
        object_min_area=args.object_min_area,
        object_max_tracks=args.max_objects,
        object_template_match_threshold=args.object_match_threshold,
        object_contact_threshold=args.object_contact_threshold,
        object_motion_speed_threshold_px_s=args.object_motion_speed_threshold,
        object_motion_displacement_threshold_px=args.object_motion_displacement_threshold,
        object_settle_radius_ratio=args.object_settle_radius_ratio,
        object_settle_radius_min_px=args.object_settle_radius_min,
        object_settle_frames=args.object_settle_frames,
        object_rebaseline_hand_distance_px=args.object_rebaseline_hand_distance,
        object_vacancy_similarity_threshold=args.object_vacancy_similarity_threshold,
        object_birth_hand_distance_px=args.object_birth_hand_distance,
        yolo_model=args.yolo_model,
        yolo_confidence=args.yolo_confidence,
        yolo_iou=args.yolo_iou,
        yolo_imgsz=args.yolo_imgsz,
        yolo_classes=parse_csv_tokens(args.yolo_classes),
        yolo_ignore_classes=parse_csv_tokens(args.yolo_ignore_classes) or (),
    )
