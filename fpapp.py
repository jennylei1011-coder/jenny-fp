import io
import os
import re
import shutil
import tempfile
import unicodedata
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import easyocr
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps


st.set_page_config(page_title="JENNY 发票识别", page_icon="📄", layout="wide")

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".pdf"}
UPLOAD_TYPES = ["png", "jpg", "jpeg", "pdf", "bmp", "tiff", "tif", "zip"]
INVOICE_FIELDS = ["发票号码", "开票日期", "金额", "销售方"]
RESULT_REQUIRED_FIELDS = ["发票号码", "开票日期", "金额（元）", "销售方"]
MAX_ZIP_FILES = 300
MAX_ZIP_TOTAL_SIZE = 800 * 1024 * 1024
MAX_PDF_PAGES = 8


def apply_custom_styles():
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #f7f8f5;
            --panel-bg: #ffffff;
            --text-main: #1f2923;
            --text-muted: #607066;
            --line: #d9e0d8;
            --accent: #2f6f5e;
            --accent-strong: #245647;
        }

        .stApp {
            background: var(--app-bg);
            color: var(--text-main);
        }

        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }

        [data-testid="stSidebar"] {
            background: #fbfcfa;
            border-right: 1px solid var(--line);
        }

        .app-hero {
            border-bottom: 1px solid var(--line);
            margin-bottom: 1.25rem;
            padding-bottom: 1rem;
        }

        .app-kicker {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            margin-bottom: 0.25rem;
            text-transform: uppercase;
        }

        .app-title {
            color: var(--text-main);
            font-size: clamp(2rem, 3.2vw, 3.25rem);
            font-weight: 760;
            letter-spacing: 0;
            line-height: 1.05;
            margin: 0;
        }

        .app-subtitle {
            color: var(--text-muted);
            font-size: 1rem;
            line-height: 1.7;
            max-width: 720px;
            margin-top: 0.75rem;
        }

        .metric-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1rem 0 1.25rem;
        }

        .metric-item {
            background: var(--panel-bg);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            min-height: 74px;
        }

        .metric-label {
            color: var(--text-muted);
            font-size: 0.78rem;
            margin-bottom: 0.25rem;
        }

        .metric-value {
            color: var(--text-main);
            font-size: 1.45rem;
            font-weight: 740;
            line-height: 1.1;
            word-break: break-word;
        }

        .stButton > button,
        .stDownloadButton > button {
            border: 1px solid var(--accent);
            border-radius: 8px;
            background: var(--accent);
            color: #ffffff;
            font-weight: 700;
            min-height: 2.75rem;
            box-shadow: none;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--accent-strong);
            background: var(--accent-strong);
            color: #ffffff;
        }

        [data-testid="stFileUploader"] section {
            background: var(--panel-bg);
            border: 1px dashed #9aaba1;
            border-radius: 8px;
            padding: 0.85rem;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid var(--line);
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        @media (max-width: 760px) {
            [data-testid="stAppViewContainer"] > .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .metric-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="正在加载本地识别模型...")
def get_reader():
    model_dir = Path(__file__).resolve().parent / "model"
    return easyocr.Reader(
        ["ch_sim", "en"],
        gpu=False,
        model_storage_directory=str(model_dir),
        download_enabled=False,
    )


def preprocess_image(image_pil):
    image_pil = ImageOps.exif_transpose(image_pil).convert("RGB")
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    return Image.fromarray(processed)


def resize_image(image_np, max_width=1400):
    h, w = image_np.shape[:2]
    if w > max_width:
        ratio = max_width / w
        image_np = cv2.resize(
            image_np,
            (max_width, max(1, int(h * ratio))),
            interpolation=cv2.INTER_AREA,
        )
    return image_np


def normalize_ocr_text(results):
    lines = [str(item).strip() for item in results if str(item).strip()]
    return "\n".join(lines)


def clean_extracted_value(value):
    value = unicodedata.normalize("NFKC", value or "")
    value = re.sub(r"[ \t\r\f\v]+", "", value)
    value = re.sub(r"[|｜]+", "", value)
    return value.strip(" :：,，.;；")


def parse_invoice_date(text):
    patterns = [
        r"(\d{4})\s*[年\-/\.]\s*(\d{1,2})\s*[月\-/\.]\s*(\d{1,2})\s*日?",
        r"(\d{4})(\d{2})(\d{2})",
    ]
    for pattern in patterns:
        for year, month, day in re.findall(pattern, text):
            if not 1990 <= int(year) <= 2100:
                continue
            try:
                parsed = datetime(int(year), int(month), int(day))
            except ValueError:
                continue
            return parsed.strftime("%Y%m%d")
    return ""


def parse_invoice_number(text):
    compact = clean_extracted_value(text)
    patterns = [
        r"(?:发票号码|票据号码|No\.?|NO\.?)[:：]?\s*([A-Z0-9]{6,24})",
        r"(?:号码)[:：]?\s*([0-9]{8,24})",
    ]
    for pattern in patterns:
        match = re.search(pattern, compact, re.IGNORECASE)
        if match:
            return re.sub(r"\D", "", match.group(1)) or match.group(1)
    return ""


def parse_amount(text):
    compact = clean_extracted_value(text)
    amount_pattern = r"([0-9]{1,3}(?:,[0-9]{3})*\.[0-9]{2}|[0-9]+\.[0-9]{2})"
    keyword_patterns = [
        rf"(?:价税合计|小写|合计金额|税价合计|总金额)[^0-9¥￥]{{0,12}}[¥￥]?{amount_pattern}",
        rf"[¥￥]\s*{amount_pattern}",
    ]
    for pattern in keyword_patterns:
        matches = re.findall(pattern, compact)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = value[-1]
            return value.replace(",", "")
    return ""


def clean_name(value, max_len=60):
    if not value:
        return "未知"
    value = unicodedata.normalize("NFKC", str(value))
    value = re.sub(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\u2028-\u202f]", "", value)
    value = re.sub(r'[\\/*?:"<>|]', "_", value)
    value = re.sub(r"\s+", "", value)
    value = value.strip().rstrip(".")
    if not value:
        return "未知"
    return value[:max_len]


def trim_company_name(value):
    value = clean_extracted_value(value)
    value = re.split(
        r"(?:纳税人识别号|统一社会信用代码|地址|电话|开户行|账号|购买方|销售方|密码区|备注)",
        value,
        maxsplit=1,
    )[0]
    match = re.search(
        r"(.{2,60}?(?:有限责任公司|股份有限公司|有限公司|公司|事务所|集团|中心|厂|店))",
        value,
    )
    if match:
        value = match.group(1)
    return clean_name(value, 60) if value else ""


def parse_seller(text):
    compact_lines = [clean_extracted_value(line) for line in text.splitlines() if line.strip()]
    compact_text = "\n".join(compact_lines)

    seller_section = re.search(
        r"(?:销售方|销货方|收款方|销售单位)(.{0,220})",
        compact_text,
        re.DOTALL,
    )
    if seller_section:
        section = seller_section.group(1)
        match = re.search(r"(?:名称|名\s*称)[:：]?([^\n]{2,80})", section)
        if not match:
            match = re.search(
                r"([^\n]{2,80}?(?:有限责任公司|股份有限公司|有限公司|公司|事务所|集团|中心|厂|店))",
                section,
            )
        if match:
            seller = trim_company_name(match.group(1))
            if seller:
                return seller

    name_matches = re.findall(r"(?:名称|名\s*称)[:：]?([^\n]{2,80})", compact_text)
    candidates = [trim_company_name(item) for item in name_matches]
    candidates = [item for item in candidates if item]
    if len(candidates) >= 2:
        return candidates[-1]
    if candidates:
        return candidates[0]
    return ""


def extract_invoice_info(text):
    return {
        "发票号码": parse_invoice_number(text),
        "开票日期": parse_invoice_date(text),
        "金额": parse_amount(text),
        "销售方": parse_seller(text),
    }


def generate_new_name(info, original_ext):
    number = clean_name(info.get("发票号码") or "未知", 24)
    date = clean_name(info.get("开票日期") or "未知", 8)
    amount = clean_name(info.get("金额") or "未知", 14)
    seller = clean_name(info.get("销售方") or "未知", 48)
    base = clean_name(f"{number}_{date}_{amount}_{seller}", 180)
    return f"{base}{original_ext.lower()}"


def get_archive_folder(info, archive_mode, custom_field=None):
    if archive_mode == "不归档":
        return ""
    if archive_mode == "按月份":
        date = info.get("开票日期", "")
        return date[:6] if len(date) >= 6 else "未知月份"
    if archive_mode == "按销售方":
        return clean_name(info.get("销售方", ""), 50)
    if archive_mode == "自定义字段" and custom_field:
        return clean_name(info.get(custom_field, ""), 50)
    return ""


def make_unique_filename(folder, filename):
    folder_path = Path(folder)
    candidate = folder_path / filename
    if not candidate.exists():
        return filename

    stem = candidate.stem
    suffix = candidate.suffix
    for index in range(2, 1000):
        next_name = f"{stem}_{index}{suffix}"
        if not (folder_path / next_name).exists():
            return next_name
    return f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"


def read_image_text(reader, image, use_preprocess):
    image = ImageOps.exif_transpose(image)
    if use_preprocess:
        image = preprocess_image(image)
    else:
        image = image.convert("RGB")
    image_np = resize_image(np.array(image), max_width=1400)
    results = reader.readtext(image_np, detail=0, paragraph=False)
    return normalize_ocr_text(results)


def ocr_file(file_bytes, filename, use_preprocess):
    reader = get_reader()
    suffix = Path(filename).suffix.lower()

    if suffix in SUPPORTED_EXTENSIONS - {".pdf"}:
        try:
            image = Image.open(io.BytesIO(file_bytes))
            ocr_text = read_image_text(reader, image, use_preprocess)
        except Exception as exc:
            raise RuntimeError(f"图片读取或识别失败：{exc}") from exc
    elif suffix == ".pdf":
        try:
            images = convert_from_bytes(
                file_bytes,
                dpi=220,
                first_page=1,
                last_page=MAX_PDF_PAGES,
            )
        except Exception as exc:
            raise RuntimeError(f"PDF 转换失败，请确认已安装 poppler：{exc}") from exc

        all_text = []
        for image in images:
            all_text.append(read_image_text(reader, image, use_preprocess))
        ocr_text = "\n".join(item for item in all_text if item)
    else:
        raise RuntimeError("不支持的文件格式")

    return ocr_text, extract_invoice_info(ocr_text)


def extract_files_from_zip(zip_bytes):
    files = []
    total_size = 0
    skipped = 0

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue

                suffix = Path(member.filename).suffix.lower()
                if suffix not in SUPPORTED_EXTENSIONS:
                    skipped += 1
                    continue

                total_size += member.file_size
                if len(files) >= MAX_ZIP_FILES or total_size > MAX_ZIP_TOTAL_SIZE:
                    skipped += 1
                    continue

                files.append((zf.read(member), clean_name(Path(member.filename).name, 160)))
    except zipfile.BadZipFile as exc:
        raise RuntimeError("ZIP 文件无法读取或已经损坏") from exc

    return files, skipped


def render_header():
    st.markdown(
        """
        <div class="app-hero">
            <div class="app-kicker">Invoice OCR Workspace</div>
            <h1 class="app-title">JENNY 发票识别</h1>
            <div class="app-subtitle">
                批量识别图片、PDF 或 ZIP 中的发票，提取号码、日期、金额和销售方，并生成可下载的归档文件。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(records):
    df = pd.DataFrame(records)
    total = len(df)
    full_hits = int((df[RESULT_REQUIRED_FIELDS] != "").all(axis=1).sum()) if total else 0
    missing = int((df[RESULT_REQUIRED_FIELDS] == "").any(axis=1).sum()) if total else 0
    failed = int(df["备注"].astype(str).str.contains("失败|错误", regex=True).sum()) if total else 0
    amount_sum = pd.to_numeric(df["金额（元）"], errors="coerce").sum() if total else 0

    st.markdown(
        f"""
        <div class="metric-strip">
            <div class="metric-item"><div class="metric-label">处理文件</div><div class="metric-value">{total}</div></div>
            <div class="metric-item"><div class="metric-label">完整识别</div><div class="metric-value">{full_hits}</div></div>
            <div class="metric-item"><div class="metric-label">待核对</div><div class="metric-value">{missing + failed}</div></div>
            <div class="metric-item"><div class="metric-label">金额合计</div><div class="metric-value">{amount_sum:,.2f}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_excel_bytes(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="发票汇总")
        worksheet = writer.sheets["发票汇总"]
        for column_cells in worksheet.columns:
            max_length = max(len(str(cell.value or "")) for cell in column_cells)
            worksheet.column_dimensions[column_cells[0].column_letter].width = min(
                max(max_length + 2, 12),
                42,
            )
    buffer.seek(0)
    return buffer.getvalue()


def reset_previous_results():
    old_dir = st.session_state.get("temp_dir")
    if old_dir:
        shutil.rmtree(old_dir, ignore_errors=True)
    st.session_state.pop("records", None)
    st.session_state.pop("temp_dir", None)
    st.session_state.pop("results_ready", None)


def process_uploads(uploaded_files, archive_mode, custom_field, use_preprocess):
    reset_previous_results()

    to_process = []
    zip_messages = []

    for uploaded_file in uploaded_files:
        name = uploaded_file.name
        data = uploaded_file.read()
        if name.lower().endswith(".zip"):
            extracted, skipped = extract_files_from_zip(data)
            zip_messages.append((name, len(extracted), skipped))
            to_process.extend(extracted)
        else:
            to_process.append((data, clean_name(name, 160)))

    if not to_process:
        raise RuntimeError("没有可处理的发票文件")

    temp_dir = tempfile.mkdtemp(prefix="jenny_invoice_")
    records = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(to_process)

    for index, (file_bytes, original_name) in enumerate(to_process, start=1):
        status_text.text(f"正在处理 {index}/{total}：{original_name}")
        original_ext = Path(original_name).suffix.lower()
        note = ""
        ocr_text = ""

        try:
            ocr_text, info = ocr_file(file_bytes, original_name, use_preprocess)
        except Exception as exc:
            info = {field: "" for field in INVOICE_FIELDS}
            note = str(exc)

        new_name = generate_new_name(info, original_ext)
        subfolder = get_archive_folder(info, archive_mode, custom_field)
        subfolder_clean = "" if not subfolder else clean_name(subfolder, 80)
        dest_folder = Path(temp_dir) / subfolder_clean if subfolder_clean else Path(temp_dir)
        dest_folder.mkdir(parents=True, exist_ok=True)

        try:
            saved_name = make_unique_filename(dest_folder, new_name)
            (dest_folder / saved_name).write_bytes(file_bytes)
        except Exception as exc:
            saved_name = f"invoice_{uuid.uuid4().hex[:8]}{original_ext}"
            try:
                (dest_folder / saved_name).write_bytes(file_bytes)
                note = f"{note}；原文件名保存失败，已使用备用名称：{exc}".strip("；")
            except Exception as save_exc:
                saved_name = "保存失败"
                note = f"{note}；保存失败：{save_exc}".strip("；")

        missing_fields = [field for field in INVOICE_FIELDS if not info.get(field)]
        if missing_fields:
            note = f"{note}；待核对：{','.join(missing_fields)}".strip("；")

        records.append(
            {
                "原文件名": original_name,
                "新文件名": saved_name,
                "归档文件夹": subfolder_clean if subfolder_clean else "根目录",
                "发票号码": info["发票号码"],
                "开票日期": info["开票日期"],
                "金额（元）": info["金额"],
                "销售方": info["销售方"],
                "备注": note,
                "OCR文本片段": ocr_text[:180],
            }
        )
        progress_bar.progress(index / total)

    status_text.text("处理完成")
    return records, temp_dir, zip_messages


def render_downloads(df, temp_dir):
    col1, col2, col3 = st.columns(3)
    with col1:
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下载 CSV",
            data=csv,
            file_name="JENNY_发票汇总.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "下载 Excel",
            data=build_excel_bytes(df),
            file_name="JENNY_发票汇总.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with col3:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(temp_dir):
                for filename in files:
                    full_path = Path(root) / filename
                    zf.write(full_path, full_path.relative_to(temp_dir))
        st.download_button(
            "下载归档 ZIP",
            data=zip_buffer.getvalue(),
            file_name="JENNY_归档发票.zip",
            mime="application/zip",
            use_container_width=True,
        )


def main():
    apply_custom_styles()
    render_header()

    with st.sidebar:
        st.header("处理设置")
        archive_mode = st.selectbox(
            "归档方式",
            ["不归档", "按月份", "按销售方", "自定义字段"],
            help="识别后的文件会按所选维度放入不同文件夹。",
        )
        custom_field = None
        if archive_mode == "自定义字段":
            custom_field = st.selectbox("归档字段", INVOICE_FIELDS)
        use_preprocess = st.checkbox(
            "图像增强",
            value=True,
            help="适合手机拍摄、灰底或轻微模糊的发票；清晰扫描件可关闭以加快处理。",
        )

    uploaded_files = st.file_uploader(
        "选择发票文件",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        help="支持图片、PDF，以及包含这些文件的 ZIP。",
    )

    if uploaded_files:
        st.caption(f"已选择 {len(uploaded_files)} 个上传项")

    start = st.button(
        "开始识别",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if start:
        try:
            records, temp_dir, zip_messages = process_uploads(
                uploaded_files,
                archive_mode,
                custom_field,
                use_preprocess,
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state["records"] = records
            st.session_state["temp_dir"] = temp_dir
            st.session_state["results_ready"] = True

            for filename, count, skipped in zip_messages:
                if skipped:
                    st.warning(f"{filename} 已提取 {count} 个可处理文件，跳过 {skipped} 个文件。")
                else:
                    st.info(f"{filename} 已提取 {count} 个可处理文件。")

    if st.session_state.get("results_ready"):
        records = st.session_state["records"]
        df = pd.DataFrame(records)
        render_metrics(records)

        st.subheader("识别结果")
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "OCR文本片段": st.column_config.TextColumn(width="large"),
                "备注": st.column_config.TextColumn(width="medium"),
            },
        )
        render_downloads(df, st.session_state["temp_dir"])


if __name__ == "__main__":
    main()
