import streamlit as st
import easyocr
import pandas as pd
import re
import os
import tempfile
import shutil
import unicodedata
import uuid
from pathlib import Path
from PIL import Image
import io
import zipfile
import numpy as np
import cv2
from pdf2image import convert_from_bytes

# -------------------------- 页面配置 --------------------------
st.set_page_config(page_title="发票识别重命名工具", layout="wide")
st.title("📋 发票批量识别、重命名与汇总表格生成")
st.markdown(
    "上传发票图片或 PDF（或包含它们的 ZIP 压缩包），自动提取关键信息，"
    "按规范重命名并导出表格。\n\n"
    "💡 **提示**：请尽量上传清晰的图片，避免反光、倾斜和阴影；"
    "对手机拍摄的发票，可开启「图像增强」提升识别率。"
    "\n\n📁 **上传整个文件夹？** 请先将文件夹压缩为 `.zip` 再上传。"
)

# -------------------------- 缓存 EasyOCR Reader（使用本地模型） --------------------------
@st.cache_resource
def get_reader():
    # 模型文件放在仓库根目录下的 model 文件夹内
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    return easyocr.Reader(
        ['ch_sim', 'en'],
        gpu=False,
        model_storage_directory=model_dir,
        download_enabled=False   # 禁止在线下载，直接使用本地模型
    )

# -------------------------- 图像预处理 --------------------------
def preprocess_image(image_pil):
    """将 PIL 图像转为自适应二值化图像，提高低质量图像的识别率"""
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    return Image.fromarray(processed)

def resize_image(image_np, max_width=1200):
    """等比缩小图片，降低 OCR 耗时，几乎不影响识别率"""
    h, w = image_np.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_w = max_width
        new_h = int(h * ratio)
        image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image_np

# -------------------------- 发票信息提取 --------------------------
def extract_invoice_info(text):
    info = {
        "发票号码": "",
        "开票日期": "",
        "金额": "",
        "销售方": ""
    }
    # 发票号码
    match = re.search(r'(?:发票号码|No)[：: ]*([\d]+)', text, re.IGNORECASE)
    if match:
        info["发票号码"] = match.group(1)
    # 开票日期
    match = re.search(r'(\d{4}[年\-]\d{1,2}[月\-]\d{1,2}[日]?)', text)
    if match:
        date_str = match.group(1)
        date_str = date_str.replace('年', '').replace('月', '').replace('日', '')
        date_str = date_str.replace('-', '')
        if len(date_str) == 8:
            info["开票日期"] = date_str
    # 金额
    match = re.search(r'(?:价税合计|合计金额|小写)[^\d]*[¥￥]?([\d,]+\.\d{2})', text)
    if not match:
        match = re.search(r'[¥￥]\s*([\d,]+\.\d{2})', text)
    if match:
        info["金额"] = match.group(1).replace(',', '')
    # 销售方
    match = re.search(r'名\s*称[：:]\s*([^\n]+)', text)
    if match:
        name = match.group(1).strip()
        info["销售方"] = re.sub(r'\s+', '', name)
    return info

# -------------------------- 安全文件名处理 --------------------------
def clean_name(s, max_len=60):
    if not s:
        return "未知"
    s = unicodedata.normalize('NFKC', s)
    s = re.sub(r'[\x00-\x1f\x7f-\x9f\u200b-\u200f\u2028-\u202f]', '', s)
    s = re.sub(r'[\\/*?:"<>|]', '_', s)
    s = s.strip().rstrip('.')
    if not s:
        return "未知"
    return s[:max_len]

def generate_new_name(info, original_ext):
    number = clean_name(info.get("发票号码") or "未知", 20)
    date = clean_name(info.get("开票日期") or "未知", 8)
    amount = clean_name(info.get("金额") or "未知", 10)
    seller = clean_name(info.get("销售方") or "未知", 30)
    base = f"{number}_{date}_{amount}_{seller}"
    base = clean_name(base, 180)
    return f"{base}{original_ext}"

# -------------------------- 归档文件夹 --------------------------
def get_archive_folder(info, archive_mode, custom_field=None):
    if archive_mode == "不归档":
        return ""
    elif archive_mode == "按月份":
        date = info.get("开票日期", "")
        if date and len(date) >= 6:
            return date[:6]
        return "未知月份"
    elif archive_mode == "按销售方":
        seller = info.get("销售方", "")
        return clean_name(seller, 50) if seller else "未知销售方"
    elif archive_mode == "自定义字段" and custom_field:
        value = info.get(custom_field, "")
        return clean_name(value, 50) if value else f"未知{custom_field}"
    return ""

# -------------------------- OCR 处理单个文件 --------------------------
def ocr_file(file_bytes, filename, use_preprocess):
    reader = get_reader()
    suffix = Path(filename).suffix.lower()
    ocr_text = ""

    if suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
        image = Image.open(io.BytesIO(file_bytes))
        if use_preprocess:
            image = preprocess_image(image)
        image_np = np.array(image)
        image_np = resize_image(image_np, max_width=1200)
        results = reader.readtext(image_np, detail=0)
        ocr_text = " ".join(results)

    elif suffix == '.pdf':
        try:
            images = convert_from_bytes(file_bytes)
        except Exception as e:
            st.error(f"PDF 转换失败: {e}。请确保服务器已安装 poppler。")
            return "", {}
        all_text = []
        for img in images:
            if use_preprocess:
                img = preprocess_image(img)
            img_np = np.array(img)
            img_np = resize_image(img_np, max_width=1200)
            results = reader.readtext(img_np, detail=0)
            all_text.extend(results)
        ocr_text = " ".join(all_text)

    info = extract_invoice_info(ocr_text)
    return ocr_text, info

# -------------------------- 从 ZIP 中提取文件 --------------------------
def extract_files_from_zip(zip_bytes):
    supported_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.pdf'}
    files = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            ext = Path(member.filename).suffix.lower()
            if ext in supported_ext:
                data = zf.read(member)
                fname = Path(member.filename).name
                files.append((data, fname))
    return files

# -------------------------- 主界面 --------------------------
def main():
    with st.sidebar:
        st.header("⚙️ 设置")
        archive_mode = st.selectbox(
            "归档方式",
            ["不归档", "按月份", "按销售方", "自定义字段"],
            help="重命名后的文件将按所选维度放入不同文件夹。"
        )
        custom_field = None
        if archive_mode == "自定义字段":
            field_options = ["发票号码", "开票日期", "金额", "销售方"]
            custom_field = st.selectbox("选择归档字段", field_options)
        use_preprocess = st.checkbox(
            "🛠️ 启用图像增强",
            value=True,
            help="对手机拍摄或扫描件进行二值化，可能提高识别率，但会稍慢。"
        )

    uploaded_files = st.file_uploader(
        "📤 选择发票文件（可多选，支持 JPG/PNG/PDF/BMP/TIFF/ZIP）",
        type=["png", "jpg", "jpeg", "pdf", "bmp", "tiff", "tif", "zip"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🚀 开始识别并处理", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 准备待处理列表
            to_process = []
            for uf in uploaded_files:
                name = uf.name
                data = uf.read()
                if name.lower().endswith('.zip'):
                    extracted = extract_files_from_zip(data)
                    if extracted:
                        st.info(f"已从 `{name}` 中提取 {len(extracted)} 个文件")
                        to_process.extend(extracted)
                    else:
                        st.warning(f"ZIP 文件 `{name}` 中未找到支持的发票文件。")
                else:
                    to_process.append((data, name))

            if not to_process:
                st.error("没有可处理的发票文件。")
                st.stop()

            total = len(to_process)
            records = []
            temp_dir = tempfile.mkdtemp()

            for idx, (file_bytes, original_name) in enumerate(to_process):
                status_text.text(f"正在处理: {original_name} ({idx+1}/{total})")
                original_ext = Path(original_name).suffix

                ocr_text, info = ocr_file(file_bytes, original_name, use_preprocess)
                new_name = generate_new_name(info, original_ext)

                # 归档子文件夹
                subfolder = get_archive_folder(info, archive_mode, custom_field)
                subfolder_clean = clean_name(subfolder, 80) if subfolder else ""
                dest_folder = os.path.join(temp_dir, subfolder_clean) if subfolder_clean else temp_dir

                os.makedirs(dest_folder, exist_ok=True)

                # 安全保存文件
                saved_name = None
                save_note = ""
                try:
                    new_path = os.path.join(dest_folder, new_name)
                    with open(new_path, "wb") as f:
                        f.write(file_bytes)
                    saved_name = new_name
                except Exception as e:
                    fallback = f"invoice_{uuid.uuid4().hex[:8]}{original_ext}"
                    fallback_path = os.path.join(dest_folder, fallback)
                    try:
                        with open(fallback_path, "wb") as f:
                            f.write(file_bytes)
                        saved_name = fallback
                        save_note = f"文件名无效已重命名，原错误：{e}"
                    except Exception as e2:
                        save_note = f"保存完全失败：{e2}"
                        saved_name = "保存失败"

                records.append({
                    "原文件名": original_name,
                    "新文件名": saved_name,
                    "归档文件夹": subfolder_clean if subfolder_clean else "根目录",
                    "发票号码": info["发票号码"],
                    "开票日期": info["开票日期"],
                    "金额（元）": info["金额"],
                    "销售方": info["销售方"],
                    "备注": save_note,
                    "OCR文本片段": ocr_text[:150]
                })

                progress_bar.progress((idx + 1) / total)

            status_text.text("✅ 处理完成！")
            st.success(f"共处理 {total} 个文件")

            # 汇总表
            st.subheader("📊 提取结果汇总")
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True)

            # 下载
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 下载汇总表格 (CSV)", data=csv,
                                   file_name="发票汇总.csv", mime="text/csv")
            with col2:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            full = os.path.join(root, file)
                            arcname = os.path.relpath(full, temp_dir)
                            zf.write(full, arcname)
                zip_buffer.seek(0)
                st.download_button("📦 下载全部文件 (ZIP)", data=zip_buffer,
                                   file_name="归档发票.zip", mime="application/zip")

            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
