"""Streamlit port of the RSLAB spectrum 👨🏻‍🔬👩🏻‍🔬 (core workflow)."""
from __future__ import annotations
import ast, io, json, csv, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
from matplotlib.colors import to_hex, to_rgb
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks, savgol_filter

st.set_page_config(page_title="RSLAB Spectrum 👨🏻‍🔬👩🏻‍🔬", layout="wide")
st.title("RSLAB Spectrum 👨🏻‍🔬👩🏻‍🔬")

FONT_DIR=Path(__file__).resolve().parent/"fonts"
BUNDLED_FONT_FAMILIES=set()
if FONT_DIR.is_dir():
    registered_paths={str(Path(font.fname).resolve()) for font in font_manager.fontManager.ttflist if getattr(font,"fname",None)}
    for font_path in sorted([*FONT_DIR.glob("*.ttf"),*FONT_DIR.glob("*.otf")]):
        try:
            resolved=str(font_path.resolve())
            if resolved not in registered_paths:
                font_manager.fontManager.addfont(resolved)
                registered_paths.add(resolved)
            BUNDLED_FONT_FAMILIES.add(font_manager.FontProperties(fname=resolved).get_name())
        except Exception as exc:
            st.warning(f"글꼴 파일을 등록하지 못했습니다 ({font_path.name}): {exc}")

def font_family_available(family):
    try:
        font_manager.findfont(font_manager.FontProperties(family=family),fallback_to_default=False)
        return True
    except (ValueError,RuntimeError):
        return False

def apply_plot_font(family):
    fallback=[family]
    if family!="Noto Sans KR" and font_family_available("Noto Sans KR"):
        fallback.append("Noto Sans KR")
    if "DejaVu Sans" not in fallback:
        fallback.append("DejaVu Sans")
    plt.rcParams["font.family"]=fallback

def read_text(upload):
    raw=upload.getvalue()
    for enc in ("utf-8-sig","utf-8","cp949","euc-kr"):
        try: return raw.decode(enc)
        except UnicodeDecodeError: pass
    return raw.decode("utf-8", errors="replace")

def choose_x_column(df):
    """Infer the spectral X column from its header and numeric shape."""
    if df is None or df.shape[1] < 2:
        raise ValueError("X/Y로 판단할 컬럼이 2개 이상 필요합니다.")
    headers=[str(c).strip().lower() for c in df.columns]
    x_hints=("raman shift","wavenumber","wave number","wavelength","lambda",
             "spectral axis","x-axis","x axis","xcoordinate","x coordinate","x (","cm-1","cm⁻¹")
    y_hints=("intensity","absorbance","counts","signal","fluorescence","y-axis","y axis")
    explicit=[]
    for i,h in enumerate(headers):
        compact=re.sub(r"[^a-z0-9µ⁻]","",h)
        if any(token in h for token in x_hints) or compact in {"x","xnm","ramanshift","wavelengthnm","nm","cm1"}:
            if not any(token in h for token in y_hints):
                explicit.append(i)
    if explicit:
        xi=explicit[0]
    else:
        scores=[]
        for i in range(df.shape[1]):
            values=pd.to_numeric(df.iloc[:,i].astype(str).str.replace(",","",regex=False),errors="coerce").to_numpy(float)
            good=values[np.isfinite(values)]
            if len(good)<2:
                scores.append((-1.0,i)); continue
            coverage=len(good)/max(1,len(values))
            diffs=np.diff(good)
            nonzero=diffs[np.isfinite(diffs)&(diffs!=0)]
            unique=len(np.unique(good))/len(good)
            if len(diffs):
                monotonic=max(np.mean(diffs>0),np.mean(diffs<0))
            else:
                monotonic=0.0
            if len(nonzero)>=2 and np.mean(np.abs(nonzero))>0:
                cv=float(np.std(np.abs(nonzero))/np.mean(np.abs(nonzero)))
                regularity=1.0/(1.0+cv)
            else:
                regularity=0.0
            header_penalty=0.35 if any(token in headers[i] for token in y_hints) else 0.0
            score=0.38*coverage+0.30*monotonic+0.22*unique+0.10*regularity-header_penalty
            scores.append((score,i))
        xi=max(scores)[1]
