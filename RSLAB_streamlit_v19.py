"""Streamlit port of the RSLAB spectrum  (core workflow)."""
from __future__ import annotations
import ast, io, json, csv, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
from matplotlib.colors import to_hex
from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks, savgol_filter

st.set_page_config(page_title="RSLAB Spectrum ", layout="wide")
st.title("RSLAB Spectrum ")

def read_text(upload):
    raw=upload.getvalue()
    for enc in ("utf-8-sig","utf-8","cp949","euc-kr"):
        try: return raw.decode(enc)
        except UnicodeDecodeError: pass
    return raw.decode("utf-8", errors="replace")

def parse_upload(upload):
    """Parse ordinary tables and files containing repeated spectrum blocks."""
    text=read_text(upload)
    lines=text.splitlines()
    x_terms=("wavelength","raman","shift","wavenumber","wave number","cm-1","cm⁻¹")
    header_rows=[]
    for i,line in enumerate(lines):
        low=line.lower()
        counts={d:line.count(d) for d in (",","\t",";")}
        sep=max(counts,key=counts.get)
        if counts[sep] and any(term in low for term in x_terms):
            header_rows.append((i,sep))

    series=[]
    file_stem=Path(upload.name).stem

    def row_is_numeric(line,sep):
        try:
            fields=next(csv.reader([line],delimiter=sep))
            values=pd.to_numeric(pd.Series(fields),errors="coerce")
            return len(fields)>=2 and values.notna().all()
        except Exception:
            return False

    # Wide mapping exports store one spectrum per row: a spectral-axis header,
    # followed by a time coordinate or X/Y coordinates and then intensity values.
    first_idx=next((i for i,line in enumerate(lines) if line.strip()),None)
    if first_idx is not None and "\t" in lines[first_idx]:
        header_cells=lines[first_idx].split("\t")
        coordinate_columns=0
        while coordinate_columns<len(header_cells) and not header_cells[coordinate_columns].strip():
            coordinate_columns+=1
        axis_values=pd.to_numeric(pd.Series(header_cells[coordinate_columns:]),errors="coerce")
        possible_mapping=(coordinate_columns in (1,2) and len(axis_values)>=50 and
                          axis_values.notna().all() and np.all(np.diff(axis_values.to_numpy(float))>0))
        if possible_mapping:
            axis_values=axis_values.to_numpy(float)
            mapped=[]
            for line in lines[first_idx+1:]:
                if not line.strip():
                    continue
                cells=line.split("\t")
                if len(cells)<coordinate_columns+len(axis_values):
                    continue
                coords=pd.to_numeric(pd.Series(cells[:coordinate_columns]),errors="coerce")
                intensities=pd.to_numeric(pd.Series(cells[coordinate_columns:coordinate_columns+len(axis_values)]),errors="coerce")
                if coords.notna().all() and intensities.notna().sum()>=2:
                    mapped.append((coords.to_numpy(float),intensities.to_numpy(float)))
            if mapped:
                for coords,intensities in mapped:
                    if coordinate_columns==1:
                        point_label=f"시간={coords[0]:g}"
                        map_type="time"
                        map_coordinate=float(coords[0])
                    else:
                        point_label=f"X={coords[0]:g}, Y={coords[1]:g}"
                        map_type="space"
                        map_coordinate=(float(coords[0]),float(coords[1]))
                    good=np.isfinite(axis_values) & np.isfinite(intensities)
                    if good.sum()>=2:
                        series.append({"name":f"{file_stem} · {point_label}",
                                       "x":axis_values[good],"raw":intensities[good],
                                       "x_label":"Raman Shift (cm⁻¹)","y_label":"Intensity",
                                       "mapping_type":map_type,"mapping_coordinate":map_coordinate,
                                       "mapping_source":file_stem})
                if series:
                    return series

    # Instrument exports may put key=value acquisition metadata before XYDATA=.
    # Parse that section explicitly instead of treating metadata lines as table rows.
    xydata_idx=next((i for i,line in enumerate(lines) if line.strip().upper()=="XYDATA="),None)
    if xydata_idx is not None:
        metadata={}
        for line in lines[:xydata_idx]:
            if "=" in line:
                key,value=line.split("=",1)
                metadata[key.strip().upper()]=value.strip()
        data_lines=[line.strip() for line in lines[xydata_idx+1:] if row_is_numeric(line.strip(),",")]
        if data_lines:
            data_df=pd.read_csv(io.StringIO("\n".join(data_lines)),header=None,sep=",",engine="python")
            if data_df.shape[1]>=2:
                x=pd.to_numeric(data_df.iloc[:,0],errors="coerce")
                y=pd.to_numeric(data_df.iloc[:,1],errors="coerce")
                valid=x.notna() & y.notna()
                x_type=(metadata.get("TYPE","")+" "+metadata.get("XUNITS","")).lower()
                is_uv=any(token in x_type for token in ("wavelength","nm"))
                series=[{"name":file_stem,"x":x[valid].to_numpy(float),"raw":y[valid].to_numpy(float),
                         "x_label":"Wavelength (nm)" if is_uv else "Raman Shift (cm⁻¹)",
                         "y_label":"Absorbance" if is_uv else "Intensity"}]
                if len(series[0]["x"])>=2:
                    return series

    def sample_label(start,header_idx,sep):
        for j in range(header_idx-1,start-1,-1):
            candidate=lines[j].strip().strip('"').strip()
            if not candidate or candidate.startswith("#"):
                continue
            if re.match(r"^\d{1,2}/\d{1,2}/\d{4}(\s|$)",candidate) or row_is_numeric(candidate,sep):
                continue
            return candidate,j
        return file_stem,None

    if header_rows:
        labels=[]
        for i,(header_idx,sep) in enumerate(header_rows):
            search_start=header_rows[i-1][0]+1 if i else 0
            labels.append(sample_label(search_start,header_idx,sep))

        for i,(header_idx,sep) in enumerate(header_rows):
            label,_=labels[i]
            if i+1<len(header_rows):
                next_label_idx=labels[i+1][1]
                block_end=next_label_idx if next_label_idx is not None else header_rows[i+1][0]
            else:
                block_end=len(lines)
            block_text="\n".join(lines[header_idx:block_end])
            df=pd.read_csv(io.StringIO(block_text),sep=sep,comment="#",on_bad_lines="skip")
            df=df.dropna(axis=1,how="all").dropna(axis=0,how="all")
            if len(df.columns)<2:
                continue
            norm=[str(c).strip().lower() for c in df.columns]
            xi=next((j for j,n in enumerate(norm) if any(k in n for k in ("raman","shift","wavelength","lambda","nm","x"))),0)
            x=pd.to_numeric(df.iloc[:,xi].astype(str).str.replace(",","",regex=False),errors="coerce")
            x_label="Wavelength (nm)" if any(k in norm[xi] for k in ("wavelength","nm")) else "Raman Shift (cm⁻¹)"
            y_columns=[j for j in range(len(df.columns)) if j!=xi]
            for j in y_columns:
                y=pd.to_numeric(df.iloc[:,j].astype(str).str.replace(",","",regex=False),errors="coerce")
                good=x.notna() & y.notna()
                if good.sum()<2:
                    continue
                curve_name=label if len(y_columns)==1 else f"{label} · {df.columns[j]}"
                if any(existing["name"]==curve_name for existing in series):
                    curve_name=f"{curve_name} ({i+1})"
                series.append({"name":curve_name,"x":x[good].to_numpy(float),"raw":y[good].to_numpy(float),
                               "x_label":x_label,"y_label":"Absorbance" if x_label=="Wavelength (nm)" else "Intensity"})
    else:
        # Retain support for headerless numeric CSV/TSV files.
        try: df=pd.read_csv(io.StringIO(text),sep=None,engine="python",comment="#",on_bad_lines="skip")
        except Exception: df=pd.read_csv(io.StringIO(text),header=None,sep=None,engine="python",comment="#",on_bad_lines="skip")
        numeric_header=len(df.columns)>=2 and all(pd.to_numeric(pd.Series([str(c).replace(",","")]),errors="coerce").notna().iloc[0] for c in df.columns[:2])
        if len(df.columns)<2 or len(df)<2 or numeric_header:
            df=pd.read_csv(io.StringIO(text),header=None,sep=None,engine="python",comment="#",on_bad_lines="skip")
        df=df.dropna(axis=1,how="all").dropna(axis=0,how="all")
        if len(df.columns)>=2:
            norm=[str(c).strip().lower() for c in df.columns]
            xi=next((j for j,n in enumerate(norm) if any(k in n for k in ("raman","shift","wavelength","lambda","nm","x"))),0)
            x=pd.to_numeric(df.iloc[:,xi].astype(str).str.replace(",","",regex=False),errors="coerce")
            x_label="Wavelength (nm)" if any(k in norm[xi] for k in ("wavelength","nm")) else "Raman Shift (cm⁻¹)"
            y_columns=[j for j in range(len(df.columns)) if j!=xi]
            for j in y_columns:
                y=pd.to_numeric(df.iloc[:,j].astype(str).str.replace(",","",regex=False),errors="coerce")
                good=x.notna() & y.notna()
                if good.sum()>=2:
                    label=file_stem if len(y_columns)==1 else f"{file_stem} · {df.columns[j]}"
                    series.append({"name":label,"x":x[good].to_numpy(float),"raw":y[good].to_numpy(float),
                                   "x_label":x_label,"y_label":"Absorbance" if x_label=="Wavelength (nm)" else "Intensity"})
    if not series: raise ValueError("사용 가능한 X/Y 숫자 쌍이 없습니다.")
    return series

def baseline(y, method, degree, lam, p, iterations):
    if method=="Polynomial":
        xx=np.arange(len(y)); return y-np.polyval(np.polyfit(xx,y,min(degree,len(y)-1)),xx)
    if method=="AsLS":
        n=len(y)
        # second difference penalty
        from scipy.sparse import diags
        D=diags([1,-2,1],[0,1,2],shape=(n-2,n),format="csc")
        w=np.ones(n)
        for _ in range(iterations):
            W=sparse.spdiags(w,0,n,n)
            z=spsolve(W+lam*(D.T@D),w*y)
            w=p*(y>z)+(1-p)*(y<z)
        return y-z
    return y.copy()

def transform(s, opts):
    y=np.asarray(s["raw"],dtype=float).copy()
    if opts.get("sg_enabled") and len(y)>=3:
        window=int(opts["sg_window"])
        if window%2==0:
            window-=1
        window=min(window,len(y) if len(y)%2 else len(y)-1)
        polyorder=min(int(opts["sg_polyorder"]),window-1)
        if window>=3:
            y=savgol_filter(y,window,polyorder)
    y=baseline(y,opts["baseline"],opts["degree"],opts["lam"],opts["p"],opts["iterations"])
    mode=opts["normalization"]
    if mode=="Min-max":
        lo,hi=np.nanmin(y),np.nanmax(y); y=(y-lo)/(hi-lo) if hi!=lo else y*0
    elif mode=="Maximum" and np.nanmax(y)!=0: y=y/np.nanmax(y)
    elif mode=="Custom divisor" and opts["divisor"]!=0: y=y/opts["divisor"]
    return y

def make_plot(width_px, height_px, size_mode, margins, dpi):
    """Create a plot with either a fixed canvas or fixed inner plot dimensions."""
    left, right, top, bottom = [float(v) for v in margins]
    if min(left, right, top, bottom) < 0:
        raise ValueError("그래프 여백은 0 이상이어야 합니다.")
    if size_mode == "플롯 영역 기준":
        plot_w, plot_h = float(width_px), float(height_px)
        fig_w, fig_h = plot_w + left + right, plot_h + top + bottom
    else:
        fig_w, fig_h = float(width_px), float(height_px)
        plot_w, plot_h = fig_w - left - right, fig_h - top - bottom
    if plot_w <= 0 or plot_h <= 0:
        raise ValueError("그래프 전체 크기는 설정한 여백보다 커야 합니다.")
    fig = plt.figure(figsize=(fig_w / dpi, fig_h / dpi), dpi=dpi)
    ax = fig.add_axes([left / fig_w, bottom / fig_h, plot_w / fig_w, plot_h / fig_h])
    return fig, ax

def render_png_fixed_size(fig, width_px, height_px, render_dpi, base_dpi=100):
    """Supersample at render_dpi, then return a PNG at the exact target size."""
    width_px, height_px = int(width_px), int(height_px)
    if width_px <= 0 or height_px <= 0:
        raise ValueError("PNG 크기는 1 px 이상이어야 합니다.")
    scale = max(1.0, float(render_dpi) / float(base_dpi))
    # Bound temporary render memory for large requested canvases.
    max_render_pixels = 8_000_000
    scale = min(scale, (max_render_pixels / (width_px * height_px)) ** 0.5)
    scale = max(1.0, scale)
    high_res = io.BytesIO()
    fig.savefig(high_res, format="png", dpi=base_dpi * scale)
    high_res.seek(0)
    with Image.open(high_res) as rendered:
        image = rendered.convert("RGBA")
        if image.size != (width_px, height_px):
            image = image.resize((width_px, height_px), Image.Resampling.LANCZOS)
        result = io.BytesIO()
        # The selected DPI is metadata; pixel dimensions remain width_px × height_px.
        image.save(result, format="PNG", dpi=(render_dpi, render_dpi))
    return result.getvalue()

def _eval_tick_expression(node, x_value):
    """Evaluate a small arithmetic expression with x, without eval()."""
    if isinstance(node, ast.Expression):
        return _eval_tick_expression(node.body, x_value)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id == "x":
        return float(x_value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_tick_expression(node.operand, x_value)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        left = _eval_tick_expression(node.left, x_value)
        right = _eval_tick_expression(node.right, x_value)
        if isinstance(node.op, ast.Add): result = left + right
        elif isinstance(node.op, ast.Sub): result = left - right
        elif isinstance(node.op, ast.Mult): result = left * right
        elif isinstance(node.op, ast.Div): result = left / right
        else:
            if abs(right) > 12: raise ValueError("수식의 지수는 -12~12 범위로 입력하세요.")
            result = left ** right
        if isinstance(result,complex) or not np.isfinite(result): raise ValueError("수식 결과가 유효한 실수가 아닙니다.")
        return float(result)
    raise ValueError("수식에는 x와 숫자, + - * / ** 및 괄호만 사용할 수 있습니다.")

def make_tick_formatter(options):
    unit_factors={"없음":1.0,"10^2":1e2,"10^3":1e3,"10^4":1e4,"10^5":1e5,"10^6":1e6,
                  "10^-2":1e-2,"10^-3":1e-3,"10^-4":1e-4,"10^-5":1e-5,"10^-6":1e-6}
    formula_tree=None
    formula=options.get("formula","").strip()
    if formula:
        formula_tree=ast.parse(formula,mode="eval")
        _eval_tick_expression(formula_tree,1.2345)  # validate before drawing
    factor=float(options.get("custom_factor",1.0)) if options.get("unit")=="사용자 지정" else unit_factors.get(options.get("unit","없음"),1.0)
    if factor<=0 or not np.isfinite(factor): raise ValueError("나누기 인수는 0보다 큰 유한한 값이어야 합니다.")

    def format_tick(value, _position):
        try:
            scaled=_eval_tick_expression(formula_tree,value) if formula_tree is not None else float(value)/factor
        except (ArithmeticError, ValueError, OverflowError, TypeError):
            return ""
        if not np.isfinite(scaled): return ""
        if options.get("decimal_enabled"):
            places=int(options.get("decimal_places",2))
            text=f"{abs(scaled):.{places}f}"
        else:
            text=f"{abs(scaled):g}"
        if np.isclose(scaled,0):
            sign=""
        elif scaled<0:
            sign=options.get("minus_sign","−")
        else:
            sign="+" if options.get("plus_sign") else ""
        return f"{options.get('prefix','')}{sign}{text}{options.get('suffix','')}"
    return FuncFormatter(format_tick)

def tick_scale_annotation(options):
    """Return a Matplotlib label showing how displayed Y ticks map to values."""
    if not options.get("show_multiplier") or options.get("formula","").strip():
        return ""
    unit_factors={"없음":1.0,"10^2":1e2,"10^3":1e3,"10^4":1e4,"10^5":1e5,"10^6":1e6,
                  "10^-2":1e-2,"10^-3":1e-3,"10^-4":1e-4,"10^-5":1e-5,"10^-6":1e-6}
    factor=float(options.get("custom_factor",1.0)) if options.get("unit")=="사용자 지정" else unit_factors.get(options.get("unit","없음"),1.0)
    if factor==1 or factor<=0 or not np.isfinite(factor):
        return ""
    exponent=round(np.log10(factor))
    if np.isclose(factor,10.0**exponent,rtol=1e-9,atol=0):
        return rf"$\times 10^{{{exponent}}}$"
    return rf"$\times {factor:g}$"

def add_y_scale_annotation(ax, options, fontsize):
    label=tick_scale_annotation(options)
    if label:
        # Anchor the scale label just above the plotting box's upper-left corner.
        ax.text(0.0,1.005,label,transform=ax.transAxes,ha="left",va="bottom",fontsize=fontsize,clip_on=False)

def apply_graph_frame(ax, visible, linewidth):
    # Keep the X/Y axes visible; the frame toggle controls only the top/right box edges.
    for side, spine in ax.spines.items():
        spine.set_visible(True if side in ("left", "bottom") else visible)
        spine.set_linewidth(linewidth)

def visible_stack_ylim(series, processed, names, plotted_names, auto_offsets, offset_step, style, x_min, x_max):
    """Return padded Y limits from only the plotted points inside the visible X range."""
    limits=[]
    by_name={item["name"]:item for item in series}
    x_min,x_max=sorted((float(x_min),float(x_max)))
    for name in plotted_names:
        item=by_name[name]
        y_values=np.asarray(processed[name],dtype=float)
        mask=(np.isfinite(item["x"]) & np.isfinite(y_values) &
              (item["x"]>=x_min) & (item["x"]<=x_max))
        if np.any(mask):
            offset=auto_offsets[name]+offset_step*names.index(name)+style[name]["offset"]
            limits.extend((y_values[mask]+offset).tolist())
    if not limits:
        return None
    low,high=float(np.nanmin(limits)),float(np.nanmax(limits))
    span=high-low
    padding=0.05*span if span>0 else max(abs(high)*0.05,0.01)
    return low-padding,high+padding

def sync_widget_value(source_key, target_key):
    st.session_state[target_key]=st.session_state[source_key]

def synced_font_control(label, minimum, maximum, default, key):
    input_key=f"{key}_input"
    if key not in st.session_state and input_key not in st.session_state:
        st.session_state[key]=default
        st.session_state[input_key]=default
    elif key not in st.session_state:
        st.session_state[key]=st.session_state[input_key]
    elif input_key not in st.session_state:
        st.session_state[input_key]=st.session_state[key]
    slider_col,input_col=st.columns([5,1])
    slider_col.slider(label,min_value=minimum,max_value=maximum,key=key,
                      on_change=sync_widget_value,args=(key,input_key))
    input_col.number_input(label,min_value=minimum,max_value=maximum,step=1,key=input_key,
                          label_visibility="collapsed",help="숫자를 직접 입력할 수 있습니다.",
                          on_change=sync_widget_value,args=(input_key,key))
    return int(st.session_state[key])

def synced_float_control(label, minimum, maximum, default, step, key, disabled=False):
    input_key=f"{key}_input"
    if key not in st.session_state and input_key not in st.session_state:
        st.session_state[key]=default
        st.session_state[input_key]=default
    elif key not in st.session_state:
        st.session_state[key]=st.session_state[input_key]
    elif input_key not in st.session_state:
        st.session_state[input_key]=st.session_state[key]
    slider_col,input_col=st.columns([5,1])
    slider_col.slider(label,min_value=minimum,max_value=maximum,step=step,key=key,disabled=disabled,
                      on_change=sync_widget_value,args=(key,input_key))
    input_col.number_input(label,min_value=minimum,max_value=maximum,step=step,format="%.1f",key=input_key,
                           disabled=disabled,label_visibility="collapsed",help="숫자를 직접 입력할 수 있습니다.",
                           on_change=sync_widget_value,args=(input_key,key))
    return float(st.session_state[key])

def axis_tick_options_ui(container, axis_name, key_prefix):
    with container:
        st.markdown(f"**{axis_name} 눈금 숫자 표시**")
        unit=st.selectbox("표시 단위 · 값 나누기",["없음","10^2","10^3","10^4","10^5","10^6","10^-2","10^-3","10^-4","10^-5","10^-6","사용자 지정"],key=f"{key_prefix}_unit")
        custom_factor=st.number_input("나누기 인수",min_value=1e-12,value=1000.0,format="%.6g",disabled=unit!="사용자 지정",key=f"{key_prefix}_factor")
        formula=st.text_input("수식 (x=원래 눈금값, 선택)",placeholder="예: x/1000",key=f"{key_prefix}_formula")
        show_multiplier=st.checkbox("Y축 위에 ×10ⁿ 배율 표시",value=True,disabled=(unit=="없음" or bool(formula.strip())),key=f"{key_prefix}_show_multiplier")
        decimal_enabled=st.checkbox("소수 자릿수 지정",value=False,key=f"{key_prefix}_decimal_enabled")
        decimal_places=st.number_input("소수 자릿수",min_value=0,max_value=10,value=2,disabled=not decimal_enabled,key=f"{key_prefix}_decimal_places")
        pcol,scol=st.columns(2)
        prefix=pcol.text_input("접두사",key=f"{key_prefix}_prefix")
        suffix=scol.text_input("접미사",key=f"{key_prefix}_suffix")
        signcol,minuscol=st.columns(2)
        plus_sign=signcol.checkbox("양수에 + 표시",value=False,key=f"{key_prefix}_plus_sign")
        minus_sign=minuscol.text_input("음수 기호",value="−",max_chars=2,key=f"{key_prefix}_minus_sign")
    return {"unit":unit,"custom_factor":custom_factor,"formula":formula,"decimal_enabled":decimal_enabled,
            "decimal_places":decimal_places,"prefix":prefix,"suffix":suffix,"plus_sign":plus_sign,"minus_sign":minus_sign,
            "show_multiplier":show_multiplier}

def format_axis_label(label):
    """Render the inverse-centimeter exponent consistently across fonts."""
    label = str(label or "")
    # Accept common plain-text spellings and use Matplotlib mathtext for -1.
    return re.sub(
        r"cm\s*(?:⁻¹|−1|-1|\^\s*\{?\s*-1\s*\}?)",
        r"cm$^{-1}$",
        label,
        flags=re.IGNORECASE,
    )

def state_json(series):
    data=[]
    for s in series:
        data.append({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in s.items()})
    return json.dumps(data,ensure_ascii=False)

def make_series_names_unique(items, existing_names=()):
    """Give every curve a unique key so per-curve arrays cannot overwrite each other."""
    used=set(existing_names)
    for item in items:
        base=str(item.get("name") or "Spectrum")
        candidate=base
        suffix=2
        while candidate in used:
            candidate=f"{base} ({suffix})"
            suffix+=1
        item["name"]=candidate
        used.add(candidate)
    return items

def export_curve_table(target_names, series, values_by_name):
    """Keep each curve's original X/Y pairs intact in a convenient table."""
    targets=[n for n in target_names if n in values_by_name]
    if not targets:
        return pd.DataFrame()
    lookup={s["name"]:s for s in series}
    first_x=np.asarray(lookup[targets[0]]["x"],dtype=float)
    same_grid=all(
        len(lookup[n]["x"])==len(first_x) and
        np.allclose(np.asarray(lookup[n]["x"],dtype=float),first_x,rtol=0,atol=1e-10,equal_nan=True)
        for n in targets[1:]
    )
    if same_grid:
        data={"X":first_x}
        for n in targets:
            data[n]=np.asarray(values_by_name[n],dtype=float)
        return pd.DataFrame(data)
    data={}
    for n in targets:
        data[f"{n} | X"]=np.asarray(lookup[n]["x"],dtype=float)
        data[f"{n} | Y"]=np.asarray(values_by_name[n],dtype=float)
    return pd.DataFrame({key:pd.Series(value) for key,value in data.items()})

def curve_archive(target_names, series, values_by_name, extension):
    """Pack one exact X/Y data file per curve into a ZIP archive."""
    lookup={s["name"]:s for s in series}
    separator="\t" if extension=="txt" else ","
    archive_buffer=io.BytesIO()
    with zipfile.ZipFile(archive_buffer,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        used=set()
        for index,name in enumerate(target_names,1):
            if name not in values_by_name or name not in lookup:
                continue
            table=pd.DataFrame({"X":lookup[name]["x"],"Y":values_by_name[name]})
            safe_name=re.sub(r'[\\/:*?"<>|]+',"_",str(name)).strip(" ._") or f"spectrum_{index}"
            filename=f"{safe_name}.{extension}"
            if filename in used:
                filename=f"{safe_name}_{index}.{extension}"
            used.add(filename)
            content=table.to_csv(index=False,sep=separator).encode("utf-8-sig")
            archive.writestr(filename,content)
    return archive_buffer.getvalue()

if "series" not in st.session_state: st.session_state.series=[]
st.session_state.series=make_series_names_unique(st.session_state.series)
with st.sidebar:
    st.header("데이터")
    uploads=st.file_uploader("CSV / TSV / TXT 파일",type=["csv","tsv","txt","dat","asc"],accept_multiple_files=True)
    if st.button("파일 불러오기", type="primary", use_container_width=True):
        loaded=[]; errors=[]
        previous_names=[s["name"] for s in st.session_state.series]
        for f in uploads or []:
            try: loaded.extend(parse_upload(f))
            except Exception as e: errors.append(f"{f.name}: {e}")
        loaded=make_series_names_unique(loaded,[s["name"] for s in st.session_state.series])
        st.session_state.series.extend(loaded)
        if loaded:
            selected_before=st.session_state.get("graph_selected",previous_names)
            st.session_state["graph_selected"]=list(dict.fromkeys([*selected_before,*[s["name"] for s in loaded]]))
        for e in errors: st.error(e)
    if st.button("전체 지우기",use_container_width=True):
        st.session_state.series=[]
        st.session_state["graph_selected"]=[]
    if st.session_state.series:
        st.download_button("현재 세션 저장 (.json)",state_json(st.session_state.series),"rslab_session.json","application/json",use_container_width=True)
        session=st.file_uploader("세션 불러오기 (.json)",type="json",key="session_upload")
        if session and st.button("세션 적용"):
            raw=json.loads(session.getvalue().decode("utf-8"))
            restored=[{**s,"x":np.asarray(s["x"]),"raw":np.asarray(s["raw"])} for s in raw]
            st.session_state.series=make_series_names_unique(restored)
            st.session_state["graph_selected"]=[s["name"] for s in st.session_state.series]
            st.rerun()
    if st.session_state.series:
        parsed_names=[s["name"] for s in st.session_state.series]
        aliases=st.session_state.pop("graph_name_aliases",{})
        if "graph_selected" not in st.session_state:
            st.session_state["graph_selected"]=parsed_names.copy()
        else:
            prior=st.session_state["graph_selected"]
            migrated=[aliases.get(n,n) for n in prior]
            valid=[n for n in migrated if n in parsed_names]
            if prior and not valid:
                valid=parsed_names.copy()
            st.session_state["graph_selected"]=valid
        mapped=[s for s in st.session_state.series if s.get("mapping_type") in ("time","space")]

if not st.session_state.series:
    st.info("왼쪽에서 데이터를 불러오세요. 첫 번째 숫자 열을 X축으로 보고 나머지 숫자 열을 각각 곡선으로 추가합니다.")
    st.stop()
series=st.session_state.series
names=[s["name"] for s in series]
mapped=[s for s in series if s.get("mapping_type") in ("time","space")]
spatial=[s for s in mapped if s.get("mapping_type")=="space" and isinstance(s.get("mapping_coordinate"),(tuple,list)) and len(s["mapping_coordinate"])>=2]
with st.sidebar:
    if spatial:
        st.markdown("**공간 위치 선택**")
        st.caption("점 클릭 또는 드래그로 여러 좌표를 선택하세요.")
        spatial_files={}
        for spec in spatial:
            source=spec.get("mapping_source",spec["name"].split(" · ")[0])
            spatial_files.setdefault(source,[]).append(spec)
        source_options=list(spatial_files)
        chosen_source=st.selectbox("공간 매핑 파일",source_options,format_func=lambda source:Path(source).name,key="spatial_map_file")
        map_specs=spatial_files[chosen_source]
        map_df=pd.DataFrame([{"X":float(s["mapping_coordinate"][0]),"Y":float(s["mapping_coordinate"][1]),"series_name":s["name"]} for s in map_specs])
        map_df["coordinate_label"]=[f"({row.X:g}, {row.Y:g})" for row in map_df.itertuples()]
        map_fig=px.scatter(map_df,x="X",y="Y",text="coordinate_label",custom_data=["series_name"])
        map_fig.update_traces(marker={"size":11,"color":"#168aad","line":{"width":1,"color":"white"}},
                              textposition="top center",
                              hovertemplate="X = %{x:g}<br>Y = %{y:g}<extra></extra>")
        map_fig.update_layout(height=300,margin={"l":8,"r":8,"t":8,"b":8},
                              xaxis_title="X",yaxis_title="Y",showlegend=False,
                              dragmode="lasso",clickmode="event+select")
        map_fig.update_yaxes(autorange="reversed",scaleanchor="x",scaleratio=1,constrain="domain")
        map_event=st.plotly_chart(map_fig,use_container_width=True,on_select="rerun",
                                  selection_mode=("points","box","lasso"),key="spatial_coordinate_map")
        try:
            selected_points=map_event.selection.points
            picked_names=[]
            for point in selected_points:
                point_data=point.get("customdata")
                picked_name=point_data[0] if isinstance(point_data,(list,tuple)) else point_data
                if picked_name in names:
                    picked_names.append(picked_name)
            if picked_names:
                mapped_names={s["name"] for s in mapped}
                ordinary_selected=[n for n in st.session_state["graph_selected"] if n not in mapped_names]
                st.session_state["graph_selected"]=list(dict.fromkeys([*ordinary_selected,*picked_names]))
                st.caption(f"선택한 좌표: {len(picked_names)}개")
        except (AttributeError,KeyError,TypeError,IndexError):
            pass

def selection_label(name):
    spec=next((item for item in mapped if item["name"]==name),None)
    if spec is None:
        return name if len(name)<=52 else f"…{name[-49:]}"
    coord=spec.get("mapping_coordinate")
    stem=Path(spec.get("mapping_source",spec["name"].split(" · ")[0])).stem
    short_stem=stem if len(stem)<=18 else f"{stem[:15]}…"
    if spec.get("mapping_type")=="space" and isinstance(coord,(tuple,list)) and len(coord)>=2:
        return f"{short_stem} · X={coord[0]:g}, Y={coord[1]:g}"
    value=coord[0] if isinstance(coord,(tuple,list)) else coord
    return f"{short_stem} · 시간={value:g}"
with st.sidebar:
    st.multiselect("그릴 스펙트럼 선택",names,key="graph_selected",format_func=selection_label,help="공간 맵에서 점 하나 또는 여러 개를 선택할 수 있습니다. 시간 매핑은 시간값으로 표시됩니다.")
tabs=st.tabs(["그래프 및 처리","피크 분석","내보내기"])
with st.sidebar:
    st.header("처리")
    normalization_tab,baseline_tab,smoothing_tab=st.tabs(["Normalization","Baseline","SG 필터"])
    with normalization_tab:
        normalization_enabled=st.checkbox("Normalization 적용",value=False,key="normalization_enabled")
        if normalization_enabled:
            normalization=st.selectbox("Normalization 종류",["Min-max","Maximum","Custom divisor"],key="normalization_method")
            divisor=st.number_input("사용자 나눗수",value=1.0,format="%.6g",key="normalization_divisor") if normalization=="Custom divisor" else 1.0
        else:
            normalization="없음"
            divisor=1.0
    with baseline_tab:
        baseline_enabled=st.checkbox("Baseline 적용",value=False,key="baseline_enabled")
        if baseline_enabled:
            baseline_method=st.selectbox("Baseline 종류",["AsLS","Polynomial"],key="baseline_method")
            if baseline_method=="Polynomial":
                degree=st.number_input("Polynomial 차수",1,10,2,key="baseline_degree")
                lam,p,iterations=100000.0,0.0001,10
            else:
                degree=2
                lam=st.number_input("AsLS λ",min_value=1.0,value=100000.0,format="%.1f",key="baseline_lambda")
                p=st.number_input("AsLS p",min_value=0.000001,max_value=0.5,value=0.0001,format="%.6f",key="baseline_p")
                iterations=st.number_input("반복 횟수",1,200,10,key="baseline_iterations")
        else:
            baseline_method="없음"
            degree,lam,p,iterations=2,100000.0,0.0001,10
    with smoothing_tab:
        min_curve_points=min(len(np.asarray(s["raw"]).reshape(-1)) for s in series)
        max_sg_window=min_curve_points if min_curve_points%2 else min_curve_points-1
        sg_enabled=st.checkbox("Savitzky–Golay 필터 적용",value=False,key="sg_enabled",disabled=max_sg_window<3)
        sg_window,sg_polyorder=11,3
        if max_sg_window<3:
            st.warning("필터를 적용하려면 각 곡선에 데이터 점이 3개 이상 필요합니다.")
            sg_enabled=False
        elif sg_enabled:
            default_window=min(11,max_sg_window)
            sg_window=st.number_input("Window length (홀수)",min_value=3,max_value=max_sg_window,
                                      value=default_window,step=2,key="sg_window_length",
                                      help="값이 클수록 더 넓은 구간을 평활화합니다.")
            if sg_window%2==0:
                sg_window=sg_window+1 if sg_window<max_sg_window else sg_window-1
                st.caption(f"Window length는 홀수여야 해서 {sg_window}를 적용합니다.")
            sg_polyorder=st.number_input("Polynomial order",min_value=1,max_value=max(1,sg_window-1),
                                         value=min(3,sg_window-1),step=1,key="sg_polyorder",
                                         help="Window length보다 작은 차수를 선택하세요.")
        st.caption("처리 순서: SG 평활화 → baseline 보정 → normalization")
    opts={"normalization":normalization,"divisor":divisor,"baseline":baseline_method,"degree":degree,"lam":lam,"p":p,"iterations":iterations,
          "sg_enabled":sg_enabled,"sg_window":sg_window,"sg_polyorder":sg_polyorder}

style={}
with tabs[0]:
    plot_col,settings_col=st.columns([2.2,1],gap="large")
    with settings_col:
        st.subheader("그래프 설정")
        size_tab,axis_tab,stack_tab,visual_tab=st.tabs(["크기·글꼴","축 설정","수직 스택","비주얼"])
        with stack_tab:
            selected=st.session_state.get("graph_selected",names.copy())
            stack=st.checkbox("수직 스택 사용",False,key="graph_stack")
            offset_step=st.number_input("곡선 간 추가 간격",value=0.0,key="graph_stack_step")
            st.caption("현재 X축 범위 안의 최대 intensity에 10%를 더해 자동 간격을 계산합니다. 첫 번째 곡선이 위쪽에 배치됩니다.")
            with st.expander("곡선별 오프셋",expanded=False):
                st.caption("각 곡선의 세로 위치를 개별 조정합니다.")
                for i,s in enumerate(series):
                    s["offset"]=st.number_input(f"{s['name']} 오프셋",value=float(s.get("offset",0)),key=f"off_{i}")
        with visual_tab:
            palette=st.selectbox("색상 팔레트",["Origin","viridis","plasma","inferno","magma","cividis","turbo","coolwarm","rainbow"],key="graph_palette")
            color_mode=st.radio("색상 설정",["팔레트 사용","곡선별 지정"],horizontal=True,key="graph_color_mode")
            width_mode=st.radio("선 굵기 설정",["전체 동일","곡선별 지정"],horizontal=True,key="graph_width_mode")
            line_width=synced_float_control("전체 선 굵기",0.5,5.0,1.8,0.1,"graph_line_width")
            show_graph_frame=st.checkbox("위쪽·오른쪽 테두리 표시",value=True,key="graph_show_frame",
                                         help="체크를 해제해도 왼쪽 Y축과 아래쪽 X축 선은 유지됩니다.")
            graph_frame_width=synced_float_control("축 / 테두리 선 두께",0.1,5.0,1.0,0.1,"graph_frame_width",disabled=not show_graph_frame)
            old_names=names.copy()
            st.markdown("**곡선 이름 변경**")
            name_cols=st.columns(2)
            for i,s in enumerate(series):
                with name_cols[i%2]:
                    s["name"]=st.text_input(f"곡선 이름 {i+1}",s["name"],key=f"nm_{i}")
            renamed={old:new["name"] for old,new in zip(old_names,series)}
            selected=[renamed.get(name,name) for name in selected]
            st.session_state["graph_name_aliases"]=renamed
            names=[s["name"] for s in series]
            st.markdown("**곡선 색상 및 선 굵기**")
            style_cols=st.columns(2)
            for i,s in enumerate(series):
                with style_cols[i%2]:
                    palette_color=to_hex(plt.get_cmap(palette)(i/max(1,len(series)-1)) if palette!="Origin" else plt.get_cmap("tab10")(i%10))
                    curve_color=st.color_picker("선 색상",value=palette_color,key=f"color_{i}") if color_mode=="곡선별 지정" else palette_color
                    curve_width=st.number_input("개별 선 굵기",min_value=0.5,max_value=10.0,value=float(line_width),step=0.1,key=f"lw_{i}") if width_mode=="곡선별 지정" else float(line_width)
                    style[s["name"]]={"color":curve_color,"offset":float(s.get("offset",0)),"linewidth":curve_width}
        with size_tab:
            size_mode="전체 이미지 기준"
            st.caption("전체 이미지 크기를 먼저 정하고, 필요하면 그 안의 플롯 박스 크기를 따로 지정합니다.")
            st.markdown("#### 크기 및 여백")
            size_c1,size_c2=st.columns(2)
            plot_width=size_c1.number_input("전체 이미지 너비 (px)",min_value=100,max_value=10000,value=600,step=50,key="graph_width_px")
            plot_height=size_c2.number_input("전체 이미지 높이 (px)",min_value=100,max_value=10000,value=400,step=50,key="graph_height_px")
            canvas_width_px=int(plot_width)
            canvas_height_px=int(plot_height)
            set_plot_box=st.checkbox("플롯 박스 크기 직접 지정",value=False,key="graph_set_inner_size",
                                     help="체크하면 전체 이미지 크기를 유지한 채 축 안쪽 박스 크기와 위치를 지정합니다.")
            if set_plot_box:
                # Keep the plot-box dimensions immediately below the toggle, side by side.
                default_left=int(st.session_state.get("graph_margin_left",65))
                default_bottom=int(st.session_state.get("graph_margin_bottom",50))
                default_inner_width=max(50,int(canvas_width_px-default_left-20))
                default_inner_height=max(50,int(canvas_height_px-default_bottom-30))
                dim_col1,dim_col2=st.columns(2)
                inner_width=dim_col1.number_input("플롯 박스 너비 (px)",min_value=50,max_value=10000,value=default_inner_width,step=25,key="graph_inner_width_px")
                inner_height=dim_col2.number_input("플롯 박스 높이 (px)",min_value=50,max_value=10000,value=default_inner_height,step=25,key="graph_inner_height_px")
            with st.expander("여백 설정",expanded=False):
                if set_plot_box:
                    st.markdown("**플롯 박스 위치 (px)**")
                    pos_col1,pos_col2=st.columns(2)
                    margin_left=pos_col1.number_input("왼쪽 위치",min_value=0,max_value=2000,value=65,step=5,key="graph_margin_left")
                    margin_bottom=pos_col2.number_input("아래쪽 위치",min_value=0,max_value=2000,value=50,step=5,key="graph_margin_bottom")
                    max_inner_width=max(1,canvas_width_px-int(margin_left)-1)
                    max_inner_height=max(1,canvas_height_px-int(margin_bottom)-1)
                    if inner_width>max_inner_width or inner_height>max_inner_height:
                        st.error("플롯 박스와 왼쪽·아래쪽 위치의 합이 전체 이미지 크기를 넘습니다. 박스 크기나 위치를 줄여주세요.")
                    safe_left=min(int(margin_left),canvas_width_px-2)
                    safe_bottom=min(int(margin_bottom),canvas_height_px-2)
                    safe_inner_width=min(int(inner_width),canvas_width_px-safe_left-1)
                    safe_inner_height=min(int(inner_height),canvas_height_px-safe_bottom-1)
                    margin_right=canvas_width_px-safe_left-safe_inner_width
                    margin_top=canvas_height_px-safe_bottom-safe_inner_height
                    plot_margins=(safe_left,margin_right,margin_top,safe_bottom)
                    st.caption(f"자동 여백 · 오른쪽 {margin_right} px · 위쪽 {margin_top} px")
                else:
                    m1,m2=st.columns(2)
                    margin_left=m1.number_input("왼쪽 여백",min_value=0,max_value=2000,value=65,step=5,key="graph_margin_left")
                    margin_right=m2.number_input("오른쪽 여백",min_value=0,max_value=2000,value=20,step=5,key="graph_margin_right")
                    m3,m4=st.columns(2)
                    margin_top=m3.number_input("위쪽 여백",min_value=0,max_value=2000,value=30,step=5,key="graph_margin_top")
                    margin_bottom=m4.number_input("아래쪽 여백",min_value=0,max_value=2000,value=50,step=5,key="graph_margin_bottom")
                    plot_margins=(margin_left,margin_right,margin_top,margin_bottom)
            plot_box_width=int(canvas_width_px-margin_left-margin_right)
            plot_box_height=int(canvas_height_px-margin_top-margin_bottom)
            export_dpi=st.number_input("미리보기 / PNG 저장 DPI",min_value=100,max_value=1200,value=300,step=50,key="graph_export_dpi",
                                       help="미리보기와 PNG 저장에 같은 DPI를 적용합니다. DPI를 바꿔도 설정한 픽셀 크기는 유지됩니다.")
            preview_dpi=export_dpi
            font_family=st.selectbox("글꼴",["Arial","DejaVu Sans","Liberation Sans","Times New Roman"],index=0,key="graph_font_family")
            axis_label_size=synced_font_control("축 제목 글자 크기",6,40,12,"graph_axis_font_size")
            title_font_size=synced_font_control("그래프 제목 글자 크기",6,48,14,"graph_title_font_size")
            tick_font_size=synced_font_control("눈금 글자 크기",6,32,10,"graph_tick_font_size")
            legend_font_size=synced_font_control("범례 글자 크기",6,32,9,"graph_legend_font_size")
        with axis_tab:
            with st.expander("축 제목 / 그래프 제목",expanded=False):
                xlab=st.text_input("X축 제목",series[0]["x_label"],key="graph_xlabel")
                ylab=st.text_input("Y축 제목","Normalized Intensity" if normalization!="없음" else series[0]["y_label"],key="graph_ylabel")
                title=st.text_input("그래프 제목",key="graph_title")
            st.markdown("**축 범위**")
            x_auto=st.checkbox("X축 자동 범위",value=True,key="graph_x_auto")
            data_xmin=float(min(np.nanmin(s["x"]) for s in series)); data_xmax=float(max(np.nanmax(s["x"]) for s in series))
            x1,x2=st.columns(2)
            xmin=x1.number_input("X 최소",value=data_xmin,format="%.6g",disabled=x_auto,key="graph_xmin")
            xmax=x2.number_input("X 최대",value=data_xmax,format="%.6g",disabled=x_auto,key="graph_xmax")
            y_auto=st.checkbox("Y축 자동 범위",value=True,key="graph_y_auto")
            proc_min=min(float(np.nanmin(transform(s,opts))) for s in series)
            proc_max=max(float(np.nanmax(transform(s,opts))) for s in series)
            y1,y2=st.columns(2)
            ymin=y1.number_input("Y 최소",value=proc_min,format="%.6g",disabled=y_auto,key="graph_ymin")
            ymax=y2.number_input("Y 최대",value=proc_max,format="%.6g",disabled=y_auto,key="graph_ymax")
            st.markdown("**눈금 표시 / 간격**")
            ticks1,ticks2=st.columns(2)
            x_bottom=ticks1.checkbox("아래 X 눈금",value=True,key="graph_x_bottom")
            x_top=ticks1.checkbox("위 X 눈금",value=False,key="graph_x_top")
            y_left=ticks2.checkbox("왼쪽 Y 눈금",value=True,key="graph_y_left")
            y_right=ticks2.checkbox("오른쪽 Y 눈금",value=False,key="graph_y_right")
            x_tick_interval=st.number_input("X 눈금 간격 (0=자동)",min_value=0.0,value=0.0,step=1.0,key="graph_x_tick_interval")
            y_tick_interval=st.number_input("Y 눈금 간격 (0=자동)",min_value=0.0,value=0.0,step=0.1,key="graph_y_tick_interval")
            with st.expander("눈금 숫자 표시 형식",expanded=False):
                y_tick_options=axis_tick_options_ui(st.container(),"Y축","y_tick_label")
                st.caption("Y축 눈금 표시 형식은 미리보기와 PNG 저장 그래프에 적용됩니다. 수식이 입력되면 표시 단위가 아닌 수식을 적용합니다.")
            try:
                y_tick_formatter=make_tick_formatter(y_tick_options)
            except (ValueError,SyntaxError,ZeroDivisionError) as e:
                st.error(f"Y축 눈금 수식 오류: {e}")
                y_tick_options["formula"]=""
                y_tick_formatter=make_tick_formatter(y_tick_options)
            grid=st.checkbox("격자 표시",False,key="graph_grid")
            legend=st.checkbox("범례 표시",True,key="graph_legend")
            legend_locations={
                "자동 (겹침 최소화)":"best",
                "오른쪽 위":"upper right",
                "왼쪽 위":"upper left",
                "왼쪽 아래":"lower left",
                "오른쪽 아래":"lower right",
                "오른쪽 가운데":"center right",
                "왼쪽 가운데":"center left",
                "위 가운데":"upper center",
                "아래 가운데":"lower center",
                "가운데":"center",
            }
            legend_position=st.selectbox("범례 위치",list(legend_locations),disabled=not legend,key="graph_legend_location")
            legend_loc=legend_locations[legend_position]
        if not x_auto and xmin>=xmax:
            st.error("X 최소값은 X 최대값보다 작아야 합니다.")
            xmin,xmax=data_xmin,data_xmax
        elif x_auto:
            xmin,xmax=data_xmin,data_xmax
        if not y_auto and ymin>=ymax:
            st.error("Y 최소값은 Y 최대값보다 작아야 합니다.")
            ymin,ymax=proc_min,proc_max
    with plot_col:
        st.subheader("그래프 미리보기")
        # Keep the selected typeface for normal text and fall back for missing glyphs.
        plt.rcParams["font.family"]=[font_family, "DejaVu Sans"]
        fig,ax=make_plot(plot_width,plot_height,size_mode,plot_margins,100)
        apply_graph_frame(ax,show_graph_frame,graph_frame_width)
        proc={s["name"]:transform(s,opts) for s in series}
        # Base the automatic vertical stack spacing on the visible X range.
        # This keeps the gaps proportional when the user zooms into a narrower range.
        x_range_min,x_range_max=sorted((float(xmin),float(xmax)))
        heights=[]
        for n in names:
            s=next(z for z in series if z["name"]==n)
            y_values=np.asarray(proc[n],dtype=float)
            visible=np.isfinite(s["x"]) & np.isfinite(y_values) & (s["x"]>=x_range_min) & (s["x"]<=x_range_max)
            values=y_values[visible] if np.any(visible) else y_values[np.isfinite(y_values)]
            peak=float(np.nanmax(values)) if values.size else 0.0
            heights.append(max(peak*1.10,0.01))
        auto={}
        for i,n in enumerate(names): auto[n]=sum(heights[i+1:]) if stack else 0
        for n in names:
            if n not in selected: continue
            s=next(z for z in series if z["name"]==n)
            off=auto[n]+offset_step*list(names).index(n)+style[n]["offset"]
            ax.plot(s["x"],proc[n]+off,label=n,color=style[n]["color"],lw=style[n]["linewidth"])
        ax.set_xlim(min(xmin,xmax),max(xmin,xmax))
        if y_auto:
            visible_limits=visible_stack_ylim(series,proc,names,selected,auto,offset_step,style,xmin,xmax)
            if visible_limits: ax.set_ylim(*visible_limits)
        else: ax.set_ylim(min(ymin,ymax),max(ymin,ymax))
        ax.set_xlabel(format_axis_label(xlab),fontsize=axis_label_size); ax.set_ylabel(format_axis_label(ylab),fontsize=axis_label_size)
        ax.set_title(title or None,fontsize=title_font_size)
        ax.tick_params(axis="x",which="both",bottom=x_bottom,top=x_top,labelbottom=x_bottom,labeltop=x_top,labelsize=tick_font_size)
        ax.tick_params(axis="y",which="both",left=y_left,right=y_right,labelleft=y_left,labelright=y_right,labelsize=tick_font_size)
        if x_tick_interval>0: ax.xaxis.set_major_locator(MultipleLocator(x_tick_interval))
        if y_tick_interval>0: ax.yaxis.set_major_locator(MultipleLocator(y_tick_interval))
        ax.yaxis.set_major_formatter(y_tick_formatter)
        add_y_scale_annotation(ax,y_tick_options,tick_font_size)
        if grid: ax.grid(True,alpha=.3)
        if legend: ax.legend(fontsize=legend_font_size,loc=legend_loc)
        preview_png=render_png_fixed_size(fig,canvas_width_px,canvas_height_px,preview_dpi)
        st.image(preview_png,width=canvas_width_px); plt.close(fig)
        st.caption(f"곡선 {len(series)}개 · 원본 데이터는 전체 해상도로 분석/내보내기합니다.")

with tabs[1]:
    peak_tab,max_tab=st.tabs(["피크 찾기","맥스 인텐시티"])
    with peak_tab:
        st.subheader("피크 찾기")
        target=st.selectbox("분석 곡선",names,key="peak_series")
        prominence_col,distance_col=st.columns(2)
        prominence=prominence_col.number_input("최소 prominence",min_value=0.0,value=0.0)
        distance=distance_col.number_input("최소 피크 간격 (points)",min_value=1,value=1)
        y=proc[target]; s=next(z for z in series if z["name"]==target)
        peaks,_=find_peaks(y,prominence=prominence,distance=int(distance))
        peaks=peaks[(s["x"][peaks]>=min(xmin,xmax))&(s["x"][peaks]<=max(xmin,xmax))]
        peak_df=pd.DataFrame({"index":peaks,"X":s["x"][peaks],"Y":y[peaks]})
        st.write(f"찾은 피크: {len(peaks)}개")
        st.dataframe(peak_df,use_container_width=True)
        if len(peaks):
            plt.rcParams["font.family"]=[font_family,"DejaVu Sans"]
            fig2,ax2=make_plot(plot_width,plot_height,size_mode,plot_margins,100)
            apply_graph_frame(ax2,show_graph_frame,graph_frame_width)
            peak_plot_names=[n for n in names if n in selected]
            if target not in peak_plot_names:
                peak_plot_names.append(target)
            peak_target_offset=auto[target]+offset_step*list(names).index(target)+style[target]["offset"]
            for n in peak_plot_names:
                peak_s=next(z for z in series if z["name"]==n)
                peak_offset=auto[n]+offset_step*list(names).index(n)+style[n]["offset"]
                ax2.plot(peak_s["x"],proc[n]+peak_offset,label=n,color=style[n]["color"],lw=style[n]["linewidth"])
            ax2.scatter(s["x"][peaks],y[peaks]+peak_target_offset,color="red",zorder=5)
            ax2.set_xlim(min(xmin,xmax),max(xmin,xmax))
            if y_auto:
                visible_limits=visible_stack_ylim(series,proc,names,peak_plot_names,auto,offset_step,style,xmin,xmax)
                if visible_limits: ax2.set_ylim(*visible_limits)
            else: ax2.set_ylim(min(ymin,ymax),max(ymin,ymax))
            ax2.set_xlabel(format_axis_label(xlab),fontsize=axis_label_size); ax2.set_ylabel(format_axis_label(ylab),fontsize=axis_label_size)
            ax2.set_title(title or None,fontsize=title_font_size)
            ax2.tick_params(axis="x",which="both",bottom=x_bottom,top=x_top,labelbottom=x_bottom,labeltop=x_top,labelsize=tick_font_size)
            ax2.tick_params(axis="y",which="both",left=y_left,right=y_right,labelleft=y_left,labelright=y_right,labelsize=tick_font_size)
            if x_tick_interval>0: ax2.xaxis.set_major_locator(MultipleLocator(x_tick_interval))
            if y_tick_interval>0: ax2.yaxis.set_major_locator(MultipleLocator(y_tick_interval))
            ax2.yaxis.set_major_formatter(y_tick_formatter)
            add_y_scale_annotation(ax2,y_tick_options,tick_font_size)
            if grid: ax2.grid(True,alpha=.3)
            if legend: ax2.legend(fontsize=legend_font_size,loc=legend_loc)
            st.image(render_png_fixed_size(fig2,canvas_width_px,canvas_height_px,preview_dpi),width=canvas_width_px); plt.close(fig2)
            st.download_button("피크 결과 CSV",peak_df.to_csv(index=False).encode("utf-8-sig"),"peaks.csv","text/csv")
    with max_tab:
        st.subheader("선택 곡선의 최대 intensity")
        st.caption("Raman 데이터는 AsLS 보정 후, UV–Vis 데이터는 원본 intensity에서 최대값을 계산합니다.")
        max_targets=st.multiselect("최대값을 계산할 곡선",names,default=[n for n in selected if n in names],key="max_targets")
        use_graph_range=st.checkbox("현재 그래프 X 범위 사용",value=True,key="max_use_graph_range")
        if use_graph_range:
            max_xmin,max_xmax=sorted((float(xmin),float(xmax)))
            st.caption(f"계산 범위: {max_xmin:g} ~ {max_xmax:g}")
        else:
            r1,r2=st.columns(2)
            data_xmin=min(float(np.nanmin(ss["x"])) for ss in series)
            data_xmax=max(float(np.nanmax(ss["x"])) for ss in series)
            max_xmin=r1.number_input("최대값 계산 X 최소",value=data_xmin,format="%.6g",key="max_xmin")
            max_xmax=r2.number_input("최대값 계산 X 최대",value=data_xmax,format="%.6g",key="max_xmax")
            if max_xmin>max_xmax:
                max_xmin,max_xmax=max_xmax,max_xmin
        max_rows=[]
        for max_name in max_targets:
            max_s=next(z for z in series if z["name"]==max_name)
            max_y=max_s["raw"].copy()
            is_uv=max_s.get("x_label")=="Wavelength (nm)"
            if not is_uv and len(max_y)>=3:
                max_y=baseline(max_y,"AsLS",degree,lam,p,iterations)
            mask=np.isfinite(max_s["x"]) & np.isfinite(max_y) & (max_s["x"]>=max_xmin) & (max_s["x"]<=max_xmax)
            if np.any(mask):
                valid_indices=np.flatnonzero(mask)
                source_idx=int(valid_indices[int(np.nanargmax(max_y[mask]))])
                max_rows.append({
                    "곡선":max_name,
                    "최대 X (nm)" if is_uv else "최대 X (cm⁻¹)":float(max_s["x"][source_idx]),
                    "최대 intensity":float(max_y[source_idx]),
                    "처리": "원본 (UV–Vis)" if is_uv else f"AsLS (λ={lam:g}, p={p:g}, 반복={iterations})",
                })
        if max_rows:
            max_df=pd.DataFrame(max_rows)
            st.dataframe(max_df,use_container_width=True,hide_index=True)
            st.download_button("최대값 결과 CSV",max_df.to_csv(index=False).encode("utf-8-sig"),"max_intensity_summary.csv","text/csv")
        elif max_targets:
            st.info("설정한 X 범위 안에 유효한 데이터가 없습니다.")

with tabs[2]:
    st.subheader("그래프 저장")
    plt.rcParams["font.family"]=[font_family, "DejaVu Sans"]
    figout,axout=make_plot(plot_width,plot_height,size_mode,plot_margins,100)
    apply_graph_frame(axout,show_graph_frame,graph_frame_width)
    for n in names:
        if n in selected:
            s=next(z for z in series if z["name"]==n)
            off=auto[n]+offset_step*list(names).index(n)+style[n]["offset"]
            axout.plot(s["x"],proc[n]+off,label=n,color=style[n]["color"],lw=style[n]["linewidth"])
    axout.set_xlim(min(xmin,xmax),max(xmin,xmax))
    if y_auto:
        visible_limits=visible_stack_ylim(series,proc,names,selected,auto,offset_step,style,xmin,xmax)
        if visible_limits: axout.set_ylim(*visible_limits)
    else: axout.set_ylim(min(ymin,ymax),max(ymin,ymax))
    axout.set_xlabel(format_axis_label(xlab),fontsize=axis_label_size); axout.set_ylabel(format_axis_label(ylab),fontsize=axis_label_size)
    axout.set_title(title or None,fontsize=title_font_size)
    axout.tick_params(axis="x",which="both",bottom=x_bottom,top=x_top,labelbottom=x_bottom,labeltop=x_top,labelsize=tick_font_size)
    axout.tick_params(axis="y",which="both",left=y_left,right=y_right,labelleft=y_left,labelright=y_right,labelsize=tick_font_size)
    if x_tick_interval>0: axout.xaxis.set_major_locator(MultipleLocator(x_tick_interval))
    if y_tick_interval>0: axout.yaxis.set_major_locator(MultipleLocator(y_tick_interval))
    axout.yaxis.set_major_formatter(y_tick_formatter)
    add_y_scale_annotation(axout,y_tick_options,tick_font_size)
    if legend: axout.legend(fontsize=legend_font_size,loc=legend_loc)
    if grid: axout.grid(True,alpha=.3)
    png_data=render_png_fixed_size(figout,canvas_width_px,canvas_height_px,export_dpi); plt.close(figout)
    st.download_button(f"그래프 PNG ({canvas_width_px} × {canvas_height_px} px)",png_data,"spectrum.png","image/png")
    st.caption(f"플롯 박스: {plot_box_width} × {plot_box_height} px · 전체 PNG: {canvas_width_px} × {canvas_height_px} px · DPI: {export_dpi}")
    st.subheader("데이터 파일 저장")
    st.markdown("<style>div[data-testid='stDownloadButton'] button{max-width:350px !important;width:100% !important;}</style>",unsafe_allow_html=True)
    selected_targets=[n for n in selected if n in names]
    raw_values={s["name"]:np.asarray(s["raw"],dtype=float) for s in series}
    corrected_values={n:np.asarray(proc[n],dtype=float) for n in names}

    raw_selected_table=export_curve_table(selected_targets,series,raw_values)
    c1,c2=st.columns(2)
    c1.download_button("선택 곡선 원본 CSV",raw_selected_table.to_csv(index=False).encode("utf-8-sig"),
                       "selected_spectra_original.csv","text/csv",disabled=not selected_targets,use_container_width=True)
    c2.download_button("선택 곡선 원본 TXT",raw_selected_table.to_csv(index=False,sep="\t").encode("utf-8-sig"),
                       "selected_spectra_original.txt","text/plain",disabled=not selected_targets,use_container_width=True)

    st.divider()
    corrected_selected_table=export_curve_table(selected_targets,series,corrected_values)
    c1,c2=st.columns(2)
    c1.download_button("선택 곡선 보정 CSV",corrected_selected_table.to_csv(index=False).encode("utf-8-sig"),
                       "selected_spectra_corrected.csv","text/csv",disabled=not selected_targets,use_container_width=True)
    c2.download_button("선택 곡선 보정 TXT",corrected_selected_table.to_csv(index=False,sep="\t").encode("utf-8-sig"),
                       "selected_spectra_corrected.txt","text/plain",disabled=not selected_targets,use_container_width=True)

    st.divider()
    c1,c2=st.columns(2)
    c1.download_button("곡선별 원본 CSV ZIP",curve_archive(names,series,raw_values,"csv"),
                       "spectra_original_csv.zip","application/zip",disabled=not names,use_container_width=True)
    c2.download_button("곡선별 원본 TXT ZIP",curve_archive(names,series,raw_values,"txt"),
                       "spectra_original_txt.zip","application/zip",disabled=not names,use_container_width=True)

    st.divider()
    c1,c2=st.columns(2)
    c1.download_button("곡선별 보정 CSV ZIP",curve_archive(names,series,corrected_values,"csv"),
                       "spectra_corrected_csv.zip","application/zip",disabled=not names,use_container_width=True)
    c2.download_button("곡선별 보정 TXT ZIP",curve_archive(names,series,corrected_values,"txt"),
                       "spectra_corrected_txt.zip","application/zip",disabled=not names,use_container_width=True)
