"""Streamlit port of the RSLAB spectrum viewer (core workflow)."""
from __future__ import annotations
import ast, io, json, csv, re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import to_hex
from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks

st.set_page_config(page_title="RSLAB Spectrum Viewer", layout="wide")
st.title("RSLAB Spectrum Viewer")

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
    y=baseline(s["raw"],opts["baseline"],opts["degree"],opts["lam"],opts["p"],opts["iterations"])
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
    for spine in ax.spines.values():
        spine.set_visible(visible)
        spine.set_linewidth(linewidth)

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

if "series" not in st.session_state: st.session_state.series=[]
with st.sidebar:
    st.header("데이터")
    uploads=st.file_uploader("CSV / TSV / TXT 파일",type=["csv","tsv","txt","dat","asc"],accept_multiple_files=True)
    if st.button("파일 불러오기", type="primary", use_container_width=True):
        loaded=[]; errors=[]
        previous_names=[s["name"] for s in st.session_state.series]
        for f in uploads or []:
            try: loaded.extend(parse_upload(f))
            except Exception as e: errors.append(f"{f.name}: {e}")
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
            raw=json.loads(session.getvalue().decode("utf-8")); st.session_state.series=[{**s,"x":np.asarray(s["x"]),"raw":np.asarray(s["raw"])} for s in raw]; st.rerun()
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
        st.multiselect("그릴 스펙트럼 선택",parsed_names,key="graph_selected",help="파일에서 파싱된 곡선이 목록에 표시됩니다. 선택한 곡선만 그래프에 그립니다.")

if not st.session_state.series:
    st.info("왼쪽에서 데이터를 불러오세요. 첫 번째 숫자 열을 X축으로 보고 나머지 숫자 열을 각각 곡선으로 추가합니다.")
    st.stop()
series=st.session_state.series
names=[s["name"] for s in series]
tabs=st.tabs(["그래프 및 처리","피크 분석","내보내기"])
with st.sidebar:
    st.header("처리")
    normalization_enabled=st.checkbox("Normalization 적용",value=False,key="normalization_enabled")
    if normalization_enabled:
        normalization=st.selectbox("Normalization 종류",["Min-max","Maximum","Custom divisor"],key="normalization_method")
        divisor=st.number_input("사용자 나눗수",value=1.0,format="%.6g",key="normalization_divisor") if normalization=="Custom divisor" else 1.0
    else:
        normalization="없음"
        divisor=1.0

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
    opts={"normalization":normalization,"divisor":divisor,"baseline":baseline_method,"degree":degree,"lam":lam,"p":p,"iterations":iterations}

style={}
with tabs[0]:
    plot_col,settings_col=st.columns([2.2,1],gap="large")
    with settings_col:
        st.subheader("그래프 설정")
        stack_tab,visual_tab,size_tab,axis_tab=st.tabs(["수직 스택","비주얼","크기·글꼴","축 설정"])
        with stack_tab:
            selected=st.session_state.get("graph_selected",names.copy())
            stack=st.checkbox("수직 스택 사용",False,key="graph_stack")
            offset_step=st.number_input("곡선 간 추가 간격",value=0.0,key="graph_stack_step")
            st.caption("첫 번째 곡선이 위쪽에 오도록 자동 간격을 계산합니다.")
            with st.expander("곡선별 오프셋",expanded=False):
                st.caption("각 곡선의 세로 위치를 개별 조정합니다.")
                for i,s in enumerate(series):
                    s["offset"]=st.number_input(f"{s['name']} 오프셋",value=float(s.get("offset",0)),key=f"off_{i}")
        with visual_tab:
            palette=st.selectbox("색상 팔레트",["Origin","viridis","plasma","inferno","magma","cividis","turbo","coolwarm","rainbow"],key="graph_palette")
            color_mode=st.radio("색상 설정",["팔레트 사용","곡선별 지정"],horizontal=True,key="graph_color_mode")
            width_mode=st.radio("선 굵기 설정",["전체 동일","곡선별 지정"],horizontal=True,key="graph_width_mode")
            line_width=st.slider("전체 선 굵기",0.5,5.0,1.8,0.1,key="graph_line_width")
            show_graph_frame=st.checkbox("그래프 박스 테두리 표시",value=True,key="graph_show_frame")
            graph_frame_width=st.slider("그래프 박스 선 두께",0.1,5.0,1.0,0.1,key="graph_frame_width",disabled=not show_graph_frame)
            style_cols=st.columns(2)
            old_names=names.copy()
            for i,s in enumerate(series):
                with style_cols[i%2]:
                    s["name"]=st.text_input(f"곡선 이름 {i+1}",s["name"],key=f"nm_{i}")
                    palette_color=to_hex(plt.get_cmap(palette)(i/max(1,len(series)-1)) if palette!="Origin" else plt.get_cmap("tab10")(i%10))
                    curve_color=st.color_picker("선 색상",value=palette_color,key=f"color_{i}") if color_mode=="곡선별 지정" else palette_color
                    curve_width=st.number_input("개별 선 굵기",min_value=0.5,max_value=10.0,value=float(line_width),step=0.1,key=f"lw_{i}") if width_mode=="곡선별 지정" else float(line_width)
                    style[s["name"]]={"color":curve_color,"offset":float(s.get("offset",0)),"linewidth":curve_width}
            renamed={old:new["name"] for old,new in zip(old_names,series)}
            selected=[renamed.get(name,name) for name in selected]
            st.session_state["graph_name_aliases"]=renamed
            names=[s["name"] for s in series]
        with size_tab:
            size_mode="전체 이미지 기준"
            st.caption("전체 이미지 크기를 먼저 정하고, 필요하면 그 안의 플롯 박스 크기를 따로 지정합니다.")
            size_c1,size_c2=st.columns(2)
            plot_width=size_c1.number_input("전체 이미지 너비 (px)",min_value=100,max_value=10000,value=600,step=50,key="graph_width_px")
            plot_height=size_c2.number_input("전체 이미지 높이 (px)",min_value=100,max_value=10000,value=400,step=50,key="graph_height_px")
            canvas_width_px=int(plot_width)
            canvas_height_px=int(plot_height)
            set_plot_box=st.checkbox("플롯 박스 크기 직접 지정",value=False,key="graph_set_inner_size",
                                     help="체크하면 전체 이미지 크기를 유지한 채 축 안쪽 박스 크기와 위치를 지정합니다.")
            st.markdown("**그래프 위치 / 여백 (px)**")
            if set_plot_box:
                pos_col,size_col=st.columns(2)
                margin_left=pos_col.number_input("왼쪽 위치",min_value=0,max_value=2000,value=65,step=5,key="graph_margin_left")
                margin_bottom=pos_col.number_input("아래쪽 위치",min_value=0,max_value=2000,value=50,step=5,key="graph_margin_bottom")
                default_inner_width=max(50,int(canvas_width_px-margin_left-20))
                default_inner_height=max(50,int(canvas_height_px-margin_bottom-30))
                inner_width=size_col.number_input("플롯 박스 너비 (px)",min_value=50,max_value=10000,value=default_inner_width,step=25,key="graph_inner_width_px")
                inner_height=size_col.number_input("플롯 박스 높이 (px)",min_value=50,max_value=10000,value=default_inner_height,step=25,key="graph_inner_height_px")
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
            axis_label_size=st.slider("축 제목 글자 크기",6,40,12,key="graph_axis_font_size")
            title_font_size=st.slider("그래프 제목 글자 크기",6,48,14,key="graph_title_font_size")
            tick_font_size=st.slider("눈금 글자 크기",6,32,10,key="graph_tick_font_size")
            legend_font_size=st.slider("범례 글자 크기",6,32,9,key="graph_legend_font_size")
        with axis_tab:
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
            xlab=st.text_input("X축 제목",series[0]["x_label"],key="graph_xlabel")
            ylab=st.text_input("Y축 제목","Normalized Intensity" if normalization!="없음" else series[0]["y_label"],key="graph_ylabel")
            title=st.text_input("그래프 제목",key="graph_title")
            grid=st.checkbox("격자 표시",False,key="graph_grid")
            legend=st.checkbox("범례 표시",True,key="graph_legend")
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
        heights=[max(float(np.nanmax(proc[n])-np.nanmin(proc[n]))*1.10,0.01) for n in names]
        auto={}
        for i,n in enumerate(names): auto[n]=sum(heights[i+1:]) if stack else 0
        for n in names:
            if n not in selected: continue
            s=next(z for z in series if z["name"]==n)
            off=auto[n]+offset_step*list(names).index(n)+style[n]["offset"]
            ax.plot(s["x"],proc[n]+off,label=n,color=style[n]["color"],lw=style[n]["linewidth"])
        ax.set_xlim(min(xmin,xmax),max(xmin,xmax))
        if not y_auto: ax.set_ylim(min(ymin,ymax),max(ymin,ymax))
        ax.set_xlabel(format_axis_label(xlab),fontsize=axis_label_size); ax.set_ylabel(format_axis_label(ylab),fontsize=axis_label_size)
        ax.set_title(title or None,fontsize=title_font_size)
        ax.tick_params(axis="x",which="both",bottom=x_bottom,top=x_top,labelbottom=x_bottom,labeltop=x_top,labelsize=tick_font_size)
        ax.tick_params(axis="y",which="both",left=y_left,right=y_right,labelleft=y_left,labelright=y_right,labelsize=tick_font_size)
        if x_tick_interval>0: ax.xaxis.set_major_locator(MultipleLocator(x_tick_interval))
        if y_tick_interval>0: ax.yaxis.set_major_locator(MultipleLocator(y_tick_interval))
        ax.yaxis.set_major_formatter(y_tick_formatter)
        add_y_scale_annotation(ax,y_tick_options,tick_font_size)
        if grid: ax.grid(True,alpha=.3)
        if legend: ax.legend(fontsize=legend_font_size)
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
            if not y_auto: ax2.set_ylim(min(ymin,ymax),max(ymin,ymax))
            ax2.set_xlabel(format_axis_label(xlab),fontsize=axis_label_size); ax2.set_ylabel(format_axis_label(ylab),fontsize=axis_label_size)
            ax2.set_title(title or None,fontsize=title_font_size)
            ax2.tick_params(axis="x",which="both",bottom=x_bottom,top=x_top,labelbottom=x_bottom,labeltop=x_top,labelsize=tick_font_size)
            ax2.tick_params(axis="y",which="both",left=y_left,right=y_right,labelleft=y_left,labelright=y_right,labelsize=tick_font_size)
            if x_tick_interval>0: ax2.xaxis.set_major_locator(MultipleLocator(x_tick_interval))
            if y_tick_interval>0: ax2.yaxis.set_major_locator(MultipleLocator(y_tick_interval))
            ax2.yaxis.set_major_formatter(y_tick_formatter)
            add_y_scale_annotation(ax2,y_tick_options,tick_font_size)
            if grid: ax2.grid(True,alpha=.3)
            if legend: ax2.legend(fontsize=legend_font_size)
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
    st.subheader("그래프와 데이터 저장")
    plt.rcParams["font.family"]=[font_family, "DejaVu Sans"]
    figout,axout=make_plot(plot_width,plot_height,size_mode,plot_margins,100)
    apply_graph_frame(axout,show_graph_frame,graph_frame_width)
    for n in names:
        if n in selected:
            s=next(z for z in series if z["name"]==n)
            off=auto[n]+offset_step*list(names).index(n)+style[n]["offset"]
            axout.plot(s["x"],proc[n]+off,label=n,color=style[n]["color"],lw=style[n]["linewidth"])
    axout.set_xlim(min(xmin,xmax),max(xmin,xmax))
    if not y_auto: axout.set_ylim(min(ymin,ymax),max(ymin,ymax))
    axout.set_xlabel(format_axis_label(xlab),fontsize=axis_label_size); axout.set_ylabel(format_axis_label(ylab),fontsize=axis_label_size)
    axout.set_title(title or None,fontsize=title_font_size)
    axout.tick_params(axis="x",which="both",bottom=x_bottom,top=x_top,labelbottom=x_bottom,labeltop=x_top,labelsize=tick_font_size)
    axout.tick_params(axis="y",which="both",left=y_left,right=y_right,labelleft=y_left,labelright=y_right,labelsize=tick_font_size)
    if x_tick_interval>0: axout.xaxis.set_major_locator(MultipleLocator(x_tick_interval))
    if y_tick_interval>0: axout.yaxis.set_major_locator(MultipleLocator(y_tick_interval))
    axout.yaxis.set_major_formatter(y_tick_formatter)
    add_y_scale_annotation(axout,y_tick_options,tick_font_size)
    if legend: axout.legend(fontsize=legend_font_size)
    if grid: axout.grid(True,alpha=.3)
    png_data=render_png_fixed_size(figout,canvas_width_px,canvas_height_px,export_dpi); plt.close(figout)
    st.download_button(f"그래프 PNG ({canvas_width_px} × {canvas_height_px} px)",png_data,"spectrum.png","image/png")
    st.caption(f"플롯 박스: {plot_box_width} × {plot_box_height} px · 전체 PNG: {canvas_width_px} × {canvas_height_px} px · DPI: {export_dpi}")
    rows=[]
    for n in selected:
        s=next(z for z in series if z["name"]==n)
        rows.append(pd.DataFrame({"X":s["x"],n:proc[n]+auto[n]+style[n]["offset"]}))
    merged=pd.concat(rows,axis=1) if rows else pd.DataFrame()
    st.download_button("선택 곡선 데이터 CSV",merged.to_csv(index=False).encode("utf-8-sig"),"spectra_processed.csv","text/csv")
    out=io.StringIO()
    for n in selected:
        s=next(z for z in series if z["name"]==n)
        pd.DataFrame({"X":s["x"],"Y":proc[n]}).to_csv(out,index=False,sep="\t")
        out.write("\n")
    st.download_button("곡선별 처리 데이터 TXT",out.getvalue().encode("utf-8-sig"),"spectra_processed.txt","text/plain")
