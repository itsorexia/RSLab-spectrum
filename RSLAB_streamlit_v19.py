"""Streamlit port of the RSLAB spectrum 👨🏻‍🔬👩🏻‍🔬 (core workflow)."""
from __future__ import annotations
import ast, io, json, csv, re, zipfile, base64
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from matplotlib.colors import to_hex, to_rgb
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

st.set_page_config(page_title="RSLAB Spectrum 생성기 👨🏻‍🔬👩🏻‍🔬", layout="wide")
st.markdown("<div style='display:flex; align-items:baseline; gap:14px; margin:0 0 0.6rem 0; flex-wrap:wrap;'><span style='font-size:2.5rem; line-height:1.25; font-weight:700; color:#262730;'>RSLAB Spectrum 생성기 👨🏻‍🔬👩🏻‍🔬</span><span style='font-size:0.78rem; color:#b8b8b8;'>created by YS Park</span></div>", unsafe_allow_html=True)

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
            score = 0.38 * coverage + 0.30 * monotonic + 0.22 * unique + 0.10 * regularity - header_penalty
            scores.append((score,i))
        xi=max(scores)[1]
        best=max(scores)[0]
        # Avoid silently treating arbitrary values as an axis when none looks like one.
        if best<0.55:
            raise ValueError("X축 컬럼을 판단하지 못했습니다. 컬럼명에 X, Wavelength, Raman Shift 등을 표시해 주세요.")
    header=headers[xi]
    if any(token in header for token in ("wavelength","lambda","nm")):
        label="Wavelength (nm)"
    elif any(token in header for token in ("raman","shift","wavenumber","wave number","cm-1","cm⁻¹")):
        label="Raman Shift (cm⁻¹)"
    elif header.strip() in {"x","x-axis","x axis"}:
        label="X"
    else:
        label=str(df.columns[xi])
    return xi,label

def parse_upload(upload):
    """Parse ordinary tables and files containing repeated spectrum blocks."""
    text=read_text(upload)
    lines=text.splitlines()
    x_terms=("wavelength","raman","shift","wavenumber","wave number","cm-1","cm⁻¹","x-axis","x axis")
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
    # Re-import the paired X/Y table used when selected curves have different X grids.
    # Each curve has its own "name | X [axis label]" and "name | Y" columns;
    # an optional "name | Error" column is attached to that curve on import.
    if first_idx is not None:
        header_line=lines[first_idx]
        paired_sep="\t" if "\t" in header_line else ","
        try:
            header_cells=next(csv.reader([header_line],delimiter=paired_sep))
            common_x_match=re.match(r"^\s*X(?:\s*\[(.*?)\])?\s*$",str(header_cells[0]),re.IGNORECASE)
            if common_x_match:
                common_df=pd.read_csv(io.StringIO(text),sep=paired_sep,comment="#",on_bad_lines="skip")
                x_values=pd.to_numeric(common_df.iloc[:,0],errors="coerce")
                axis_label=(common_x_match.group(1) or "X").strip()
                common_errors={}
                for column_index,cell in enumerate(header_cells[1:],1):
                    match=re.match(r"^(.+?)\s*\|\s*Error$",str(cell).strip(),re.IGNORECASE)
                    if match:
                        common_errors[match.group(1).strip()]=column_index
                common_series=[]
                common_y_columns=[(column_index,str(cell).strip())
                                  for column_index,cell in enumerate(header_cells[1:],1)
                                  if not re.search(r"\s\|\sError$",str(cell).strip(),re.IGNORECASE)]
                for column_index,source_column_name in common_y_columns:
                    curve_name=(file_stem if len(common_y_columns)==1
                                else f"{file_stem} · {source_column_name}")
                    y_values=pd.to_numeric(common_df.iloc[:,column_index],errors="coerce")
                    valid=x_values.notna() & y_values.notna()
                    if valid.sum()<2:
                        continue
                    item={"name":curve_name,"x":x_values[valid].to_numpy(float),"raw":y_values[valid].to_numpy(float),
                          "source_file":upload.name,
                          "x_label":axis_label,"y_label":"Intensity"}
                    if source_column_name in common_errors:
                        error_values=pd.to_numeric(common_df.iloc[:,common_errors[source_column_name]],errors="coerce")
                        item["error"]=np.abs(error_values[valid].to_numpy(float))
                    common_series.append(item)
                if common_series:
                    return common_series
            x_pattern=re.compile(r"^(.+?)\s*\|\s*X(?:\s*\[(.*?)\])?$",re.IGNORECASE)
            y_pattern=re.compile(r"^(.+?)\s*\|\s*Y$",re.IGNORECASE)
            error_pattern=re.compile(r"^(.+?)\s*\|\s*Error$",re.IGNORECASE)
            paired_x={}; paired_y={}; paired_error={}; paired_axis_labels={}
            for column_index,cell in enumerate(header_cells):
                header=str(cell).strip()
                match=x_pattern.match(header)
                if match:
                    curve_name=match.group(1).strip()
                    paired_x[curve_name]=column_index
                    paired_axis_labels[curve_name]=(match.group(2) or "X").strip()
                    continue
                match=y_pattern.match(header)
                if match:
                    paired_y[match.group(1).strip()]=column_index
                    continue
                match=error_pattern.match(header)
                if match:
                    paired_error[match.group(1).strip()]=column_index
            paired_names=[name for name in paired_x if name in paired_y]
            if paired_names:
                paired_df=pd.read_csv(io.StringIO(text),sep=paired_sep,comment="#",on_bad_lines="skip")
                paired_series=[]
                for source_curve_name in paired_names:
                    curve_name=(file_stem if len(paired_names)==1
                                else f"{file_stem} · {source_curve_name}")
                    x=pd.to_numeric(paired_df.iloc[:,paired_x[source_curve_name]],errors="coerce")
                    y=pd.to_numeric(paired_df.iloc[:,paired_y[source_curve_name]],errors="coerce")
                    valid=x.notna() & y.notna()
                    if valid.sum()<2:
                        continue
                    item={"name":curve_name,"x":x[valid].to_numpy(float),"raw":y[valid].to_numpy(float),
                          "source_file":upload.name,
                          "x_label":paired_axis_labels.get(source_curve_name,"X"),"y_label":"Intensity"}
                    if source_curve_name in paired_error:
                        error=pd.to_numeric(paired_df.iloc[:,paired_error[source_curve_name]],errors="coerce")
                        item["error"]=np.abs(error[valid].to_numpy(float))
                    paired_series.append(item)
                if paired_series:
                    return paired_series
        except (ValueError,IndexError,pd.errors.ParserError):
            pass
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
                                       "mapping_source":upload.name})
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
                         "source_file":upload.name,
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
            xi,x_label=choose_x_column(df)
            x=pd.to_numeric(df.iloc[:,xi].astype(str).str.replace(",","",regex=False),errors="coerce")
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
                               "source_file":upload.name,
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
            xi,x_label=choose_x_column(df)
            x=pd.to_numeric(df.iloc[:,xi].astype(str).str.replace(",","",regex=False),errors="coerce")
            y_columns=[j for j in range(len(df.columns)) if j!=xi]
            for j in y_columns:
                y=pd.to_numeric(df.iloc[:,j].astype(str).str.replace(",","",regex=False),errors="coerce")
                good=x.notna() & y.notna()
                if good.sum()>=2:
                    label=file_stem if len(y_columns)==1 else f"{file_stem} · {df.columns[j]}"
                    series.append({"name":label,"x":x[good].to_numpy(float),"raw":y[good].to_numpy(float),
                                   "source_file":upload.name,
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
    reference_y=y
    normalization_range=opts.get("normalization_range")
    if normalization_range is not None:
        x=np.asarray(s["x"],dtype=float)
        range_min,range_max=sorted((float(normalization_range[0]),float(normalization_range[1])))
        in_range=np.isfinite(x)&np.isfinite(y)&(x>=range_min)&(x<=range_max)
        if np.any(in_range):
            reference_y=y[in_range]
        elif mode=="Maximum in X range":
            return y
    if mode=="Min-max":
        lo,hi=np.nanmin(reference_y),np.nanmax(reference_y)
        y=(y-lo)/(hi-lo) if hi!=lo else y*0
    elif mode in ("Maximum","Maximum in X range"):
        reference_max=np.nanmax(reference_y)
        if reference_max!=0: y=y/reference_max
    elif mode=="Custom divisor" and opts["divisor"]!=0: y=y/opts["divisor"]
    return y

def processed_error(s, opts):
    """Return supplied Y uncertainties in the same units as the processed curve."""
    error=s.get("error")
    if error is None:
        return None
    error=np.abs(np.asarray(error,dtype=float))
    mode=opts.get("normalization","없음")
    if mode=="없음":
        return error
    before_norm={**opts,"normalization":"없음","normalization_range":None}
    y=transform(s,before_norm)
    reference=y
    x=np.asarray(s["x"],dtype=float)
    selected_range=opts.get("normalization_range")
    if selected_range is not None:
        lo,hi=sorted((float(selected_range[0]),float(selected_range[1])))
        mask=np.isfinite(x)&np.isfinite(y)&(x>=lo)&(x<=hi)
        if np.any(mask): reference=y[mask]
    if mode=="Min-max":
        span=float(np.nanmax(reference)-np.nanmin(reference))
        return error/span if span>0 else error
    if mode in ("Maximum","Maximum in X range"):
        scale=abs(float(np.nanmax(reference)))
        if selected_range is not None:
            x_values=np.asarray(s["x"],dtype=float)
            range_min,range_max=sorted((float(selected_range[0]),float(selected_range[1])))
            in_range=np.isfinite(x_values)&np.isfinite(y)&(x_values>=range_min)&(x_values<=range_max)
            if not np.any(in_range) and mode=="Maximum in X range":
                return error
            if np.any(in_range):
                scale=abs(float(np.nanmax(y[in_range])))
        return error/scale if scale>0 else error
    if mode=="Custom divisor":
        scale=abs(float(opts.get("divisor",1.0)))
        return error/scale if scale>0 else error
    return error

def draw_curve(ax, s, y, label, color, linewidth, plot_style, error=None,
               scatter_marker=("o",True), error_capsize=3.0, error_capthick=1.0, error_linewidth=1.0):
    """Draw one series; scatter plots include supplied uncertainties as error bars."""
    x=np.asarray(s["x"],dtype=float)
    y=np.asarray(y,dtype=float)
    if plot_style=="산점도":
        marker,filled=scatter_marker
        facecolor=color if filled else "none"
        artist=ax.scatter(x,y,label=label,marker=marker,s=34,
                          facecolors=facecolor,edgecolors=color,linewidths=max(0.7,linewidth))
        if error is not None:
            err=np.asarray(error,dtype=float)
            if len(err)==len(y) and np.isfinite(err).any():
                ax.errorbar(x,y,yerr=err,fmt="none",ecolor=color,capsize=error_capsize,
                            capthick=error_capthick,elinewidth=error_linewidth,zorder=artist.get_zorder()-0.1)
        return artist
    return ax.plot(x,y,label=label,color=color,lw=linewidth)

def fit_curve_model(x, y, purpose, model, sigma=None):
    """Fit one predefined curve model and return predictions, parameters, and errors."""
    x=np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float)
    sigma_array=np.asarray(sigma,dtype=float) if sigma is not None else None
    fit_kwargs={"maxfev":50000}
    if sigma_array is not None and np.all(np.isfinite(sigma_array)&(sigma_array>0)):
        fit_kwargs.update({"sigma":sigma_array,"absolute_sigma":True})
    else:
        sigma_array=None

    if model in ("Linear","Quadratic","Cubic"):
        degree={"Linear":1,"Quadratic":2,"Cubic":3}[model]
        weights=(1/sigma_array) if sigma_array is not None else None
        coeffs,covariance=np.polyfit(x,y,degree,w=weights,cov=True)
        fitted=np.polyval(coeffs,x)
        params=[(f"a{degree-i}",float(value)) for i,value in enumerate(coeffs)]
        errors=np.sqrt(np.maximum(0,np.diag(covariance)))
        return np.polyval,coeffs,params,fitted,errors

    if model in ("Gaussian","Lorentzian","Pseudo-Voigt"):
        span=max(float(np.ptp(x)),np.finfo(float).eps)
        baseline0=float(np.percentile(y,10))
        peak_i=int(np.nanargmax(y))
        amplitude0=max(float(y[peak_i]-baseline0),np.finfo(float).eps)
        center0=float(x[peak_i])
        width0=max(span/10,np.finfo(float).eps*100)
        center_bounds=(float(np.min(x)),float(np.max(x)))
        if model=="Gaussian":
            names=["baseline","amplitude","center","sigma"]
            def fn(xx,b,a,c,w): return b+a*np.exp(-0.5*((xx-c)/w)**2)
            p0=[baseline0,amplitude0,center0,width0]
            lower=[-np.inf,-np.inf,center_bounds[0],np.finfo(float).eps]
            upper=[np.inf,np.inf,center_bounds[1],span*10]
        elif model=="Lorentzian":
            names=["baseline","amplitude","center","gamma"]
            def fn(xx,b,a,c,w): return b+a*(w*w)/((xx-c)**2+w*w)
            p0=[baseline0,amplitude0,center0,width0]
            lower=[-np.inf,-np.inf,center_bounds[0],np.finfo(float).eps]
            upper=[np.inf,np.inf,center_bounds[1],span*10]
        else:
            names=["baseline","amplitude","center","width","eta"]
            def fn(xx,b,a,c,w,eta):
                u=(xx-c)/w
                return b+a*(eta/(1+u*u)+(1-eta)*np.exp(-4*np.log(2)*u*u))
            p0=[baseline0,amplitude0,center0,width0,0.5]
            lower=[-np.inf,-np.inf,center_bounds[0],np.finfo(float).eps,0.0]
            upper=[np.inf,np.inf,center_bounds[1],span*10,1.0]
        popt,pcov=curve_fit(fn,x,y,p0=p0,bounds=(lower,upper),**fit_kwargs)
        fitted=fn(x,*popt)
        parameter_rows=[(name,float(value)) for name,value in zip(names,popt)]
        parameter_errors=np.sqrt(np.maximum(0,np.diag(pcov))) if np.all(np.isfinite(pcov)) else np.full(len(popt),np.nan)
        return fn,popt,parameter_rows,fitted,parameter_errors

    if model=="Exponential":
        center=float(np.mean(x))
        scale=max(float(np.ptp(x))/2,np.finfo(float).eps)
        z=(x-center)/scale
        baseline0=float(y[np.argmin(y)])
        amplitude0=float(np.max(y)-np.min(y))
        slope0=float(np.polyfit(z,y,1)[0])
        rate0=float(np.clip(slope0/max(abs(amplitude0),1e-12),-2,2))
        def fn(zv,a,k,c): return c+a*np.exp(np.clip(k*zv,-700,700))
        p0=[amplitude0,rate0,baseline0]
        popt,pcov=curve_fit(fn,z,y,p0=p0,bounds=([-np.inf,-50,-np.inf],[np.inf,50,np.inf]),**fit_kwargs)
        fitted=fn(z,*popt)
        rows=[("amplitude",float(popt[0])),("rate per half-range",float(popt[1])),("offset",float(popt[2]))]
        errors=np.sqrt(np.maximum(0,np.diag(pcov))) if np.all(np.isfinite(pcov)) else np.full(len(popt),np.nan)
        evaluator=lambda xx: fn((np.asarray(xx,dtype=float)-center)/scale,*popt)
        return evaluator,popt,rows,fitted,errors

    if model=="4PL":
        if np.any(x<0) or not np.any(x>0):
            raise ValueError("4PL 피팅에는 0 이상의 X값과 적어도 하나의 양수 X값이 필요합니다.")
        y_at_low_x=float(y[np.argmin(x)])
        y_at_high_x=float(y[np.argmax(x)])
        correlation=float(np.corrcoef(x,y)[0,1])
        direction=1.0 if not np.isfinite(correlation) or correlation>=0 else -1.0
        bottom0,top0=((y_at_low_x,y_at_high_x) if direction>0 else (y_at_high_x,y_at_low_x))
        positive_x=x[x>0]
        ec50_0=float(np.median(positive_x))
        def fn(xx,bottom,top,ec50,hill):
            xx=np.asarray(xx,dtype=float)
            exponent=np.empty_like(xx)
            positive=xx>0
            exponent[positive]=hill*(np.log(ec50)-np.log(xx[positive]))
            exponent[~positive]=700 if hill>=0 else -700
            exponent=np.clip(exponent,-700,700)
            return bottom+(top-bottom)/(1+np.exp(exponent))
        p0=[bottom0,top0,ec50_0,direction]
        lower=[-np.inf,-np.inf,max(float(np.min(positive_x))*1e-4,np.finfo(float).tiny),-20]
        upper=[np.inf,np.inf,float(np.max(positive_x))*1e4,20]
        popt,pcov=curve_fit(fn,x,y,p0=p0,bounds=(lower,upper),**fit_kwargs)
        fitted=fn(x,*popt)
        names=["bottom","top","EC50","Hill slope"]
        rows=[(name,float(value)) for name,value in zip(names,popt)]
        errors=np.sqrt(np.maximum(0,np.diag(pcov))) if np.all(np.isfinite(pcov)) else np.full(len(popt),np.nan)
        return fn,popt,rows,fitted,errors

    raise ValueError(f"지원하지 않는 피팅 모델입니다: {model}")

def stacked_curve_y(values, name, plotted_names, auto_offsets, offset_step, style,
                    log_stack=False, log_lifts=None):
    """Apply additive offsets for linear stacks or multiplicative offsets for log stacks."""
    values=np.asarray(values,dtype=float)
    offset=auto_offsets[name]+offset_step*plotted_names.index(name)+style[name]["offset"]
    if log_stack:
        lift=(log_lifts or {}).get(name,0.0)
        return (values+lift)*np.power(10.0,offset)
    return values+offset

def extend_ylim_for_fit(limits, fits, visible_ids, plotted_names, plot_names, auto_offsets,
                        offset_step, style, x_min, x_max, log_scale=False,
                        log_lifts=None, log_stack=False):
    """Expand automatic graph limits to include visible fit curves."""
    if limits is None:
        return None
    values=[]
    x_min,x_max=sorted((float(x_min),float(x_max)))
    for fit in fits:
        name=fit.get("curve_name")
        if fit.get("id") not in visible_ids or name not in plotted_names:
            continue
        xx=np.asarray(fit["x_fit"],dtype=float)
        yy=np.asarray(fit["y_fit"],dtype=float)
        mask=np.isfinite(xx)&np.isfinite(yy)&(xx>=x_min)&(xx<=x_max)
        plotted_y=stacked_curve_y(yy,name,plotted_names,auto_offsets,offset_step,style,log_stack,log_lifts)
        if log_scale: mask &= plotted_y>0
        if np.any(mask): values.extend(plotted_y[mask].tolist())
    if not values:
        return limits
    low=min(float(limits[0]),float(np.min(values)))
    high=max(float(limits[1]),float(np.max(values)))
    if log_scale:
        if low<=0: low=min(v for v in values if v>0)
        if high<=low: return limits
        ratio=(high/low)**0.05
        return low/ratio,high*ratio
    span=high-low
    padding=0.05*span if span>0 else max(abs(high)*0.05,0.01)
    return low-padding,high+padding

def draw_fit_overlays(ax, fits, visible_ids, plotted_names, plot_names, auto_offsets,
                      offset_step, style, log_stack=False, log_lifts=None):
    for fit in fits:
        name=fit.get("curve_name")
        if fit.get("id") not in visible_ids or name not in plotted_names:
            continue
        fit_style=fit.get("style",{})
        fit_y=stacked_curve_y(fit["y_fit"],name,plotted_names,auto_offsets,offset_step,
                              style,log_stack,log_lifts)
        ax.plot(fit["x_fit"],fit_y,
                linestyle=fit_style.get("linestyle","--"),
                linewidth=float(fit_style.get("linewidth",2.0)),
                color=fit_style.get("color",style[name]["color"]),
                label=f"Fit · {name} · {fit['model']}")

def compute_average_curve(target_names, series, values_by_name, axis_label):
    """Average selected curves on the first selected curve's grid over their shared X range."""
    lookup={item["name"]:item for item in series}
    curves=[]
    for name in target_names:
        item=lookup.get(name)
        if item is None or str(item.get("x_label","X"))!=str(axis_label) or name not in values_by_name:
            continue
        x=np.asarray(item["x"],dtype=float)
        y=np.asarray(values_by_name[name],dtype=float)
        valid=np.isfinite(x)&np.isfinite(y)
        x,y=x[valid],y[valid]
        if len(x)<2:
            continue
        order=np.argsort(x)
        x,y=x[order],y[order]
        unique_x,inverse=np.unique(x,return_inverse=True)
        if len(unique_x)<2:
            continue
        sums=np.bincount(inverse,weights=y)
        counts=np.bincount(inverse)
        curves.append((unique_x,sums/counts))
    if len(curves)<2:
        return None
    overlap_min=max(float(x[0]) for x,_ in curves)
    overlap_max=min(float(x[-1]) for x,_ in curves)
    if overlap_min>=overlap_max:
        return None
    reference_x=curves[0][0]
    reference_x=reference_x[(reference_x>=overlap_min)&(reference_x<=overlap_max)]
    if len(reference_x)<2:
        return None
    aligned=np.vstack([np.interp(reference_x,x,y) for x,y in curves])
    return {"name":f"평균 ({len(curves)}개 곡선)","x":reference_x,
            "y":np.mean(aligned,axis=0),"count":len(curves),"axis_label":axis_label}

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

def add_y_scale_annotation(ax, options, fontsize, color="black"):
    label=tick_scale_annotation(options)
    if label:
        # Anchor the scale label just above the plotting box's upper-left corner.
        ax.text(0.0,1.005,label,transform=ax.transAxes,ha="left",va="bottom",fontsize=fontsize,color=color,clip_on=False)

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
            offset=auto_offsets[name]+offset_step*plotted_names.index(name)+style[name]["offset"]
            limits.extend((y_values[mask]+offset).tolist())
    if not limits:
        return None
    low,high=float(np.nanmin(limits)),float(np.nanmax(limits))
    span=high-low
    padding=0.05*span if span>0 else max(abs(high)*0.05,0.01)
    return low-padding,high+padding

def visible_stack_log_ylim(series, processed, names, plotted_names, auto_offsets, offset_step,
                           style, x_min, x_max, log_lifts=None, log_stack=False):
    """Return positive, multiplicatively padded Y limits for a log axis."""
    values=[]
    by_name={item["name"]:item for item in series}
    x_min,x_max=sorted((float(x_min),float(x_max)))
    for name in plotted_names:
        item=by_name[name]
        x_values=np.asarray(item["x"],dtype=float)
        y_values=np.asarray(processed[name],dtype=float)
        plotted=stacked_curve_y(y_values,name,plotted_names,auto_offsets,offset_step,
                                style,log_stack,log_lifts)
        mask=(np.isfinite(x_values)&np.isfinite(plotted)&(x_values>=x_min)&(x_values<=x_max)&(plotted>0))
        values.extend(plotted[mask].tolist())
    if not values:
        return None
    low,high=float(np.min(values)),float(np.max(values))
    if np.isclose(low,high):
        return low/1.1,high*1.1
    ratio=(high/low)**0.05
    return low/ratio,high*ratio

def sync_widget_value(source_key, target_key):
    st.session_state[target_key]=st.session_state[source_key]

def synced_font_control(label, minimum, maximum, default, key):
    input_key=f"{key}_input"
    # Reconcile stale values before either keyed widget is instantiated.
    current=st.session_state.get(key,st.session_state.get(input_key,default))
    current=max(minimum,min(maximum,int(current)))
    st.session_state[key]=current
    st.session_state[input_key]=current
    slider_col,input_col=st.columns([5,1])
    slider_col.slider(label,min_value=minimum,max_value=maximum,key=key,
                      on_change=sync_widget_value,args=(key,input_key))
    input_col.number_input(label,min_value=minimum,max_value=maximum,step=1,key=input_key,
                          label_visibility="collapsed",help="숫자를 직접 입력할 수 있습니다.",
                          on_change=sync_widget_value,args=(input_key,key))
    return int(st.session_state[key])

def synced_float_control(label, minimum, maximum, default, step, key, disabled=False):
    input_key=f"{key}_input"
    # Reconcile stale values before either keyed widget is instantiated.
    current=float(st.session_state.get(key,st.session_state.get(input_key,default)))
    current=max(float(minimum),min(float(maximum),current))
    st.session_state[key]=current
    st.session_state[input_key]=current
    slider_col,input_col=st.columns([5,1])
    slider_col.slider(label,min_value=minimum,max_value=maximum,step=step,key=key,disabled=disabled,
                      on_change=sync_widget_value,args=(key,input_key))
    input_col.number_input(label,min_value=minimum,max_value=maximum,step=step,format="%.1f",key=input_key,
                           disabled=disabled,label_visibility="collapsed",help="숫자를 직접 입력할 수 있습니다.",
                           on_change=sync_widget_value,args=(input_key,key))
    return float(st.session_state[key])

def validate_curve_name(index):
    """Reject empty or duplicate curve names before the editor reruns."""
    series=st.session_state.get("series",[])
    if not 0 <= index < len(series):
        return
    key=f"nm_{index}"
    proposed=str(st.session_state.get(key,"")).strip()
    duplicate=any(i != index and str(item.get("name"," ")).strip().casefold()==proposed.casefold()
                  for i,item in enumerate(series))
    if not proposed or duplicate:
        st.session_state[key]=series[index].get("name","")
        st.session_state["curve_name_error"]=("곡선 이름은 비워둘 수 없습니다." if not proposed
                                               else f"'{proposed}' 이름을 이미 사용 중입니다. 다른 이름을 입력하세요.")
        return
    series[index]["name"]=proposed

def move_fit(index, direction):
    """Move a fit result in the overlay draw and legend order."""
    fits=st.session_state.get("fit_results",[])
    target=index+direction
    if not 0 <= index < len(fits) or not 0 <= target < len(fits):
        return
    fits[index],fits[target]=fits[target],fits[index]
    st.session_state["fit_results"]=fits

def swap_curve_positions(index, target):
    """Swap two curves and keep their position-keyed appearance settings attached."""
    series=st.session_state.get("series",[])
    if not 0 <= index < len(series) or not 0 <= target < len(series):
        return
    series[index],series[target]=series[target],series[index]
    for prefix in ("nm", "color", "lw", "lw_input", "off"):
        left,right=f"{prefix}_{index}",f"{prefix}_{target}"
        left_exists,right_exists=left in st.session_state,right in st.session_state
        left_value=st.session_state.get(left)
        right_value=st.session_state.get(right)
        if right_exists:
            st.session_state[left]=right_value
        elif left_exists:
            del st.session_state[left]
        if left_exists:
            st.session_state[right]=left_value
        elif right_exists:
            del st.session_state[right]

def move_curve_to(index, target):
    """Move a curve to a target position while preserving its appearance settings."""
    series=st.session_state.get("series",[])
    if not 0 <= index < len(series) or not 0 <= target < len(series):
        return
    step=1 if target>index else -1
    while index!=target:
        next_index=index+step
        swap_curve_positions(index,next_index)
        index=next_index

def move_curve(index, direction):
    move_curve_to(index,index+direction)

def reverse_curve_order():
    """Reverse all curves and preserve each curve's keyed appearance settings."""
    series=st.session_state.get("series",[])
    for index in range(len(series)//2):
        swap_curve_positions(index,len(series)-1-index)

def move_selected_curves(direction):
    """Move selected curves together while retaining their relative order."""
    series=st.session_state.get("series",[])
    selected=set(st.session_state.get("curve_order_selection",[]))
    selected_order=[curve["name"] for curve in series if curve["name"] in selected]
    if not selected_order:
        return
    if direction=="up":
        for index in range(1,len(series)):
            if series[index]["name"] in selected and series[index-1]["name"] not in selected:
                swap_curve_positions(index,index-1)
    elif direction=="down":
        for index in range(len(series)-2,-1,-1):
            if series[index]["name"] in selected and series[index+1]["name"] not in selected:
                swap_curve_positions(index,index+1)
    elif direction=="top":
        desired_order=selected_order+[curve["name"] for curve in series if curve["name"] not in selected]
    elif direction=="bottom":
        desired_order=[curve["name"] for curve in series if curve["name"] not in selected]+selected_order
    else:
        return
    if direction in ("top","bottom"):
        for target,name in enumerate(desired_order):
            current=next((i for i,item in enumerate(series) if item["name"]==name),None)
            if current is not None:
                move_curve_to(current,target)

def apply_axis_label_positions(ax,x_mode,x_pad,x_coords,y_mode,y_pad,y_coords):
    """Apply axis-title distance from ticks or axes-relative placement."""
    if x_mode=="그래프 좌표 지정":
        ax.xaxis.set_label_coords(*x_coords)
    else:
        ax.xaxis.labelpad=x_pad
    if y_mode=="그래프 좌표 지정":
        ax.yaxis.set_label_coords(*y_coords)
    else:
        ax.yaxis.labelpad=y_pad

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

def extract_uploaded_palette(upload):
    """Read an ordered hex list or extract dominant colors from an uploaded image."""
    if upload is None:
        return []
    suffix=Path(upload.name).suffix.lower()
    if suffix in (".png",".jpg",".jpeg",".webp"):
        with Image.open(io.BytesIO(upload.getvalue())) as source:
            rgba=source.convert("RGBA")
            background=Image.new("RGBA",rgba.size,"white")
            background.alpha_composite(rgba)
            quantized=background.convert("RGB").quantize(colors=16,method=Image.Quantize.MEDIANCUT)
            palette=quantized.getpalette()
            counts=sorted(quantized.getcolors(maxcolors=256) or [],reverse=True)
            colors=[]
            for count,index in counts:
                base=index*3
                color=to_hex(tuple(channel/255 for channel in palette[base:base+3]))
                if color not in colors:
                    colors.append(color)
            return colors
    text=read_text(upload)
    tokens=re.findall(r"#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3}(?![0-9a-fA-F])",text)
    colors=[]
    for token in tokens:
        if len(token)==4:
            token="#"+"".join(char*2 for char in token[1:])
        token=token.upper()
        if token not in colors:
            colors.append(token)
    return colors

def interpolate_palette(colors,count,index):
    """Sample an uploaded palette evenly, interpolating when there are few swatches."""
    if len(colors)==1 or count<=1:
        return colors[0]
    points=np.asarray([to_rgb(color) for color in colors],dtype=float)
    position=index*(len(points)-1)/(count-1)
    left=int(np.floor(position))
    right=min(left+1,len(points)-1)
    fraction=position-left
    rgb=points[left]*(1-fraction)+points[right]*fraction
    return to_hex(rgb)

def state_json(series):
    data=[]
    for s in series:
        data.append({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in s.items()})
    return json.dumps(data,ensure_ascii=False)

def refresh_linked_errors(items):
    """Attach imported error-curve Y values to targets on the target X grid."""
    lookup={item["name"]:item for item in items}
    source_names={item.get("error_source") for item in items if item.get("error_source")}
    for target in items:
        source_name=target.get("error_source")
        # Preserve errors entered directly in the spreadsheet. Only refresh an
        # error array when this target is explicitly linked to a separate curve.
        if not source_name:
            continue
        source=lookup.get(source_name)
        if source is None:
            continue
        sx=np.asarray(source["x"],dtype=float)
        sy=np.abs(np.asarray(source["raw"],dtype=float))
        good=np.isfinite(sx)&np.isfinite(sy)
        if good.sum()<2:
            target.pop("error",None)
            continue
        sx,sy=sx[good],sy[good]
        order=np.argsort(sx,kind="stable")
        sx,sy=sx[order],sy[order]
        unique_x,inverse=np.unique(sx,return_inverse=True)
        if len(unique_x)!=len(sx):
            sums=np.bincount(inverse,weights=sy)
            counts=np.bincount(inverse)
            sy=sums/counts
            sx=unique_x
        tx=np.asarray(target["x"],dtype=float)
        err=np.interp(tx,sx,sy,left=np.nan,right=np.nan)
        target["error"]=err
    return source_names

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

def export_curve_table(target_names, series, values_by_name, errors_by_name=None):
    """Keep each curve's original X/Y pairs intact in a convenient table."""
    targets=[n for n in target_names if n in values_by_name]
    if not targets:
        return pd.DataFrame()
    lookup={s["name"]:s for s in series}
    first_x=np.asarray(lookup[targets[0]]["x"],dtype=float)
    same_grid=all(
        len(lookup[n]["x"])==len(first_x) and
        np.array_equal(np.asarray(lookup[n]["x"],dtype=float),first_x,equal_nan=True)
        for n in targets[1:]
    )
    if same_grid:
        x_label=str(lookup[targets[0]].get("x_label","X")).strip() or "X"
        x_header="X" if x_label.casefold()=="x" else f"X [{x_label}]"
        data={x_header:first_x}
        for n in targets:
            data[n]=np.asarray(values_by_name[n],dtype=float)
            error=(errors_by_name or {}).get(n,lookup[n].get("error"))
            if error is not None:
                data[f"{n} | Error"]=np.asarray(error,dtype=float)
        return pd.DataFrame(data)
    data={}
    for n in targets:
        x_label=str(lookup[n].get("x_label","X")).strip() or "X"
        data[f"{n} | X [{x_label}]"]=np.asarray(lookup[n]["x"],dtype=float)
        data[f"{n} | Y"]=np.asarray(values_by_name[n],dtype=float)
        error=(errors_by_name or {}).get(n,lookup[n].get("error"))
        if error is not None:
            data[f"{n} | Error"]=np.asarray(error,dtype=float)
    return pd.DataFrame({key:pd.Series(value) for key,value in data.items()})

def curve_archive(target_names, series, values_by_name, extension, errors_by_name=None):
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
            error=(errors_by_name or {}).get(name,lookup[name].get("error"))
            if error is not None:
                table["Error"]=np.asarray(error,dtype=float)
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
st.session_state.setdefault("fit_results",[])

def render_blank_spectrum(prefix):
    """Provide a spreadsheet-like grid for direct cell editing and multi-cell paste."""
    st.caption("표의 첫 번째 셀(X)에 커서를 두고 엑셀에서 복사한 행렬을 Ctrl+V로 붙여넣으세요. 에러바를 쓸 때는 Y 열과 오차 열 수를 설정하고, 각 Y에 연결할 오차 열을 선택하세요.")
    new_name=st.text_input("곡선 이름",value="새 스펙트럼",key=f"{prefix}_name")
    label1,label2=st.columns(2)
    x_label=label1.text_input("X축 이름",value="X",key=f"{prefix}_xlabel")
    y_label=label2.text_input("Y축 이름",value="Intensity",key=f"{prefix}_ylabel")
    y_count=st.number_input("붙여넣을 Y 데이터 열 수",min_value=1,max_value=20,value=1,step=1,
                            key=f"{prefix}_y_column_count",
                            help="X열을 제외한 Y열 수입니다. 예: X와 신호 3개를 붙여넣으려면 3을 선택하세요.")
    y_columns=["Y"] if y_count==1 else [f"Y{i+1}" for i in range(int(y_count))]
    error_enabled=st.checkbox("에러바 데이터 열 포함",value=False,key=f"{prefix}_error_enabled")
    error_count=0
    error_columns=[]
    if error_enabled:
        error_count=st.number_input("붙여넣을 오차 데이터 열 수",min_value=1,max_value=20,
                                    value=int(y_count),step=1,key=f"{prefix}_error_column_count")
        error_columns=["Error"] if error_count==1 else [f"Error{i+1}" for i in range(int(error_count))]
    columns=["X",*y_columns,*error_columns]
    editor_key=f"{prefix}_sheet_editor_{int(y_count)}_{int(error_count)}"
    sheet_data=pd.DataFrame(np.nan,index=range(20),columns=columns)
    config={col:st.column_config.NumberColumn(str(col),format="%.8g") for col in columns}
    edited=st.data_editor(sheet_data,num_rows="dynamic",use_container_width=True,hide_index=True,
                          height=420,key=editor_key,column_config=config,placeholder="")
    error_for_y={}
    if error_columns:
        st.markdown("**Y 열과 에러바 열 연결**")
        for yi,y_col in enumerate(y_columns):
            default_error=error_columns[yi] if yi<len(error_columns) else "없음"
            options=["없음",*error_columns]
            default_index=options.index(default_error)
            error_for_y[y_col]=st.selectbox(f"{y_col}의 에러바 열",options,index=default_index,
                                            key=f"{prefix}_error_for_{yi}")
    if st.button("입력한 데이터로 그래프 그리기",type="primary",key=f"{prefix}_add"):
        x=pd.to_numeric(edited["X"],errors="coerce")
        base_name=str(new_name).strip() or "새 스펙트럼"
        new_items=[]
        for col in y_columns:
            y=pd.to_numeric(edited[col],errors="coerce")
            valid=x.notna() & y.notna()
            if int(valid.sum())<2:
                continue
            curve_name=base_name if len(y_columns)==1 else f"{base_name} · {col}"
            item={"name":curve_name,"x":x[valid].to_numpy(float),"raw":y[valid].to_numpy(float),
                  "x_label":str(x_label).strip() or "X",
                  "y_label":str(y_label).strip() or "Intensity"}
            err_col=error_for_y.get(col,"없음")
            if err_col!="없음":
                err=pd.to_numeric(edited[err_col],errors="coerce")
                item["error"]=np.abs(err[valid].to_numpy(float))
            new_items.append(item)
        if not new_items:
            st.error("그래프를 그리려면 X와 Y 열에 숫자 데이터가 2행 이상 필요합니다.")
        else:
            old_count=len(st.session_state.series)
            st.session_state.series=make_series_names_unique([*st.session_state.series,*new_items])
            added_names=[item["name"] for item in st.session_state.series[old_count:]]
            st.session_state["graph_selected"]=list(dict.fromkeys([*st.session_state.get("graph_selected",[]),*added_names]))
            st.rerun()

with st.sidebar:
    st.header("데이터")
    uploads=st.file_uploader("CSV / TSV / TXT 파일",type=["csv","tsv","txt","dat","asc"],accept_multiple_files=True)
    if st.button("데이터 불러오기", type="primary", use_container_width=True):
        loaded=[]; errors=[]
        previous_names=[s["name"] for s in st.session_state.series]
        for f in uploads or []:
            try:
                parsed_curves=parse_upload(f)
                for item in parsed_curves:
                    item["source_file"]=f.name
                loaded.extend(parsed_curves)
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
        st.session_state["fit_results"]=[]
        st.session_state["graph_visible_fit_ids"]=[]
    if st.session_state.series:
        st.download_button("현재 세션 저장 (.json)",state_json(st.session_state.series),"rslab_session.json","application/json",use_container_width=True)
        session=st.file_uploader("세션 불러오기 (.json)",type="json",key="session_upload")
        if session and st.button("세션 적용"):
            raw=json.loads(session.getvalue().decode("utf-8"))
            restored=[{**s,"x":np.asarray(s["x"]),"raw":np.asarray(s["raw"]),
                       **({"error":np.asarray(s["error"],dtype=float)} if s.get("error") is not None else {})}
                      for s in raw]
            st.session_state.series=make_series_names_unique(restored)
            st.session_state["fit_results"]=[]
            st.session_state["graph_visible_fit_ids"]=[]
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
        all_names=[s["name"] for s in st.session_state.series]
        linked_error_sources={s.get("error_source") for s in st.session_state.series if s.get("error_source")}
        ordinary_candidates=[n for n in all_names if n not in linked_error_sources and
                             not next((item.get("mapping_type") in ("time","space") for item in st.session_state.series if item["name"]==n),False)]
        with st.expander("불러온 곡선에서 에러 데이터 지정",expanded=False):
            st.caption("업로드 파일에서 오차값이 별도 곡선으로 읽혔다면, 그 곡선을 오차 데이터로 지정해 다른 곡선에 연결하세요. X값이 다르면 공통 범위에서 보간합니다.")
            if len(ordinary_candidates)>=2:
                target_name=st.selectbox("오차를 적용할 신호 곡선",ordinary_candidates,key="loaded_error_target")
                error_choices=[n for n in all_names if n!=target_name and
                               not next((item.get("mapping_type") in ("time","space") for item in st.session_state.series if item["name"]==n),False)]
                source_name=st.selectbox("오차값으로 사용할 곡선",error_choices,key="loaded_error_source")
                target_item=next(item for item in st.session_state.series if item["name"]==target_name)
                if target_item.get("error_source"):
                    st.caption(f"현재 연결: {target_item['error_source']}")
                map_col,unlink_col=st.columns(2)
                if map_col.button("에러 데이터 연결",key="loaded_error_link",use_container_width=True):
                    target_item["error_source"]=source_name
                    st.session_state.series=st.session_state.series
                    st.rerun()
                if unlink_col.button("연결 해제",key="loaded_error_unlink",disabled=not target_item.get("error_source"),use_container_width=True):
                    target_item.pop("error_source",None)
                    target_item.pop("error",None)
                    st.session_state.series=st.session_state.series
                    st.rerun()
            else:
                st.caption("에러 곡선을 연결하려면 일반 데이터 곡선이 2개 이상 필요합니다.")

if not st.session_state.series:
    st.info("왼쪽에서 파일을 불러오거나 아래 빈 스프레드시트에 직접 데이터를 입력하세요.")
    st.subheader("빈 스프레드시트")
    render_blank_spectrum("blank_start")
    st.stop()
series=st.session_state.series
error_source_names=refresh_linked_errors(series)
names=[s["name"] for s in series]
plot_names=[n for n in names if n not in error_source_names]

@st.dialog("스프레드시트 편집",width="large")
def show_curve_editor():
    curve_name=st.session_state.get("curve_editor_name")
    curve=next((item for item in st.session_state.series if item["name"]==curve_name),None)
    if curve is None:
        st.warning("편집할 곡선을 찾을 수 없습니다.")
        return
    st.caption(f"{curve_name}의 X/Y와 선택적 Error 데이터를 편집합니다. 빈 행은 저장할 때 제외됩니다.")
    edit_key=st.session_state.get("curve_editor_table_key","curve_editor_table")
    edited=st.data_editor(st.session_state.get("curve_editor_table_data",pd.DataFrame()),
                          num_rows="dynamic",use_container_width=True,hide_index=True,key=edit_key,
                          column_config={"X":st.column_config.NumberColumn("X",format="%.8g"),
                                         "Y":st.column_config.NumberColumn("Y",format="%.8g"),
                                         "Error":st.column_config.NumberColumn("Error (에러바)",format="%.8g")})
    apply_col,close_col=st.columns(2)
    if apply_col.button("수정 내용 적용",type="primary",key="curve_editor_apply"):
        x=pd.to_numeric(edited.get("X",pd.Series(dtype=float)),errors="coerce")
        y=pd.to_numeric(edited.get("Y",pd.Series(dtype=float)),errors="coerce")
        valid=x.notna() & y.notna()
        if int(valid.sum())<2:
            st.error("그래프 데이터는 숫자로 된 X/Y 쌍이 2개 이상 필요합니다.")
        else:
            curve["x"]=x[valid].to_numpy(float)
            curve["raw"]=y[valid].to_numpy(float)
            error=pd.to_numeric(edited.get("Error",pd.Series(np.nan,index=edited.index)),errors="coerce")
            error_values=np.abs(error[valid].to_numpy(float))
            if np.isfinite(error_values).any():
                curve["error"]=error_values
            else:
                curve.pop("error",None)
            curve.pop("mapping_type",None)
            curve.pop("mapping_coordinate",None)
            curve.pop("mapping_source",None)
            st.session_state.series=st.session_state.series
            st.session_state["curve_editor_open"]=False
            st.rerun()
    if close_col.button("닫기",key="curve_editor_close"):
        st.session_state["curve_editor_open"]=False
        st.rerun()

if st.session_state.get("curve_editor_open"):
    show_curve_editor()
mapped=[s for s in series if s.get("mapping_type") in ("time","space")]
spatial=[s for s in mapped if s.get("mapping_type")=="space" and isinstance(s.get("mapping_coordinate"),(tuple,list)) and len(s["mapping_coordinate"])>=2]
mapped_source_groups={}
for spec in mapped:
    source=spec.get("mapping_source",spec["name"].split(" · ")[0])
    mapped_source_groups.setdefault(source,[]).append(spec)
mapped_sources=list(mapped_source_groups)
mapped_names={s["name"] for s in mapped}
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
                file_names={s["name"] for s in map_specs}
                other_selected=[n for n in st.session_state.get("graph_selected",[]) if n not in file_names]
                st.session_state["graph_selected"]=list(dict.fromkeys([*other_selected,*picked_names]))
                group_key=f"mapped_curve_selection::{chosen_source}"
                st.session_state[group_key]=picked_names
                st.caption(f"선택한 좌표: {len(picked_names)}개")
        except (AttributeError,KeyError,TypeError,IndexError):
            pass

def selection_label(name):
    spec=next((item for item in mapped if item["name"]==name),None)
    if spec is None:
        spec=next((item for item in series if item["name"]==name),None)
        label=name
        if spec is not None:
            source=spec.get("source_file")
            if source:
                stem=Path(source).stem
                for prefix in (f"{stem} · ",f"{stem} - ",f"{stem}_"):
                    if label.startswith(prefix):
                        label=label[len(prefix):]
                        break
                if label==stem:
                    label="전체 스펙트럼"
        return label if len(label)<=42 else f"…{label[-39:]}"
    coord=spec.get("mapping_coordinate")
    stem=Path(spec.get("mapping_source",spec["name"].split(" · ")[0])).stem
    short_stem=stem if len(stem)<=18 else f"{stem[:15]}…"
    if spec.get("mapping_type")=="space" and isinstance(coord,(tuple,list)) and len(coord)>=2:
        return f"{short_stem} · X={coord[0]:g}, Y={coord[1]:g}"
    value=coord[0] if isinstance(coord,(tuple,list)) else coord
    return f"{short_stem} · 시간={value:g}"
with st.sidebar:
    st.markdown("**그릴 스펙트럼 선택**")
    regular_specs=[s for s in series if s["name"] not in mapped_names and s["name"] not in error_source_names]
    regular_source_groups={}
    for spec in regular_specs:
        source=spec.get("source_file") or "직접 입력 / 기타 데이터"
        regular_source_groups.setdefault(source,[]).append(spec)
    if st.button("전체 그리기",key="select_all_graph_curves",use_container_width=True):
        for source,group_specs in regular_source_groups.items():
            st.session_state[f"regular_curve_selection::{source}"]=[s["name"] for s in group_specs]
        for source,group_specs in mapped_source_groups.items():
            st.session_state[f"mapped_curve_selection::{source}"]=[
                s["name"] for s in group_specs if s["name"] not in error_source_names
            ]
        st.session_state["graph_selected_regular"]=[s["name"] for s in regular_specs]
    prior_regular=st.session_state.get("graph_selected_regular")
    if prior_regular is None:
        prior_regular=st.session_state.get("graph_selected",names)
    selected_for_plot=[]
    for source,group_specs in regular_source_groups.items():
        group_names=[s["name"] for s in group_specs]
        group_key=f"regular_curve_selection::{source}"
        group_values=st.session_state.get(group_key)
        if group_values is None:
            group_values=[name for name in prior_regular if name in group_names]
        else:
            group_values=[name for name in group_values if name in group_names]
        st.session_state[group_key]=group_values
        heading=f"{Path(source).name} · {len(group_names)}개" if source!="직접 입력 / 기타 데이터" else f"{source} · {len(group_names)}개"
        with st.expander(heading,expanded=False):
            st.multiselect("이 파일에서 그릴 곡선",group_names,key=group_key,
                           format_func=selection_label,label_visibility="collapsed")
        selected_for_plot.extend(st.session_state[group_key])
    st.session_state["graph_selected_regular"]=[name for group in regular_source_groups.values() for name in st.session_state[f"regular_curve_selection::{group[0].get('source_file') or '직접 입력 / 기타 데이터'}"]]
    for source,group_specs in mapped_source_groups.items():
        group_names=[s["name"] for s in group_specs if s["name"] not in error_source_names]
        group_key=f"mapped_curve_selection::{source}"
        group_values=st.session_state.get(group_key)
        if group_values is None or any(name not in group_names for name in group_values):
            group_values=[name for name in st.session_state.get("graph_selected",[]) if name in group_names]
            st.session_state[group_key]=group_values
        with st.expander(f"{Path(source).name} · {len(group_names)}개",expanded=False):
            st.multiselect("이 파일에서 그릴 곡선",group_names,key=group_key,
                           format_func=selection_label,label_visibility="collapsed")
        selected_for_plot.extend(st.session_state[group_key])
    st.session_state["graph_selected"]=list(dict.fromkeys(selected_for_plot))
tabs=st.tabs(["그래프 및 처리","곡선 평균","피크 분석","피팅","내보내기","데이터 시트"])
with st.sidebar:
    st.header("처리")
    normalization_tab,baseline_tab,smoothing_tab=st.tabs(["Normalization","Baseline","SG 필터"])
    with normalization_tab:
        normalization_enabled=st.checkbox("Normalization 적용",value=False,key="normalization_enabled")
        if normalization_enabled:
            normalization=st.selectbox("Normalization 종류",["Min-max","Maximum","지정 X 범위 최대값으로 나누기","Custom divisor"],key="normalization_method")
            if normalization=="지정 X 범위 최대값으로 나누기":
                st.caption("각 스펙트럼의 지정 X 범위 내 최대값으로 나눈 뒤, 같은 배율을 스펙트럼 전체에 적용합니다.")
                norm_default_specs=[s for s in series if s["name"] in st.session_state.get("graph_selected",[]) and s["name"] not in error_source_names]
                if not norm_default_specs:
                    norm_default_specs=[s for s in series if s["name"] not in error_source_names]
                norm_x_arrays=[np.asarray(s["x"],dtype=float) for s in norm_default_specs]
                norm_x_values=np.concatenate(norm_x_arrays) if norm_x_arrays else np.asarray([],dtype=float)
                norm_x_values=norm_x_values[np.isfinite(norm_x_values)]
                norm_x_min_default=float(np.min(norm_x_values)) if norm_x_values.size else 0.0
                norm_x_max_default=float(np.max(norm_x_values)) if norm_x_values.size else 1.0
                norm_x1,norm_x2=st.columns(2)
                normalization_x_min=norm_x1.number_input("정규화 X 최소",value=norm_x_min_default,format="%.6g",key="normalization_custom_xmin")
                normalization_x_max=norm_x2.number_input("정규화 X 최대",value=norm_x_max_default,format="%.6g",key="normalization_custom_xmax")
                normalization_scope="지정 X 범위"
                divisor=1.0
            else:
                normalization_x_min,normalization_x_max=None,None
                divisor=st.number_input("사용자 나눗수",value=1.0,format="%.6g",key="normalization_divisor") if normalization=="Custom divisor" else 1.0
            if normalization in ("Min-max","Maximum"):
                normalization_scope=st.selectbox("정규화 기준 범위",["전체 데이터","현재 그래프 X 범위"],key="normalization_scope",
                                                 help="선택한 범위에서 정규화 기준값을 계산한 뒤 곡선 전체에 적용합니다.")
                st.caption("현재 그래프 X 범위를 선택하면 축 설정의 X 최소·최대 사이 데이터로 기준값을 계산합니다.")
            elif normalization!="지정 X 범위 최대값으로 나누기":
                normalization_scope="전체 데이터"
        else:
            normalization="없음"
            normalization_scope="전체 데이터"
            normalization_x_min,normalization_x_max=None,None
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
        with st.container(height=650,border=False):
            st.subheader("그래프 설정")
            size_tab,axis_tab,stack_tab,visual_tab=st.tabs(["크기·글꼴","축 설정","수직 스택","비주얼"])
            with stack_tab:
                selected=st.session_state.get("graph_selected",names.copy())
                stack=st.checkbox("수직 스택 사용",False,key="graph_stack")
                log_stack_ui=bool(st.session_state.get("graph_y_log",False)) and stack
                offset_label="곡선 간 추가 간격 (log₁₀ 단위)" if log_stack_ui else "곡선 간 추가 간격"
                offset_step=st.number_input(offset_label,value=0.0,key="graph_stack_step")
                if log_stack_ui:
                    st.caption("로그 스케일 수직 스택은 곡선을 양수로 보정한 뒤 log₁₀ 단위로 쌓습니다. 첫 번째 곡선이 위쪽에 배치됩니다.")
                else:
                    st.caption("현재 X축 범위 안의 최대 intensity에 10%를 더해 자동 간격을 계산합니다. 첫 번째 곡선이 위쪽에 배치됩니다.")
                selected_stack_names=[name for name in selected if name in plot_names]
                st.caption(f"자동 간격 계산 대상: 선택한 곡선 {len(selected_stack_names)}개")
                with st.expander("곡선별 오프셋",expanded=False):
                    st.caption("각 곡선의 세로 위치를 개별 조정합니다."+(" 로그 스택에서는 log₁₀ 단위입니다." if log_stack_ui else ""))
                    selected_offset_curves=[(i,s) for i,s in enumerate(series) if s["name"] in selected_stack_names]
                    if not selected_offset_curves:
                        st.caption("곡선별 오프셋을 설정하려면 먼저 곡선을 선택하세요.")
                    for i,s in selected_offset_curves:
                        offset_name=f"{s['name']} 오프셋 (log₁₀ 단위)" if log_stack_ui else f"{s['name']} 오프셋"
                        s["offset"]=st.number_input(offset_name,value=float(s.get("offset",0)),key=f"off_{i}")
            with visual_tab:
                if st.session_state.get("graph_plot_style")=="에러바":
                    st.session_state["graph_plot_style"]="산점도"
                plot_style=st.selectbox("플롯 스타일",["라인 플롯","산점도"],key="graph_plot_style")
                scatter_marker=("o",True)
                error_capsize,error_capthick,error_linewidth=3.0,1.0,1.0
                if plot_style=="산점도":
                    marker_options={"꽉 찬 원":("o",True),"안 찬 원":("o",False),
                                    "꽉 찬 네모":("s",True),"안 찬 네모":("s",False),
                                    "꽉 찬 삼각형":("^",True),"안 찬 삼각형":("^",False),
                                    "꽉 찬 마름모":("D",True),"안 찬 마름모":("D",False)}
                    marker_label=st.selectbox("점 모양",list(marker_options),key="graph_scatter_marker")
                    scatter_marker=marker_options[marker_label]
                    has_error_data=any(s["name"] in selected and s.get("error") is not None and
                                       np.isfinite(np.asarray(s["error"],dtype=float)).any() for s in series)
                    st.caption("연결된 오차 데이터가 있으면 산점도 점에 에러바가 함께 표시됩니다.")
                    if has_error_data:
                        st.caption("정규화 배율은 오차에도 반영됩니다. baseline·SG 필터의 오차 전파는 계산하지 않습니다.")
                        cap_col,thickness_col=st.columns(2)
                        error_capsize=cap_col.number_input("에러바 머리 길이 (pt)",min_value=0.0,max_value=15.0,value=3.0,step=0.5,key="graph_error_capsize")
                        error_linewidth=thickness_col.number_input("에러바 두께 (머리+선)",min_value=0.1,max_value=5.0,value=1.0,step=0.1,key="graph_error_thickness")
                        error_capthick=error_linewidth
                fit_results=st.session_state.get("fit_results",[])
                fit_ids=[fit["id"] for fit in fit_results]
                pending_fit_id=st.session_state.pop("_pending_fit_overlay_id",None)
                if pending_fit_id in fit_ids:
                    st.session_state["graph_show_fit_overlay"]=True
                    current_before=st.session_state.get("graph_visible_fit_ids",[])
                    st.session_state["graph_visible_fit_ids"]=list(dict.fromkeys([*current_before,pending_fit_id]))
                if fit_ids:
                    valid_fit_ids=set(fit_ids)
                    current_fit_ids=st.session_state.get("graph_visible_fit_ids")
                    if current_fit_ids is None:
                        st.session_state["graph_visible_fit_ids"]=fit_ids.copy()
                    else:
                        st.session_state["graph_visible_fit_ids"]=[fit_id for fit_id in current_fit_ids if fit_id in valid_fit_ids]
                    show_fit_overlay=st.checkbox("피팅 곡선 그래프에 표시",value=True,key="graph_show_fit_overlay")
                    fit_labels={fit["id"]:f"{fit['curve_name']} · {fit['model']} · {fit['purpose']}" for fit in fit_results}
                    visible_fit_ids=(st.multiselect("표시할 피팅 결과",fit_ids,
                                                    format_func=lambda fit_id:fit_labels.get(fit_id,fit_id),
                                                    key="graph_visible_fit_ids") if show_fit_overlay else [])
                else:
                    show_fit_overlay=False
                    visible_fit_ids=[]
                    st.caption("피팅 탭에서 결과를 계산하면 이 그래프에 겹쳐 표시할 수 있습니다.")
                palette_upload=st.file_uploader("색상 팔레트 업로드 (이미지 / TXT / CSV)",
                                                 type=["png","jpg","jpeg","webp","txt","csv"],
                                                 key="graph_palette_upload",
                                                 help="이미지는 주요 색상 최대 16개를 추출하고, TXT/CSV는 #RRGGBB 코드를 파일 순서대로 읽습니다.")
                palette_template=("# RSLAB color palette template\n"
                                  "# 한 줄에 HEX 색상 코드를 하나씩 입력하세요.\n"
                                  "#R4E4F2\n#76B7B2\n#59A14F\n#EDC948\n"
                                  "#F28E2B\n#E15759\n#B07AA1\n#4E79A7\n")
                st.download_button("색상 팔레트 템플릿 다운로드",palette_template,
                                   "color_palette_template.txt","text/plain",key="palette_template_download")
                try:
                    uploaded_palette=extract_uploaded_palette(palette_upload)
                except Exception as e:
                    uploaded_palette=[]
                    st.warning(f"팔레트를 읽을 수 없습니다: {e}")
                palette_options=["Origin","viridis","plasma","inferno","magma","cividis","turbo","coolwarm","rainbow"]
                if uploaded_palette:
                    palette_options.append("업로드 팔레트")
                    st.caption(f"업로드 팔레트 색상 {len(uploaded_palette)}개")
                elif st.session_state.get("graph_palette")=="업로드 팔레트":
                    st.session_state["graph_palette"]="Origin"
                palette=st.selectbox("색상 팔레트",palette_options,key="graph_palette")
                if palette=="업로드 팔레트":
                    preview_cols=st.columns(min(len(uploaded_palette),8))
                    for j,color in enumerate(uploaded_palette):
                        with preview_cols[j%len(preview_cols)]:
                            st.markdown(f"<div title='{color}' style='height:22px;background:{color};border:1px solid #d1d5db;border-radius:5px'></div><small>{color}</small>",unsafe_allow_html=True)
                    st.caption("각 선택 곡선의 메뉴에서 사용할 팔레트 색을 직접 고를 수 있습니다.")
                color_mode=st.radio("색상 설정",["팔레트 사용","곡선별 지정"],horizontal=True,key="graph_color_mode")
                width_mode=st.radio("선 굵기 설정",["전체 동일","곡선별 지정"],horizontal=True,key="graph_width_mode")
                line_width=synced_float_control("전체 선 굵기",0.5,5.0,1.8,0.1,"graph_line_width")
                show_graph_frame=st.checkbox("위쪽·오른쪽 테두리 표시",value=True,key="graph_show_frame",
                                             help="체크를 해제해도 왼쪽 Y축과 아래쪽 X축 선은 유지됩니다.")
                graph_frame_width=synced_float_control("축 / 테두리 선 두께",0.1,5.0,1.0,0.1,"graph_frame_width",disabled=not show_graph_frame)
                old_names=names.copy()
                with st.expander("곡선 순서 변경",expanded=False):
                    st.caption("곡선을 여러 개 선택한 뒤 이동 버튼을 누르세요. 이동 후에도 선택은 유지됩니다.")
                    st.button("순서 전체 뒤집기",key="reverse_curve_order",
                              on_click=reverse_curve_order,use_container_width=True)
                    curve_order_names=[curve["name"] for curve in series]
                    if "curve_order_selection" in st.session_state:
                        st.session_state["curve_order_selection"]=[
                            name for name in st.session_state["curve_order_selection"] if name in curve_order_names
                        ]
                    st.multiselect("이동할 곡선 (여러 개 선택 가능)",curve_order_names,
                                   key="curve_order_selection",label_visibility="collapsed",
                                   placeholder="이동할 곡선을 선택하세요")
                    selected_order_names=st.session_state.get("curve_order_selection",[])
                    top_col,up_col=st.columns(2)
                    bottom_col,down_col=st.columns(2)
                    top_col.button("↑ 맨 위로",key="selected_curves_top",disabled=not selected_order_names,
                                   on_click=move_selected_curves,args=("top",),use_container_width=True)
                    up_col.button("↑ 한 칸 위로",key="selected_curves_up",disabled=not selected_order_names,
                                  on_click=move_selected_curves,args=("up",),use_container_width=True)
                    bottom_col.button("↓ 맨 아래로",key="selected_curves_bottom",disabled=not selected_order_names,
                                      on_click=move_selected_curves,args=("bottom",),use_container_width=True)
                    down_col.button("↓ 한 칸 아래로",key="selected_curves_down",disabled=not selected_order_names,
                                    on_click=move_selected_curves,args=("down",),use_container_width=True)
                st.markdown("**곡선 색상 및 선 굵기**")
                selected_style_names=set(selected)
                for i,s in enumerate(series):
                    if palette=="업로드 팔레트" and uploaded_palette:
                        palette_index=round(i*(len(uploaded_palette)-1)/max(1,len(series)-1))
                        palette_color=uploaded_palette[palette_index]
                    else:
                        palette_color=to_hex(plt.get_cmap(palette)(i/max(1,len(series)-1)) if palette!="Origin" else plt.get_cmap("tab10")(i%10))
                    if s["name"] in selected_style_names:
                        with st.container(border=True):
                            if palette=="업로드 팔레트" and uploaded_palette:
                                name_col=st.container()
                                s["name"]=name_col.text_input("곡선 이름",value=s["name"],key=f"nm_{i}",
                                                              label_visibility="collapsed",on_change=validate_curve_name,args=(i,))
                                palette_key=f"uploaded_palette_curve_color_{i}"
                                if st.session_state.get(palette_key) not in uploaded_palette:
                                    st.session_state[palette_key]=palette_color
                                st.caption("사용할 색상을 눌러 선택하세요.")
                                for row_start in range(0,len(uploaded_palette),8):
                                    row_colors=uploaded_palette[row_start:row_start+8]
                                    swatch_cols=st.columns(len(row_colors),gap="small")
                                    css_rules=[]
                                    for row_index,color in enumerate(row_colors):
                                        color_index=row_start+row_index
                                        swatch_key=f"uploaded_palette_swatch_{i}_{color_index}"
                                        is_current=st.session_state[palette_key]==color
                                        rgb=np.asarray(to_rgb(color))
                                        foreground="#111827" if float(np.dot(rgb,[0.299,0.587,0.114]))>0.58 else "#ffffff"
                                        border="#17202a" if is_current else "#d1d5db"
                                        css_rules.append(
                                            f".st-key-{swatch_key} button{{background-color:{color}!important;"
                                            f"color:{foreground}!important;border:2px solid {border}!important;"
                                            "height:32px;min-height:32px;padding:0!important;border-radius:6px!important;"
                                            "font-size:17px!important;line-height:1!important;}"
                                            f".st-key-{swatch_key} button:hover{{filter:brightness(.88);border-color:#111827!important;}}")
                                    st.markdown("<style>"+"".join(css_rules)+"</style>",unsafe_allow_html=True)
                                    for row_index,color in enumerate(row_colors):
                                        color_index=row_start+row_index
                                        swatch_key=f"uploaded_palette_swatch_{i}_{color_index}"
                                        with swatch_cols[row_index]:
                                            if st.button("✓" if st.session_state[palette_key]==color else " ",
                                                          key=swatch_key,help=f"팔레트 색상 {color_index+1}: {color}",
                                                          use_container_width=True):
                                                st.session_state[palette_key]=color
                                                st.rerun()
                                curve_color=st.session_state[palette_key]
                            else:
                                name_col,color_col=st.columns([8,1],vertical_alignment="center")
                                s["name"]=name_col.text_input("곡선 이름",value=s["name"],key=f"nm_{i}",
                                                              label_visibility="collapsed",on_change=validate_curve_name,args=(i,))
                                if color_mode=="곡선별 지정":
                                    curve_color=color_col.color_picker("선 색상",value=palette_color,key=f"color_{i}",
                                                                       label_visibility="collapsed")
                                else:
                                    curve_color=palette_color
                                    color_col.markdown(f"<div title='{palette_color}' style='width:34px;height:28px;background:{palette_color};border:1px solid #cbd5e1;border-radius:6px;margin:4px auto'></div>",unsafe_allow_html=True)
                            curve_width=(synced_float_control("개별 선 굵기",0.5,10.0,float(line_width),0.1,f"lw_{i}")
                                         if width_mode=="곡선별 지정" else float(line_width))
                    else:
                        # Keep hidden curves' style available to other views without rendering
                        # editors for spectra that are not currently selected.
                        if palette=="업로드 팔레트" and uploaded_palette:
                            palette_key=f"uploaded_palette_curve_color_{i}"
                            saved_palette_color=st.session_state.get(palette_key)
                            curve_color=saved_palette_color if saved_palette_color in uploaded_palette else palette_color
                        else:
                            curve_color=(st.session_state.get(f"color_{i}",palette_color)
                                         if color_mode=="곡선별 지정" else palette_color)
                        curve_width=(float(st.session_state.get(f"lw_{i}",st.session_state.get(f"lw_{i}_input",line_width)))
                                     if width_mode=="곡선별 지정" else float(line_width))
                    style[s["name"]]={"color":curve_color,"offset":float(s.get("offset",0)),"linewidth":curve_width}
                if not selected_style_names:
                    st.caption("색상과 굵기를 설정할 스펙트럼을 먼저 선택하세요.")
                curve_name_error=st.session_state.pop("curve_name_error",None)
                if curve_name_error:
                    st.error(curve_name_error)
                renamed={old:new["name"] for old,new in zip(old_names,series)}
                for item in series:
                    if item.get("error_source") in renamed:
                        item["error_source"]=renamed[item["error_source"]]
                for fit in st.session_state.get("fit_results",[]):
                    if fit.get("curve_name") in renamed:
                        fit["curve_name"]=renamed[fit["curve_name"]]
                selected=[renamed.get(name,name) for name in selected]
                st.session_state["graph_name_aliases"]=renamed
                if "graph_average_targets" in st.session_state:
                    st.session_state["graph_average_targets"]=[renamed.get(name,name) for name in st.session_state["graph_average_targets"]]
                # Curve renaming changes the valid identifiers used by the preview,
                # so refresh linked-error names and the plot list after applying aliases.
                error_source_names=refresh_linked_errors(series)
                names=[s["name"] for s in series]
                plot_names=[n for n in names if n not in error_source_names]
                signal_selected=[n for n in selected if n not in error_source_names]
                plot_selected=signal_selected
                fit_results=st.session_state.get("fit_results",[])
                if fit_results:
                    with st.expander("피팅 곡선 순서 변경",expanded=False):
                        st.caption("위·아래 버튼으로 피팅 곡선의 그리기 및 범례 순서를 바꿉니다. 뒤에 그려지는 곡선이 겹친 부분에서 위에 표시됩니다.")
                        for fit_index,fit in enumerate(fit_results):
                            fit_label=f"{fit['curve_name']} · {fit['model']} · {fit['purpose']}"
                            order_col,up_col,down_col=st.columns([6,2,2],vertical_alignment="center")
                            order_col.write(fit_label)
                            up_col.button("위로",key=f"fit_up_{abs(hash(fit['id']))}",disabled=fit_index==0,
                                           on_click=move_fit,args=(fit_index,-1),use_container_width=True)
                            down_col.button("아래로",key=f"fit_down_{abs(hash(fit['id']))}",disabled=fit_index==len(fit_results)-1,
                                             on_click=move_fit,args=(fit_index,1),use_container_width=True)
                    with st.expander("피팅 곡선 스타일",expanded=False):
                        st.caption("각 피팅 결과의 색상, 선 종류, 선 굵기를 따로 설정합니다. 미리보기와 PNG 저장 그래프에 함께 적용됩니다.")
                        line_styles={"실선":"-","파선":"--","점선":":","일점쇄선":"-."}
                        reverse_line_styles={value:key for key,value in line_styles.items()}
                        for fit_index,fit in enumerate(fit_results):
                            fit_color_default=style.get(fit.get("curve_name"),{}).get("color","#d62728")
                            saved_fit_style=fit.get("style",{})
                            color_col,line_col,width_col=st.columns([1,1,1])
                            color=color_col.color_picker(f"{fit['curve_name']} 색상",value=saved_fit_style.get("color",fit_color_default),
                                                         key=f"fit_style_color_{abs(hash(fit['id']))}")
                            line_label=line_col.selectbox(f"{fit['curve_name']} 선 종류",list(line_styles),
                                                          index=list(line_styles).index(reverse_line_styles.get(saved_fit_style.get("linestyle","--"),"파선")),
                                                          key=f"fit_style_line_{abs(hash(fit['id']))}")
                            line_width=width_col.number_input(f"{fit['curve_name']} 굵기",min_value=0.5,max_value=6.0,
                                                              value=float(saved_fit_style.get("linewidth",2.0)),step=0.1,
                                                              key=f"fit_style_width_{abs(hash(fit['id']))}")
                            fit["style"]={"color":color,"linestyle":line_styles[line_label],"linewidth":line_width}
                overlay_fit_results=[fit for fit in st.session_state.get("fit_results",[])
                                     if fit.get("id") in visible_fit_ids and fit.get("curve_name") in plot_selected]

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
                        margin_left=pos_col1.number_input("왼쪽 위치",min_value=0,max_value=2000,value=90,step=5,key="graph_margin_left")
                        margin_bottom=pos_col2.number_input("아래쪽 위치",min_value=0,max_value=2000,value=60,step=5,key="graph_margin_bottom")
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
                        margin_left=m1.number_input("왼쪽 여백",min_value=0,max_value=2000,value=90,step=5,key="graph_margin_left")
                        margin_right=m2.number_input("오른쪽 여백",min_value=0,max_value=2000,value=30,step=5,key="graph_margin_right")
                        m3,m4=st.columns(2)
                        margin_top=m3.number_input("위쪽 여백",min_value=0,max_value=2000,value=30,step=5,key="graph_margin_top")
                        margin_bottom=m4.number_input("아래쪽 여백",min_value=0,max_value=2000,value=60,step=5,key="graph_margin_bottom")
                        plot_margins=(margin_left,margin_right,margin_top,margin_bottom)
                plot_box_width=int(canvas_width_px-margin_left-margin_right)
                plot_box_height=int(canvas_height_px-margin_top-margin_bottom)
                export_dpi=st.number_input("미리보기 / PNG 저장 DPI",min_value=100,max_value=1200,value=300,step=50,key="graph_export_dpi",
                                           help="미리보기와 PNG 저장에 같은 DPI를 적용합니다. DPI를 바꿔도 설정한 픽셀 크기는 유지됩니다.")
                preview_dpi=export_dpi
                base_font_options=["Arial","DejaVu Sans","Liberation Sans","Times New Roman"]
                font_options=list(dict.fromkeys([*base_font_options,*sorted(BUNDLED_FONT_FAMILIES)]))
                font_family=st.selectbox("글꼴",font_options,index=font_options.index("Arial"),key="graph_font_family")
                if not font_family_available(font_family):
                    st.warning(f"{font_family} 글꼴 파일을 찾을 수 없습니다. 저장소의 fonts 폴더에 글꼴 파일을 넣어야 실제로 적용됩니다.")
                axis_text_color=st.color_picker("축 글씨 색상",value="#000000",key="graph_axis_text_color")
                axis_label_size=synced_font_control("축 제목 글자 크기",6,40,12,"graph_axis_font_size")
                title_font_size=synced_font_control("그래프 제목 글자 크기",6,48,14,"graph_title_font_size")
                tick_font_size=synced_font_control("눈금 글자 크기",6,32,10,"graph_tick_font_size")
                legend_font_size=synced_font_control("범례 글자 크기",6,32,9,"graph_legend_font_size")
            with axis_tab:
                with st.expander("축 제목 / 그래프 제목",expanded=False):
                    xlab=st.text_input("X축 제목",series[0]["x_label"],key="graph_xlabel")
                    ylab=st.text_input("Y축 제목","Normalized Intensity" if normalization!="없음" else series[0]["y_label"],key="graph_ylabel")
                    title=st.text_input("그래프 제목",key="graph_title")
                with st.expander("축 제목 위치",expanded=False):
                    st.caption("간격 조절은 눈금에서 떨어지는 거리를 바꾸고, 그래프 좌표는 플롯 영역을 기준으로 제목 위치를 직접 지정합니다.")
                    x_label_position_mode=st.selectbox("X축 제목 위치 방식",["눈금과 간격 조절","그래프 좌표 지정"],key="graph_xlabel_position_mode")
                    if x_label_position_mode=="눈금과 간격 조절":
                        x_label_pad=st.number_input("X축 제목과 눈금 사이 간격 (pt)",min_value=-100.0,max_value=200.0,value=4.0,step=1.0,key="graph_xlabel_pad")
                        x_label_coords=(0.5,-0.15)
                    else:
                        x_coord_col,y_coord_col=st.columns(2)
                        x_label_x=x_coord_col.number_input("X축 제목 가로 위치",min_value=-2.0,max_value=3.0,value=0.5,step=0.05,format="%.2f",key="graph_xlabel_coord_x")
                        x_label_y=y_coord_col.number_input("X축 제목 세로 위치",min_value=-2.0,max_value=3.0,value=-0.15,step=0.05,format="%.2f",key="graph_xlabel_coord_y")
                        x_label_coords=(x_label_x,x_label_y)
                        x_label_pad=4.0
                    st.caption("가로 0은 왼쪽, 0.5는 가운데, 1은 오른쪽입니다. 세로 0은 플롯 아래, 1은 위이며 범위 밖 숫자도 입력할 수 있습니다.")
                    y_label_position_mode=st.selectbox("Y축 제목 위치 방식",["눈금과 간격 조절","그래프 좌표 지정"],key="graph_ylabel_position_mode")
                    if y_label_position_mode=="눈금과 간격 조절":
                        y_label_pad=st.number_input("Y축 제목과 눈금 사이 간격 (pt)",min_value=-100.0,max_value=200.0,value=4.0,step=1.0,key="graph_ylabel_pad")
                        y_label_coords=(-0.15,0.5)
                    else:
                        x_coord_col,y_coord_col=st.columns(2)
                        y_label_x=x_coord_col.number_input("Y축 제목 가로 위치",min_value=-2.0,max_value=3.0,value=-0.15,step=0.05,format="%.2f",key="graph_ylabel_coord_x")
                        y_label_y=y_coord_col.number_input("Y축 제목 세로 위치",min_value=-2.0,max_value=3.0,value=0.5,step=0.05,format="%.2f",key="graph_ylabel_coord_y")
                        y_label_coords=(y_label_x,y_label_y)
                        y_label_pad=4.0
                    st.caption("Y축 제목의 가로 좌표 0은 플롯 왼쪽, 1은 오른쪽이고 세로 좌표 0은 아래, 0.5는 가운데, 1은 위입니다.")
                st.markdown("**축 범위**")
                x_auto=st.checkbox("X축 자동 범위",value=True,key="graph_x_auto")
                x_log=st.checkbox("X축 로그 스케일",value=False,key="graph_x_log")
                range_series=[s for s in series if s["name"] in selected and s["name"] not in error_source_names]
                if not range_series:
                    range_series=[s for s in series if s["name"] not in error_source_names]
                all_x=np.concatenate([np.asarray(s["x"],dtype=float) for s in range_series])
                finite_x=all_x[np.isfinite(all_x)]
                positive_x=finite_x[finite_x>0]
                data_xmin=float(np.min(positive_x)) if x_log and positive_x.size else float(np.min(finite_x))
                data_xmax=float(np.max(positive_x)) if x_log and positive_x.size else float(np.max(finite_x))
                x1,x2=st.columns(2)
                xmin=x1.number_input("X 최소",value=data_xmin,format="%.6g",disabled=x_auto,key="graph_xmin")
                xmax=x2.number_input("X 최대",value=data_xmax,format="%.6g",disabled=x_auto,key="graph_xmax")
                if normalization=="지정 X 범위 최대값으로 나누기":
                    opts["normalization"]="Maximum in X range"
                    opts["normalization_range"]=(float(normalization_x_min),float(normalization_x_max))
                    range_lo,range_hi=sorted((float(normalization_x_min),float(normalization_x_max)))
                    missing_range_curves=[s["name"] for s in range_series
                                          if not np.any(np.isfinite(np.asarray(s["x"],dtype=float)) &
                                                        (np.asarray(s["x"],dtype=float)>=range_lo) &
                                                        (np.asarray(s["x"],dtype=float)<=range_hi))]
                    if missing_range_curves:
                        shown_names=", ".join(missing_range_curves[:3])
                        extra=f" 외 {len(missing_range_curves)-3}개" if len(missing_range_curves)>3 else ""
                        st.warning(f"지정 X 범위에 데이터가 없는 곡선은 나누기 정규화를 적용하지 않습니다: {shown_names}{extra}")
                else:
                    opts["normalization_range"]=(tuple(sorted((float(xmin),float(xmax))))
                                                  if normalization_scope=="현재 그래프 X 범위" else None)
                y_auto=st.checkbox("Y축 자동 범위",value=True,key="graph_y_auto")
                y_log=st.checkbox("Y축 로그 스케일",value=False,key="graph_y_log")
                if x_log or y_log:
                    st.caption("로그 스케일에서는 해당 축의 0 이하 값이 표시되지 않습니다.")
                all_y=np.concatenate([np.asarray(transform(s,opts),dtype=float) for s in range_series])
                finite_y=all_y[np.isfinite(all_y)]
                positive_y=finite_y[finite_y>0]
                proc_min=float(np.min(positive_y)) if y_log and positive_y.size else float(np.min(finite_y))
                proc_max=float(np.max(positive_y)) if y_log and positive_y.size else float(np.max(finite_y))
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
            if x_log and xmin<=0:
                st.error("X축 로그 스케일에서는 X 최소값이 0보다 커야 합니다.")
                xmin=data_xmin
            if y_log and not y_auto and (ymin<=0 or ymax<=0):
                st.error("Y축 로그 스케일에서는 Y축 범위가 모두 0보다 커야 합니다.")
                ymin,ymax=(float(np.min(positive_y)),float(np.max(positive_y))) if positive_y.size>1 else (1.0,10.0)
            if x_log and not positive_x.size:
                st.error("X축 로그 스케일에 사용할 양수 데이터가 없습니다.")
            if y_log and not positive_y.size and not stack:
                st.error("Y축 로그 스케일에 사용할 양수 데이터가 없습니다.")
    with plot_col:
        with st.container(height=650,border=False):
            st.subheader("그래프 미리보기")
            crosshair_enabled=st.checkbox("X/Y 좌표 십자선 표시",value=True,key="graph_crosshair_enabled")
            proc={s["name"]:transform(s,opts) for s in series}
            # Base the automatic vertical stack spacing on the visible X range.
            # This keeps the gaps proportional when the user zooms into a narrower range.
            x_range_min,x_range_max=sorted((float(xmin),float(xmax)))
            # Only selected signal curves determine the stack heights and their order.
            visible_plot_names=[n for n in plot_names if n in plot_selected]
            log_stack=bool(stack and y_log)
            heights=[]
            log_stack_lifts={}
            for n in visible_plot_names:
                s=next(z for z in series if z["name"]==n)
                y_values=np.asarray(proc[n],dtype=float)
                visible=np.isfinite(s["x"]) & np.isfinite(y_values) & (s["x"]>=x_range_min) & (s["x"]<=x_range_max)
                values=y_values[visible] if np.any(visible) else y_values[np.isfinite(y_values)]
                finite_values=values[np.isfinite(values)]
                peak=float(np.max(finite_values)) if finite_values.size else 0.0
                if log_stack and finite_values.size:
                    low=float(np.min(finite_values))
                    positive_floor=max(np.finfo(float).tiny,max(abs(low),abs(peak),np.finfo(float).tiny)*1e-6)
                    log_stack_lifts[n]=max(0.0,positive_floor-low)
                    shifted_values=finite_values+log_stack_lifts[n]
                    shifted_values=shifted_values[shifted_values>0]
                    log_span=(float(np.log10(np.max(shifted_values))-np.log10(np.min(shifted_values)))
                              if shifted_values.size else 0.0)
                    heights.append(max(log_span+0.12,0.12))
                else:
                    log_stack_lifts[n]=0.0
                    heights.append(max(peak*1.10,0.01))
            auto={n:0 for n in plot_names}
            for i,n in enumerate(visible_plot_names):
                if stack:
                    auto[n]=sum(heights[i+1:])
                else:
                    auto[n]=0
            # Draw the preview with the same Matplotlib renderer and settings as the PNG export.
            # The HTML layer only adds pointer crosshairs over the rendered image, so font
            # metrics, tick styling, labels, margins, and line placement remain unchanged.
            apply_plot_font(font_family)
            preview_mpl_fig,preview_ax=make_plot(plot_width,plot_height,size_mode,plot_margins,100)
            apply_graph_frame(preview_ax,show_graph_frame,graph_frame_width)
            hover_curves=[]
            for n in names:
                if n in plot_selected:
                    s=next(z for z in series if z["name"]==n)
                    curve_x=np.asarray(s["x"],dtype=float)
                    curve_y=stacked_curve_y(proc[n],n,visible_plot_names,auto,offset_step,style,
                                            log_stack,log_stack_lifts)
                    error_values=processed_error(s,opts)
                    draw_curve(preview_ax,s,curve_y,n,style[n]["color"],style[n]["linewidth"],
                               plot_style,error_values,scatter_marker,error_capsize,error_capthick,error_linewidth)
                    valid=np.isfinite(curve_x)&np.isfinite(curve_y)
                    valid_indices=np.flatnonzero(valid)
                    if len(valid_indices)>20000:
                        valid_indices=valid_indices[np.linspace(0,len(valid_indices)-1,20000,dtype=int)]
                    order=np.argsort(curve_x[valid_indices])
                    hover_curves.append({"name":n,"x":curve_x[valid_indices][order].tolist(),
                                         "y":curve_y[valid_indices][order].tolist()})
            if show_fit_overlay:
                draw_fit_overlays(preview_ax,overlay_fit_results,visible_fit_ids,visible_plot_names,
                                  plot_names,auto,offset_step,style,log_stack,log_stack_lifts)
            if x_log: preview_ax.set_xscale("log")
            if y_log: preview_ax.set_yscale("log")
            preview_ax.set_xlim(min(xmin,xmax),max(xmin,xmax))
            if y_auto:
                visible_limits=(visible_stack_log_ylim(series,proc,plot_names,visible_plot_names,auto,offset_step,style,
                                                       xmin,xmax,log_stack_lifts,log_stack)
                                if y_log else visible_stack_ylim(series,proc,plot_names,visible_plot_names,auto,offset_step,style,xmin,xmax))
                visible_limits=extend_ylim_for_fit(visible_limits,overlay_fit_results,visible_fit_ids,
                                                   visible_plot_names,plot_names,auto,offset_step,style,
                                                   xmin,xmax,y_log,log_stack_lifts,log_stack)
                if visible_limits:
                    preview_ax.set_ylim(*visible_limits)
            else:
                preview_ax.set_ylim(min(ymin,ymax),max(ymin,ymax))
            preview_ax.set_xlabel(format_axis_label(xlab),fontsize=axis_label_size,color=axis_text_color)
            preview_ax.set_ylabel(format_axis_label(ylab),fontsize=axis_label_size,color=axis_text_color)
            apply_axis_label_positions(preview_ax,x_label_position_mode,x_label_pad,x_label_coords,
                                      y_label_position_mode,y_label_pad,y_label_coords)
            preview_ax.set_title(title or None,fontsize=title_font_size,color=axis_text_color)
            preview_ax.tick_params(axis="x",which="both",bottom=x_bottom,top=x_top,
                                   labelbottom=x_bottom,labeltop=x_top,labelsize=tick_font_size,labelcolor=axis_text_color)
            preview_ax.tick_params(axis="y",which="both",left=y_left,right=y_right,
                                   labelleft=y_left,labelright=y_right,labelsize=tick_font_size,labelcolor=axis_text_color)
            if x_tick_interval>0 and not x_log:
                preview_ax.xaxis.set_major_locator(MultipleLocator(x_tick_interval))
            if y_tick_interval>0 and not y_log:
                preview_ax.yaxis.set_major_locator(MultipleLocator(y_tick_interval))
            preview_ax.yaxis.set_major_formatter(y_tick_formatter)
            add_y_scale_annotation(preview_ax,y_tick_options,tick_font_size,axis_text_color)
            if legend:
                preview_ax.legend(fontsize=legend_font_size,loc=legend_loc)
            if grid:
                preview_ax.grid(True,alpha=.3)
            preview_ax.figure.canvas.draw()
            preview_x_limits=preview_ax.get_xlim()
            preview_y_limits=preview_ax.get_ylim()
            axes_bbox=preview_ax.get_window_extent()
            figure_bbox=preview_mpl_fig.bbox
            overlay_box={
                "left":float(axes_bbox.x0/figure_bbox.width*canvas_width_px),
                "top":float((figure_bbox.height-axes_bbox.y1)/figure_bbox.height*canvas_height_px),
                "width":float(axes_bbox.width/figure_bbox.width*canvas_width_px),
                "height":float(axes_bbox.height/figure_bbox.height*canvas_height_px),
            }
            preview_png=render_png_fixed_size(preview_mpl_fig,canvas_width_px,canvas_height_px,preview_dpi)
            plt.close(preview_mpl_fig)
            preview_image_b64=base64.b64encode(preview_png).decode("ascii")
            crosshair_script="""
            const stage=document.getElementById('stage');
            const svg=document.getElementById('crosshair');
            const vx=document.getElementById('vertical');
            const hy=document.getElementById('horizontal');
            const readout=document.getElementById('readout');
            const plot={plot};
            const curves={curves};
            const width={width}, height={height};
            const xlim={xlim}, ylim={ylim};
            const enabled={enabled};
            const xlog={xlog}, ylog={ylog};
            const toAxis=(v,log)=>log?Math.log10(v):v;
            const fromAxis=(v,log)=>log?Math.pow(10,v):v;
            function lowerBound(values,target){{
              let lo=0,hi=values.length;
              while(lo<hi){{const mid=(lo+hi)>>1;if(values[mid]<target)lo=mid+1;else hi=mid;}}
              return lo;
            }}
            if(enabled){{
              stage.addEventListener('pointermove',(event)=>{{
                const rect=stage.getBoundingClientRect();
                const x=(event.clientX-rect.left)*width/rect.width;
                const y=(event.clientY-rect.top)*height/rect.height;
                if(x<plot.left || x>plot.left+plot.width || y<plot.top || y>plot.top+plot.height){{
                  svg.style.display='none'; readout.style.display='none'; return;
                }}
                const valueX=fromAxis(toAxis(xlim[0],xlog)+(x-plot.left)/plot.width*(toAxis(xlim[1],xlog)-toAxis(xlim[0],xlog)),xlog);
                const valueY=fromAxis(toAxis(ylim[1],ylog)-(y-plot.top)/plot.height*(toAxis(ylim[1],ylog)-toAxis(ylim[0],ylog)),ylog);
                let closest=null,closestDistance=20*20;
                for(const curve of curves){{
                  const i=lowerBound(curve.x,valueX);
                  for(let j=Math.max(0,i-2);j<Math.min(curve.x.length,i+2);j++){{
                    if((xlog&&curve.x[j]<=0)||(ylog&&curve.y[j]<=0)) continue;
                    const px=plot.left+(toAxis(curve.x[j],xlog)-toAxis(xlim[0],xlog))/(toAxis(xlim[1],xlog)-toAxis(xlim[0],xlog))*plot.width;
                    const py=plot.top+(toAxis(ylim[1],ylog)-toAxis(curve.y[j],ylog))/(toAxis(ylim[1],ylog)-toAxis(ylim[0],ylog))*plot.height;
                    const distance=(px-x)*(px-x)+(py-y)*(py-y);
                    if(distance<closestDistance){{closestDistance=distance;closest={{name:curve.name,x:curve.x[j],y:curve.y[j]}};}}
                  }}
                }}
                vx.setAttribute('x1',x); vx.setAttribute('x2',x);
                vx.setAttribute('y1',plot.top); vx.setAttribute('y2',plot.top+plot.height);
                hy.setAttribute('x1',plot.left); hy.setAttribute('x2',plot.left+plot.width);
                hy.setAttribute('y1',y); hy.setAttribute('y2',y);
                svg.style.display='block';
                readout.textContent='커서  X: '+Number(valueX.toPrecision(6))+'   Y: '+Number(valueY.toPrecision(6))+
                  (closest?'\\n'+closest.name+'  X: '+Number(closest.x.toPrecision(6))+'   Y: '+Number(closest.y.toPrecision(6)):'');
                readout.style.display='block';
              }});
              stage.addEventListener('pointerleave',()=>{{svg.style.display='none';readout.style.display='none';}});
            }}
            """.format(plot=json.dumps(overlay_box),curves=json.dumps(hover_curves,ensure_ascii=False,separators=(",",":")),
                       width=canvas_width_px,height=canvas_height_px,
                       xlim=json.dumps([float(preview_x_limits[0]),float(preview_x_limits[1])]),
                       ylim=json.dumps([float(preview_y_limits[0]),float(preview_y_limits[1])]),
                       enabled="true" if crosshair_enabled else "false",
                       xlog="true" if x_log else "false",ylog="true" if y_log else "false")
            preview_html=f"""<!doctype html><html><head><meta charset="utf-8"><style>
            html,body{{margin:0;padding:0;overflow:hidden;background:transparent}}
            #frame{{display:inline-block;padding:12px;background:#f3f6fa;border:1px solid #e1e7ef;
              border-radius:10px;box-sizing:border-box}}
            #stage{{position:relative;width:{canvas_width_px}px;height:{canvas_height_px}px;overflow:hidden}}
            #graph{{display:block;width:{canvas_width_px}px;height:{canvas_height_px}px}}
            #crosshair{{position:absolute;inset:0;width:100%;height:100%;display:none;pointer-events:none}}
            #crosshair line{{stroke:#566573;stroke-width:1;stroke-dasharray:4 3;vector-effect:non-scaling-stroke}}
            #readout{{display:none;position:absolute;right:7px;top:7px;padding:4px 7px;border:1px solid #aeb8c2;
              border-radius:4px;background:rgba(255,255,255,.94);color:#263238;font:12px Arial,sans-serif;
              white-space:pre-line;pointer-events:none}}
            </style></head><body><div id="frame"><div id="stage">
            <img id="graph" src="data:image/png;base64,{preview_image_b64}" alt="그래프 미리보기">
            <svg id="crosshair" viewBox="0 0 {canvas_width_px} {canvas_height_px}" preserveAspectRatio="none">
              <line id="vertical"></line><line id="horizontal"></line></svg>
            <div id="readout"></div></div></div><script>{crosshair_script}</script></body></html>"""
            components.html(preview_html,width=canvas_width_px+26,height=canvas_height_px+26,scrolling=False)
            st.caption("그래프 위에 마우스를 올리면 십자선의 X/Y 좌표와 가까운 곡선의 이름 및 실제 데이터 X/Y 값이 표시됩니다.")
            st.caption(f"캔버스 {canvas_width_px} × {canvas_height_px} px · 플롯 박스 {plot_box_width} × {plot_box_height} px · 곡선 {len(series)}개")

with tabs[1]:
    st.subheader("선택 곡선 평균")
    st.caption("현재 사이드바의 SG 필터·baseline·normalization 처리를 반영해 평균을 계산합니다. X 간격이 서로 다르면 첫 곡선의 X 좌표에 맞춰 보간하고, 공통 X 범위만 사용합니다.")
    average_axis_labels=list(dict.fromkeys(str(s.get("x_label","X")) for s in series))
    average_axis=st.selectbox("평균 계산 X축",average_axis_labels,key="graph_average_axis")
    average_candidates=[s["name"] for s in series if s["name"] not in error_source_names and str(s.get("x_label","X"))==average_axis]
    average_default=[n for n in selected if n in average_candidates]
    if "graph_average_targets" in st.session_state:
        st.session_state["graph_average_targets"]=[n for n in st.session_state["graph_average_targets"] if n in average_candidates]
    else:
        st.session_state["graph_average_targets"]=average_default
    average_targets=st.multiselect("평균에 포함할 곡선",average_candidates,key="graph_average_targets")
    average_enabled=st.checkbox("평균 곡선 계산",value=True,key="graph_average_enabled")
    average_curve=compute_average_curve(average_targets,series,proc,average_axis) if average_enabled else None
    if not average_enabled:
        st.info("평균 곡선 계산을 켜면 선택한 곡선의 평균 그래프와 다운로드를 표시합니다.")
    elif average_curve is None:
        st.warning("평균을 계산하려면 같은 X축을 사용하는 곡선을 2개 이상 선택하고, 서로 겹치는 X 범위가 있어야 합니다.")
    else:
        apply_plot_font(font_family)
        average_fig,average_ax=plt.subplots(figsize=(9,4.5))
        average_ax.plot(average_curve["x"],average_curve["y"],color="#d62728",linewidth=2,label=average_curve["name"])
        average_ax.set_xlabel(format_axis_label(average_axis),fontsize=axis_label_size,color=axis_text_color)
        average_ax.set_ylabel(format_axis_label(str(next((s.get("y_label","Intensity") for s in series if s["name"] in average_targets),"Intensity"))),fontsize=axis_label_size,color=axis_text_color)
        apply_axis_label_positions(average_ax,x_label_position_mode,x_label_pad,x_label_coords,
                                  y_label_position_mode,y_label_pad,y_label_coords)
        average_ax.tick_params(labelsize=tick_font_size,labelcolor=axis_text_color)
        average_ax.grid(True,alpha=.3)
        average_ax.legend()
        st.pyplot(average_fig,use_container_width=True)
        plt.close(average_fig)
        curve_lookup={s["name"]:s for s in series}
        source_stems=[]
        for curve_name in average_targets:
            source_file=curve_lookup.get(curve_name,{}).get("source_file")
            source_stems.append(Path(source_file).stem if source_file
                                 else str(curve_name).split(" · ",1)[0])
        source_stems=list(dict.fromkeys(source_stems))
        average_export_name=(source_stems[0] if len(source_stems)==1 else "_and_".join(source_stems))
        average_export_name=re.sub(r'[\\/:*?"<>|]+',"_",average_export_name).strip(" ._") or "average_curve"
        average_table=pd.DataFrame({"X":average_curve["x"],average_export_name:average_curve["y"]})
        avg_csv_col,avg_txt_col=st.columns(2)
        avg_csv_col.download_button("평균 곡선 CSV",average_table.to_csv(index=False).encode("utf-8-sig"),
                                        f"{average_export_name}.csv","text/csv",key="average_curve_csv",use_container_width=True)
        avg_txt_col.download_button("평균 곡선 TXT",average_table.to_csv(index=False,sep="\t").encode("utf-8-sig"),
                                        f"{average_export_name}.txt","text/plain",key="average_curve_txt",use_container_width=True)
        st.caption(f"평균 포함 곡선 {average_curve['count']}개 · 공통 X 범위 {average_curve['x'][0]:g}–{average_curve['x'][-1]:g}")

with tabs[2]:
    peak_tab,max_tab,range_stats_tab=st.tabs(["피크 찾기","맥스 인텐시티","범위 통계"])
    with peak_tab:
        st.subheader("피크 찾기")
        target=st.selectbox("분석 곡선",plot_names,key="peak_series")
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
            apply_plot_font(font_family)
            fig2,ax2=make_plot(plot_width,plot_height,size_mode,plot_margins,100)
            apply_graph_frame(ax2,show_graph_frame,graph_frame_width)
            peak_plot_names=[n for n in plot_names if n in plot_selected]
            if target not in peak_plot_names and target in plot_names:
                peak_plot_names.append(target)
            peak_log_lifts=dict(log_stack_lifts)
            if log_stack and target not in log_stack_lifts:
                target_values=np.asarray(y,dtype=float)
                target_values=target_values[np.isfinite(target_values)]
                if target_values.size:
                    target_low=float(np.min(target_values)); target_high=float(np.max(target_values))
                    target_floor=max(np.finfo(float).tiny,max(abs(target_low),abs(target_high),np.finfo(float).tiny)*1e-6)
                    peak_log_lifts[target]=max(0.0,target_floor-target_low)
            peak_target_y=stacked_curve_y(y,target,peak_plot_names,auto,offset_step,style,
                                          log_stack,peak_log_lifts)
            for n in peak_plot_names:
                peak_s=next(z for z in series if z["name"]==n)
                peak_y=stacked_curve_y(proc[n],n,peak_plot_names,auto,offset_step,style,
                                       log_stack,peak_log_lifts)
                error_values=processed_error(peak_s,opts)
                draw_curve(ax2,peak_s,peak_y,n,style[n]["color"],style[n]["linewidth"],
                           plot_style,error_values,scatter_marker,error_capsize,error_capthick,error_linewidth)
            ax2.scatter(s["x"][peaks],peak_target_y[peaks],color="red",zorder=5)
            if x_log: ax2.set_xscale("log")
            if y_log: ax2.set_yscale("log")
            ax2.set_xlim(min(xmin,xmax),max(xmin,xmax))
            if y_auto:
                visible_limits=(visible_stack_log_ylim(series,proc,plot_names,peak_plot_names,auto,offset_step,style,
                                                       xmin,xmax,peak_log_lifts,log_stack)
                               if y_log else visible_stack_ylim(series,proc,plot_names,peak_plot_names,auto,offset_step,style,xmin,xmax))
                if visible_limits: ax2.set_ylim(*visible_limits)
            else: ax2.set_ylim(min(ymin,ymax),max(ymin,ymax))
            ax2.set_xlabel(format_axis_label(xlab),fontsize=axis_label_size,color=axis_text_color); ax2.set_ylabel(format_axis_label(ylab),fontsize=axis_label_size,color=axis_text_color)
            apply_axis_label_positions(ax2,x_label_position_mode,x_label_pad,x_label_coords,
                                       y_label_position_mode,y_label_pad,y_label_coords)
            ax2.set_title(title or None,fontsize=title_font_size,color=axis_text_color)
            ax2.tick_params(axis="x",which="both",bottom=x_bottom,top=x_top,labelbottom=x_bottom,labeltop=x_top,labelsize=tick_font_size,labelcolor=axis_text_color)
            ax2.tick_params(axis="y",which="both",left=y_left,right=y_right,labelleft=y_left,labelright=y_right,labelsize=tick_font_size,labelcolor=axis_text_color)
            if x_tick_interval>0 and not x_log: ax2.xaxis.set_major_locator(MultipleLocator(x_tick_interval))
            if y_tick_interval>0 and not y_log: ax2.yaxis.set_major_locator(MultipleLocator(y_tick_interval))
            ax2.yaxis.set_major_formatter(y_tick_formatter)
            add_y_scale_annotation(ax2,y_tick_options,tick_font_size,axis_text_color)
            if grid: ax2.grid(True,alpha=.3)
            if legend: ax2.legend(fontsize=legend_font_size,loc=legend_loc)
            st.image(render_png_fixed_size(fig2,canvas_width_px,canvas_height_px,preview_dpi),width=canvas_width_px); plt.close(fig2)
            st.download_button("피크 결과 CSV",peak_df.to_csv(index=False).encode("utf-8-sig"),"peaks.csv","text/csv")
    with max_tab:
        st.subheader("선택 곡선의 최대 intensity")
        st.caption("왼쪽 처리 설정의 SG 필터, baseline 보정, normalization 결과에서 최대값을 계산합니다.")
        max_targets=st.multiselect("최대값을 계산할 곡선",plot_names,default=[n for n in selected if n in plot_names],key="max_targets")
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
            is_uv=max_s.get("x_label")=="Wavelength (nm)"
            max_y=np.asarray(proc[max_name],dtype=float)
            mask=np.isfinite(max_s["x"]) & np.isfinite(max_y) & (max_s["x"]>=max_xmin) & (max_s["x"]<=max_xmax)
            if np.any(mask):
                valid_indices=np.flatnonzero(mask)
                source_idx=int(valid_indices[int(np.nanargmax(max_y[mask]))])
                processing_steps=[]
                if sg_enabled:
                    processing_steps.append(f"SG (window={sg_window}, order={sg_polyorder})")
                if baseline_method!="없음":
                    processing_steps.append(f"{baseline_method} baseline")
                if normalization!="없음":
                    processing_steps.append(f"{normalization} normalization")
                max_rows.append({
                    "곡선":max_name,
                    "최대 X (nm)" if is_uv else "최대 X (cm⁻¹)":float(max_s["x"][source_idx]),
                    "최대 intensity":float(max_y[source_idx]),
                    "처리": " → ".join(processing_steps) if processing_steps else "원본",
                })
        if max_rows:
            max_df=pd.DataFrame(max_rows)
            st.dataframe(max_df,use_container_width=True,hide_index=True)
            st.download_button("최대값 결과 CSV",max_df.to_csv(index=False).encode("utf-8-sig"),"max_intensity_summary.csv","text/csv")
        elif max_targets:
            st.info("설정한 X 범위 안에 유효한 데이터가 없습니다.")

    with range_stats_tab:
        st.subheader("X 범위 내 평균 및 표준편차")
        st.caption("선택한 X 범위에 실제로 포함되는 데이터 포인트만 사용합니다. 현재 처리 설정 또는 원본 intensity를 기준으로 계산할 수 있습니다.")
        axis_labels=list(dict.fromkeys(str(s.get("x_label","X")) for s in series))
        stats_axis=st.selectbox("X축 종류",axis_labels,key="range_stats_axis")
        axis_series=[s for s in series if s["name"] not in error_source_names and str(s.get("x_label","X"))==stats_axis]
        axis_names=[s["name"] for s in axis_series]
        default_stats=[n for n in selected if n in axis_names] or axis_names
        stats_targets=st.multiselect("계산할 곡선",axis_names,default=default_stats,key="range_stats_targets")
        if stats_targets:
            selected_stats=[s for s in axis_series if s["name"] in stats_targets]
            range_min=min(float(np.nanmin(s["x"])) for s in selected_stats)
            range_max=max(float(np.nanmax(s["x"])) for s in selected_stats)
        else:
            range_min=min(float(np.nanmin(s["x"])) for s in axis_series)
            range_max=max(float(np.nanmax(s["x"])) for s in axis_series)
        range_cols=st.columns(2)
        stats_xmin=range_cols[0].number_input("X 범위 최소",value=range_min,format="%.8g",key="range_stats_xmin")
        stats_xmax=range_cols[1].number_input("X 범위 최대",value=range_max,format="%.8g",key="range_stats_xmax")
        basis_col,std_col=st.columns(2)
        stats_basis=basis_col.radio("계산할 Y값",["현재 처리 결과","원본 intensity"],horizontal=True,key="range_stats_basis")
        std_kind=std_col.radio("표준편차 계산",["표본 (n−1)","모집단 (n)"],horizontal=True,key="range_stats_std_kind",
                               help="표본 표준편차는 n−1, 모집단 표준편차는 n으로 나눕니다.")
        if stats_xmin>stats_xmax:
            st.error("X 범위 최소값은 최대값보다 작아야 합니다.")
        elif not stats_targets:
            st.info("계산할 곡선을 선택하세요.")
        else:
            rows=[]
            for stats_name in stats_targets:
                stats_curve=next(s for s in axis_series if s["name"]==stats_name)
                y_stats=np.asarray(proc[stats_name] if stats_basis=="현재 처리 결과" else stats_curve["raw"],dtype=float)
                x_stats=np.asarray(stats_curve["x"],dtype=float)
                mask=np.isfinite(x_stats)&np.isfinite(y_stats)&(x_stats>=stats_xmin)&(x_stats<=stats_xmax)
                values=y_stats[mask]
                if len(values):
                    ddof=1 if std_kind=="표본 (n−1)" else 0
                    std_value=float(np.std(values,ddof=ddof)) if len(values)>ddof else np.nan
                    rows.append({"곡선":stats_name,"X축":stats_axis,"범위 최소":float(stats_xmin),
                                 "범위 최대":float(stats_xmax),"포인트 수":int(len(values)),
                                 "평균":float(np.mean(values)),"표준편차":std_value,"Y 기준":stats_basis,
                                 "표준편차 방식":std_kind})
            if rows:
                stats_df=pd.DataFrame(rows)
                st.dataframe(stats_df,use_container_width=True,hide_index=True)
                st.download_button("범위 통계 CSV 저장",stats_df.to_csv(index=False).encode("utf-8-sig"),
                                   "range_statistics.csv","text/csv",key="range_stats_download")
                if any(row["포인트 수"]==1 for row in rows) and std_kind=="표본 (n−1)":
                    st.caption("포인트가 1개인 곡선은 표본 표준편차를 계산할 수 없어 NaN으로 표시됩니다.")
            else:
                st.info("입력한 X 범위 안에 유효한 데이터가 없습니다.")

with tabs[3]:
    st.subheader("곡선 피팅")
    st.caption("현재 처리 설정(SG 필터·baseline·normalization)이 적용된 Y 데이터를 피팅합니다. 설정을 바꾼 뒤에는 다시 피팅해 주세요. 입력 오차가 모두 유효하면 가중 피팅을 적용합니다. baseline·SG 필터에 따른 오차 전파는 계산하지 않습니다.")
    fit_purpose=st.selectbox("피팅 목적",["스펙트럼 피크","검량선"],key="fit_purpose")
    fit_models={"스펙트럼 피크":["Gaussian","Lorentzian","Pseudo-Voigt"],
                "검량선":["Linear","Quadratic","Cubic","Exponential","4PL"]}
    if st.session_state.get("fit_model_name") not in fit_models[fit_purpose]:
        st.session_state["fit_model_name"]=fit_models[fit_purpose][0]
    fit_model_name=st.selectbox("피팅 모델",fit_models[fit_purpose],key="fit_model_name")
    fit_target=st.selectbox("피팅할 곡선",plot_names,key="fit_target_curve")
    fit_series=next(item for item in series if item["name"]==fit_target)
    fit_x_all=np.asarray(fit_series["x"],dtype=float)
    fit_y_all=np.asarray(proc[fit_target],dtype=float)
    fit_x_finite=fit_x_all[np.isfinite(fit_x_all)]
    fit_x_data_min=float(np.min(fit_x_finite)); fit_x_data_max=float(np.max(fit_x_finite))
    if st.session_state.get("fit_range_curve")!=fit_target:
        st.session_state["fit_x_min"]=fit_x_data_min
        st.session_state["fit_x_max"]=fit_x_data_max
        st.session_state["fit_range_curve"]=fit_target
    fit_range_col1,fit_range_col2=st.columns(2)
    fit_x_min=fit_range_col1.number_input("피팅 X 최소",value=fit_x_data_min,format="%.8g",key="fit_x_min")
    fit_x_max=fit_range_col2.number_input("피팅 X 최대",value=fit_x_data_max,format="%.8g",key="fit_x_max")
    if st.button("피팅 실행",type="primary",key="run_curve_fit"):
        fit_mask=(np.isfinite(fit_x_all)&np.isfinite(fit_y_all)&
                  (fit_x_all>=min(fit_x_min,fit_x_max))&(fit_x_all<=max(fit_x_min,fit_x_max)))
        x_fit_data=fit_x_all[fit_mask]
        y_fit_data=fit_y_all[fit_mask]
        minimum_points={"Linear":3,"Quadratic":4,"Cubic":5,"Exponential":5,
                        "4PL":5,"Gaussian":5,"Lorentzian":5,"Pseudo-Voigt":6}[fit_model_name]
        if len(x_fit_data)<minimum_points:
            st.error(f"{fit_model_name} 피팅에는 현재 범위에 유효한 데이터가 {minimum_points}점 이상 필요합니다.")
        elif fit_x_min>=fit_x_max:
            st.error("피팅 X 최소값은 최대값보다 작아야 합니다.")
        else:
            sigma_all=processed_error(fit_series,opts)
            sigma_fit=sigma_all[fit_mask] if sigma_all is not None else None
            use_weights=(sigma_fit is not None and np.all(np.isfinite(sigma_fit)&(sigma_fit>0)))
            try:
                evaluator,fit_params,param_rows,fitted_values,param_errors=fit_curve_model(
                    x_fit_data,y_fit_data,fit_purpose,fit_model_name,sigma_fit if use_weights else None)
                dense_x=np.linspace(float(np.min(x_fit_data)),float(np.max(x_fit_data)),400)
                if fit_model_name in ("Linear","Quadratic","Cubic"):
                    dense_y=evaluator(dense_x,fit_params)
                else:
                    dense_y=evaluator(dense_x,*fit_params)
                residuals=y_fit_data-fitted_values
                ss_res=float(np.sum(residuals**2))
                ss_tot=float(np.sum((y_fit_data-np.mean(y_fit_data))**2))
                r_squared=1-ss_res/ss_tot if ss_tot>0 else np.nan
                rmse=float(np.sqrt(np.mean(residuals**2)))
                fit_id=f"{fit_target}|{fit_purpose}|{fit_model_name}|{min(fit_x_min,fit_x_max):.10g}|{max(fit_x_min,fit_x_max):.10g}"
                parameter_records=[]
                for i,(param_name,param_value) in enumerate(param_rows):
                    standard_error=(float(param_errors[i]) if param_errors is not None and i<len(param_errors) else np.nan)
                    parameter_records.append({"Parameter":param_name,"Estimate":param_value,"Std. error":standard_error})
                prior_fit=next((fit for fit in st.session_state.get("fit_results",[]) if fit.get("id")==fit_id),None)
                default_fit_style={"color":style.get(fit_target,{}).get("color","#d62728"),
                                   "linestyle":"--","linewidth":2.0}
                new_fit={"id":fit_id,"curve_name":fit_target,"purpose":fit_purpose,"model":fit_model_name,
                         "x_fit":dense_x,"y_fit":np.asarray(dense_y,dtype=float),
                         "parameters":parameter_records,"r_squared":r_squared,"rmse":rmse,
                         "n_points":int(len(x_fit_data)),"weighted":bool(use_weights),
                         "x_min":float(min(fit_x_min,fit_x_max)),"x_max":float(max(fit_x_min,fit_x_max)),
                         "style":prior_fit.get("style",default_fit_style) if prior_fit else default_fit_style}
                current=[fit for fit in st.session_state.get("fit_results",[]) if fit.get("id")!=fit_id]
                st.session_state["fit_results"]=[*current,new_fit]
                st.session_state["_pending_fit_overlay_id"]=fit_id
                st.rerun()
            except (RuntimeError,ValueError,OverflowError,FloatingPointError) as exc:
                st.error(f"피팅에 실패했습니다: {exc}")
    fit_results=st.session_state.get("fit_results",[])
    if fit_results:
        st.markdown("**저장된 피팅 결과**")
        for fit in reversed(fit_results):
            with st.expander(f"{fit['curve_name']} · {fit['model']} · {fit['purpose']} ({fit['x_min']:g}–{fit['x_max']:g})",expanded=False):
                metric1,metric2,metric3=st.columns(3)
                metric1.metric("R²",f"{fit['r_squared']:.6g}")
                metric2.metric("RMSE",f"{fit['rmse']:.6g}")
                metric3.metric("포인트 수",fit["n_points"])
                st.dataframe(pd.DataFrame(fit["parameters"]),use_container_width=True,hide_index=True)
                st.caption("오차값 가중" if fit.get("weighted") else "비가중 피팅")
                if st.button("이 피팅 결과 삭제",key=f"delete_fit_{fit['id']}"):
                    st.session_state["fit_results"]=[item for item in fit_results if item["id"]!=fit["id"]]
                    st.rerun()

with tabs[4]:
    st.subheader("그래프 저장")
    apply_plot_font(font_family)
    figout,axout=make_plot(plot_width,plot_height,size_mode,plot_margins,100)
    apply_graph_frame(axout,show_graph_frame,graph_frame_width)
    for n in names:
        if n in plot_selected:
            s=next(z for z in series if z["name"]==n)
            plotted_y=stacked_curve_y(proc[n],n,visible_plot_names,auto,offset_step,style,
                                      log_stack,log_stack_lifts)
            error_values=processed_error(s,opts)
            draw_curve(axout,s,plotted_y,n,style[n]["color"],style[n]["linewidth"],
                       plot_style,error_values,scatter_marker,error_capsize,error_capthick,error_linewidth)
    if show_fit_overlay:
        draw_fit_overlays(axout,overlay_fit_results,visible_fit_ids,visible_plot_names,
                          plot_names,auto,offset_step,style,log_stack,log_stack_lifts)
    if x_log: axout.set_xscale("log")
    if y_log: axout.set_yscale("log")
    axout.set_xlim(min(xmin,xmax),max(xmin,xmax))
    if y_auto:
        visible_limits=(visible_stack_log_ylim(series,proc,plot_names,visible_plot_names,auto,offset_step,style,
                                               xmin,xmax,log_stack_lifts,log_stack)
                       if y_log else visible_stack_ylim(series,proc,plot_names,visible_plot_names,auto,offset_step,style,xmin,xmax))
        visible_limits=extend_ylim_for_fit(visible_limits,overlay_fit_results,visible_fit_ids,
                                           visible_plot_names,plot_names,auto,offset_step,style,
                                           xmin,xmax,y_log,log_stack_lifts,log_stack)
        if visible_limits:
            ylo,yhi=visible_limits
            axout.set_ylim(ylo,yhi)
    else: axout.set_ylim(min(ymin,ymax),max(ymin,ymax))
    axout.set_xlabel(format_axis_label(xlab),fontsize=axis_label_size,color=axis_text_color); axout.set_ylabel(format_axis_label(ylab),fontsize=axis_label_size,color=axis_text_color)
    apply_axis_label_positions(axout,x_label_position_mode,x_label_pad,x_label_coords,
                               y_label_position_mode,y_label_pad,y_label_coords)
    axout.set_title(title or None,fontsize=title_font_size,color=axis_text_color)
    axout.tick_params(axis="x",which="both",bottom=x_bottom,top=x_top,labelbottom=x_bottom,labeltop=x_top,labelsize=tick_font_size,labelcolor=axis_text_color)
    axout.tick_params(axis="y",which="both",left=y_left,right=y_right,labelleft=y_left,labelright=y_right,labelsize=tick_font_size,labelcolor=axis_text_color)
    if x_tick_interval>0 and not x_log: axout.xaxis.set_major_locator(MultipleLocator(x_tick_interval))
    if y_tick_interval>0 and not y_log: axout.yaxis.set_major_locator(MultipleLocator(y_tick_interval))
    axout.yaxis.set_major_formatter(y_tick_formatter)
    add_y_scale_annotation(axout,y_tick_options,tick_font_size,axis_text_color)
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
    corrected_errors={n:processed_error(s,opts) for s in series if s.get("error") is not None}

    raw_selected_table=export_curve_table(selected_targets,series,raw_values)
    c1,c2=st.columns(2)
    c1.download_button("선택 곡선 원본 CSV",raw_selected_table.to_csv(index=False).encode("utf-8-sig"),
                       "selected_spectra_original.csv","text/csv",disabled=not selected_targets,use_container_width=True)
    c2.download_button("선택 곡선 원본 TXT",raw_selected_table.to_csv(index=False,sep="\t").encode("utf-8-sig"),
                       "selected_spectra_original.txt","text/plain",disabled=not selected_targets,use_container_width=True)

    st.divider()
    corrected_selected_table=export_curve_table(selected_targets,series,corrected_values,corrected_errors)
    c1,c2=st.columns(2)
    c1.download_button("선택 곡선 보정 CSV",corrected_selected_table.to_csv(index=False).encode("utf-8-sig"),
                       "selected_spectra_corrected.csv","text/csv",disabled=not selected_targets,use_container_width=True)
    c2.download_button("선택 곡선 보정 TXT",corrected_selected_table.to_csv(index=False,sep="\t").encode("utf-8-sig"),
                       "selected_spectra_corrected.txt","text/plain",disabled=not selected_targets,use_container_width=True)

    st.divider()
    c1,c2=st.columns(2)
    exported_curve_names=[n for n in names if n not in error_source_names]
    c1.download_button("곡선별 원본 CSV ZIP",curve_archive(exported_curve_names,series,raw_values,"csv"),
                       "spectra_original_csv.zip","application/zip",disabled=not names,use_container_width=True)
    c2.download_button("곡선별 원본 TXT ZIP",curve_archive(exported_curve_names,series,raw_values,"txt"),
                       "spectra_original_txt.zip","application/zip",disabled=not names,use_container_width=True)

    st.divider()
    c1,c2=st.columns(2)
    c1.download_button("곡선별 보정 CSV ZIP",curve_archive(exported_curve_names,series,corrected_values,"csv",corrected_errors),
                       "spectra_corrected_csv.zip","application/zip",disabled=not names,use_container_width=True)
    c2.download_button("곡선별 보정 TXT ZIP",curve_archive(exported_curve_names,series,corrected_values,"txt",corrected_errors),
                       "spectra_corrected_txt.zip","application/zip",disabled=not names,use_container_width=True)

with tabs[5]:
    st.subheader("불러온 데이터 편집")
    st.caption("곡선을 선택하고 편집 창을 열면 X/Y 값을 스프레드시트처럼 수정할 수 있습니다. 수정한 값은 그래프와 내보내기에 반영됩니다.")
    edit_names=[s["name"] for s in st.session_state.series]
    selected_curve=st.selectbox("편집할 곡선",edit_names,key="sheet_curve_select")
    if st.button("선택 곡선 스프레드시트 편집 창 열기",key="open_curve_editor"):
        curve=next(s for s in st.session_state.series if s["name"]==selected_curve)
        st.session_state["curve_editor_name"]=selected_curve
        editor_data={"X":np.asarray(curve["x"],dtype=float),
                     "Y":np.asarray(curve["raw"],dtype=float),
                     "Error":np.asarray(curve.get("error",np.full(len(curve["x"]),np.nan)),dtype=float)}
        st.session_state["curve_editor_table_data"]=pd.DataFrame(editor_data)
        st.session_state["curve_editor_table_key"]=f"curve_editor_table_{edit_names.index(selected_curve)}_{st.session_state.get('curve_editor_revision',0)}"
        st.session_state["curve_editor_revision"]=st.session_state.get("curve_editor_revision",0)+1
        st.session_state["curve_editor_open"]=True
        st.rerun()
    st.divider()
    st.subheader("빈 스프레드시트")
    st.caption("새 데이터도 아래 표에 입력한 뒤 그래프에 추가할 수 있습니다.")
    render_blank_spectrum("blank_tab")
