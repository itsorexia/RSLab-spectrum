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
