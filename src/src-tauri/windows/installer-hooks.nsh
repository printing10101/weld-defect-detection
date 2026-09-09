; NSIS 安装器钩子（tauri.conf.json -> bundle.windows.nsis.installerHooks 引用）。
; 仅支持 ; / # / /* */ 注释（NSIS 不认 //）。
;
; 数据保护：历史版本把业务数据（SQLite 检测记录 / 报告 PDF / 影像副本 /
; 主密钥 .crypto_key）写在安装目录 data\ 下。新版数据目录已重定向到
; %APPDATA%\com.scandetection.sd\data（壳经 SCANDETECTION_USER_DATA_DIR 传
; 给后端）。两个钩子分别覆盖两条路径：
;   - 覆盖式升级（装新版不触发旧版卸载器）：POSTINSTALL 把 $INSTDIR\data
;     迁移到用户数据目录（仅在目标不存在时，防覆盖新版已写入的数据）；
;   - 直接卸载旧版：PREUNINSTALL 把 $INSTDIR\data 备份到用户数据目录
;     （目标已存在时备份到独立目录，绝不原地覆盖）。
!macro NSIS_HOOK_POSTINSTALL
  IfFileExists "$INSTDIR\data\*.*" 0 sd_post_done
    IfFileExists "$APPDATA\com.scandetection.sd\data\*.*" sd_post_done 0
      CreateDirectory "$APPDATA\com.scandetection.sd"
      CopyFiles /SILENT "$INSTDIR\data" "$APPDATA\com.scandetection.sd"
      DetailPrint "已迁移旧版业务数据到 %APPDATA%\com.scandetection.sd\data"
  sd_post_done:
  ; 字节码预编译（首启加速）：壳侧以 PYTHONPYCACHEPREFIX=%TEMP%\ScanDetection\pycache
  ; 运行后端（缓存不进安装目录，卸载清单干净）。安装期按同一前缀预编译一次，
  ; 首次启动即免去全量源码重编译（重依赖导入从分钟级降到秒级）。
  ; 预编译失败不阻断安装——首次启动自会边导入边编译。
  IfFileExists "$INSTDIR\python_embed\python.exe" 0 sd_precomp_done
    CreateDirectory "$TEMP\ScanDetection"
    DetailPrint "正在预编译 Python 字节码（约 1~2 分钟，加速首次启动）…"
    nsExec::ExecToLog 'cmd /c set PYTHONPYCACHEPREFIX=$TEMP\ScanDetection\pycache&& "$INSTDIR\python_embed\python.exe" -m compileall -q "$INSTDIR\backend" "$INSTDIR\python_embed\Lib\site-packages"'
    Pop $0
    DetailPrint "字节码预编译结束（退出码 $0，非零不影响使用）"
  sd_precomp_done:
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  IfFileExists "$INSTDIR\data\*.*" 0 sd_pre_done
    IfFileExists "$APPDATA\com.scandetection.sd\data\*.*" 0 sd_pre_migrate
      ; 用户数据目录已有数据（新版在用）：旧数据备份到独立目录，不覆盖
      CreateDirectory "$APPDATA\com.scandetection.sd"
      CopyFiles /SILENT "$INSTDIR\data" "$APPDATA\com.scandetection.sd\data.pre-uninstall-backup"
      DetailPrint "旧版数据已备份到 %APPDATA%\com.scandetection.sd\data.pre-uninstall-backup"
      Goto sd_pre_done
  sd_pre_migrate:
      CreateDirectory "$APPDATA\com.scandetection.sd"
      CopyFiles /SILENT "$INSTDIR\data" "$APPDATA\com.scandetection.sd"
      DetailPrint "已备份业务数据到 %APPDATA%\com.scandetection.sd\data"
  sd_pre_done:
!macroend
