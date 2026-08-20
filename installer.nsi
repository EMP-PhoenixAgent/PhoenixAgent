; ============================================================================
;  Phoenix Agent — NSIS Installer Wizard
; ============================================================================
;  This script is used by Tauri's NSIS bundler as a "installer hook" via the
;  `bundle.windows.nsis.installerHooks` config. Tauri generates the main
;  installer body; we provide custom preinstall and postinstall hooks that:
;    Step 1: Check + install missing dependencies (Ollama, WebView2, ripgrep)
;    Step 2: (Tauri handles the file copy)
;    Step 3: Launch the app after install
;
;  The full standalone NSIS script below is an alternative that can be used
;  directly with makensis if Tauri's built-in NSIS bundler is unavailable.
; ============================================================================

!define APPNAME "Phoenix Agent"
!define APPVERSION "0.3.0"
!define APPPUBLISHER "Phoenix Agent"
!define APPID "com.phoenix.agent"
!define APPURL "https://github.com/phoenix-agent"

; --- Install directory (per-user, no admin needed) -------------------------
InstallDir "$LOCALAPPDATA\Programs\PhoenixAgent"

Name "${APPNAME}"
OutFile "..\dist\PhoenixAgent-Setup.exe"
Unicode True
RequestExecutionLevel user
ShowInstDetails show

; --- Modern UI -------------------------------------------------------------
!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "x64.nsh"

!define MUI_ICON "..\phoenix-agent\icons\icon.ico"
!define MUI_ABORTWARNING

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

; ============================================================================
;  Sections
; ============================================================================

Section "Phoenix Agent" SecMain
    SectionIn RO
    SetOutPath "$INSTDIR"

    ; --- Step 1: Dependency checks + install --------------------------------
    DetailPrint "Step 1: Checking dependencies..."

    ; Check Ollama
    nsExec::ExecToStack 'where ollama'
    Pop $0
    ${If} $0 != 0
        DetailPrint "Ollama not found. Installing..."
        ; Run the bundled Ollama installer silently
        File "..\installer\deps\OllamaSetup.exe"
        nsExec::ExecWait '"$INSTDIR\OllamaSetup.exe" /S' $0
        DetailPrint "Ollama installer exited with code $0"
        Delete "$INSTDIR\OllamaSetup.exe"
    ${Else}
        DetailPrint "Ollama already installed."
    ${EndIf}

    ; Check WebView2 (registry)
    ReadRegStr $0 HKLM "SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" "pv"
    ${If} $0 == ""
        ReadRegStr $0 HKCU "Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" "pv"
    ${EndIf}
    ${If} $0 == ""
        DetailPrint "WebView2 not found. Installing..."
        File "..\installer\deps\MicrosoftEdgeWebview2Setup.exe"
        nsExec::ExecWait '"$INSTDIR\MicrosoftEdgeWebview2Setup.exe" /silent /install' $0
        DetailPrint "WebView2 installer exited with code $0"
        Delete "$INSTDIR\MicrosoftEdgeWebview2Setup.exe"
    ${Else}
        DetailPrint "WebView2 already installed (v$0)."
    ${EndIf}

    ; Install ripgrep
    CreateDirectory "$INSTDIR\bin"
    File /oname=bin\rg.exe "..\installer\deps\rg.exe"
    DetailPrint "ripgrep installed to $INSTDIR\bin\rg.exe"

    ; --- Step 2: Install the app --------------------------------------------
    DetailPrint "Step 2: Installing Phoenix Agent..."
    ; The main binary (built by cargo tauri build)
    File "..\phoenix-agent\target\release\phoenix.exe"

    ; Frontend resources (Tauri embeds these in the binary, but we also
    ; copy bundled background images as standalone resources)
    CreateDirectory "$INSTDIR\scripts\assets\backgrounds"
    File /nonfatal "..\phoenix-agent\scripts\assets\backgrounds\*.png"

    ; --- Shortcuts + PATH ---
    CreateDirectory "$SMPROGRAMS\Phoenix Agent"
    CreateShortcut "$SMPROGRAMS\Phoenix Agent\Phoenix Agent.lnk" "$INSTDIR\phoenix.exe"
    CreateShortcut "$DESKTOP\Phoenix Agent.lnk" "$INSTDIR\phoenix.exe"

    ; Add bin dir to user PATH (for rg.exe)
    nsExec::ExecToStack 'echo %PATH%'
    Pop $0
    Pop $1
    ${If} $1 not contains "$INSTDIR\bin"
        ReadRegStr $1 HKCU "Environment" "PATH"
        StrCpy $1 "$1;$INSTDIR\bin"
        WriteRegExpandStr HKCU "Environment" "PATH" "$1"
        SendMessage ${HWND_BROADCAST} ${WM_SETTINGCHANGE} 0 "STR:Environment" /TIMEOUT=5000
        DetailPrint "Added $INSTDIR\bin to PATH"
    ${EndIf}

    ; --- Uninstaller ---
    WriteUninstaller "$INSTDIR\Uninstall.exe"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPID}" "DisplayName" "${APPNAME}"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPID}" "UninstallString" "$INSTDIR\Uninstall.exe"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPID}" "Publisher" "${APPPUBLISHER}"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPID}" "DisplayVersion" "${APPVERSION}"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPID}" "InstallLocation" "$INSTDIR"
SectionEnd

; --- Launch the app after install ------------------------------------------
Section -PostInstall
    DetailPrint "Step 3: Launching Phoenix Agent..."
    Exec '"$INSTDIR\phoenix.exe"'
SectionEnd

; ============================================================================
;  Uninstaller
; ============================================================================

Section "Uninstall"
    Delete "$INSTDIR\phoenix.exe"
    Delete "$INSTDIR\Uninstall.exe"
    RMDir /r "$INSTDIR\bin"
    RMDir /r "$INSTDIR\scripts"
    RMDir "$INSTDIR"

    Delete "$SMPROGRAMS\Phoenix Agent\Phoenix Agent.lnk"
    RMDir "$SMPROGRAMS\Phoenix Agent"
    Delete "$DESKTOP\Phoenix Agent.lnk"

    DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPID}"
SectionEnd

Function .onInstSuccess
    DetailPrint "Installation complete!"
FunctionEnd
