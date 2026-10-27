; Food Optimizer's Windows installer (Inno Setup 6.3 or later; spec section 5).
; build_installer.ps1 compiles it against the built image:
;
;   ISCC /Qp "/DAppVersion=<ver>" "/DImageDir=<image>" "/DDiskFreeGB=<n>"
;        "/O<dist>" "/F<name>" setup.iss
;
; Per-user by default (%LOCALAPPDATA%\Programs\Food Optimizer, no admin);
; /ALLUSERS installs to Program Files for IT. Three pages: Welcome, Install,
; Finish. Every message a user can reach is in installer_messages.isl.
; No PowerShell, no network and no download: Pascal and Shell COM, and
; icacls for an all-users /DIR outside Program Files (LockDownAppDir).
; ASCII only. tests/test_windows_installer.py pins the directives below.

#if VER < EncodeVer(6,3,0)
  #error Inno Setup 6.3 or later is required
#endif

#ifndef AppVersion
  #define VersionHandle FileOpen(AddBackslash(SourcePath) + "..\VERSION")
  #define AppVersion Trim(FileRead(VersionHandle))
  #expr FileClose(VersionHandle)
#endif
#ifndef ImageDir
  #error ImageDir is required: ISCC /DImageDir=<image folder>
#endif
#ifndef DiskFreeGB
  #error DiskFreeGB is required: ISCC /DDiskFreeGB=<GB>
#endif
; The install root's budget, its trailing "\" included (build_env.ROOT_BUDGET).
#define MaxRoot 80
; Python ends the pre-warm itself after this many seconds (faulthandler).
#define PrewarmSecs 600

[Setup]
AppId={{DE876ED9-927A-45AD-AD9B-D6E4B4C9A4A3}
AppName=Food Optimizer
AppVersion={#AppVersion}
AppVerName=Food Optimizer {#AppVersion}
AppPublisher=Food System Innovations
AppContact=sohum.patnaik@fsi.org
AppSupportURL=mailto:sohum.patnaik@fsi.org
VersionInfoVersion={#AppVersion}.0
VersionInfoProductVersion={#AppVersion}
VersionInfoCompany=Food System Innovations
VersionInfoProductName=Food Optimizer
VersionInfoDescription=Food Optimizer Setup
; Per-user unless the command line says /ALLUSERS; the choice is never
; remembered from an earlier install.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline
UsePreviousPrivileges=no
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Windows 10 22H2 (build 19045) or later.
MinVersion=10.0.19045
DefaultDirName={autopf}\Food Optimizer
DisableDirPage=yes
DisableProgramGroupPage=yes
DisableReadyPage=yes
DisableWelcomePage=no
ShowLanguageDialog=no
; No AppMutex: it is one list for both scopes. The code below checks the
; running app's name per scope instead (RunningName, AppClosed). SetupMutex
; is per scope too (SetupMutexNames; Inno expands it after InitializeSetup).
SetupMutex={code:SetupMutexNames}
CloseApplications=yes
CloseApplicationsFilter=*.exe,*.dll,*.pyd
RestartApplications=no
SetupLogging=yes
UninstallDisplayName=Food Optimizer
UninstallDisplayIcon={app}\FoodOptimizer.exe
SetupIconFile=icon.ico
WizardStyle=modern
WizardImageFile=installer_art\wizard_100.bmp,installer_art\wizard_150.bmp,installer_art\wizard_200.bmp
WizardSmallImageFile=installer_art\small_100.bmp,installer_art\small_150.bmp,installer_art\small_200.bmp
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4
OutputDir=dist
OutputBaseFilename=FoodOptimizer-{#AppVersion}-Windows

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl,installer_messages.isl"

[Files]
Source: "{#ImageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; An upgrade starts from a clean Python and app: a package an older lock had
; must not linger. runtimes and the root's *.dll: a shell runtime file a
; newer shell no longer ships.
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\app"
Type: filesandordirs; Name: "{app}\shared"
Type: filesandordirs; Name: "{app}\runtimes"
Type: files; Name: "{app}\*.dll"

[Icons]
Name: "{autoprograms}\Food Optimizer"; Filename: "{app}\FoodOptimizer.exe"; WorkingDir: "{app}"; AppUserModelID: "FoodSystemInnovations.FoodOptimizer"
Name: "{autoprograms}\Food Optimizer - Start Here"; Filename: "{app}\Start Here.txt"
Name: "{autodesktop}\Food Optimizer"; Filename: "{app}\FoodOptimizer.exe"; WorkingDir: "{app}"; AppUserModelID: "FoodSystemInnovations.FoodOptimizer"; Check: WantDesktopIcon

[Run]
Filename: "{app}\FoodOptimizer.exe"; Description: "{cm:OpenApp}"; Flags: postinstall nowait skipifsilent runasoriginaluser
Filename: "{app}\Start Here.txt"; Description: "{cm:OpenStartHere}"; Flags: postinstall shellexec nowait skipifsilent unchecked runasoriginaluser

[UninstallDelete]
Type: files; Name: "{app}\all-users"
Type: dirifempty; Name: "{app}"

[Code]
const
  UninstallKey = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{DE876ED9-927A-45AD-AD9B-D6E4B4C9A4A3}_is1';
  QuickAccess = 'shell:::{679f85cb-0220-4080-b29b-5540cc05aab6}';
  MaxRoot = {#MaxRoot};
  DiskFreeGB = {#DiskFreeGB};
  GiB = 1073741824;

var
  DesktopCheck: TNewCheckBox;
  PrewarmFailed: Boolean;
  Upgrading: Boolean;

function GetTickCount: DWORD; external 'GetTickCount@kernel32.dll stdcall';

function EndsText(const Tail, S: String): Boolean;
begin
  Result := (Length(S) >= Length(Tail)) and
    (CompareText(Copy(S, Length(S) - Length(Tail) + 1, Length(Tail)), Tail) = 0);
end;

function EnvOr(const Name, Fallback: String): String;
begin
  Result := GetEnv(Name);
  if Result = '' then
    Result := Fallback;
end;

{ The user's folders; FOODOPT_* override them as they do for the shell. }
function ProjectsDir: String;
begin
  Result := EnvOr('FOODOPT_PROJECTS_DIR', ExpandConstant('{%USERPROFILE}') + '\FoodOptimizer');
end;

function StateDir: String;
begin
  Result := EnvOr('FOODOPT_STATE_DIR', ExpandConstant('{localappdata}') + '\FoodOptimizer');
end;

{ Whether Inner is Outer or a folder inside it, whatever the case, with
  or without a trailing backslash, and however written: ExpandFileName
  (GetFullPathName) folds doubled backslashes, "." and ".." and "/". }
function SameOrInside(const Inner, Outer: String): Boolean;
var
  I, O: String;
begin
  I := AnsiLowercase(RemoveBackslashUnlessRoot(ExpandFileName(Inner)));
  O := AnsiLowercase(RemoveBackslashUnlessRoot(ExpandFileName(Outer)));
  Result := (I = O) or (Pos(AddBackslash(O), I) = 1);
end;

function IsRunningAsSystem: Boolean;
begin
  Result := (CompareText(GetUserNameString, 'SYSTEM') = 0) or
    EndsText('\config\systemprofile', GetEnv('USERPROFILE'));
end;

{ The name this scope's running app holds (AppIdentity.cs): a per-user
  copy's only in its own session (Local), so no other user's app can hold
  up a per-user install or uninstall; the all-users copy's in every
  session (Global), so an IT upgrade waits for every user's app. }
function RunningName: String;
begin
  if IsAdminInstallMode then
    Result := 'Global\FoodOptimizer.Running.AllUsers.3CC8CB77-0313-4A7D-9582-9896F0A3A8B3'
  else
    Result := 'Local\FoodOptimizer.Running.PerUser.3CC8CB77-0313-4A7D-9582-9896F0A3A8B3';
end;

{ One installer at a time, per scope: a per-user installer checks and
  holds only this session's name, so no other user can hold it up; an
  all-users one also the Global name, so two all-users installers never
  run at once, whatever their sessions. }
function SetupMutexNames(Param: String): String;
begin
  Result := 'FoodOptimizer.Setup.0E00CC2F-4E51-4EBF-A229-A910D5742A39';
  if IsAdminInstallMode then
    Result := Result + ',Global\FoodOptimizer.Setup.0E00CC2F-4E51-4EBF-A229-A910D5742A39';
end;

{ Inno's AppMutex check, for this scope's name only: OK checks again;
  Cancel, and every run with /SUPPRESSMSGBOXES, stops before anything
  changes (Setup exits 1, as with AppMutex). Quiet (a silent uninstall,
  which the all-users app starts for this user's per-user copy) stops
  without a box, whatever the switches. }
function AppClosed(const Msg: String; const Quiet: Boolean): Boolean;
begin
  Result := True;
  while CheckForMutexes(RunningName) do
    if Quiet or (SuppressibleMsgBox(FmtMessage(Msg, ['Food Optimizer']), mbError, MB_OKCANCEL, IDCANCEL) <> IDOK) then begin
      Log('refused: app_running');
      Result := False;
      exit;
    end;
end;

{ Inno runs this just before its own AppMutex check. }
function InitializeSetup: Boolean;
begin
  Result := AppClosed(SetupMessage(msgSetupAppRunningError), False);
end;

function MachineCopyExists: Boolean;
begin
  Result := RegKeyExists(HKLM64, UninstallKey);
end;

function HasParam(const Name: String): Boolean;
var
  I: Integer;
begin
  Result := False;
  for I := 1 to ParamCount do
    if CompareText(ParamStr(I), Name) = 0 then
      Result := True;
end;

{ Whether a folder holds anything at all. }
function DirHasEntries(const Dir: String): Boolean;
var
  FindRec: TFindRec;
begin
  Result := False;
  if FindFirst(AddBackslash(Dir) + '*', FindRec) then begin
    try
      repeat
        if (FindRec.Name <> '.') and (FindRec.Name <> '..') then begin
          Result := True;
          exit;
        end;
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;
end;

function DirBytes(const Dir: String): Int64;
var
  FindRec: TFindRec;
begin
  Result := 0;
  if FindFirst(AddBackslash(Dir) + '*', FindRec) then begin
    try
      repeat
        if (FindRec.Name <> '.') and (FindRec.Name <> '..') then begin
          if FindRec.Attributes and FILE_ATTRIBUTE_DIRECTORY <> 0 then begin
            if FindRec.Attributes and FILE_ATTRIBUTE_REPARSE_POINT = 0 then
              Result := Result + DirBytes(AddBackslash(Dir) + FindRec.Name);
          end else
            Result := Result + Int64(FindRec.SizeHigh) * 4294967296 + Int64(FindRec.SizeLow);
        end;
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;
end;

{ The first refusal that applies, or ''. Silent: exit 7; interactive: the
  Preparing page with Cancel. }
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  App: String;
  Free, Total, Need: Int64;
begin
  Result := '';
  if IsRunningAsSystem and not IsAdminInstallMode then begin
    Log('refused: system_needs_allusers');
    Result := CustomMessage('SystemNeedsAllUsers');
    exit;
  end;
  if not IsAdminInstallMode and MachineCopyExists then begin
    Log('refused: installed_by_it');
    Result := CustomMessage('InstalledByIT');
    exit;
  end;
  App := ExpandConstant('{app}');
  if Length(AddBackslash(App)) > MaxRoot then begin
    Log('refused: folder_too_long');
    Result := CustomMessage('FolderTooLong');
    exit;
  end;
  { [InstallDelete] would delete python, app, shared and *.dll of any
    folder /DIR names: install only into an empty folder or our own. }
  Upgrading := FileExists(AddBackslash(App) + 'FoodOptimizer.exe');
  if DirExists(App) and DirHasEntries(App) and not Upgrading and
      not FileExists(AddBackslash(App) + 'unins000.dat') then begin
    Log('refused: folder_not_ours');
    Result := CustomMessage('FolderNotOurs');
    exit;
  end;
  { An upgrade frees the copy it replaces ([InstallDelete] runs first). }
  Need := Int64(DiskFreeGB) * GiB;
  if Upgrading then
    Need := Need - DirBytes(App);
  if GetSpaceOnDisk64(AddBackslash(ExtractFileDrive(App)), Free, Total) then begin
    Log(Format('free space: %d MB free, %d MB needed', [Free div 1048576, Need div 1048576]));
    if Free < Need then begin
      Log('refused: needs_space');
      Result := FmtMessage(CustomMessage('NeedsSpace'), [IntToStr(DiskFreeGB)]);
      exit;
    end;
  end else
    Log('free space: unknown for ' + App);
end;

{ Our Welcome text alone: Inno would append its own "Click Next..." line,
  and its messages know no [ver]. }
function WelcomeText: String;
begin
  Result := SetupMessage(msgWelcomeLabel2);
  StringChangeEx(Result, '[name]', 'Food Optimizer', True);
  StringChangeEx(Result, '[ver]', '{#AppVersion}', True);
  StringChangeEx(Result, '%n', #13#10, True);
end;

procedure InitializeWizard;
begin
  WizardForm.WelcomeLabel2.Caption := WelcomeText;
  WizardForm.WelcomeLabel2.AdjustHeight;
  DesktopCheck := TNewCheckBox.Create(WizardForm);
  DesktopCheck.Parent := WizardForm.WelcomePage;
  DesktopCheck.Caption := CustomMessage('DesktopShortcut');
  DesktopCheck.Checked := True;
  DesktopCheck.Left := WizardForm.WelcomeLabel2.Left;
  DesktopCheck.Top := WizardForm.WelcomeLabel2.Top + WizardForm.WelcomeLabel2.Height + ScaleY(16);
  DesktopCheck.Width := WizardForm.WelcomeLabel2.Width;
  DesktopCheck.Height := ScaleY(20);
end;

{ Silent installs add the shortcut unless /NODESKTOPSHORTCUT (IT often
  wants none); interactive ones follow the Welcome page's checkbox. }
function WantDesktopIcon: Boolean;
begin
  if WizardSilent then
    Result := not HasParam('/NODESKTOPSHORTCUT')
  else
    Result := DesktopCheck.Checked;
end;

function QuickAccessItem(Shell: Variant; const Dir: String): Variant;
var
  Items: Variant;
  I: Integer;
begin
  Result := Unassigned;
  Items := Shell.Namespace(QuickAccess).Items;
  for I := 0 to Items.Count - 1 do
    if CompareText(Items.Item(I).Path, Dir) = 0 then begin
      Result := Items.Item(I);
      exit;
    end;
end;

procedure UnpinFromQuickAccess(const Dir: String);
var
  Shell, Item: Variant;
begin
  try
    Shell := CreateOleObject('Shell.Application');
    Item := QuickAccessItem(Shell, Dir);
    if VarIsEmpty(Item) then begin
      Log('quickaccess: not listed ' + Dir);
      exit;
    end;
    Item.InvokeVerb('unpinfromhome');
    Log('quickaccess: unpinned ' + Dir);
  except
    Log('quickaccess failed: ' + GetExceptionMessage);
  end;
end;

{ Whether the app pinned the projects folder (D2): it pins it at its first
  launch and records that in settings.json (ShellSettings writes exactly
  this text). A pin the user made stays. }
function AppPinnedProjects: Boolean;
var
  S: AnsiString;
begin
  Result := LoadStringFromFile(AddBackslash(StateDir) + 'settings.json', S) and
    (Pos('"quickAccessPinned": true', String(S)) > 0);
end;

{ Pays Defender's first scan at install time and catches a blocked runtime
  with a plain message. A failure keeps the install and exits 9. }
function Prewarm: Integer;
var
  Rc: Integer;
  Started: DWORD;
begin
  WizardForm.StatusLabel.Caption := CustomMessage('PrewarmStatus');
  WizardForm.FilenameLabel.Caption := '';
  WizardForm.ProgressGauge.Style := npbstMarquee;
  Started := GetTickCount;
  if not ExecAndLogOutput(ExpandConstant('{app}\python\python.exe'),
      '-I -X utf8 -B -c "import faulthandler; faulthandler.dump_traceback_later({#PrewarmSecs}, exit=True); import torch, botorch, gpytorch, pandas, openpyxl, streamlit"',
      ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, Rc, nil) then begin
    Log('prewarm could not start: ' + SysErrorMessage(Rc));
    Rc := -1;
  end;
  Log(Format('prewarm rc=%d secs=%d', [Rc, (GetTickCount - Started) div 1000]));
  WizardForm.ProgressGauge.Style := npbstNormal;
  if Rc <> 0 then begin
    PrewarmFailed := True;
    SuppressibleMsgBox(CustomMessage('PrewarmFailed'), mbError, MB_OK, IDOK);
  end;
  Result := Rc;
end;

{ Under Program Files an all-users copy inherits its rule: users read and
  run it, never change it. A /DIR elsewhere would inherit the drive's, and
  C:\ lets every signed-in user change a new folder, so the app's files
  could be swapped under everyone (M4). icacls replaces the inherited rule
  with Program Files' own: SYSTEM and Administrators full, Users read and
  run. A failure is logged; the install goes on. }
procedure LockDownAppDir;
var
  App: String;
  Code: Integer;
begin
  App := RemoveBackslashUnlessRoot(ExpandConstant('{app}'));
  if SameOrInside(App, ExpandConstant('{commonpf64}')) then
    exit;
  if Exec(ExpandConstant('{sys}\icacls.exe'),
      '"' + App + '" /inheritance:r /grant:r *S-1-5-18:(OI)(CI)F *S-1-5-32-544:(OI)(CI)F *S-1-5-32-545:(OI)(CI)RX',
      '', SW_HIDE, ewWaitUntilTerminated, Code) and (Code = 0) then
    Log('acl: locked ' + App)
  else
    Log('acl: failed ' + IntToStr(Code));
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then begin
    { An all-users install never changes a per-user copy (D1): that
      user's next launch of this copy runs the per-user copy's own
      uninstaller. The app pins the projects folder (D2). }
    if IsAdminInstallMode then begin
      { The shell reads it: this copy's app holds the all-users name and
        looks for this user's per-user copy. }
      if not SaveStringToFile(ExpandConstant('{app}\all-users'), 'all-users' + #13#10, False) then
        Log('could not write ' + ExpandConstant('{app}\all-users'));
      { Before the pre-warm runs Python from the folder as SYSTEM. }
      LockDownAppDir;
    end;
    if not IsAdminInstallMode then begin
      if not ForceDirectories(ProjectsDir) then
        Log('could not create ' + ProjectsDir);
    end;
    Prewarm;
  end;
end;

{ An upgrade's [InstallDelete] has already deleted the old Python and app
  when the copy starts, and rollback can't bring them back: once
  installing, an upgrade runs to the end. }
procedure CancelButtonClick(CurPageID: Integer; var Cancel, Confirm: Boolean);
begin
  if (CurPageID = wpInstalling) and Upgrading then begin
    Log('cancel refused: an upgrade cannot stop part-way');
    Cancel := False;
    Confirm := False;
  end;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  { Welcome is the last page before installing (no Ready page). }
  if CurPageID = wpWelcome then
    WizardForm.NextButton.Caption := SetupMessage(msgButtonInstall);
  if (CurPageID = wpFinished) and PrewarmFailed and (WizardForm.RunList.Items.Count > 0) then begin
    WizardForm.RunList.Checked[0] := False;
    WizardForm.RunList.ItemEnabled[0] := False;
  end;
end;

function GetCustomSetupExitCode: Integer;
begin
  if PrewarmFailed then
    Result := 9
  else
    Result := 0;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Dir: String;
  I: Integer;
begin
  { After the user confirms, where Inno checks AppMutex; Abort here ends
    the uninstall before anything changes. }
  if (CurUninstallStep = usAppMutexCheck) and
      not AppClosed(SetupMessage(msgUninstallAppRunningError), UninstallSilent) then
    Abort;
  { The all-users app runs a per-user copy's uninstaller (D1) while it
    uses the same state folder and pin: both stay while an all-users copy
    is installed. }
  if (CurUninstallStep = usUninstall) and not IsAdminInstallMode then begin
    if MachineCopyExists then
      Log('quickaccess: kept (an all-users copy uses it)')
    else if AppPinnedProjects then
      UnpinFromQuickAccess(ProjectsDir)
    else
      Log('quickaccess: not pinned by the app');
  end;
  if CurUninstallStep = usPostUninstall then begin
    Dir := RemoveBackslashUnlessRoot(StateDir);
    { A state folder is a per-user location: an all-users (elevated)
      uninstall deletes none (D1). }
    if IsAdminInstallMode then
      Log('state folder kept (all-users uninstall): ' + Dir)
    else if MachineCopyExists then
      Log('state folder kept (an all-users copy uses it): ' + Dir)
    else if SameOrInside(ProjectsDir, Dir) then
      Log('state folder kept (it holds the projects folder): ' + Dir)
    else if CompareText(ExtractFileName(Dir), 'FoodOptimizer') = 0 then begin
      for I := 1 to 5 do begin
        if not DirExists(Dir) then
          break;
        DelTree(Dir, True, True, True);
        if not DirExists(Dir) then
          break;
        Sleep(1000);
      end;
      if DirExists(Dir) then
        Log('state folder left: ' + Dir)
      else
        Log('state folder deleted: ' + Dir);
    end else
      Log('state folder kept (its name is not FoodOptimizer): ' + Dir);
  end;
end;
