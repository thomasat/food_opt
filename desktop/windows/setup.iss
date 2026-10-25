; Food Optimizer's Windows installer (Inno Setup 6.3 or later; spec section 5).
; build_installer.ps1 compiles it against the built image:
;
;   ISCC /Qp "/DAppVersion=<ver>" "/DImageDir=<image>" "/DDiskFreeGB=<n>"
;        "/O<dist>" "/F<name>" setup.iss
;
; Per-user by default (%LOCALAPPDATA%\Programs\Food Optimizer, no admin);
; /ALLUSERS installs to Program Files for IT. Three pages: Welcome, Install,
; Finish. Every message a user can reach is in installer_messages.isl.
; No PowerShell, no network and no download: Pascal and Shell COM only.
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
; The shell holds AppIdentity.SingleInstanceMutex (this session) and
; AppIdentity.InstallerMutex (every session, so a SYSTEM install sees it).
AppMutex=Local\FoodOptimizer.SingleInstance.3CC8CB77-0313-4A7D-9582-9896F0A3A8B3,Global\FoodOptimizer.Running.3CC8CB77-0313-4A7D-9582-9896F0A3A8B3
SetupMutex=FoodOptimizer.Setup.0E00CC2F-4E51-4EBF-A229-A910D5742A39,Global\FoodOptimizer.Setup.0E00CC2F-4E51-4EBF-A229-A910D5742A39
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
; must not linger. *.dll at the root: a shell runtime file a newer shell no
; longer ships.
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\app"
Type: filesandordirs; Name: "{app}\shared"
Type: files; Name: "{app}\*.dll"

[Icons]
Name: "{autoprograms}\Food Optimizer"; Filename: "{app}\FoodOptimizer.exe"; WorkingDir: "{app}"; AppUserModelID: "FoodSystemInnovations.FoodOptimizer"
Name: "{autoprograms}\Food Optimizer - Start Here"; Filename: "{app}\Start Here.txt"
Name: "{autodesktop}\Food Optimizer"; Filename: "{app}\FoodOptimizer.exe"; WorkingDir: "{app}"; AppUserModelID: "FoodSystemInnovations.FoodOptimizer"; Check: WantDesktopIcon

[Run]
Filename: "{app}\FoodOptimizer.exe"; Description: "{cm:OpenApp}"; Flags: postinstall nowait skipifsilent runasoriginaluser
Filename: "{app}\Start Here.txt"; Description: "{cm:OpenStartHere}"; Flags: postinstall shellexec nowait skipifsilent unchecked runasoriginaluser

[UninstallDelete]
Type: dirifempty; Name: "{app}"

[Code]
const
  UninstallKey = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{DE876ED9-927A-45AD-AD9B-D6E4B4C9A4A3}_is1';
  ProfileListKey = 'SOFTWARE\Microsoft\Windows NT\CurrentVersion\ProfileList';
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

function IsRunningAsSystem: Boolean;
begin
  Result := (CompareText(GetUserNameString, 'SYSTEM') = 0) or
    EndsText('\config\systemprofile', GetEnv('USERPROFILE'));
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

procedure DeleteLogged(const Path: String);
begin
  if FileExists(Path) then begin
    if DeleteFile(Path) then
      Log('replaced per-user copy: deleted ' + Path)
    else
      Log('replaced per-user copy: could not delete ' + Path);
  end;
end;

{ An all-users install replaces every per-user copy: a per-user and an
  all-users install can't coexist (spec section 5). It runs once the
  all-users copy is in place, so a cancelled or failed IT install leaves
  the user's copy alone. Never touches projects or state. A user whose
  hive isn't loaded keeps an orphan Apps entry. }
procedure ReplacePerUserCopies;
var
  Sids: TArrayOfString;
  I: Integer;
  Sid, Profile, Dir, Desktop, ShellFolders: String;
begin
  if not RegGetSubkeyNames(HKLM64, ProfileListKey, Sids) then begin
    Log('replaced per-user copy: no profile list');
    exit;
  end;
  for I := 0 to GetArrayLength(Sids) - 1 do begin
    Sid := Sids[I];
    { Local and domain accounts (S-1-5-21-*) and Entra ID accounts
      (S-1-12-1-*, the users of a cloud-joined Intune PC). }
    if (Pos('S-1-5-21-', Sid) <> 1) and (Pos('S-1-12-1-', Sid) <> 1) then
      continue;
    if not RegQueryStringValue(HKLM64, ProfileListKey + '\' + Sid, 'ProfileImagePath', Profile) then
      continue;
    StringChangeEx(Profile, '%SystemDrive%', GetEnv('SystemDrive'), True);
    Dir := Profile + '\AppData\Local\Programs\Food Optimizer';
    if FileExists(Dir + '\FoodOptimizer.exe') then begin
      if DelTree(Dir, True, True, True) then
        Log('replaced per-user copy: deleted ' + Dir)
      else
        Log('replaced per-user copy: could not delete ' + Dir);
    end;
    DeleteLogged(Profile + '\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Food Optimizer.lnk');
    DeleteLogged(Profile + '\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Food Optimizer - Start Here.lnk');
    if RegKeyExists(HKU, Sid) then begin
      if RegKeyExists(HKU, Sid + '\' + UninstallKey) then begin
        if RegDeleteKeyIncludingSubkeys(HKU, Sid + '\' + UninstallKey) then
          Log('replaced per-user copy: deleted the uninstall entry of ' + Sid)
        else
          Log('replaced per-user copy: could not delete the uninstall entry of ' + Sid);
      end;
      ShellFolders := Sid + '\Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders';
      if RegQueryStringValue(HKU, ShellFolders, 'Desktop', Desktop) then begin
        StringChangeEx(Desktop, '%USERPROFILE%', Profile, True);
        DeleteLogged(Desktop + '\Food Optimizer.lnk');
      end;
    end else
      DeleteLogged(Profile + '\Desktop\Food Optimizer.lnk');
  end;
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

procedure PinToQuickAccess(const Dir: String);
var
  Shell: Variant;
begin
  try
    Shell := CreateOleObject('Shell.Application');
    if not VarIsEmpty(QuickAccessItem(Shell, Dir)) then begin
      Log('quickaccess: already listed ' + Dir);
      exit;
    end;
    Shell.Namespace(Dir).Self.InvokeVerb('pintohome');
    Log('quickaccess: pinned ' + Dir);
  except
    Log('quickaccess failed: ' + GetExceptionMessage);
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

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then begin
    if IsAdminInstallMode then
      ReplacePerUserCopies;
    if not IsAdminInstallMode then begin
      if not ForceDirectories(ProjectsDir) then
        Log('could not create ' + ProjectsDir);
      PinToQuickAccess(ProjectsDir);
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
  if (CurUninstallStep = usUninstall) and not IsAdminInstallMode then
    UnpinFromQuickAccess(ProjectsDir);
  if CurUninstallStep = usPostUninstall then begin
    Dir := RemoveBackslashUnlessRoot(StateDir);
    if CompareText(ExtractFileName(Dir), 'FoodOptimizer') = 0 then begin
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
