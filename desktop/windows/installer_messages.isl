; Food Optimizer's installer copy (owner-approved 2026-10-24), layered on
; Inno Setup's Default.isl by setup.iss:
;   MessagesFile: "compiler:Default.isl,installer_messages.isl"
; Every message our install and uninstall flows reach is rewritten to the
; copy rules: status <= 10 words a sentence, titles <= 8, labels <= 5.
; tests/test_windows_installer.py pins each value. ASCII only, LF line endings. [name] is Food Optimizer,
; [ver] its version, %1 Inno's argument, %n a new line, & an accelerator.

[Messages]
SetupAppTitle=Food Optimizer Setup
SetupWindowTitle=Install %1
UninstallAppTitle=Uninstall Food Optimizer
UninstallAppFullTitle=Uninstall %1
WelcomeLabel1=Install [name]
WelcomeLabel2=Version [ver]. Works offline.%n%nUpdating keeps your projects.
WizardPreparing=Getting ready
PreparingDesc=Getting ready to install [name].
CannotContinue=Click Cancel to close.
ApplicationsFound=These apps use files the installer must update.
ApplicationsFound2=These apps use files the installer must update.
CloseApplications=&Close them automatically
DontCloseApplications=&Leave them open
ErrorCloseApplications=Some apps could not be closed. Close them, then click Next.
WizardInstalling=Installing
InstallingLabel=Installing [name] on this PC.
StatusClosingApplications=Closing Food Optimizer
StatusCreateDirs=Creating folders
StatusExtractFiles=Copying files
StatusCreateIcons=Adding shortcuts
StatusCreateIniEntries=Saving settings
StatusCreateRegistryEntries=Saving settings
StatusSavingUninstall=Saving uninstall details
StatusRunProgram=Finishing
StatusRollback=Undoing changes
FinishedHeadingLabel=[name] is installed
FinishedLabelNoIcons=Open it from Start.
FinishedLabel=Open it from Start.
ClickFinish=Click Finish to close.
ExitSetupTitle=Stop installing?
ExitSetupMessage=Food Optimizer is not installed yet. Stop installing?
SetupAppRunningError=Close %1, then click OK.
UninstallAppRunningError=Close %1, then click OK.
ConfirmUninstall=Uninstall %1? Your projects are kept.
UninstallStatusLabel=Uninstalling %1 from this PC.
UninstalledAll=%1 is uninstalled.%n%nYour projects stay in your user folder, under FoodOptimizer.
UninstalledMost=%1 is uninstalled. Restart this PC to delete the last files.
UninstalledAndNeedsRestart=Restart this PC to finish uninstalling %1.
WinVersionTooLowError=This PC is not supported.%n%nNeeds 64-bit Windows 10 (22H2) or Windows 11.
OnlyOnTheseArchitectures=This PC is not supported.%n%nNeeds 64-bit Windows 10 (22H2) or Windows 11.
AdminPrivilegesRequired=Installing for all users needs admin rights.
PowerUserPrivilegesRequired=Installing for all users needs admin rights.
SetupFileCorrupt=This installer is damaged. Download it again.
SetupFileCorruptOrWrongVer=This installer is damaged. Download it again.
LdrCannotCreateTemp=The installer could not start. Free some disk space, then try again.
LdrCannotExecTemp=The installer could not start. Free some disk space, then try again.
ErrorReplacingExistingFile=A file could not be replaced. Close Food Optimizer, then try again.
DiskSpaceWarningTitle=Not enough space
DiskSpaceWarning=Needs %1 KB free. This drive has %2 KB.%n%nContinue anyway?
WizardUninstalling=Uninstalling
StatusUninstalling=Uninstalling %1
AbortRetryIgnoreSelectAction=Choose what to do
AbortRetryIgnoreRetry=&Try again
AbortRetryIgnoreIgnore=&Continue anyway
AbortRetryIgnoreCancel=Stop installing
FileAbortRetryIgnoreSkipNotRecommended=&Skip (app may not work)
RetryCancelSelectAction=Choose what to do
RetryCancelRetry=&Try again
RetryCancelCancel=&Cancel
StatusRestartingApplications=Reopening closed apps
ShutdownBlockReasonInstallingApp=Installing Food Optimizer.
ShutdownBlockReasonUninstallingApp=Uninstalling Food Optimizer.
UninstallOpenError=Food Optimizer could not be uninstalled.%nEmail sohum.patnaik@fsi.org.
; Settings > Apps: an all-users install beside the same account's per-user
; copy is named "Food Optimizer (all users)" (Inno's own format, kept here).
UninstallDisplayNameMark=%1 (%2)
UninstallDisplayNameMarkAllUsers=all users

[CustomMessages]
DesktopShortcut=Add a desktop shortcut
OpenApp=Open Food Optimizer
OpenStartHere=Open Start Here
PrewarmStatus=Loading the app's components
PrewarmFailed=Food Optimizer could not load its components.%n%nAsk IT to allow Food Optimizer, or email sohum.patnaik@fsi.org.
InstalledByIT=Installed by your IT. Ask IT to update.
SystemNeedsAllUsers=Installing as SYSTEM needs /ALLUSERS.
FolderTooLong=This install folder path is too long. Use at most 80 characters.
NeedsSpace=Needs %1 GB free on this PC.
FolderNotOurs=This folder holds other files. Use an empty folder.
