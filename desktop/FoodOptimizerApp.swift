// Food Optimizer — native window wrapper.
// Compiled by build_dmg.sh into Contents/MacOS/FoodOptimizer (the bundle
// executable). Runs launcher.sh for environment setup and the Streamlit
// server, shows a status page meanwhile, then hosts the UI in a WKWebView.
// Quitting (Cmd-Q or closing the window) terminates the launcher, which
// shuts the server down.
import Cocoa
import WebKit

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    var window: NSWindow!
    var webView: WKWebView!
    var launcher: Process?
    var pollTimer: Timer?
    var watchTimer: Timer?
    var loaded = false
    var healthStrikes = 0
    var downloadDestinations: [ObjectIdentifier: URL] = [:]
    var pollTicks = 0
    var deferDeadline: Int?   // pollTicks limit after our launcher deferred to another launch
    // When this launch's launcher process was started. A server.port file can
    // survive a previous, force-quit launch for a few seconds (launcher.sh
    // removes it before writing its own), so only a port file written at or
    // after this moment belongs to the launcher we are actually watching.
    var launchStartedAt: Date?
    // The tick the CURRENT launch's server.port file first appeared, once the
    // launcher is done with its own setup steps and Streamlit is expected to
    // answer any moment. nil while still waiting for that file (or while a
    // stale one from a previous launch is all that is on disk — see
    // portFileIsFromThisLaunch()). Bounds the otherwise unbounded wait for
    // the first health check to succeed, so a Streamlit that started but
    // never answers still fails kindly instead of leaving the last step
    // spinning forever.
    var awaitingHealthSince: Int?
    // Set once the 180s bound above has been reached and the give-up page
    // has been painted, so poll() does not repaint it every tick — but
    // polling itself continues, so a late 200 still recovers into the web
    // view instead of being stranded on a page that told the user to quit.
    var timedOutWaitingForHealth = false
    var lastStatusLine: String?   // raw "<text>|<percent>" last read from status.txt
    var lastStepText: String?     // its text half, so a re-render keeps the step
    var lastProgress: Int?        // its percent half, so a re-render keeps the bar
    // ---- the starting page ----
    // One page covers every launch, and it lists what a launch actually does:
    // the components are downloaded (first run or upgrade only), then loaded,
    // then the projects are opened. The launcher's status lines say which of
    // the three is running; this wrapper's own health poll says when the last
    // one ends. Nothing here is estimated — the page shows the step, an
    // honest range, and how long the step has actually been running.
    enum LaunchStep: Int { case download = 0, load = 1, open = 2 }
    var stepsPageShowing = false    // the steps page is the page on screen
    var stepsHasDownload = false    // ...and it was built with the download row
    var activeStep: LaunchStep = .load
    var activeStepStarted = Date()  // when it became active, for the counter
    // True once a launcher line — not merely the window opening — put the
    // active step on screen. The counter times the work, so a warm launch
    // that shows "Loading" from its first frame still starts counting when
    // the launcher says the import has begun.
    var activeStepConfirmed = false
    // A slow step stays acknowledged for the rest of this launch.
    var activeStepStalled = false
    var stepDetail = ""             // the launcher's own line, under the active step
    // The six-minute "Still setting up" page, which is not a steps page: it
    // must not be rebuilt over by the next status line.
    var stalledPageShowing = false
    var retryToken = 0            // cancels a pending retry when Try again is clicked again
    var pendingOldLauncher: Process?   // the launcher a retry is waiting on; a second click must keep waiting on it
    var setupIsUpgrade = false    // the marker existed when this launch began
    // Bumped per launch so a terminated launcher's handler can be ignored.
    var launchGeneration = 0

    // Lines the launcher publishes while it is doing setup work. "Starting the
    // app" is deliberately absent: it arrives on EVERY launch, warm ones
    // included, and must never turn an ordinary opening page into a setup page
    // (it may only update that page's step line).
    let setupStepPrefixes = ["Downloading Python", "Creating environment",
                             "Updating components", "Installing components",
                             "Finishing setup"]

    // The three steps, as the user reads them, and the lines under them. The
    // load step's label doubles as the prefix of the launcher's own line for
    // it, so the two stay consistent. Avoid duration promises: startup time
    // depends on the computer and whether components need downloading.
    let downloadStepLabel = "Downloading the app's components"
    let loadStepLabel = "Loading the app's components"
    let openStepLabel = "Opening your projects"
    let usualWaitLine = "Preparing the app. Please keep this window open."
    let stillLoadingLine = "Still loading. Please keep this window open."
    // The one line that says what the app is for, so the wait has something
    // to read that is not about waiting.
    let nextUpLine = "Define your ingredients, process settings, and measurements. Then generate your first set of formulations."

    let appVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"]
        as? String ?? ""

    let supportDir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/FoodOptimizer")
    let dataDir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("FoodOptimizer")

    // Runtime files the launcher and this wrapper share.
    func supportPath(_ name: String) -> String {
        supportDir.appendingPathComponent(name).path
    }
    // No marker => the launcher is about to do the long one-time setup.
    var isFirstRun: Bool {
        !FileManager.default.fileExists(atPath: supportPath("setup_complete"))
    }
    // Quote arbitrary text as a JavaScript string literal (status text comes
    // from a file, so it must never be able to break out of the script).
    func jsString(_ s: String) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: [s]),
              let arr = String(data: data, encoding: .utf8), arr.count >= 2
        else { return "\"\"" }
        return String(arr.dropFirst().dropLast())   // ["x"] -> "x"
    }

    func applicationDidFinishLaunching(_ note: Notification) {
        buildMenus()

        webView = WKWebView(frame: .zero)
        webView.uiDelegate = self
        webView.navigationDelegate = self
        webView.pageZoom = CGFloat(UserDefaults.standard.double(forKey: "pageZoom"))
        if webView.pageZoom <= 0.25 { webView.pageZoom = 1.0 }

        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1280, height: 860),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered, defer: false)
        window.title = "Food Optimizer"
        window.minSize = NSSize(width: 700, height: 500)
        window.setFrameAutosaveName("FoodOptimizerMain")
        window.contentView = webView
        window.delegate = self
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)

        if offerMoveToApplications() { return }   // relaunching from /Applications

        showInitialStatus()
        startLauncher()
        // Half-second ticks so the health check that decides when to swap in
        // the web view (see poll()) notices the first HTTP 200 promptly —
        // the status screen must never sit on a stale page once the app is
        // actually ready. Every tick-counted threshold below is doubled to
        // match, so the real-world timing they describe is unchanged.
        pollTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) {
            [weak self] _ in self?.poll()
        }
    }

    // ---------- menus ----------

    func buildMenus() {
        let main = NSMenu()

        let appItem = NSMenuItem(); main.addItem(appItem)
        let appMenu = NSMenu()
        appMenu.addItem(NSMenuItem(
            title: "About Food Optimizer",
            action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)),
            keyEquivalent: ""))
        appMenu.addItem(.separator())
        appMenu.addItem(NSMenuItem(
            title: "Quit Food Optimizer",
            action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
        appItem.submenu = appMenu

        let fileItem = NSMenuItem(); main.addItem(fileItem)
        let fileMenu = NSMenu(title: "File")
        fileMenu.addItem(NSMenuItem(
            title: "Open Projects Folder",
            action: #selector(openProjectsFolder), keyEquivalent: "o"))
        fileMenu.addItem(.separator())
        fileMenu.addItem(NSMenuItem(
            title: "Close Window",
            action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w"))
        fileItem.submenu = fileMenu

        let editItem = NSMenuItem(); main.addItem(editItem)
        let editMenu = NSMenu(title: "Edit")
        editMenu.addItem(NSMenuItem(title: "Undo",
            action: Selector(("undo:")), keyEquivalent: "z"))
        editMenu.addItem(NSMenuItem(title: "Redo",
            action: Selector(("redo:")), keyEquivalent: "Z"))
        editMenu.addItem(.separator())
        editMenu.addItem(NSMenuItem(title: "Cut",
            action: #selector(NSText.cut(_:)), keyEquivalent: "x"))
        editMenu.addItem(NSMenuItem(title: "Copy",
            action: #selector(NSText.copy(_:)), keyEquivalent: "c"))
        editMenu.addItem(NSMenuItem(title: "Paste",
            action: #selector(NSText.paste(_:)), keyEquivalent: "v"))
        editMenu.addItem(NSMenuItem(title: "Select All",
            action: #selector(NSText.selectAll(_:)), keyEquivalent: "a"))
        editItem.submenu = editMenu

        let viewItem = NSMenuItem(); main.addItem(viewItem)
        let viewMenu = NSMenu(title: "View")
        viewMenu.addItem(NSMenuItem(title: "Bigger Text",
            action: #selector(zoomIn), keyEquivalent: "+"))
        viewMenu.addItem(NSMenuItem(title: "Smaller Text",
            action: #selector(zoomOut), keyEquivalent: "-"))
        viewMenu.addItem(NSMenuItem(title: "Actual Size",
            action: #selector(zoomReset), keyEquivalent: "0"))
        viewItem.submenu = viewMenu

        let windowItem = NSMenuItem(); main.addItem(windowItem)
        let windowMenu = NSMenu(title: "Window")
        windowMenu.addItem(NSMenuItem(title: "Minimize",
            action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m"))
        windowItem.submenu = windowMenu
        NSApp.windowsMenu = windowMenu

        let helpItem = NSMenuItem(); main.addItem(helpItem)
        let helpMenu = NSMenu(title: "Help")
        helpMenu.addItem(NSMenuItem(title: "Get Help",
            action: #selector(getHelp), keyEquivalent: ""))
        helpMenu.addItem(NSMenuItem(title: "Show Log File",
            action: #selector(showLogFile), keyEquivalent: ""))
        helpItem.submenu = helpMenu
        NSApp.helpMenu = helpMenu

        NSApp.mainMenu = main
    }

    @objc func openProjectsFolder() {
        try? FileManager.default.createDirectory(
            at: dataDir, withIntermediateDirectories: true)
        NSWorkspace.shared.open(dataDir)
    }
    @objc func getHelp() {
        let alert = NSAlert()
        alert.messageText = "Food Optimizer support"
        alert.informativeText =
            "Write to us at https://github.com/thomasat/food_opt/issues. "
            + "Describe the problem and what you expected. Do not attach project "
            + "files, saved copies or formulations, because that page is public. "
            + "Attaching the app's log file helps — Help › Show Log File "
            + "finds it for you."
        alert.runModal()
    }
    @objc func showLogFile() {
        let log = supportDir.appendingPathComponent("launcher.log")
        if FileManager.default.fileExists(atPath: log.path) {
            NSWorkspace.shared.activateFileViewerSelecting([log])
        } else {
            let alert = NSAlert()
            alert.messageText = "There is no log file yet"
            alert.informativeText =
                "The log is created when the app starts up. Quit and open "
                + "Food Optimizer again, then try this menu item once more."
            alert.runModal()
        }
    }
    @objc func zoomIn()    { setZoom(webView.pageZoom + 0.1) }
    @objc func zoomOut()   { setZoom(webView.pageZoom - 0.1) }
    @objc func zoomReset() { setZoom(1.0) }
    func setZoom(_ z: CGFloat) {
        let clamped = min(max(z, 0.5), 2.5)
        webView.pageZoom = clamped
        UserDefaults.standard.set(Double(clamped), forKey: "pageZoom")
    }

    // ---------- first-run placement ----------

    // If running from the mounted dmg (or Gatekeeper's translocated copy of
    // it), offer to copy to /Applications so the app doesn't "vanish" when
    // the disk image is ejected. Returns true when relaunching from the copy.
    func offerMoveToApplications() -> Bool {
        let path = Bundle.main.bundlePath
        guard path.hasPrefix("/Volumes/") || path.contains("/AppTranslocation/")
        else { return false }
        let alert = NSAlert()
        alert.messageText = "Copy Food Optimizer to your Applications folder?"
        alert.informativeText =
            "The app is running from the downloaded disk image. Copying it to "
            + "Applications keeps it on your Mac so you can open it anytime."
        alert.addButton(withTitle: "Copy to Applications")
        alert.addButton(withTitle: "Not Now")
        guard alert.runModal() == .alertFirstButtonReturn else { return false }

        let dest = "/Applications/Food Optimizer.app"
        try? FileManager.default.removeItem(atPath: dest)
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        p.arguments = [path, dest]
        do { try p.run() } catch { return false }
        p.waitUntilExit()
        guard p.terminationStatus == 0 else {
            showStatus("The copy did not work",
                       "Please drag Food Optimizer onto the Applications folder "
                       + "in the installer window, then open it from Applications.")
            return false
        }
        // Relaunch via a detached helper. Opening the copy directly from this
        // still-running instance can make Launch Services just activate THIS
        // instance (same bundle id) instead of launching the copy — the app
        // would quit and never come back. The helper waits for this process
        // to exit (bounded at ~10s), then opens the fresh copy.
        let pid = ProcessInfo.processInfo.processIdentifier
        let helper = Process()
        helper.executableURL = URL(fileURLWithPath: "/bin/sh")
        helper.arguments = ["-c",
            "i=0; while /bin/kill -0 \(pid) 2>/dev/null && [ $i -lt 50 ]; do "
            + "sleep 0.2; i=$((i+1)); done; /usr/bin/open \"\(dest)\""]
        do { try helper.run() } catch {
            NSWorkspace.shared.open(URL(fileURLWithPath: dest))
        }
        NSApp.terminate(nil)
        return true
    }

    // ---------- status pages ----------

    // The page shown while the launcher works, from the first moment to the
    // web view. Its steps are set here: the download row exists only for a
    // launch that has one to do.
    func showInitialStatus() {
        // An upgrade still has the marker at this point (the launcher deletes
        // it only after announcing itself), so remember which job this is;
        // the first setup line adds the download row to a launch that looked
        // warm.
        setupIsUpgrade = !isFirstRun
        stepsHasDownload = isFirstRun
        stepsPageShowing = false
        activeStep = isFirstRun ? .download : .load
        activeStepStarted = Date()
        activeStepConfirmed = false
        activeStepStalled = false
        stepDetail = ""
        showStepsPage()
        updateStepsPage()
    }

    // The sentence a download-carrying launch reads under its step, kept
    // word for word as it has always been.
    var setupBody: String {
        setupIsUpgrade
            ? "Food Optimizer is downloading an update. "
              + "This usually takes under a minute; on a slow network, a few minutes. Leave this window open."
            : "The first time it opens, Food Optimizer downloads about 1 GB. "
              + "This usually takes under a minute; on a slow network, a few minutes. Leave this window open."
    }

    var stepsPageTitle: String {
        guard activeStep == .download else { return "Starting Food Optimizer" }
        return setupIsUpgrade ? "Updating Food Optimizer" : "Setting up Food Optimizer"
    }

    var stepLabels: [String] {
        stepsHasDownload ? [downloadStepLabel, loadStepLabel, openStepLabel]
                         : [loadStepLabel, openStepLabel]
    }

    // Which row a step is drawn in: without the download row every step
    // moves up one.
    func rowOf(_ step: LaunchStep) -> Int {
        stepsHasDownload ? step.rawValue : step.rawValue - 1
    }

    // Make `step` the one being worked on. A step that is genuinely new
    // restarts the counter, and so does the first launcher line to confirm
    // the step the window opened on; the same step reported again (the
    // launcher publishes its download line once a second) does not.
    func setActiveStep(_ step: LaunchStep, detail: String? = nil) {
        if let d = detail { stepDetail = d }
        if step == .download && !stepsHasDownload {
            stepsHasDownload = true      // a launch that turned out to be an upgrade
            stepsPageShowing = false     // the row set changed: rebuild
        }
        if step != activeStep || !activeStepConfirmed {
            activeStep = step
            activeStepStarted = Date()
        }
        activeStepConfirmed = true
        if !stepsPageShowing { showStepsPage() }
        updateStepsPage()
    }

    // How long the active step has been running, and the line under it. Both
    // are read by the paint and by the tick, so the page says the same thing
    // whichever drew it last.
    var elapsedOnActiveStep: Int {
        max(0, Int(Date().timeIntervalSince(activeStepStarted)))
    }

    // Past the range we promised: say so, and keep saying it for the rest of
    // the launch. Only the two second-scale steps make that promise — the
    // download has its own sentence and its own bar.
    func latchStalled() {
        if elapsedOnActiveStep >= 30 && activeStep != .download {
            activeStepStalled = true
        }
    }

    var stepsNote: String {
        if activeStep == .download { return setupBody }
        return activeStepStalled ? stillLoadingLine : usualWaitLine
    }

    // Text bound for the page's HTML rather than for a JS string. The detail
    // line is the launcher's own, so it is escaped rather than trusted.
    func htmlText(_ s: String) -> String {
        s.replacingOccurrences(of: "&", with: "&amp;")
         .replacingOccurrences(of: "<", with: "&lt;")
         .replacingOccurrences(of: ">", with: "&gt;")
    }

    // Draw the page in the state it is in. The JS that maintains it cannot
    // run until WebKit has loaded this string, so anything left blank here
    // would BE blank for the first frames — including, on a first run, the
    // sentence about the download. updateStepsPage() only maintains what is
    // painted here.
    func showStepsPage() {
        latchStalled()
        let active = rowOf(activeStep)
        var rowsHTML = ""
        for (i, label) in stepLabels.enumerated() {
            let done = i < active, running = i == active
            rowsHTML += """
            <li id="row\(i)" style="display:flex;align-items:center;gap:10px;
                                    margin:0 0 12px;opacity:\(done || running ? "1" : ".45")">
              <span style="width:16px;height:16px;display:inline-flex;
                           align-items:center;justify-content:center;flex:none">
                <span id="dot\(i)" style="color:#9aa39b;
                      display:\(done || running ? "none" : "inline")">•</span>
                <span id="spin\(i)" style="display:\(running ? "inline-block" : "none");
                     width:13px;height:13px;
                     border:2px solid #cdd6ce;border-top-color:#2E6E4E;
                     border-radius:50%;animation:spin 1s linear infinite"></span>
                <span id="tick\(i)" style="display:\(done ? "inline" : "none");
                      color:#2E6E4E">✓</span>
              </span>
              <span>\(label)</span>
            </li>
            """
        }
        // The measured bar belongs to the download alone — it is the only
        // step with a real measure. It is hidden once that step is done.
        let barHTML = stepsHasDownload ? """
            <div id="barwrap" style="max-width:320px;height:6px;margin:0 auto 16px;
                                     background:#e2e6e1;border-radius:3px;overflow:hidden;
                                     display:\(activeStep == .download ? "block" : "none")">
              <div id="bar" style="width:\(lastProgress ?? 0)%;height:100%;
                                   background:#2E6E4E;border-radius:3px;
                                   transition:width .4s ease"></div>
            </div>
            """ : ""
        let html = """
        <html><head><meta charset="utf-8">
        <style>@keyframes spin{to{transform:rotate(360deg)}}</style></head>
        <body style="font-family:-apple-system,sans-serif;background:#f7f6f2;color:#2d3a2e;
                     display:flex;align-items:center;justify-content:center;height:96vh;margin:0">
          <div style="text-align:center;max-width:460px">
            <h1 id="title" style="font-weight:600">\(stepsPageTitle)</h1>
            <ul id="steps" style="list-style:none;padding:0;
                                  margin:26px auto 18px;display:inline-block;
                                  text-align:left;font-size:15px">\(rowsHTML)</ul>
            \(barHTML)
            <p id="detail" style="font-size:13px;color:#7c867e;margin:0 0 8px">
              \(htmlText(activeStep == .download ? stepDetail : ""))</p>
            <p id="note" style="font-size:15px;line-height:1.5;color:#556;margin:0">
              \(htmlText(stepsNote))</p>
            <p id="elapsed" style="font-size:13px;color:#9aa39b;margin:10px 0 0">
              \(elapsedOnActiveStep) s</p>
            <p style="margin-top:34px;font-size:13px;color:#7c867e">\(nextUpLine)</p>
            <p style="margin-top:26px;font-size:12px;color:#9aa39b">
              \(appVersion.isEmpty ? "Food Optimizer" : "Food Optimizer " + appVersion)</p>
          </div>
        </body></html>
        """
        webView.loadHTMLString(html, baseURL: nil)
        stepsPageShowing = true
        stalledPageShowing = false
        // Forget the last line we pushed: the DOM is new, so the next tick
        // must re-apply the current step even if the launcher has not moved on.
        lastStatusLine = nil
    }

    // One tick of the page: which rows are done, what the note says, and how
    // long the active step has been running. Called from every poll, so the
    // counter moves even when the launcher has nothing new to say.
    func updateStepsPage() {
        guard stepsPageShowing else { return }
        latchStalled()
        let active = rowOf(activeStep)
        var js = ""
        for i in 0..<stepLabels.count {
            let done = i < active, running = i == active
            js += "(function(){var r=document.getElementById('row\(i)');if(!r)return;"
                + "r.style.opacity='\(done || running ? "1" : ".45")';"
                + "document.getElementById('dot\(i)').style.display="
                + "'\(done || running ? "none" : "inline")';"
                + "document.getElementById('spin\(i)').style.display="
                + "'\(running ? "inline-block" : "none")';"
                + "document.getElementById('tick\(i)').style.display="
                + "'\(done ? "inline" : "none")';})();"
        }
        js += "var t=document.getElementById('title');"
            + "if(t){t.textContent=\(jsString(stepsPageTitle))}"
        js += "var n=document.getElementById('note');"
            + "if(n){n.textContent=\(jsString(stepsNote))}"
        js += "var d=document.getElementById('detail');"
            + "if(d){d.textContent=\(jsString(activeStep == .download ? stepDetail : ""))}"
        js += "var e=document.getElementById('elapsed');"
            + "if(e){e.textContent=\(jsString("\(elapsedOnActiveStep) s"))}"
        if stepsHasDownload {
            js += "var w=document.getElementById('barwrap');"
                + "if(w){w.style.display='\(activeStep == .download ? "block" : "none")'}"
            if let pct = lastProgress {
                js += "var b=document.getElementById('bar');"
                    + "if(b){b.style.width='\(pct)%'}"
            }
        }
        webView.evaluateJavaScript(js, completionHandler: nil)
    }

    // The generic "we don't know why, but it never came up" failure. One
    // page for every poll-driven give-up — setup finished but Streamlit
    // never answered, another launch's server never appeared, the plain
    // default when the launcher itself exits unexpectedly — so the three
    // near-identical pages that used to say this each their own way can't
    // drift apart. `detail`, when given, is a short sentence naming what was
    // different about this particular give-up, said before the shared advice.
    func showCouldNotStartStatus(detail: String? = nil) {
        let body = (detail.map { $0 + " " } ?? "")
            + "Click Try again. If the problem persists, contact "
            + "the Food Intelligence Lab and attach the file from Help › Show "
            + "Log File."
        showStatus("The app could not start", body, retry: true)
    }

    // `indeterminate` draws a looping bar for work with no measurable
    // progress; it is what marks a page as the opening page.
    func showStatus(_ title: String, _ body: String, spinner: Bool = false,
                    step: String? = nil, progress: Int? = nil,
                    indeterminate: Bool = false, retry: Bool = false) {
        let spinnerHTML = spinner ? """
            <div style="margin:24px auto;width:28px;height:28px;border:3px solid #cdd6ce;
                        border-top-color:#2E6E4E;border-radius:50%;
                        animation:spin 1s linear infinite"></div>
            <style>@keyframes spin{to{transform:rotate(360deg)}}</style>
            """ : ""
        let stepHTML = step.map {
            """
            <p id="step" style="font-size:14px;color:#2E6E4E;font-weight:600;
                                margin:0 0 14px">\($0)</p>
            """
        } ?? ""
        // A measured bar where there is a percentage to show (poll() widens
        // #bar as the launcher reports progress), otherwise a looping one when
        // the caller asked for it. Same track and same green either way, so
        // the two pages read as one family.
        let indeterminateHTML = """
            <div style="max-width:320px;height:4px;margin:0 auto 20px;
                        background:#e2e6e1;border-radius:2px;overflow:hidden">
              <div style="width:40%;height:100%;background:#2E6E4E;border-radius:2px;
                          animation:slide 1.5s ease-in-out infinite"></div>
            </div>
            <style>@keyframes slide{0%{transform:translateX(-105%)}
                                    100%{transform:translateX(255%)}}</style>
            """
        let barHTML: String
        if progress != nil {
            barHTML = """
            <div id="barwrap" style="max-width:320px;height:6px;margin:0 auto 20px;
                                     background:#e2e6e1;border-radius:3px;overflow:hidden">
              <div id="bar" style="width:\(max(0, min(progress ?? 0, 100)))%;height:100%;
                                   background:#2E6E4E;transition:width .4s ease"></div>
            </div>
            """
        } else {
            barHTML = indeterminate ? indeterminateHTML : ""
        }
        let retryHTML = retry ? """
            <p style="margin-top:28px"><a href="foodopt://retry"
               style="display:inline-block;padding:10px 22px;background:#2E6E4E;color:#fff;
                      border-radius:6px;text-decoration:none;font-weight:600">Try again</a></p>
            """ : ""
        let html = """
        <html><head><meta charset="utf-8"></head>
        <body style="font-family:-apple-system,sans-serif;background:#f7f6f2;color:#2d3a2e;
                     display:flex;align-items:center;justify-content:center;height:96vh;margin:0">
          <div style="text-align:center;max-width:460px">
            <h1 style="font-weight:600">\(title)</h1>
            \(spinnerHTML)
            \(stepHTML)
            \(barHTML)
            <p style="font-size:15px;line-height:1.5;color:#556">\(body)</p>
            \(retryHTML)
            <p style="margin-top:36px;font-size:12px;color:#9aa39b">
              \(appVersion.isEmpty ? "Food Optimizer" : "Food Optimizer " + appVersion)</p>
          </div>
        </body></html>
        """
        webView.loadHTMLString(html, baseURL: nil)
        // Whatever this page is, it is not the steps page: the next status
        // line must update it in place, not rebuild the steps over it. Nor is
        // it the six-minute page — poll() sets that flag itself, right after
        // the one call that paints it.
        stepsPageShowing = false
        stalledPageShowing = false
        // Forget the last line we pushed: the DOM is new, so the next tick
        // must re-apply the current step even if the launcher has not moved on.
        lastStatusLine = nil
    }

    // ---------- launcher lifecycle ----------

    func startLauncher() {
        launchGeneration += 1
        let generation = launchGeneration
        // Anything server.port carries from before this moment belongs to a
        // launch we are not watching (a previous, possibly force-quit one).
        launchStartedAt = Date()
        let p = Process()
        p.executableURL = URL(fileURLWithPath:
            Bundle.main.bundlePath + "/Contents/Resources/launcher.sh")
        // Detach stdio: the launcher logs to its own file, and piping through
        // this app would SIGPIPE the launcher's shutdown when we exit first.
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        p.terminationHandler = { [weak self] proc in
            DispatchQueue.main.async {
                // A launcher we deliberately replaced must never paint an
                // error over the launch that succeeded it.
                guard let self, generation == self.launchGeneration else { return }
                self.launcherEnded(code: proc.terminationStatus)
            }
        }
        do { try p.run() } catch {
            showStatus("The app could not start",
                       "Please contact us and mention: launcher failed to run.")
            return
        }
        launcher = p
    }

    // Every failure page offers this instead of "quit and open the app
    // again": start the whole launch over, in place.
    func retryLaunch() {
        loaded = false
        pollTicks = 0
        deferDeadline = nil
        awaitingHealthSince = nil
        timedOutWaitingForHealth = false
        lastStatusLine = nil
        lastStepText = nil
        lastProgress = nil
        healthStrikes = 0
        watchTimer?.invalidate()
        watchTimer = nil
        // Stay stopped until the replacement is actually running: a leftover
        // port file from the launcher we are killing would otherwise look
        // like a healthy server for a tick or two.
        pollTimer?.invalidate()
        pollTimer = nil
        // A failed setup leaves its last line behind — the launcher clears
        // status.txt only once the NEXT launch owns the lock — so drop it
        // here, or the restarted poll paints that stale percentage first.
        try? FileManager.default.removeItem(atPath: supportPath("status.txt"))
        try? FileManager.default.removeItem(atPath: supportPath("status.txt.tmp"))
        // Detach the handler BEFORE terminating: a launcher killed mid-setup
        // exits by signal, and its handler would otherwise report that as a
        // failure of the launch we are about to start.
        // A second click during the wait must keep waiting on the same old
        // process instead of seeing `launcher == nil` and starting at once.
        let old = launcher ?? pendingOldLauncher
        pendingOldLauncher = old
        if let l = old, l.isRunning {
            l.terminationHandler = nil
            l.terminate()
        }
        launcher = nil
        showInitialStatus()   // instant feedback: the click must feel like one
        retryToken += 1
        startWhenOldLauncherExits(old, token: retryToken,
                                  deadline: Date().addingTimeInterval(5.0))
    }

    // The launcher we just terminated can hold launch.lock for up to a second
    // (bash runs its trap only after the current `sleep` returns). Starting on
    // top of it makes the new launcher take the "another launch in progress"
    // exit-0 path, and the window then waits in silence for the deferral
    // deadline. So wait for the old process to actually go.
    func startWhenOldLauncherExits(_ old: Process?, token: Int, deadline: Date) {
        guard token == retryToken else { return }   // a newer click supersedes us
        if let l = old, l.isRunning, Date() < deadline {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) { [weak self] in
                self?.startWhenOldLauncherExits(old, token: token, deadline: deadline)
            }
            return
        }
        pendingOldLauncher = nil
        startLauncher()
        pollTimer?.invalidate()
        pollTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) {
            [weak self] _ in self?.poll()
        }
    }

    func launcherEnded(code: Int32) {
        guard !loaded else { return }   // normal shutdown later is fine
        // Exit 0 is not a failure: it means another launcher instance is
        // bringing the server up (e.g. this one deferred to it). Keep polling
        // — the health check will load the UI as soon as the server answers.
        // But never wait forever: if the launch we deferred to never produces
        // a server (e.g. a stale launch lock that wrongly looks alive), fail
        // with guidance instead of spinning indefinitely.
        if code == 0 {
            deferDeadline = pollTicks + 1800   // ~15 minutes, at 500ms ticks
            return
        }
        pollTimer?.invalidate()
        switch code {
        case 2:
            showStatus("This Mac is not supported",
                       "Food Optimizer needs a Mac with an Apple chip (2020 or "
                       + "newer) running macOS 13 or later. Please contact us for help.")
        case 3:
            showStatus("Setup needs the internet, just this once",
                       "The first time it opens, Food Optimizer downloads about "
                       + "1 GB. Please connect to the internet, "
                       + "then click Try again. After that, no internet is "
                       + "needed. If you are connected but this message keeps "
                       + "coming back (some office networks block downloads), "
                       + "reach out to the Food Intelligence Lab.",
                       retry: true)
        case 4:
            showStatus("Not enough free space to set up",
                       "Food Optimizer needs about 6 GB of free space the first "
                       + "time it opens. Please free up some space, then click "
                       + "Try again.",
                       retry: true)
        default:
            showCouldNotStartStatus()
        }
    }

    // A clean numeric port from the (possibly mid-write) port file.
    func readServerPort() -> String? {
        let portFile = supportDir.appendingPathComponent("server.port")
        guard let contents = try? String(contentsOf: portFile, encoding: .utf8),
              let token = contents.split(separator: " ").first.map(String.init),
              !token.isEmpty,
              token.allSatisfy({ $0.isNumber }) else { return nil }
        return token
    }

    // launcher.sh removes an orphaned server.port from a previous, possibly
    // force-quit launch, but that can take up to ~5s (it waits for the old
    // server to die before deleting the file). Until then, a freshly started
    // launcher and a leftover file from the last one are indistinguishable by
    // content alone — only the file's own age says which launch it belongs
    // to.
    func portFileIsFromThisLaunch() -> Bool {
        guard let started = launchStartedAt else { return true }
        let portFile = supportDir.appendingPathComponent("server.port")
        guard let values = try? portFile.resourceValues(forKeys: [.contentModificationDateKey]),
              let modified = values.contentModificationDate
        else { return false }
        // A 2s tolerance: a small backward clock step, or a volume whose
        // mtimes are coarser than our own clock, must never reject a file
        // this very launch just wrote — that would starve poll() of a port
        // to watch forever, not just for the few seconds the check exists to
        // cover.
        return modified >= started.addingTimeInterval(-2)
    }

    // The previous launch's last line survives until the NEXT launch owns the
    // lock and clears it — up to ~10 s in. Read as this launch's, it would
    // drive the steps backwards ("Finishing setup" after "Loading") and
    // could add a download row to a launch with nothing to download. Same 2 s
    // tolerance as portFileIsFromThisLaunch(), and for the same reasons.
    func statusFileIsFromThisLaunch() -> Bool {
        guard let started = launchStartedAt else { return true }
        let statusFile = supportDir.appendingPathComponent("status.txt")
        guard let values = try? statusFile.resourceValues(
                  forKeys: [.contentModificationDateKey]),
              let modified = values.contentModificationDate
        else { return false }
        return modified >= started.addingTimeInterval(-2)
    }

    // The launcher publishes "<text>|<percent>" (percent may be empty) once
    // a second. Its text says which step is running, and the page is updated
    // in place rather than reloaded, so the spinner, the bar and the elapsed
    // counter animate instead of restarting.
    func readStatusFile() {
        // Once the port file exists there are no more launcher lines coming —
        // poll() has already moved the page on to its last step — so a stray
        // leftover status.txt line must not paint over it.
        guard !loaded, awaitingHealthSince == nil, statusFileIsFromThisLaunch(),
              let raw = try? String(contentsOfFile: supportPath("status.txt"),
                                    encoding: .utf8)
        else { return }
        let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !line.isEmpty, line != lastStatusLine else { return }
        lastStatusLine = line
        var text = line
        var percent = ""
        if let sep = line.lastIndex(of: "|") {
            text = String(line[line.startIndex..<sep])
                .trimmingCharacters(in: .whitespaces)
            percent = String(line[line.index(after: sep)...])
                .trimmingCharacters(in: .whitespaces)
        }
        guard !text.isEmpty else { return }
        lastStepText = text
        var pct: Int? = nil
        if let n = Int(percent) {
            pct = max(0, min(n, 100))
            lastProgress = pct
        }
        // Which of the three steps this line is about. A percent or one of
        // the known setup prefixes is the download (an ordinary launch
        // publishes neither); the load step's line starts with its own label;
        // anything else — "Starting the app" — is the server coming up,
        // which is where "Opening your projects" begins.
        let step: LaunchStep
        if pct != nil || setupStepPrefixes.contains(where: { text.hasPrefix($0) }) {
            step = .download
        } else if text.hasPrefix(loadStepLabel) {
            step = .load
        } else {
            step = .open
        }
        // The six-minute page is not a steps page, but it carries a step line
        // and a bar of its own: keep those moving rather than painting the
        // steps back over the explanation it exists to give.
        if stalledPageShowing {
            var js = "var s=document.getElementById('step');"
                + "if(s){s.textContent=\(jsString(text))}"
            if let pct = lastProgress {
                js += ";var b=document.getElementById('bar');"
                    + "if(b){b.style.width='\(pct)%'}"
            }
            webView.evaluateJavaScript(js, completionHandler: nil)
            return
        }
        setActiveStep(step, detail: text)
    }

    func poll() {
        // Keep the setup message honest on slow connections.
        pollTicks += 1
        readStatusFile()
        updateStepsPage()   // the counter ticks whether or not anything was said
        // Only while still waiting on setup itself: once a port file exists
        // (armed or not — see below) this message's own copy is wrong, and
        // painting it here would silently cover the steps page's last step
        // or, worse, the give-up page — which, unlike this one, has a Try
        // again link — stranding the user with no way forward.
        if pollTicks == 720, awaitingHealthSince == nil, !timedOutWaitingForHealth {
            showStatus("Still setting up",
                       "The downloads are taking a while — slow connections "
                       + "can take longer than usual. Leave this window open; "
                       + "the app will appear as soon as it's ready.",
                       spinner: true, step: lastStepText ?? "Still working",
                       progress: lastProgress, indeterminate: lastProgress == nil)
            stalledPageShowing = true
        }
        if let deadline = deferDeadline, pollTicks >= deadline {
            pollTimer?.invalidate()
            showCouldNotStartStatus(detail: "Another copy of the app seemed "
                                    + "to be starting, but it never finished.")
            return
        }
        guard let port = readServerPort(), portFileIsFromThisLaunch(),
              let health = URL(string: "http://127.0.0.1:\(port)/_stcore/health")
        else {
            // Nothing from THIS launch to wait on right now — either no port
            // file yet, or the one on disk is a leftover launcher.sh has not
            // removed yet. Disarm rather than let a since-vanished file's old
            // tick count linger: readStatusFile() (above) is already showing
            // setup progress again, and we re-arm cleanly once a fresh file
            // appears.
            awaitingHealthSince = nil
            return
        }
        // The launcher is done and Streamlit should answer any moment, which
        // is the page's last step. Never a blank window: the status page
        // stays up until the health check below actually succeeds.
        if awaitingHealthSince == nil {
            awaitingHealthSince = pollTicks
            timedOutWaitingForHealth = false
            // The server is up; what is left is Streamlit answering, which is
            // the last step. (It is usually already active: the launcher
            // publishes "Starting the app" as it spawns the server.)
            if !stalledPageShowing { setActiveStep(.open) }
        } else if let since = awaitingHealthSince, !timedOutWaitingForHealth,
                  pollTicks - since >= 360 {   // matches launcher.sh's own 180s patience
            // Say so, but keep polling: launcher.sh itself gives up around
            // now (~180s), but a late 200 must still swap in the web view
            // rather than strand the user on a page that told them to quit.
            timedOutWaitingForHealth = true
            showCouldNotStartStatus()
        }
        URLSession.shared.dataTask(with: health) { [weak self] _, resp, _ in
            guard let self, !self.loaded,
                  let http = resp as? HTTPURLResponse, http.statusCode == 200
            else { return }
            DispatchQueue.main.async {
                guard !self.loaded, let ui = URL(string: "http://localhost:\(port)")
                else { return }
                self.loaded = true
                self.pollTimer?.invalidate()
                self.webView.load(URLRequest(url: ui))
                self.startHealthWatch(port: port)
            }
        }.resume()
    }

    // After the app is up, notice if the server ever dies and say so kindly.
    func startHealthWatch(port: String) {
        guard let health = URL(string: "http://127.0.0.1:\(port)/_stcore/health")
        else { return }
        watchTimer = Timer.scheduledTimer(withTimeInterval: 10.0, repeats: true) {
            [weak self] _ in
            URLSession.shared.dataTask(with: health) { _, resp, _ in
                DispatchQueue.main.async {
                    guard let self else { return }
                    if let http = resp as? HTTPURLResponse, http.statusCode == 200 {
                        self.healthStrikes = 0
                    } else {
                        self.healthStrikes += 1
                        if self.healthStrikes >= 3 {
                            self.watchTimer?.invalidate()
                            self.showStatus("Food Optimizer stopped unexpectedly",
                                "Your projects are saved. Please click Try again. "
                                + "If this keeps happening, reach out to the "
                                + "Food Intelligence Lab.",
                                retry: true)
                        }
                    }
                }
            }.resume()
        }
    }

    // Closing the window quits the app; quitting stops the server.
    func applicationShouldTerminateAfterLastWindowClosed(_ app: NSApplication) -> Bool { true }
    func applicationWillTerminate(_ note: Notification) {
        guard let l = launcher, l.isRunning else { return }
        l.terminate()   // SIGTERM -> launcher trap stops Streamlit and cleans up
        // Wait for a clean shutdown, but never hang the app: if the launcher is
        // stuck in a long setup step (bash defers its trap until the current
        // command returns), force-kill after a short grace period so quitting
        // is always prompt.
        let deadline = Date().addingTimeInterval(3.0)
        while l.isRunning && Date() < deadline {
            usleep(100_000)   // 0.1s
        }
        if l.isRunning {
            kill(l.processIdentifier, SIGKILL)
        }
    }
}

// File-upload support (st.file_uploader) — WKWebView needs this explicitly.
extension AppDelegate: WKUIDelegate {
    func webView(_ webView: WKWebView,
                 runOpenPanelWith parameters: WKOpenPanelParameters,
                 initiatedByFrame frame: WKFrameInfo,
                 completionHandler: @escaping ([URL]?) -> Void) {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.beginSheetModal(for: window) { resp in
            completionHandler(resp == .OK ? panel.urls : nil)
        }
    }

    // Without these, WebKit silently drops JS alert/confirm/prompt — a click
    // that should show a message would appear to do nothing.
    func webView(_ webView: WKWebView,
                 runJavaScriptAlertPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo,
                 completionHandler: @escaping () -> Void) {
        let alert = NSAlert()
        alert.messageText = "Food Optimizer"
        alert.informativeText = message
        alert.beginSheetModal(for: window) { _ in completionHandler() }
    }
    func webView(_ webView: WKWebView,
                 runJavaScriptConfirmPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo,
                 completionHandler: @escaping (Bool) -> Void) {
        let alert = NSAlert()
        alert.messageText = "Food Optimizer"
        alert.informativeText = message
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: "Cancel")
        alert.beginSheetModal(for: window) { resp in
            completionHandler(resp == .alertFirstButtonReturn)
        }
    }
    func webView(_ webView: WKWebView,
                 runJavaScriptTextInputPanelWithPrompt prompt: String,
                 defaultText: String?,
                 initiatedByFrame frame: WKFrameInfo,
                 completionHandler: @escaping (String?) -> Void) {
        let alert = NSAlert()
        alert.messageText = "Food Optimizer"
        alert.informativeText = prompt
        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 260, height: 24))
        field.stringValue = defaultText ?? ""
        alert.accessoryView = field
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: "Cancel")
        alert.beginSheetModal(for: window) { resp in
            completionHandler(resp == .alertFirstButtonReturn ? field.stringValue : nil)
        }
    }

    // target="_blank" / window.open() — open in the real browser rather than
    // silently doing nothing. Returning nil means "no new WebView created".
    func webView(_ webView: WKWebView,
                 createWebViewWith configuration: WKWebViewConfiguration,
                 for navigationAction: WKNavigationAction,
                 windowFeatures: WKWindowFeatures) -> WKWebView? {
        if let url = navigationAction.request.url { NSWorkspace.shared.open(url) }
        return nil
    }
}

// Download support (st.download_button) — asks where to save via NSSavePanel.
extension AppDelegate: WKNavigationDelegate, WKDownloadDelegate {
    func webView(_ webView: WKWebView,
                 decidePolicyFor navigationAction: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        // Our own status pages' "Try again" button. Never a real navigation.
        if let url = navigationAction.request.url,
           url.scheme?.lowercased() == "foodopt" {
            decisionHandler(.cancel)
            if url.host == "retry" { retryLaunch() }
            return
        }
        // Keep the app window on our own local server. Any external http(s)
        // link (docs, footer, an embedded link) opens in the user's real
        // browser instead of hijacking the app with no way back.
        if let url = navigationAction.request.url,
           let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https" {
            let host = url.host ?? ""
            if host != "127.0.0.1" && host != "localhost" {
                decisionHandler(.cancel)
                NSWorkspace.shared.open(url)
                return
            }
        }
        decisionHandler(navigationAction.shouldPerformDownload ? .download : .allow)
    }
    func webView(_ webView: WKWebView,
                 decidePolicyFor navigationResponse: WKNavigationResponse,
                 decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void) {
        decisionHandler(navigationResponse.canShowMIMEType ? .allow : .download)
    }
    // A blank window is never acceptable: if the page itself fails to load
    // after the server was healthy, say so kindly.
    func webView(_ webView: WKWebView,
                 didFailProvisionalNavigation navigation: WKNavigation!,
                 withError error: Error) { handleLoadFailure(error) }
    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!,
                 withError error: Error) { handleLoadFailure(error) }
    // If WebKit's content process dies (memory pressure, renderer crash),
    // the view goes blank white with no error callback. Reload immediately —
    // the server is still running, so this recovers invisibly.
    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        webView.reload()
    }
    func handleLoadFailure(_ error: Error) {
        let code = (error as NSError).code
        // Cancellations are normal (e.g. a navigation became a download).
        guard code != NSURLErrorCancelled, loaded else { return }
        watchTimer?.invalidate()
        showStatus("Food Optimizer stopped unexpectedly",
                   "Your projects are saved. Please click Try again. If this "
                   + "keeps happening, reach out to the Food Intelligence Lab.",
                   retry: true)
    }

    func webView(_ webView: WKWebView, navigationAction: WKNavigationAction,
                 didBecome download: WKDownload) { download.delegate = self }
    func webView(_ webView: WKWebView, navigationResponse: WKNavigationResponse,
                 didBecome download: WKDownload) { download.delegate = self }

    // Ask where to save (a user-chosen location is always writable — macOS
    // privacy protection silently blocks unsigned apps from writing straight
    // into ~/Downloads), then reveal the saved file in Finder so it is
    // obvious that something happened and where it went.
    func download(_ download: WKDownload,
                  decideDestinationUsing response: URLResponse,
                  suggestedFilename: String,
                  completionHandler: @escaping (URL?) -> Void) {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = suggestedFilename
        panel.directoryURL = FileManager.default.urls(
            for: .downloadsDirectory, in: .userDomainMask).first
        panel.beginSheetModal(for: window) { resp in
            guard resp == .OK, let url = panel.url else {
                completionHandler(nil)   // user chose Cancel
                return
            }
            try? FileManager.default.removeItem(at: url)  // panel confirmed overwrite
            self.downloadDestinations[ObjectIdentifier(download)] = url
            completionHandler(url)
        }
    }

    func downloadDidFinish(_ download: WKDownload) {
        NSSound(named: "Glass")?.play()
        if let dest = downloadDestinations.removeValue(forKey: ObjectIdentifier(download)) {
            NSWorkspace.shared.activateFileViewerSelecting([dest])
        }
    }

    func download(_ download: WKDownload, didFailWithError error: Error,
                  resumeData: Data?) {
        downloadDestinations.removeValue(forKey: ObjectIdentifier(download))
        let alert = NSAlert()
        alert.messageText = "The file could not be saved"
        alert.informativeText =
            "Something interrupted the save. Please try again and pick a "
            + "different folder — your FoodOptimizer folder always works."
        alert.beginSheetModal(for: window)
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
