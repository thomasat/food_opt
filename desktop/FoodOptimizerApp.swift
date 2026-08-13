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

    let supportDir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/FoodOptimizer")
    let dataDir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("FoodOptimizer")

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

        showStatus("Starting Food Optimizer…",
                   "The very first time, setup usually takes 1 to 5 minutes "
                   + "depending on your internet speed. None of your data is "
                   + "sent anywhere.", spinner: true)
        startLauncher()
        pollTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) {
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
        helpMenu.addItem(NSMenuItem(title: "Email Support",
            action: #selector(emailSupport), keyEquivalent: ""))
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
    @objc func emailSupport() {
        let url = URL(string:
            "mailto:sohum.patnaik@fsi.org?subject=Food%20Optimizer%20help")!
        NSWorkspace.shared.open(url)
    }
    @objc func showLogFile() {
        NSWorkspace.shared.activateFileViewerSelecting(
            [supportDir.appendingPathComponent("launcher.log")])
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
        NSWorkspace.shared.open(URL(fileURLWithPath: dest))
        NSApp.terminate(nil)
        return true
    }

    // ---------- status pages ----------

    func showStatus(_ title: String, _ body: String, spinner: Bool = false) {
        let spinnerHTML = spinner ? """
            <div style="margin:24px auto;width:28px;height:28px;border:3px solid #cdd6ce;
                        border-top-color:#2E6E4E;border-radius:50%;
                        animation:spin 1s linear infinite"></div>
            <style>@keyframes spin{to{transform:rotate(360deg)}}</style>
            """ : ""
        let html = """
        <html><head><meta charset="utf-8"></head>
        <body style="font-family:-apple-system,sans-serif;background:#f7f6f2;color:#2d3a2e;
                     display:flex;align-items:center;justify-content:center;height:96vh;margin:0">
          <div style="text-align:center;max-width:460px">
            <h1 style="font-weight:600">\(title)</h1>
            \(spinnerHTML)
            <p style="font-size:15px;line-height:1.5;color:#556">\(body)</p>
          </div>
        </body></html>
        """
        webView.loadHTMLString(html, baseURL: nil)
    }

    // ---------- launcher lifecycle ----------

    func startLauncher() {
        let p = Process()
        p.executableURL = URL(fileURLWithPath:
            Bundle.main.bundlePath + "/Contents/MacOS/launcher.sh")
        // Detach stdio: the launcher logs to its own file, and piping through
        // this app would SIGPIPE the launcher's shutdown when we exit first.
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        p.terminationHandler = { [weak self] proc in
            DispatchQueue.main.async { self?.launcherEnded(code: proc.terminationStatus) }
        }
        do { try p.run() } catch {
            showStatus("The app could not start",
                       "Please contact us and mention: launcher failed to run.")
            return
        }
        launcher = p
    }

    func launcherEnded(code: Int32) {
        guard !loaded else { return }   // normal shutdown later is fine
        pollTimer?.invalidate()
        switch code {
        case 2:
            showStatus("This Mac is not supported",
                       "Food Optimizer needs a Mac with an Apple chip (2020 or "
                       + "newer) running macOS 13 or later. Please contact us for help.")
        case 3:
            showStatus("Setup needs the internet, just this once",
                       "The first time it opens, Food Optimizer downloads its "
                       + "software components. Please connect to the internet, "
                       + "then quit (press Cmd-Q) and open Food Optimizer again. "
                       + "After that, no internet is needed.")
        default:
            showStatus("The app could not start",
                       "Please quit and open Food Optimizer again. If this keeps "
                       + "happening, use Help › Email Support and attach the file "
                       + "from Help › Show Log File.")
        }
    }

    func poll() {
        let portFile = supportDir.appendingPathComponent("server.port")
        guard let contents = try? String(contentsOf: portFile, encoding: .utf8),
              let port = contents.split(separator: " ").first.map(String.init),
              !port.isEmpty else { return }
        let health = URL(string: "http://127.0.0.1:\(port)/_stcore/health")!
        URLSession.shared.dataTask(with: health) { [weak self] _, resp, _ in
            guard let self, !self.loaded,
                  let http = resp as? HTTPURLResponse, http.statusCode == 200
            else { return }
            DispatchQueue.main.async {
                guard !self.loaded else { return }
                self.loaded = true
                self.pollTimer?.invalidate()
                self.webView.load(URLRequest(url:
                    URL(string: "http://localhost:\(port)")!))
                self.startHealthWatch(port: port)
            }
        }.resume()
    }

    // After the app is up, notice if the server ever dies and say so kindly.
    func startHealthWatch(port: String) {
        let health = URL(string: "http://127.0.0.1:\(port)/_stcore/health")!
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
                                "Your projects are saved. Please quit (press Cmd-Q) "
                                + "and open Food Optimizer again. If this keeps "
                                + "happening, use Help › Email Support.")
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
        l.terminate()      // SIGTERM -> launcher trap stops Streamlit
        l.waitUntilExit()  // let its cleanup (server + port file) finish
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
}

// Download support (st.download_button) — saves into ~/Downloads.
extension AppDelegate: WKNavigationDelegate, WKDownloadDelegate {
    func webView(_ webView: WKWebView,
                 decidePolicyFor navigationAction: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        decisionHandler(navigationAction.shouldPerformDownload ? .download : .allow)
    }
    func webView(_ webView: WKWebView,
                 decidePolicyFor navigationResponse: WKNavigationResponse,
                 decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void) {
        decisionHandler(navigationResponse.canShowMIMEType ? .allow : .download)
    }
    func webView(_ webView: WKWebView, navigationAction: WKNavigationAction,
                 didBecome download: WKDownload) { download.delegate = self }
    func webView(_ webView: WKWebView, navigationResponse: WKNavigationResponse,
                 didBecome download: WKDownload) { download.delegate = self }

    func download(_ download: WKDownload,
                  decideDestinationUsing response: URLResponse,
                  suggestedFilename: String,
                  completionHandler: @escaping (URL?) -> Void) {
        let downloads = FileManager.default.urls(
            for: .downloadsDirectory, in: .userDomainMask)[0]
        var dest = downloads.appendingPathComponent(suggestedFilename)
        var n = 2
        while FileManager.default.fileExists(atPath: dest.path) {
            let base = (suggestedFilename as NSString).deletingPathExtension
            let ext = (suggestedFilename as NSString).pathExtension
            let name = ext.isEmpty ? "\(base) \(n)" : "\(base) \(n).\(ext)"
            dest = downloads.appendingPathComponent(name)
            n += 1
        }
        completionHandler(dest)
    }
    func downloadDidFinish(_ download: WKDownload) {
        NSSound(named: "Glass")?.play()
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
