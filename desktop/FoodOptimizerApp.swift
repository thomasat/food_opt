// Food Optimizer — native window wrapper.
// Compiled by build_dmg.sh into Contents/MacOS/FoodOptimizer (the bundle
// executable). Runs launcher.sh (headless) for environment setup and the
// Streamlit server, shows a friendly status page meanwhile, then hosts the
// UI in a WKWebView. Quitting (Cmd-Q or closing the window) terminates the
// launcher, which shuts the server down.
import Cocoa
import WebKit

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    var window: NSWindow!
    var webView: WKWebView!
    var launcher: Process?
    var pollTimer: Timer?
    var loaded = false

    let supportDir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/FoodOptimizer")

    func applicationDidFinishLaunching(_ note: Notification) {
        webView = WKWebView(frame: .zero)
        webView.uiDelegate = self
        webView.navigationDelegate = self

        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1280, height: 860),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered, defer: false)
        window.title = "Food Optimizer"
        window.minSize = NSSize(width: 700, height: 500)
        window.center()
        window.contentView = webView
        window.delegate = self
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)

        showStatus("Starting Food Optimizer…",
                   "The very first time, setup usually takes 1 to 5 minutes "
                   + "depending on your internet speed. None of your data is "
                   + "sent anywhere.")
        startLauncher()
        pollTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) {
            [weak self] _ in self?.poll()
        }
    }

    func showStatus(_ title: String, _ body: String) {
        let html = """
        <html><head><meta charset="utf-8"></head>
        <body style="font-family:-apple-system,sans-serif;background:#f7f6f2;color:#2d3a2e;
                     display:flex;align-items:center;justify-content:center;height:96vh;margin:0">
          <div style="text-align:center;max-width:440px">
            <h1 style="font-weight:600">\(title)</h1>
            <p style="font-size:15px;line-height:1.5;color:#556">\(body)</p>
          </div>
        </body></html>
        """
        webView.loadHTMLString(html, baseURL: nil)
    }

    func startLauncher() {
        let p = Process()
        p.executableURL = URL(fileURLWithPath:
            Bundle.main.bundlePath + "/Contents/MacOS/launcher.sh")
        var env = ProcessInfo.processInfo.environment
        env["FOODOPT_HEADLESS"] = "1"   // window replaces dialogs and browser
        p.environment = env
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
        if code == 2 {
            showStatus("This Mac is not supported",
                       "Food Optimizer needs a Mac with an Apple chip (2020 or "
                       + "newer) running macOS 13 or later. Please contact us for help.")
        } else {
            showStatus("The app could not start",
                       "Please quit and open Food Optimizer again. If this keeps "
                       + "happening, contact us and attach the file at: Library › "
                       + "Application Support › FoodOptimizer › launcher.log")
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
            }
        }.resume()
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
