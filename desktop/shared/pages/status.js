// The setup and error page, an exact port of the page the Swift window
// builds (FoodOptimizerApp.swift: showInitialStatus ... updateStepsPage,
// readStatusFile). It holds no text of its own: every string comes from
// boot.messages, by its messages.json id. The shell calls window.FoodOpt.
(function () {
  // Lines the launcher publishes while it is doing setup work; the Swift
  // list setupStepPrefixes, word for word.
  var SETUP_STEP_PREFIXES = ["Downloading Python", "Creating environment",
                             "Updating components", "Installing components",
                             "Finishing setup"];

  // The error pages: the message ids each one shows, and whether it offers
  // Try again (FoodOptimizerApp.swift launcherEnded, startLauncher,
  // showCouldNotStartStatus, startHealthWatch).
  var ERRORS = {
    could_not_start: {title: "error.could_not_start.title", body: "error.could_not_start.body", retry: true},
    missing_part: {title: "error.missing_part.title", body: "error.missing_part.body", retry: false},
    unsupported: {title: "error.unsupported.title", body: "error.unsupported.body", retry: false},
    needs_internet: {title: "error.needs_internet.title", body: "error.needs_internet.body", retry: true},
    no_space: {title: "error.no_space.title", body: "error.no_space.body", retry: true},
    stopped: {title: "error.stopped.title", body: "error.stopped.body", retry: true}
  };

  // The sentences a shell may put before an error body (showError's
  // detailId): what was different about this give-up.
  var DETAILS = ["error.could_not_start.another_copy"];

  var DOWNLOAD = 0, LOAD = 1, OPEN = 2;
  var TICK_MS = 500;

  var M = {};
  var version = "";
  var s = freshState({firstRun: false});

  function freshState(view) {
    var firstRun = !!(view && view.firstRun);
    return {
      firstOpening: !!(view && view.firstOpening),
      upgrade: !firstRun,
      hasDownload: firstRun,
      activeStep: firstRun ? DOWNLOAD : LOAD,
      started: Date.now(),
      confirmed: false,
      stalled: false,
      detail: "",
      lastStatusLine: null,
      lastStepText: null,
      lastProgress: null,
      stepsShowing: false,
      stalledShowing: false
    };
  }

  function msg(id) {
    return Object.prototype.hasOwnProperty.call(M, id) ? M[id] : "";
  }
  function el(id) { return document.getElementById(id); }
  function setText(id, value) {
    var e = el(id);
    if (e) { e.textContent = value; }
  }
  function show(id, on, how) {
    var e = el(id);
    if (e) { e.style.display = on ? (how || "") : "none"; }
  }
  function footerText() {
    return version ? msg("footer.name") + " " + version : msg("footer.name");
  }

  // ---------- steps view ----------

  function stepLabels() {
    var labels = [msg("steps.load"), msg("steps.open")];
    return s.hasDownload ? [msg("steps.download")].concat(labels) : labels;
  }
  function rowOf(step) { return s.hasDownload ? step : step - 1; }
  function elapsed() {
    return Math.max(0, Math.floor((Date.now() - s.started) / 1000));
  }
  function latchStalled() {
    if (elapsed() >= 30 && s.activeStep !== DOWNLOAD) { s.stalled = true; }
  }
  function title() {
    if (s.activeStep === DOWNLOAD) {
      return msg(s.upgrade ? "steps.title.updating" : "steps.title.setting_up");
    }
    return msg(s.firstOpening ? "steps.title.setting_up" : "steps.title.starting");
  }
  function note() {
    if (s.activeStep === DOWNLOAD) {
      return msg(s.upgrade ? "steps.note.upgrade" : "steps.note.first_run");
    }
    if (s.stalled) { return msg("steps.note.still_loading"); }
    return msg(s.firstOpening ? "steps.note.first_opening" : "steps.note.usual");
  }

  function span(cls, id, glyph) {
    var e = document.createElement("span");
    e.className = cls;
    e.id = id;
    if (glyph) { e.textContent = glyph; }
    return e;
  }

  // Build the rows for the current row set, then paint them.
  function buildSteps() {
    var list = el("steps");
    while (list.firstChild) { list.removeChild(list.firstChild); }
    stepLabels().forEach(function (label, i) {
      var li = document.createElement("li");
      li.className = "step";
      li.id = "row" + i;
      var mark = document.createElement("span");
      mark.className = "mark";
      mark.appendChild(span("dot", "dot" + i, "•"));
      mark.appendChild(span("spin", "spin" + i));
      mark.appendChild(span("tick", "tick" + i, "✓"));
      li.appendChild(mark);
      li.appendChild(span("label", "label" + i));
      li.lastChild.textContent = label;
      list.appendChild(li);
    });
    setText("nextup", msg("steps.next_up"));
    setText("footer", footerText());
    show("message-view", false);
    show("steps-view", true);
    s.stepsShowing = true;
    s.stalledShowing = false;
    s.lastStatusLine = null;
    paintSteps();
  }

  // One tick of the steps view: which rows are done, the note, the counter.
  function paintSteps() {
    if (!s.stepsShowing) { return; }
    latchStalled();
    var active = rowOf(s.activeStep);
    var n = stepLabels().length;
    for (var i = 0; i < n; i++) {
      var done = i < active, running = i === active;
      var row = el("row" + i);
      if (!row) { continue; }
      row.style.opacity = done || running ? "1" : ".45";
      show("dot" + i, !(done || running), "inline");
      show("spin" + i, running, "inline-block");
      show("tick" + i, done, "inline");
    }
    setText("title", title());
    setText("note", note());
    setText("detail", s.activeStep === DOWNLOAD ? s.detail : "");
    setText("elapsed", msg("steps.elapsed").replace("{seconds}", String(elapsed())));
    show("barwrap", s.hasDownload && s.activeStep === DOWNLOAD, "block");
    var bar = el("bar");
    if (bar) { bar.style.width = (s.lastProgress === null ? 0 : s.lastProgress) + "%"; }
  }

  // Make `step` the one being worked on. A new step restarts the counter,
  // and so does the first launcher line confirming the step the page
  // opened on; the same step reported again does not.
  function setActiveStep(step, detail) {
    if (detail !== undefined) { s.detail = detail; }
    var rebuild = !s.stepsShowing;
    if (step === DOWNLOAD && !s.hasDownload) {
      s.hasDownload = true;
      rebuild = true;
    }
    if (step !== s.activeStep || !s.confirmed) {
      s.activeStep = step;
      s.started = Date.now();
    }
    s.confirmed = true;
    if (rebuild) { buildSteps(); } else { paintSteps(); }
  }

  // ---------- message view: error pages and the long-wait page ----------

  // Port of showStatus: a title, an optional spinner, step line and bar
  // (measured, or sliding when there is no percent), the body, Try again.
  function showMessage(o) {
    setText("msg-title", o.title);
    show("msg-spinner", !!o.spinner, "block");
    setText("msg-step", o.step || "");
    show("msg-step", o.step !== undefined && o.step !== null, "block");
    var measured = o.progress !== undefined && o.progress !== null;
    show("msg-barwrap", measured, "block");
    show("msg-slide", !measured && !!o.indeterminate, "block");
    var bar = el("msg-bar");
    if (bar) { bar.style.width = (measured ? Math.max(0, Math.min(o.progress, 100)) : 0) + "%"; }
    setText("msg-body", o.body);
    setText("retry", msg("button.try_again"));
    show("retry", !!o.retry, "inline-block");
    setText("msg-footer", footerText());
    show("steps-view", false);
    show("message-view", true);
    s.stepsShowing = false;
    s.stalledShowing = false;
    s.lastStatusLine = null;
  }

  function showError(kind, detailId) {
    var e = ERRORS[kind] || ERRORS.could_not_start;
    var body = msg(e.body);
    if (detailId && DETAILS.indexOf(detailId) >= 0) { body = msg(detailId) + " " + body; }
    showMessage({title: msg(e.title), body: body, retry: e.retry});
  }

  // The long-wait page. Not a steps page: later status lines update its
  // step line and bar in place instead of painting the steps back over it.
  function showStalled() {
    showMessage({
      title: msg("stalled.title"),
      body: msg("stalled.body"),
      spinner: true,
      step: s.lastStepText === null ? msg("stalled.step_fallback") : s.lastStepText,
      progress: s.lastProgress,
      indeterminate: s.lastProgress === null
    });
    s.stalledShowing = true;
  }

  // Send a message to the shell: WebView2 on Windows, WKWebView on the Mac.
  function post(message) {
    var c = window.chrome;
    if (c && c.webview && typeof c.webview.postMessage === "function") {
      c.webview.postMessage(message);
      return;
    }
    var w = window.webkit;
    if (w && w.messageHandlers && w.messageHandlers.foodopt) {
      w.messageHandlers.foodopt.postMessage(message);
    }
  }

  // ---------- the API the shells call ----------

  function init(boot) {
    boot = boot || {};
    M = boot.messages || {};
    version = boot.version || "";
    document.documentElement.setAttribute("data-os", boot.os || "");
    if (boot.view) { showSteps(boot.view); }
  }

  function showSteps(view) {
    s = freshState(view || {});
    buildSteps();
  }

  // One "<text>|<percent>" line from status.txt; the percent may be empty.
  function statusLine(raw) {
    var line = String(raw === undefined || raw === null ? "" : raw).trim();
    if (!line || line === s.lastStatusLine) { return; }
    s.lastStatusLine = line;
    var text = line, percent = "";
    var sep = line.lastIndexOf("|");
    if (sep >= 0) {
      text = line.slice(0, sep).trim();
      percent = line.slice(sep + 1).trim();
    }
    if (!text) { return; }
    s.lastStepText = text;
    var pct = null;
    if (/^[+-]?\d+$/.test(percent)) {
      pct = Math.max(0, Math.min(parseInt(percent, 10), 100));
      s.lastProgress = pct;
    }
    var step;
    var isSetup = SETUP_STEP_PREFIXES.some(function (p) { return text.indexOf(p) === 0; });
    if (pct !== null || isSetup) {
      step = DOWNLOAD;
    } else if (text.indexOf(msg("steps.load")) === 0) {
      step = LOAD;
    } else {
      step = OPEN;
    }
    if (s.stalledShowing) {
      setText("msg-step", text);
      var bar = el("msg-bar");
      if (bar && s.lastProgress !== null) { bar.style.width = s.lastProgress + "%"; }
      return;
    }
    setActiveStep(step, text);
  }

  // The server is up; what is left is Streamlit answering: the last step.
  function serverStarted() {
    if (!s.stalledShowing) { setActiveStep(OPEN); }
  }

  function tick() {
    paintSteps();
  }

  window.FoodOpt = {
    init: init,
    showSteps: showSteps,
    statusLine: statusLine,
    serverStarted: serverStarted,
    tick: tick,
    showStalled: showStalled,
    showError: showError,
    post: post
  };

  document.addEventListener("DOMContentLoaded", function () {
    el("retry").addEventListener("click", function () { post("retry"); });
    if (window.FOODOPT_BOOT) { init(window.FOODOPT_BOOT); }
    setInterval(tick, TICK_MS);
  });
})();
