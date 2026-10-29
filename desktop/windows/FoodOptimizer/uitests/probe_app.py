"""A small Streamlit app the Windows shell's UI tests load in place of the
real app (uitests/stage_layout.ps1 -App probe). Each widget is something a
test finds by its label: typing, a download, an upload, links, script
dialogs and a message to the shell."""
import streamlit as st

st.text_input("Type here")
st.download_button("Download a CSV", "a,b\n1,2\n", file_name="probe.csv")
st.download_button("Download a numbered CSV", "a,b\n3,4\n", file_name="probe (2).csv")
st.download_button("Download an accented CSV", "a,b\n5,6\n", file_name="Crème (2).csv")
files = st.file_uploader("Upload files", accept_multiple_files=True)
names = [f.name for f in files or []]
st.write("uploaded: " + ", ".join(sorted(names)))
st.markdown("[Blank link](https://example.com/foodopt-blank)")
# Inline handlers and file: links don't survive st.html's sanitiser, so a
# script wires the buttons and sets the file link's address.
st.html(
    """
    <p><a href="https://example.com/foodopt-same" target="_self">Same link</a></p>
    <p><a id="file-link" target="_self">File link</a></p>
    <p>
      <button id="b-open">Open window</button>
      <button id="b-alert">Alert</button>
      <button id="b-confirm">Confirm</button>
      <button id="b-prompt">Prompt</button>
      <button id="b-retry">Post retry</button>
    </p>
    <span id="r"></span>
    <img src="https://example.com/foodopt-probe.png" alt="">
    <script>
      (function () {
        function on(id, f) { document.getElementById(id).addEventListener("click", f); }
        var r = function (t) { document.getElementById("r").textContent = t; };
        document.getElementById("file-link").setAttribute("href", "file:///C:/Windows/win.ini");
        on("b-open", function () { window.open("https://example.com/foodopt-open"); });
        on("b-alert", function () { alert("probe alert"); });
        on("b-confirm", function () { r("confirm: " + confirm("probe confirm")); });
        on("b-prompt", function () { r("prompt: " + prompt("probe prompt", "")); });
        on("b-retry", function () { window.chrome.webview.postMessage("retry"); });
      })();
    </script>
    """,
    unsafe_allow_javascript=True,
)
