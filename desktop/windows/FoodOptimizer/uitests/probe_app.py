"""A small Streamlit app the Windows shell's UI tests load in place of the
real app (uitests/stage_layout.ps1 -App probe). Each widget is something a
test finds by its label: typing, a download, an upload, links, script
dialogs and a message to the shell."""
import streamlit as st

st.text_input("Type here")
st.download_button("Download a CSV", "a,b\n1,2\n", file_name="probe.csv")
files = st.file_uploader("Upload files", accept_multiple_files=True)
names = [f.name for f in files or []]
st.write("uploaded: " + ", ".join(sorted(names)))
st.markdown("[Blank link](https://example.com/foodopt-blank)")
st.html(
    """
    <p><a href="https://example.com/foodopt-same" target="_self">Same link</a></p>
    <p><a href="file:///C:/Windows/win.ini" target="_self">File link</a></p>
    <p>
      <button onclick="window.open('https://example.com/foodopt-open')">Open window</button>
      <button onclick="alert('probe alert')">Alert</button>
      <button onclick="document.getElementById('r').textContent = 'confirm: ' + confirm('probe confirm')">Confirm</button>
      <button onclick="document.getElementById('r').textContent = 'prompt: ' + prompt('probe prompt', '')">Prompt</button>
      <button onclick="window.chrome.webview.postMessage('retry')">Post retry</button>
    </p>
    <span id="r"></span>
    <img src="https://example.com/foodopt-probe.png" alt="">
    """,
    unsafe_allow_javascript=True,
)
