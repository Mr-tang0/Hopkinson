import os
import sys
from GUI.MainAPI import Api
import webview


if __name__ == '__main__':
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    print("exe_dir:", exe_dir)
    html_path = os.path.join(exe_dir, "GUI", "index.html")
    print("html_path:", html_path)

    if "conda" in sys.prefix.lower() or "program" in html_path.lower():
        html_path = "GUI/index.html"

    if not os.path.exists(html_path):
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    api = Api()
    window = webview.create_window("SHPB Data Processor",
                                   html_path,
                                   width=1200,
                                   height=800,
                                   js_api=api,
                                   )
    webview.start()

# python -m PyInstaller main.spec
