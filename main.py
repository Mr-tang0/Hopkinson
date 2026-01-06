import os
import sys
from GUI.MainAPI import Api
import webview


def get_resource_path(relative_path):
    """ 获取资源绝对路径，兼容开发环境和 PyInstaller 打包环境 """
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller 打包后的临时目录
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


if __name__ == '__main__':
    # 获取 HTML 的绝对路径
    html_path = get_resource_path(os.path.join("GUI", "index.html"))

    if not os.path.exists(html_path):
        # 如果是调试模式，尝试直接使用相对路径
        html_path = "GUI/index.html"

    api = Api()
    print("ready")
    window = webview.create_window(
        "SHPB Data Processor",
        html_path,
        js_api=api,
        width=1200,
        height=800
    )
    webview.start(debug=True, http_server=True)  # 开启调试模式，打包后按 F12 可以看 Console 报错

# python -m PyInstaller main.spec
