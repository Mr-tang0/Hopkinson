import webview
from GUI.MainAPI import Api

if __name__ == '__main__':
    api = Api()
    window = webview.create_window("SHPB Data Processor",
                                   "GUI/index.html",
                                   width=1200,
                                   height=800,
                                   js_api=api)
    webview.start()


# 打包exe
# pyinstaller -F -w -i GUI/icon.ico main.py

