def get_stylesheet():
    return """
        QWidget {
            background-color: #161b22;
            color: #cdd9e5;
            font-size: 15px;
            font-family: 'Segoe UI', 'Roboto', 'Helvetica', sans-serif;
        }
        QMainWindow {
            border: 1px solid #21262d;
            border-radius: 8px;
            box-shadow: 0px 4px 16px rgba(20,23,28,0.15);
        }
        QTabWidget::pane {
            border: 1px solid #21262d;
            border-radius: 8px;
            margin: 2px;
            background: #1c2128;
        }
        QTabBar::tab {
            background: #21262d;
            padding: 12px 30px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            font-weight: bold;
            color: #768390;
            margin-right: 2px;
            transition: background 0.2s, color 0.2s;
        }
        QTabBar::tab:selected {
            background: #2d333b;
            color: #539bf5;
            border-bottom: 3px solid #539bf5;
        }
        QTabBar::tab:!selected {
            background: #21262d;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #21262d;
            border-radius: 8px;
            margin-top: 1.2ex;
            font-size: 16px;
            background: #161b22;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 12px;
            font-size: 16px;
        }
        QPushButton {
            background-color: #238636;
            color: #fff;
            border: none;
            padding: 12px 20px;
            border-radius: 8px;
            font-weight: bold;
            box-shadow: 0px 2px 8px rgba(20,23,28,0.07);
            transition: background-color 0.2s, box-shadow 0.2s;
        }
        QPushButton:hover {
            background-color: #2ea043;
            box-shadow: 0px 4px 12px rgba(83,155,245,0.08);
        }
        QPushButton:pressed {
            background-color: #2da44e;
        }
        QPushButton:disabled {
            background-color: #555;
            color: #999;
            box-shadow: none;
        }
        QPushButton#DeleteButton {
            background-color: #da3633;
        }
        QPushButton#DeleteButton:hover {
            background-color: #f04747;
        }
        QLineEdit, QListWidget {
            background-color: #21262d;
            border: 1px solid #333c4a;
            border-radius: 8px;
            padding: 9px 12px;
            font-size: 15px;
            color: #cdd9e5;
        }
        QLineEdit:focus, QListWidget:focus {
            border: 1.5px solid #539bf5;
            outline: none;
        }
        QListWidget::item:hover {
            background-color: #2d333b;
        }
        QListWidget::item:selected {
            background-color: #539bf5;
            color: #161b22;
            font-weight: bold;
            border-radius: 6px;
        }
        QLabel#ImagePreview, QLabel#CameraView {
            background-color: #000;
            border: 2px dashed #21262d;
            border-radius: 8px;
        }
        QStatusBar {
            background-color: #21262d;
            font-weight: bold;
            color: #539bf5;
            border-top: 1px solid #333c4a;
        }
        QMessageBox, QProgressDialog {
             background-color: #2d333b;
             border: 1px solid #539bf5;
             border-radius: 8px;
        }
        QProgressDialog QLabel {
            color: #cdd9e5;
            font-weight: bold;
        }
        QProgressDialog QPushButton {
            background-color: #da3633;
            color: #fff;
            font-weight: bold;
        }
        QProgressDialog QPushButton:hover {
            background-color: #f04747;
        }
    """
