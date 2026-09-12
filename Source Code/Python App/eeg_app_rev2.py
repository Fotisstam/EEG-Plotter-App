import sys
import time
import csv
import struct
import serial
import serial.tools.list_ports
import numpy as np
import pyqtgraph as pg
from pyqtgraph import InfiniteLine
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QFrame, QCheckBox, QScrollArea,
    QToolBar, QColorDialog, QFileDialog, QGridLayout, QSlider,
    QDoubleSpinBox, QSizePolicy, QDesktopWidget, QSpinBox,
    QDialog, QDialogButtonBox, QGroupBox, QRadioButton, QButtonGroup,
    QLineEdit, QProgressBar, QStackedWidget
)
from PyQt5.QtCore import QTimer, Qt, QMimeData, QVariantAnimation, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QColor, QDrag, QPixmap

# ── Signal constants ──────────────────────────────────────────────────────────
FFT_LENGTH   = 256
DISPLAY_BINS = FFT_LENGTH // 2
RAW_WINDOW_BINS = DISPLAY_BINS * 4
NUM_CHANNELS = 32
SAMPLE_RATE  = 256          # Hz — change to match your STM32 config
RAW_SAMPLES_PER_FRAME = max(1, round(SAMPLE_RATE * 0.020))

# ── Binary frame protocol (must match firmware) ───────────────────────────────
PROTO_SYNC0      = 0xAA
PROTO_SYNC1      = 0x55
PROTO_HDR_SIZE   = 7                          # sync(2)+seq(2)+nch(1)+nbins(2)
PROTO_DATA_BYTES = NUM_CHANNELS * DISPLAY_BINS * 2   # uint16 magnitudes
PROTO_FRAME_SIZE = PROTO_HDR_SIZE + PROTO_DATA_BYTES + 2   # +CRC16
PROTO_SCALE      = 1.0 / 65535.0             # normalise uint16 → 0..1 float

# Frequency axis: bin i  →  i * (SAMPLE_RATE / FFT_LENGTH)  Hz
BIN_TO_HZ = SAMPLE_RATE / FFT_LENGTH
HZ_AXIS   = np.arange(DISPLAY_BINS, dtype=np.float32) * BIN_TO_HZ

# EEG band definitions  (name, lo_hz, hi_hz, colour)
EEG_BANDS = [
    ("δ Delta",  0.5,  4.0,  "#569CD6"),
    ("θ Theta",  4.0,  8.0,  "#4EC9B0"),
    ("α Alpha",  8.0, 13.0,  "#00B37E"),
    ("β Beta",  13.0, 30.0,  "#DCDCAA"),
    ("γ Gamma", 30.0, SAMPLE_RATE / 2, "#F48771"),
]

TRACE_PALETTE = [
    "#569CD6", "#4EC9B0", "#DCDCAA", "#F48771",
    "#9CDCFE", "#CE9178", "#B5CEA8", "#D7BA7D",
]

EEG_10_20_LABELS = [
    "Fp1", "Fpz", "Fp2", "F7",  "F3",  "Fz",  "F4",  "F8",
    "FT7", "FC3", "FCz", "FC4", "FT8", "T7",  "C3",  "Cz",
    "C4",  "T8",  "TP7", "CP3", "CPz", "CP4", "TP8", "P7",
    "P3",  "Pz",  "P4",  "P8",  "PO7", "O1",  "Oz",  "O2",
]

BAUD_RATES = ["9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"]

# ── Flash LUT ─────────────────────────────────────────────────────────────────
_FLASH_LUT = []
for _i in range(256):
    _p = _i / 255.0
    _br = int(0x00 + (0x2d - 0x00) * _p); _bg = int(0xB3 + (0x2d - 0xB3) * _p); _bb = int(0x7E + (0x30 - 0x7E) * _p)
    _gr = int(0x0A + (0x1c - 0x0A) * _p); _gg = int(0x2F + (0x1c - 0x2F) * _p); _gb = int(0x24 + (0x1f - 0x24) * _p)
    _FLASH_LUT.append((f"#{_br:02x}{_bg:02x}{_bb:02x}", f"#{_gr:02x}{_gg:02x}{_gb:02x}"))



# ── Helpers ───────────────────────────────────────────────────────────────────
def make_separator():
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setStyleSheet("color: #2b2b2b; background-color: #2b2b2b; max-height: 1px;")
    return sep

def section_label(text):
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight: bold; color: #858585; font-size: 10px; letter-spacing: 0.8px;")
    return lbl

def add_band_lines(plot_widget, show=True):
    """Add labeled EEG bands and return all items so visibility can be controlled."""
    items = []
    y_top = plot_widget.viewRange()[1][1] * 0.96
    for band_index, (name, lo, hi, color) in enumerate(EEG_BANDS):
        lo_bin = lo / BIN_TO_HZ
        hi_bin = hi / BIN_TO_HZ
        band_color = QColor(color)
        band_brush = QColor(band_color)
        band_brush.setAlpha(28)
        band_pen = QColor(band_color)
        band_pen.setAlpha(170)
        region = pg.LinearRegionItem(
            values=(lo_bin, hi_bin),
            brush=pg.mkBrush(band_brush),
            movable=False,
            pen=pg.mkPen(band_pen, width=1, style=Qt.DotLine),
        )
        region.setZValue(-10)
        region.setVisible(show)
        region._eeg_band_index = band_index
        plot_widget.addItem(region)
        items.append(region)

        label = pg.TextItem(
            text=name.split()[-1].upper(),
            color=QColor(color).lighter(125),
            anchor=(0.5, 1.0),
        )
        label.setPos((lo_bin + hi_bin) / 2, y_top)
        label.setZValue(-5)
        label.setVisible(show)
        label._eeg_band_index = band_index
        plot_widget.addItem(label)
        items.append(label)
    return items


# ── Draggable plot panel ───────────────────────────────────────────────────────
class DraggablePlotWrapper(QFrame):
    def __init__(self, channel_idx, plot_widget, initial_height=180, parent=None):
        super().__init__(parent)
        self.channel_idx = channel_idx
        self.plot_widget  = plot_widget

        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("PlotWrapper")
        self._base_style = "QFrame#PlotWrapper {{ background-color: {bg}; border: 1px solid {bd}; border-radius: 3px; }}"
        self.reset_style("#3e3e42", "#1e1e1e")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        ch_label = EEG_10_20_LABELS[channel_idx]

        title_row = QHBoxLayout()
        self.title_bar = QLabel(f" ☰  {ch_label}  (CH {channel_idx+1:02d})")
        self.title_bar.setStyleSheet("""
            QLabel { background-color: #252526; color: #cccccc; font-weight: bold;
                     font-size: 11px; padding: 5px; border-radius: 2px; }
        """)
        self.title_bar.setCursor(Qt.OpenHandCursor)
        title_row.addWidget(self.title_bar, stretch=1)

        self.peak_label = QLabel("peak: –")
        self.peak_label.setStyleSheet(
            "color: #4EC9B0; font-family: 'Cascadia Code', Consolas; font-size: 10px; padding: 5px 6px;"
            " background-color: #252526; border-radius: 2px;"
        )
        title_row.addWidget(self.peak_label)

        self.quality_label = QLabel("NO DATA")
        self.quality_label.setStyleSheet(
            "color: #858585; font-family: Consolas; font-size: 9px; font-weight: bold; padding: 5px 6px;"
            " background-color: #252526; border-radius: 2px;"
        )
        title_row.addWidget(self.quality_label)
        layout.addLayout(title_row)
        layout.addWidget(self.plot_widget)

        self.setFixedHeight(initial_height)

        self.slide_anim = QPropertyAnimation(self, b"pos")
        self.slide_anim.setDuration(250)
        self.slide_anim.setEasingCurve(QEasingCurve.OutCubic)

        self.flash_anim = QVariantAnimation(self)
        self.flash_anim.setDuration(400)
        self.flash_anim.setStartValue(0.0)
        self.flash_anim.setEndValue(1.0)
        self.flash_anim.valueChanged.connect(self._on_flash)

    def reset_style(self, border_hex, bg_hex):
        self.setStyleSheet(self._base_style.format(bg=bg_hex, bd=border_hex))

    def _on_flash(self, progress):
        idx = min(int(progress * 255), 255)
        self.reset_style(*_FLASH_LUT[idx])

    def update_peak(self, data_row, display_mode="FFT"):
        peak_bin = int(np.argmax(data_row))
        if display_mode == "Raw Data":
            self.peak_label.setText(f"bin: {peak_bin:03d}")
        else:
            peak_hz = peak_bin * BIN_TO_HZ
            self.peak_label.setText(f"peak: {peak_hz:.1f} Hz")

    def update_quality(self, data_row):
        peak = float(np.max(data_row))
        average = float(np.mean(data_row))
        if peak <= 0.001:
            quality, color = "NO DATA", "#858585"
        elif average > 0 and peak / average > 80:
            quality, color = "CHECK", "#FFB800"
        else:
            quality, color = "OK", "#00B37E"
        self.quality_label.setText(quality)
        self.quality_label.setStyleSheet(
            f"color: {color}; font-family: Consolas; font-size: 9px; font-weight: bold; padding: 5px 6px;"
            " background-color: #252526; border-radius: 2px;"
        )

    def animate_to_pos(self, target_pos):
        self.slide_anim.stop()
        self.slide_anim.setStartValue(self.pos())
        self.slide_anim.setEndValue(target_pos)
        self.slide_anim.start()

    def trigger_drop_flash(self):
        self.flash_anim.stop()
        self.flash_anim.start()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.title_bar.geometry().contains(event.pos()):
            self.title_bar.setCursor(Qt.ClosedHandCursor)
            window = self.window()
            if hasattr(window, 'set_active_drag_origin'):
                window.set_active_drag_origin(self.channel_idx)
            drag      = QDrag(self)
            mime_data = QMimeData()
            mime_data.setText(str(self.channel_idx))
            drag.setMimeData(mime_data)
            pixmap = QPixmap(self.size())
            self.render(pixmap)
            drag.setPixmap(pixmap)
            drag.setHotSpot(event.pos())
            self.setVisible(False)
            drag.exec_(Qt.MoveAction)
            self.setVisible(True)
            self.title_bar.setCursor(Qt.OpenHandCursor)


# ── Grid layout ────────────────────────────────────────────────────────────────
class RearrangeableGridLayout(QGridLayout):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent

    def handle_drag_enter(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def handle_drag_move(self, event):
        if not event.mimeData().hasText():
            return
        source_idx = int(event.mimeData().text())
        target_idx = None
        for i in range(self.count()):
            w = self.itemAt(i).widget()
            if w and w.geometry().contains(event.pos()) and isinstance(w, DraggablePlotWrapper):
                if w.channel_idx != source_idx:
                    target_idx = w.channel_idx
                    break
        if target_idx is not None:
            win = self.parent_widget.window()
            if hasattr(win, 'live_swap_channels'):
                win.live_swap_channels(source_idx, target_idx)
                event.acceptProposedAction()

    def handle_drop(self, event):
        source_idx = int(event.mimeData().text())
        win = self.parent_widget.window()
        if hasattr(win, 'finalize_drop_execution'):
            win.finalize_drop_execution(source_idx)
        event.acceptProposedAction()




# ── Connection dialog ──────────────────────────────────────────────────────────
class ConnectionDialog(QDialog):
    """
    Modal dialog that handles port scanning, selection, baud rate,
    and connection status in one place.
    """
    def __init__(self, current_ser, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connect to Hardware")
        self.setMinimumWidth(420)
        self.setModal(True)
        self._ser = current_ser   # may be None or already open
        self.result_ser = current_ser

        self.setStyleSheet("""
            QDialog      { background-color: #1e1e1e; color: #cccccc; }
            QLabel       { color: #cccccc; font-size: 13px; }
            QGroupBox    { color: #858585; font-size: 10px; font-weight: bold;
                           border: 1px solid #3e3e42; border-radius: 3px;
                           margin-top: 8px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; }
            QComboBox    { background-color: #252526; color: #cccccc;
                           border: 1px solid #3e3e42; border-radius: 3px; padding: 5px; }
            QPushButton  { background-color: #252526; color: #cccccc; font-weight: 500;
                           border: 1px solid #3e3e42; border-radius: 3px; padding: 6px 14px; }
            QPushButton:hover { background-color: #333333; border-color: #007acc; }
            QPushButton:pressed { background-color: #094771; }
            QLabel#status_lbl { font-family: Consolas; font-size: 12px; padding: 6px;
                                 border-radius: 4px; background-color: #18181B; }
        """)

        root = QVBoxLayout(self)
        root.setSpacing(14)
        root.setContentsMargins(16, 16, 16, 16)

        # ── Port group ────────────────────────────────────────────────────
        port_group = QGroupBox("SERIAL PORT")
        port_lay   = QVBoxLayout(port_group)

        scan_row = QHBoxLayout()
        self._port_combo = QComboBox()
        self._port_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        scan_row.addWidget(self._port_combo, stretch=1)
        self._scan_btn = QPushButton("🔍 Scan")
        self._scan_btn.setFixedWidth(90)
        self._scan_btn.clicked.connect(self._scan_ports)
        scan_row.addWidget(self._scan_btn)
        port_lay.addLayout(scan_row)

        self._desc_label = QLabel("")
        self._desc_label.setStyleSheet("color:#7C7C8A; font-size:11px; padding: 2px 0;")
        port_lay.addWidget(self._desc_label)
        self._port_combo.currentIndexChanged.connect(self._on_port_selected)

        root.addWidget(port_group)

        # ── Baud rate group ───────────────────────────────────────────────
        baud_group = QGroupBox("BAUD RATE")
        baud_lay   = QVBoxLayout(baud_group)

        self._baud_combo = QComboBox()
        for b in BAUD_RATES:
            self._baud_combo.addItem(b)
        self._baud_combo.setCurrentText("921600")
        baud_lay.addWidget(self._baud_combo)

        hint = QLabel("Tip: firmware uses 921600 — match this exactly.")
        hint.setStyleSheet("color:#7C7C8A; font-size:11px;")
        baud_lay.addWidget(hint)
        root.addWidget(baud_group)

        # ── Info group ────────────────────────────────────────────────────
        info_group = QGroupBox("PROTOCOL INFO")
        info_lay   = QGridLayout(info_group)
        info_lay.setSpacing(6)

        def info_pair(row, key, val):
            k = QLabel(key)
            k.setStyleSheet("color:#7C7C8A; font-size:11px;")
            v = QLabel(val)
            v.setStyleSheet("color:#C4C4CC; font-size:11px; font-family:Consolas;")
            info_lay.addWidget(k, row, 0)
            info_lay.addWidget(v, row, 1)

        info_pair(0, "Frame sync",    "0xAA 0x55")
        info_pair(1, "Channels",      f"{NUM_CHANNELS}")
        info_pair(2, "Bins / channel",f"{DISPLAY_BINS}")
        info_pair(3, "Frame size",    f"{2+2+1+2+NUM_CHANNELS*DISPLAY_BINS*2+2} bytes")
        info_pair(4, "CRC config",    "CCITT / poly 0x1021 / init 0xFFFF")
        info_pair(5, "CRC scope",     "Payload data bytes only")
        info_pair(6, "Sample rate",   f"{SAMPLE_RATE} Hz")
        root.addWidget(info_group)

        # ── Status label ──────────────────────────────────────────────────
        self._status_lbl = QLabel("Not connected.")
        self._status_lbl.setObjectName("status_lbl")
        self._status_lbl.setAlignment(Qt.AlignCenter)
        root.addWidget(self._status_lbl)

        # ── Action buttons ────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        self._conn_btn = QPushButton("🔌 Connect")
        self._conn_btn.setStyleSheet(
            "QPushButton { background-color:#007acc; border:1px solid #1a9fff; color:white; font-weight:bold; }"
            "QPushButton:hover { background-color:#1a9fff; }"
        )
        self._conn_btn.clicked.connect(self._do_connect)

        self._disc_btn = QPushButton("⏏ Disconnect")
        self._disc_btn.setStyleSheet(
            "QPushButton { background-color:#252526; border:1px solid #3e3e42; color:#cccccc; font-weight:bold; }"
            "QPushButton:hover { background-color:#333333; border-color:#007acc; }"
        )
        self._disc_btn.clicked.connect(self._do_disconnect)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        btn_row.addWidget(self._conn_btn)
        btn_row.addWidget(self._disc_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        # Initial scan and state sync
        self._scan_ports()
        self._sync_ui_state()

    # ── Helpers ───────────────────────────────────────────────────────────
    def _scan_ports(self):
        self._port_combo.blockSignals(True)
        self._port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for p in ports:
            self._port_combo.addItem(p.device, userData=p.description)
        self._port_combo.blockSignals(False)
        if ports:
            self._port_combo.setCurrentIndex(0)
            self._on_port_selected(0)
            self._set_status(f"{len(ports)} port(s) found. Select one and click Connect.", "#FFB800")
        else:
            self._desc_label.setText("")
            self._set_status("No serial ports found. Check USB cable and drivers.", "#E25C5C")

    def _on_port_selected(self, idx):
        desc = self._port_combo.itemData(idx)
        self._desc_label.setText(desc or "")

    def _do_connect(self):
        port = self._port_combo.currentText()
        if not port:
            self._set_status("No port selected.", "#E25C5C")
            return
        baud = int(self._baud_combo.currentText())
        try:
            if self._ser and self._ser.is_open:
                self._ser.close()
            self._ser = serial.Serial(port, baudrate=baud, timeout=0.02)
            self._ser.reset_input_buffer()
            self.result_ser = self._ser
            self._set_status(f"✓  Connected: {port} @ {baud} baud", "#00B37E")
        except Exception as e:
            self._ser = None
            self.result_ser = None
            self._set_status(f"✗  {e}", "#E25C5C")
        self._sync_ui_state()

    def _do_disconnect(self):
        try:
            if self._ser and self._ser.is_open:
                self._ser.close()
        except Exception:
            pass
        self._ser = None
        self.result_ser = None
        self._set_status("Disconnected.", "#FFB800")
        self._sync_ui_state()

    def _set_status(self, text, color):
        self._status_lbl.setText(text)
        self._status_lbl.setStyleSheet(
            f"QLabel#status_lbl {{ color:{color}; font-family:Consolas; font-size:12px;"
            f" padding:6px; border-radius:4px; background-color:#18181B; }}"
        )

    def _sync_ui_state(self):
        connected = bool(self._ser and self._ser.is_open)
        self._conn_btn.setEnabled(not connected)
        self._disc_btn.setEnabled(connected)
        self._port_combo.setEnabled(not connected)
        self._baud_combo.setEnabled(not connected)
        self._scan_btn.setEnabled(not connected)

    # ── Read-back accessors ───────────────────────────────────────────────
    @property
    def serial_port(self):
        return self.result_ser

    @property
    def selected_port_name(self):
        return self._port_combo.currentText()

    @property
    def selected_baud(self):
        return int(self._baud_combo.currentText())

# ── Recording setup dialog ─────────────────────────────────────────────────────
class RecordingDialog(QDialog):
    """Modal dialog to configure a recording session before it starts."""

    def __init__(self, channel_labels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Recording Session")
        self.setMinimumWidth(460)
        self.setStyleSheet("""
            QDialog        { background-color: #1e1e1e; color: #cccccc; }
            QLabel         { color: #cccccc; font-size: 13px; }
            QGroupBox      { color: #858585; font-size: 10px; font-weight: bold;
                             border: 1px solid #3e3e42; border-radius: 3px;
                             margin-top: 8px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; }
            QCheckBox      { color: #C4C4CC; font-size: 12px; padding: 2px; }
            QCheckBox:hover{ color: white; }
            QRadioButton   { color: #C4C4CC; font-size: 12px; padding: 2px; }
            QRadioButton:hover { color: white; }
            QSpinBox, QLineEdit {
                background-color: #202024; color: white;
                border: 1px solid #323238; border-radius: 4px; padding: 4px; }
            QPushButton    { background-color: #252526; color: #cccccc; font-weight: 500;
                             border: 1px solid #3e3e42; border-radius: 3px; padding: 6px 14px; }
            QPushButton:hover { background-color: #333333; border-color: #007acc; }
        """)

        root = QVBoxLayout(self)
        root.setSpacing(14)
        root.setContentsMargins(16, 16, 16, 16)

        # ── Duration ──────────────────────────────────────────────────────
        dur_group = QGroupBox("DURATION")
        dur_layout = QVBoxLayout(dur_group)
        dur_layout.setSpacing(8)

        self._dur_radio_group = QButtonGroup(self)
        self._rb_unlimited = QRadioButton("Record until manually stopped")
        self._rb_timed     = QRadioButton("Stop automatically after:")
        self._rb_unlimited.setChecked(True)
        self._dur_radio_group.addButton(self._rb_unlimited)
        self._dur_radio_group.addButton(self._rb_timed)
        dur_layout.addWidget(self._rb_unlimited)

        timed_row = QHBoxLayout()
        timed_row.addWidget(self._rb_timed)
        self._dur_spin = QSpinBox()
        self._dur_spin.setRange(1, 86400)
        self._dur_spin.setValue(60)
        self._dur_spin.setFixedWidth(72)
        self._dur_spin.setEnabled(False)
        timed_row.addWidget(self._dur_spin)
        timed_row.addWidget(QLabel("seconds"))
        timed_row.addStretch()
        dur_layout.addLayout(timed_row)

        self._rb_timed.toggled.connect(self._dur_spin.setEnabled)
        root.addWidget(dur_group)

        # ── Channels ──────────────────────────────────────────────────────
        ch_group = QGroupBox("CHANNELS TO RECORD")
        ch_vlay  = QVBoxLayout(ch_group)
        ch_vlay.setSpacing(6)

        macro_row = QHBoxLayout()
        sel_all = QPushButton("Select All")
        sel_none = QPushButton("Clear All")
        sel_active = QPushButton("Active Only")
        for btn in (sel_all, sel_none, sel_active):
            btn.setFixedHeight(26)
            macro_row.addWidget(btn)
        macro_row.addStretch()
        ch_vlay.addLayout(macro_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(160)
        scroll.setStyleSheet("QScrollArea { border: 1px solid #2d2d30; }")
        inner = QWidget()
        inner.setStyleSheet("background-color: #18181B;")
        grid  = QGridLayout(inner)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setSpacing(4)

        self._ch_checks = []
        for i, lbl in enumerate(channel_labels):
            cb = QCheckBox(f"({lbl}) Ch {i+1:02d}")
            cb.setChecked(True)
            grid.addWidget(cb, i // 4, i % 4)
            self._ch_checks.append(cb)

        scroll.setWidget(inner)
        ch_vlay.addWidget(scroll)
        root.addWidget(ch_group)

        sel_all.clicked.connect(lambda: [cb.setChecked(True)  for cb in self._ch_checks])
        sel_none.clicked.connect(lambda: [cb.setChecked(False) for cb in self._ch_checks])
        # "Active Only" is wired by the caller after construction via set_active_channels()

        self._sel_active_btn = sel_active

        # ── Output file ───────────────────────────────────────────────────
        file_group = QGroupBox("OUTPUT FILE")
        file_lay   = QVBoxLayout(file_group)

        file_row = QHBoxLayout()
        self._path_edit = QLineEdit(f"eeg_{int(time.time())}.csv")
        self._path_edit.setPlaceholderText("output path…")
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse)
        file_row.addWidget(self._path_edit, stretch=1)
        file_row.addWidget(browse_btn)
        file_lay.addLayout(file_row)
        root.addWidget(file_group)

        # ── Buttons ───────────────────────────────────────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("▶  Start Recording")
        btns.button(QDialogButtonBox.Ok).setStyleSheet(
            "QPushButton { background-color:#007acc; border:1px solid #1a9fff; color:white; }"
            "QPushButton:hover { background-color:#1a9fff; }"
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def _browse(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Recording", self._path_edit.text(), "CSV (*.csv)"
        )
        if path:
            self._path_edit.setText(path)

    def set_active_channels(self, active_indices):
        """Pre-select only the currently visible channels when 'Active Only' is clicked."""
        self._sel_active_btn.clicked.connect(
            lambda: [cb.setChecked(i in active_indices) for i, cb in enumerate(self._ch_checks)]
        )

    # ── Result accessors ──────────────────────────────────────────────────
    @property
    def output_path(self):
        return self._path_edit.text().strip()

    @property
    def duration_seconds(self):
        """Returns None for unlimited, int for timed."""
        return self._dur_spin.value() if self._rb_timed.isChecked() else None

    @property
    def selected_channels(self):
        return [i for i, cb in enumerate(self._ch_checks) if cb.isChecked()]


# ── Main application ───────────────────────────────────────────────────────────
class MultiChannelFFTApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("32-Channel EEG Signal Analyzer")

        screen = QDesktopWidget().availableGeometry()
        dw = int(screen.width()  * 0.85)
        dh = int(screen.height() * 0.85)
        self.resize(dw, dh)
        self.setAcceptDrops(True)

        self.setStyleSheet("""
            QMainWindow  { background-color: #181818; }
            QToolBar     { background-color: #202024; border-bottom: 1px solid #3e3e42; padding: 6px 10px; }
            QLabel       { color: #cccccc; font-family: 'Segoe UI', Arial; font-size: 13px; }
            QComboBox    { background-color: #252526; color: #cccccc; border: 1px solid #3e3e42;
                           border-radius: 3px; padding: 5px 8px; min-width: 80px; }
            QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: #007acc; }
            QComboBox QAbstractItemView { background-color: #252526; color: #cccccc; selection-background-color: #094771; }
            QPushButton  { background-color: #252526; color: #cccccc; font-weight: 500;
                           border: 1px solid #3e3e42; border-radius: 3px; padding: 6px 12px; }
            QPushButton:hover { background-color: #333333; border-color: #007acc; }
            QPushButton:pressed { background-color: #094771; }
            QScrollArea  { border: none; background-color: transparent; }
            QCheckBox    { color: #cccccc; font-family: 'Segoe UI'; font-size: 12px; padding: 3px; }
            QCheckBox:hover { color: white; }
            QDoubleSpinBox, QSpinBox {
                background-color: #252526; color: white; border: 1px solid #3e3e42;
                border-radius: 3px; padding: 4px; min-width: 65px; }
            QSlider::groove:horizontal { border:1px solid #3e3e42; height:6px; background:#252526; border-radius:3px; }
            QSlider::handle:horizontal { background:#007acc; width:14px; margin:-4px 0; border-radius:7px; }
            QSlider::handle:horizontal:hover { background:#1a9fff; }
            QScrollBar:vertical { background:#181818; width:10px; margin:0; }
            QScrollBar::handle:vertical { background:#424242; min-height:24px; border-radius:3px; }
            QScrollBar::handle:vertical:hover { background:#5a5a5a; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
            QFrame#HomePage { background-color:#1e1e1e; border:1px solid #2b2b2b; border-radius:3px; }
            QFrame#HomeHero { background-color:#252526; border:1px solid #3e3e42; border-left:3px solid #007acc; border-radius:3px; }
            QFrame#MetricCard { background-color:#252526; border:1px solid #3e3e42; border-radius:3px; }
            QFrame#MetricCard:hover { border-color:#007acc; }
            QFrame#HomeActions { background-color:#252526; border:1px solid #3e3e42; border-radius:3px; }
            QLabel#HomeEyebrow { color:#858585; font-size:10px; font-weight:bold; letter-spacing:1px; }
            QLabel#HomeTitle { color:#ffffff; font-size:24px; font-weight:bold; }
            QLabel#HomeHeroDetail { color:#858585; font-size:12px; }
            QLabel#HomeStatus { color:#4ec9b0; background:#183b3b; border:1px solid #245b59; border-radius:3px; padding:6px 10px; font-size:10px; font-weight:bold; }
            QLabel#MetricValue { color:#ffffff; font-size:22px; font-weight:bold; }
            QLabel#MetricLabel { color:#858585; font-size:10px; font-weight:bold; letter-spacing:0.6px; }
            QLabel#MetricDetail { color:#cccccc; font-size:11px; }
            QPushButton#PrimaryAction { background-color:#007acc; color:white; border:1px solid #1a9fff; font-weight:bold; padding:9px 16px; }
            QPushButton#PrimaryAction:hover { background-color:#1a9fff; }
            QPushButton#SecondaryAction { background-color:#252526; color:#cccccc; border:1px solid #3e3e42; padding:9px 16px; }
            QPushButton#SecondaryAction:hover { background-color:#333333; border-color:#007acc; }
        """)

        # ── State ──────────────────────────────────────────────────────────────
        self.ser              = None
        self.stream_active    = False
        self.is_split_view    = False
        self.is_frozen        = False
        self.show_bands       = True
        self.split_columns    = 2
        self.current_plot_height = int(dh * 0.22)
        self.calibration_scalar  = 1.0
        self.packet_count     = 0
        self.dropped_packets  = 0
        self.last_fps_time    = time.time()
        self.frame_count      = 0
        self.session_runtime  = 0.0
        self.last_update_time = time.time()
        self.active_drag_origin_idx = None
        self.is_animating_layout    = False
        self.grid_layout      = None
        self.nav_visible      = True
        self._recording       = False
        self._csv_writer      = None
        self._csv_file        = None
        self._peak_tick       = 0      # update peak labels every N frames
        self._serial_buf      = bytearray()   # raw bytes from UART
        self._last_seq        = -1            # for out-of-order detection
        self._crc_errors      = 0
        self._record_duration = None   # None = unlimited
        self._record_channels = list(range(NUM_CHANNELS))
        self._record_elapsed  = 0.0
        self._record_timer    = None
        self.display_mode     = "FFT"
        self.show_peak_markers = True
        self.band_visibility   = [True] * len(EEG_BANDS)
        self.peak_lines        = {}
        self.peak_labels       = {}
        self.plot_signal_proxies = []
        self.cursor_readout    = None
        self.auto_scale_y      = False
        self._frame_ready      = False

        self.channel_render_order = list(range(NUM_CHANNELS))
        self.channel_colors = [
            QColor(TRACE_PALETTE[i % len(TRACE_PALETTE)]).lighter(
                100 + 12 * (i // len(TRACE_PALETTE))
            )
            for i in range(NUM_CHANNELS)
        ]

        self.fft_data_matrix = np.zeros((NUM_CHANNELS, DISPLAY_BINS), dtype=np.float32)
        self.raw_data_matrix = np.zeros((NUM_CHANNELS, RAW_WINDOW_BINS), dtype=np.float32)
        self._raw_pending = np.empty((NUM_CHANNELS, 0), dtype=np.float32)
        self._raw_sample_accumulator = 0.0

        # ── Build UI ──────────────────────────────────────────────────────────
        self.init_control_toolbar()

        main_workspace = QWidget()
        self.setCentralWidget(main_workspace)
        self.layout_core = QHBoxLayout(main_workspace)
        self.layout_core.setContentsMargins(12, 12, 12, 12)
        self.layout_core.setSpacing(12)

        self.content_stack = QStackedWidget()
        self.layout_core.addWidget(self.content_stack, stretch=1)

        self.plot_container = QWidget()
        self.plot_container.setObjectName("PlotWorkspace")
        self.plot_container.setStyleSheet(
            "QWidget#PlotWorkspace { background-color: #1e1e1e; border: 1px solid #2b2b2b; border-radius: 3px; }"
        )
        self.plot_container_layout = QVBoxLayout(self.plot_container)
        self.plot_container_layout.setContentsMargins(0, 0, 0, 0)
        self.content_stack.addWidget(self.plot_container)

        self.init_plot_display()

        self.curves           = {}
        self.individual_widgets = {}
        self.color_buttons    = []
        self.band_region_items = []   # for merged view

        self.init_left_navigation_menu(self.layout_core)
        self.init_home_dashboard()
        self.show_home()
        self.update_channel_visibility()

        self.serial_timer = QTimer()
        self.serial_timer.timeout.connect(self._poll_serial)
        self.serial_timer.setTimerType(Qt.PreciseTimer)
        self.serial_timer.start(1)

        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.update_plots)
        self.render_timer.setTimerType(Qt.PreciseTimer)
        self.render_timer.start(4)

    # ══════════════════════════════════════════════════════════════════════════
    # Home dashboard
    # ══════════════════════════════════════════════════════════════════════════
    def init_home_dashboard(self):
        self.home_page = QFrame()
        self.home_page.setObjectName("HomePage")

        root = QVBoxLayout(self.home_page)
        root.setContentsMargins(24, 22, 24, 22)
        root.setSpacing(12)

        hero = QFrame()
        hero.setObjectName("HomeHero")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(16, 12, 16, 12)
        hero_layout.setSpacing(4)

        eyebrow = QLabel("EEG STUDIO  /  ACQUISITION MONITOR")
        eyebrow.setObjectName("HomeEyebrow")
        hero_layout.addWidget(eyebrow)

        hero_row = QHBoxLayout()
        title = QLabel("Acquisition overview")
        title.setObjectName("HomeTitle")
        hero_row.addWidget(title)
        hero_row.addStretch()

        self.home_status_label = QLabel("●  SYSTEM READY")
        self.home_status_label.setObjectName("HomeStatus")
        hero_row.addWidget(self.home_status_label)
        hero_layout.addLayout(hero_row)

        subtitle = QLabel(
            "Serial telemetry and session state from the connected STM32 device."
        )
        subtitle.setObjectName("HomeHeroDetail")
        hero_layout.addWidget(subtitle)
        root.addWidget(hero)

        cards = QGridLayout()
        cards.setHorizontalSpacing(10)
        cards.setVerticalSpacing(10)

        def add_metric(row, col, label, value, detail, accent):
            card = QFrame()
            card.setObjectName("MetricCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(14, 12, 14, 12)
            card_layout.setSpacing(4)

            metric_label = QLabel(label)
            metric_label.setObjectName("MetricLabel")
            card_layout.addWidget(metric_label)

            value_label = QLabel(value)
            value_label.setObjectName("MetricValue")
            value_label.setStyleSheet(f"color:{accent}; font-size:22px; font-weight:bold;")
            card_layout.addWidget(value_label)

            detail_label = QLabel(detail)
            detail_label.setObjectName("MetricDetail")
            card_layout.addWidget(detail_label)
            cards.addWidget(card, row, col)
            return value_label, detail_label

        self.home_connection_value, self.home_connection_detail = add_metric(
            0, 0, "HARDWARE LINK", "OFFLINE", "Scan a port to begin", "#f48771"
        )
        self.home_channels_value, self.home_channels_detail = add_metric(
            0, 1, "ACTIVE CHANNELS", "04 / 32", "Visible in the analyzer", "#4ec9b0"
        )
        self.home_packets_value, self.home_packets_detail = add_metric(
            0, 2, "PACKETS RECEIVED", "0", "CRC errors: 0", "#569cd6"
        )
        self.home_runtime_value, self.home_runtime_detail = add_metric(
            0, 3, "SESSION RUNTIME", "00:00:00", "Ready for acquisition", "#dcdcaa"
        )
        root.addLayout(cards)

        link_panel = QFrame()
        link_panel.setObjectName("HomeActions")
        link_layout = QVBoxLayout(link_panel)
        link_layout.setContentsMargins(14, 12, 14, 12)
        link_layout.setSpacing(8)

        link_title = QLabel("SERIAL LINK PROFILE")
        link_title.setObjectName("MetricLabel")
        link_layout.addWidget(link_title)

        link_grid = QGridLayout()
        link_grid.setHorizontalSpacing(18)
        link_grid.setVerticalSpacing(4)
        link_values = [
            ("Frame sync", "0xAA  0x55"),
            ("Payload", f"{NUM_CHANNELS} channels  /  {DISPLAY_BINS} bins"),
            ("CRC config", "CRC-16/CCITT  |  poly 0x1021  |  init 0xFFFF"),
            ("CRC scope", "Payload data bytes only"),
            ("Sample rate", f"{SAMPLE_RATE} Hz"),
        ]
        for index, (label_text, value_text) in enumerate(link_values):
            label = QLabel(label_text.upper())
            label.setObjectName("MetricLabel")
            value = QLabel(value_text)
            value.setObjectName("MetricDetail")
            link_grid.addWidget(label, index, 0)
            link_grid.addWidget(value, index, 1)
        link_grid.setColumnStretch(1, 1)
        link_layout.addLayout(link_grid)
        root.addWidget(link_panel)

        actions = QFrame()
        actions.setObjectName("HomeActions")
        actions_layout = QVBoxLayout(actions)
        actions_layout.setContentsMargins(14, 12, 14, 12)
        actions_layout.setSpacing(8)

        actions_title = QLabel("QUICK ACTIONS")
        actions_title.setObjectName("MetricLabel")
        actions_layout.addWidget(actions_title)

        actions_detail = QLabel("Use the existing serial stream to inspect, record, or export data.")
        actions_detail.setObjectName("HomeHeroDetail")
        actions_layout.addWidget(actions_detail)

        actions_row = QHBoxLayout()
        dashboard_action = QPushButton("Open Live Dashboard")
        dashboard_action.setObjectName("PrimaryAction")
        dashboard_action.clicked.connect(self.show_dashboard)
        actions_row.addWidget(dashboard_action)

        connect_action = QPushButton("Connect Hardware")
        connect_action.setObjectName("SecondaryAction")
        connect_action.clicked.connect(self.open_connection_dialog)
        actions_row.addWidget(connect_action)

        record_action = QPushButton("Configure Recording")
        record_action.setObjectName("SecondaryAction")
        record_action.clicked.connect(self.toggle_recording)
        actions_row.addWidget(record_action)
        actions_row.addStretch()
        actions_layout.addLayout(actions_row)
        root.addWidget(actions)
        root.addStretch()

        self.content_stack.addWidget(self.home_page)

    def show_home(self):
        self.content_stack.setCurrentWidget(self.home_page)
        self.home_nav_btn.setChecked(True)
        self.dashboard_nav_btn.setChecked(False)
        self.workspace_label.setText("OVERVIEW")
        self._set_dashboard_controls_visible(False)
        self.nav_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Maximum)
        self.nav_frame.setFixedHeight(188)
        self.layout_core.setAlignment(self.nav_frame, Qt.AlignTop)
        self.nav_frame.adjustSize()

    def show_dashboard(self):
        self.content_stack.setCurrentWidget(self.plot_container)
        self.home_nav_btn.setChecked(False)
        self.dashboard_nav_btn.setChecked(True)
        self.workspace_label.setText("SPECTRUM WORKSPACE")
        self._set_dashboard_controls_visible(True)
        self.nav_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.nav_frame.setMinimumHeight(0)
        self.nav_frame.setMaximumHeight(16777215)
        self.layout_core.setAlignment(self.nav_frame, Qt.Alignment())

    @staticmethod
    def _set_layout_item_visible(item, visible):
        widget = item.widget()
        if widget is not None:
            widget.setVisible(visible)
            return
        child_layout = item.layout()
        if child_layout is not None:
            for child_index in range(child_layout.count()):
                MultiChannelFFTApp._set_layout_item_visible(
                    child_layout.itemAt(child_index), visible
                )

    def _set_dashboard_controls_visible(self, visible):
        for widget in self.dashboard_toolbar_widgets:
            widget.setVisible(visible)

        nav_layout = self.nav_frame.layout()
        for item_index in range(self.dashboard_nav_start_index, nav_layout.count()):
            self._set_layout_item_visible(nav_layout.itemAt(item_index), visible)

    # ══════════════════════════════════════════════════════════════════════════
    # Toolbar
    # ══════════════════════════════════════════════════════════════════════════
    def init_control_toolbar(self):
        toolbar = QToolBar("Top Control Deck")
        toolbar.setMovable(False)
        toolbar.setFixedHeight(56)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(6)

        brand = QLabel("◈  EEG STUDIO")
        brand.setStyleSheet(
            "color:#4EC9B0; font-size:13px; font-weight:bold; letter-spacing:0.8px; padding-right:12px;"
        )
        layout.addWidget(brand)

        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setStyleSheet("color:#3e3e42; background:#3e3e42; max-width:1px;")
        layout.addWidget(divider)

        workspace_label = QLabel("SPECTRUM WORKSPACE")
        self.workspace_label = workspace_label
        workspace_label.setStyleSheet(
            "color:#858585; font-size:10px; font-weight:bold; letter-spacing:0.7px; padding:0 8px;"
        )
        layout.addWidget(workspace_label)

        mode_label = QLabel("MODE")
        mode_label.setStyleSheet("color:#858585; font-size:10px; font-weight:bold; padding-left:8px;")
        layout.addWidget(mode_label)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["FFT", "Raw Data"])
        self.mode_combo.setFixedWidth(88)
        self.mode_combo.currentTextChanged.connect(self.set_display_mode)
        layout.addWidget(self.mode_combo)

        self.dashboard_toolbar_widgets = []
        self.dashboard_toolbar_widgets.extend([mode_label, self.mode_combo])

        self.split_btn = QPushButton("Split View")
        self.split_btn.setFixedHeight(30)
        self.split_btn.clicked.connect(self.toggle_split_view)
        layout.addWidget(self.split_btn)
        self.dashboard_toolbar_widgets.append(self.split_btn)

        self.revert_layout_btn = QPushButton("Reset Order")
        self.revert_layout_btn.clicked.connect(self.revert_channels_to_original)
        self.revert_layout_btn.setStyleSheet(
            "QPushButton { background-color:#252526; border:1px solid #3e3e42; }"
            "QPushButton:hover { background-color:#333333; border-color:#007acc; }"
        )
        self.revert_layout_btn.setEnabled(False)
        layout.addWidget(self.revert_layout_btn)
        self.dashboard_toolbar_widgets.append(self.revert_layout_btn)

        # Column count for split view
        cols_label = QLabel("Cols:")
        layout.addWidget(cols_label)
        self.dashboard_toolbar_widgets.append(cols_label)
        self.col_spin = QSpinBox()
        self.col_spin.setRange(1, 4)
        self.col_spin.setValue(2)
        self.col_spin.setFixedWidth(48)
        self.col_spin.valueChanged.connect(self._on_col_count_changed)
        layout.addWidget(self.col_spin)
        self.dashboard_toolbar_widgets.append(self.col_spin)

        self.freeze_btn = QPushButton("⏸ Freeze")
        self.freeze_btn.setFixedHeight(30)
        self.freeze_btn.setStyleSheet(
            "QPushButton { background-color:#252526; border:1px solid #3e3e42; color:#dcdcaa; }"
            "QPushButton:hover { background-color:#333333; border-color:#007acc; }"
        )
        self.freeze_btn.clicked.connect(self.toggle_freeze)
        layout.addWidget(self.freeze_btn)
        self.dashboard_toolbar_widgets.append(self.freeze_btn)

        self.band_btn = QPushButton("⚡ Bands ON")
        self.band_btn.setFixedHeight(30)
        self.band_btn.setStyleSheet(
            "QPushButton { background-color:#094771; border:1px solid #007acc; color:white; }"
            "QPushButton:hover { background-color:#0e639c; }"
        )
        self.band_btn.clicked.connect(self.toggle_bands)
        layout.addWidget(self.band_btn)
        self.dashboard_toolbar_widgets.append(self.band_btn)

        self.peak_btn = QPushButton("⌖ Peaks ON")
        self.peak_btn.setFixedHeight(30)
        self.peak_btn.clicked.connect(self.toggle_peak_markers)
        layout.addWidget(self.peak_btn)
        self.dashboard_toolbar_widgets.append(self.peak_btn)

        self.record_btn = QPushButton("⏺ Record")
        self.record_btn.setFixedHeight(30)
        self.record_btn.setStyleSheet(
            "QPushButton { background-color:#252526; border:1px solid #3e3e42; color:#cccccc; }"
            "QPushButton:hover { background-color:#333333; border-color:#007acc; }"
        )
        self.record_btn.clicked.connect(self.toggle_recording)
        layout.addWidget(self.record_btn)
        self.dashboard_toolbar_widgets.append(self.record_btn)

        self.export_btn = QPushButton("📷 Snapshot")
        self.export_btn.setFixedHeight(30)
        self.export_btn.clicked.connect(self.export_plot_image)
        layout.addWidget(self.export_btn)
        self.dashboard_toolbar_widgets.append(self.export_btn)

        self.sidebar_btn = QPushButton("◀ Hide Panel")
        self.sidebar_btn.setFixedHeight(30)
        self.sidebar_btn.clicked.connect(self.toggle_sidebar)
        layout.addWidget(self.sidebar_btn)
        self.dashboard_toolbar_widgets.append(self.sidebar_btn)

        layout.addStretch()

        self.runtime_label = QLabel("T+ 00:00.00")
        self.runtime_label.setStyleSheet(
            "color:#00B37E; font-family:'Consolas',monospace; font-size:13px; font-weight:bold; margin-right:10px;"
        )
        layout.addWidget(self.runtime_label)

        self.cursor_readout = QLabel("Hover a trace for details")
        self.cursor_readout.setStyleSheet(
            "color:#858585; font-family:'Consolas',monospace; font-size:11px; padding:0 10px;"
        )
        layout.addWidget(self.cursor_readout)

        self.telemetry_label = QLabel("Waiting for hardware — click Scan then Connect.")
        self.telemetry_label.setStyleSheet(
            "color:#FFB800; font-family:'Consolas',monospace; font-size:12px; font-weight:bold;"
        )
        layout.addWidget(self.telemetry_label)
        self.dashboard_toolbar_widgets.extend([
            self.runtime_label, self.cursor_readout, self.telemetry_label
        ])

        toolbar.addWidget(container)

    # ══════════════════════════════════════════════════════════════════════════
    # Left navigation panel
    # ══════════════════════════════════════════════════════════════════════════
    def init_left_navigation_menu(self, parent_layout):
        self.nav_frame = QFrame()
        self.nav_frame.setFixedWidth(260)
        self.nav_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.nav_frame.setStyleSheet(
            "background-color: #252526; border-radius: 3px; border: 1px solid #2b2b2b;"
        )

        nav_layout = QVBoxLayout(self.nav_frame)
        nav_layout.setContentsMargins(14, 16, 14, 16)
        nav_layout.setSpacing(10)

        app_title = QLabel("EEG STUDIO")
        app_title.setStyleSheet(
            "color:#ffffff; font-size:18px; font-weight:bold; letter-spacing:0.6px; padding:2px 0 0;"
        )
        nav_layout.addWidget(app_title)

        app_subtitle = QLabel("32-CHANNEL SIGNAL ANALYZER")
        app_subtitle.setStyleSheet(
            "color:#858585; font-size:9px; font-weight:bold; letter-spacing:0.8px; padding-bottom:6px;"
        )
        nav_layout.addWidget(app_subtitle)
        nav_layout.addWidget(make_separator())

        self.home_nav_btn = QPushButton("⌂  Home")
        self.home_nav_btn.setCheckable(True)
        self.home_nav_btn.setStyleSheet(
            "QPushButton { text-align:left; padding:5px 10px; background:#252526; border:1px solid #3e3e42; }"
            "QPushButton:checked { background:#094771; border-color:#007acc; color:white; }"
            "QPushButton:hover { background:#333333; border-color:#007acc; }"
        )
        self.home_nav_btn.setFixedHeight(34)
        self.home_nav_btn.clicked.connect(self.show_home)
        nav_layout.addWidget(self.home_nav_btn)

        self.dashboard_nav_btn = QPushButton("▦  Live Dashboard")
        self.dashboard_nav_btn.setCheckable(True)
        self.dashboard_nav_btn.setStyleSheet(
            "QPushButton { text-align:left; padding:5px 10px; background:#252526; border:1px solid #3e3e42; }"
            "QPushButton:checked { background:#094771; border-color:#007acc; color:white; }"
            "QPushButton:hover { background:#333333; border-color:#007acc; }"
        )
        self.dashboard_nav_btn.setFixedHeight(34)
        self.dashboard_nav_btn.clicked.connect(self.show_dashboard)
        nav_layout.addWidget(self.dashboard_nav_btn)
        nav_layout.addWidget(make_separator())

        # ── HARDWARE LINK ──────────────────────────────────────────────────
        self.dashboard_nav_start_index = nav_layout.count()
        nav_layout.addWidget(section_label("01  HARDWARE LINK"))

        self.hw_status_label = QLabel("⬤  Not connected")
        self.hw_status_label.setStyleSheet(
            "color:#E25C5C; font-family:Consolas; font-size:12px; font-weight:bold;"
        )
        nav_layout.addWidget(self.hw_status_label)

        self.open_conn_btn = QPushButton("🔌  CONNECT HARDWARE")
        self.open_conn_btn.setStyleSheet(
            "QPushButton { background-color:#007acc; border:1px solid #1a9fff;"
            " color:white; font-weight:bold; padding:8px; }"
            "QPushButton:hover { background-color:#1a9fff; border-color:#1a9fff; }"
        )
        self.open_conn_btn.clicked.connect(self.open_connection_dialog)
        nav_layout.addWidget(self.open_conn_btn)

        nav_layout.addWidget(make_separator())

        # ── DISPLAY GEOMETRY ───────────────────────────────────────────────
        nav_layout.addWidget(section_label("02  DISPLAY GEOMETRY"))

        cal_row = QHBoxLayout()
        cal_row.addWidget(QLabel("Scale:"))
        self.cal_spin = QDoubleSpinBox()
        self.cal_spin.setRange(0.01, 100.0)
        self.cal_spin.setValue(1.0)
        self.cal_spin.setSingleStep(0.1)
        self.cal_spin.valueChanged.connect(self.update_calibration_scalar)
        cal_row.addWidget(self.cal_spin)
        nav_layout.addLayout(cal_row)

        nav_layout.addWidget(QLabel("Plot Height:"))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(110)
        self.size_slider.setMaximum(500)
        self.size_slider.setValue(self.current_plot_height)
        self.size_slider.valueChanged.connect(self.adjust_plot_sizes)
        nav_layout.addWidget(self.size_slider)

        # X axis max
        x_row = QHBoxLayout()
        self.x_max_label = QLabel("X Max (Hz):")
        x_row.addWidget(self.x_max_label)
        self.x_max_spin = QDoubleSpinBox()
        self.x_max_spin.setRange(1, SAMPLE_RATE / 2)
        self.x_max_spin.setDecimals(1)
        self.x_max_spin.setValue(SAMPLE_RATE / 2)
        self.x_max_spin.setSingleStep(1.0)
        self.x_max_spin.valueChanged.connect(self.update_axis_ranges)
        x_row.addWidget(self.x_max_spin)
        nav_layout.addLayout(x_row)

        # Y axis max
        y_row = QHBoxLayout()
        self.y_max_label = QLabel("Y Max (mag):")
        y_row.addWidget(self.y_max_label)
        self.y_max_spin = QDoubleSpinBox()
        self.y_max_spin.setRange(1, 100000)
        self.y_max_spin.setDecimals(0)
        self.y_max_spin.setValue(80)
        self.y_max_spin.setSingleStep(10)
        self.y_max_spin.valueChanged.connect(self.update_axis_ranges)
        y_row.addWidget(self.y_max_spin)
        nav_layout.addLayout(y_row)

        fit_btn = QPushButton("⊡  Fit to Data")
        fit_btn.setStyleSheet(
            "QPushButton { background-color:#252526; border:1px solid #3e3e42; color:#cccccc; }"
            "QPushButton:hover { background-color:#333333; border-color:#007acc; }"
        )
        fit_btn.clicked.connect(self.fit_axes_to_data)
        nav_layout.addWidget(fit_btn)

        auto_scale = QCheckBox("Auto Y scale")
        auto_scale.setChecked(False)
        auto_scale.toggled.connect(self.set_auto_scale)
        self.auto_scale_check = auto_scale
        nav_layout.addWidget(auto_scale)

        nav_layout.addWidget(section_label("03  DISPLAY PRESET"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Clinical", "Alpha Focus", "Full Spectrum", "Presentation"])
        self.preset_combo.currentTextChanged.connect(self.apply_display_preset)
        nav_layout.addWidget(self.preset_combo)

        nav_layout.addWidget(section_label("04  EEG BANDS"))
        self.band_checks = []
        for band_index, (name, _lo, _hi, color) in enumerate(EEG_BANDS):
            band_check = QCheckBox(name)
            band_check.setChecked(True)
            band_check.setStyleSheet(f"QCheckBox {{ color:{color}; }}")
            band_check.toggled.connect(
                lambda checked, index=band_index: self.set_band_visibility(index, checked)
            )
            self.band_checks.append(band_check)
            nav_layout.addWidget(band_check)

        nav_layout.addWidget(make_separator())

        # ── ELECTRODE MATRIX ───────────────────────────────────────────────
        nav_layout.addWidget(section_label("05  ELECTRODE MATRIX"))

        macro_row = QHBoxLayout()
        all_on  = QPushButton("All On")
        all_off = QPushButton("All Off")
        all_on.clicked.connect(lambda: self.set_all_checkboxes(True))
        all_off.clicked.connect(lambda: self.set_all_checkboxes(False))
        macro_row.addWidget(all_on)
        macro_row.addWidget(all_off)
        nav_layout.addLayout(macro_row)

        scroll = QScrollArea()
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        self.scroll_layout.setContentsMargins(0, 5, 0, 5)
        self.scroll_layout.setSpacing(4)

        self.checkboxes = []
        for i in range(NUM_CHANNELS):
            row_w  = QWidget()
            row_l  = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)

            cb = QCheckBox(f"({EEG_10_20_LABELS[i]}) Ch {i+1:02d}")
            cb.setChecked(i < 4)
            cb.stateChanged.connect(self.update_channel_visibility)
            row_l.addWidget(cb)
            self.checkboxes.append(cb)

            row_l.addStretch()

            col_btn = QPushButton()
            col_btn.setFixedSize(14, 14)
            col_btn.setStyleSheet(
                f"background-color:{self.channel_colors[i].name()}; border:1px solid #444; border-radius:2px;"
            )
            col_btn.clicked.connect(lambda _, idx=i: self.pick_custom_color(idx))
            row_l.addWidget(col_btn)
            self.color_buttons.append(col_btn)

            self.scroll_layout.addWidget(row_w)

        scroll.setWidget(scroll_content)
        scroll.setWidgetResizable(True)
        nav_layout.addWidget(scroll, stretch=1)

        parent_layout.insertWidget(0, self.nav_frame)

    # ══════════════════════════════════════════════════════════════════════════
    # Plot init
    # ══════════════════════════════════════════════════════════════════════════
    def _hz_to_bin(self, hz):
        return hz / BIN_TO_HZ

    def _x_max_bin(self):
        if not hasattr(self, "x_max_spin"):
            return DISPLAY_BINS
        if self.display_mode == "Raw Data":
            return self.x_max_spin.value()
        return self._hz_to_bin(self.x_max_spin.value())

    def _display_axis(self):
        return (
            np.arange(RAW_WINDOW_BINS, dtype=np.float32)
            if self.display_mode == "Raw Data" else HZ_AXIS
        )

    def _plot_data_matrix(self):
        return self.raw_data_matrix if self.display_mode == "Raw Data" else self.fft_data_matrix

    def _plot_channel_values(self, channel_idx):
        data = self._plot_data_matrix()[channel_idx]
        if self.display_mode != "Raw Data":
            return data

        visible_channels = list(self.curves)
        if self.is_split_view:
            return (data - 32768.0) / 24000.0 * 0.38 + 0.5
        try:
            lane = visible_channels.index(channel_idx)
        except ValueError:
            lane = channel_idx
        baseline = max(0, len(visible_channels) - lane - 1)
        centered = (data - 32768.0) / 24000.0
        return baseline + centered * 0.38

    def _y_max(self):
        if self.display_mode == "Raw Data":
            return 1.0 if self.is_split_view else max(1.0, float(len(self.curves)))
        return self.y_max_spin.value() if hasattr(self, "y_max_spin") else 80

    def init_plot_display(self):
        self.plot_widget = pg.PlotWidget()
        self._style_plot_widget(self.plot_widget, step_hz=8)

        self.band_region_items = add_band_lines(self.plot_widget, show=self.show_bands)
        self.plot_legend = self.plot_widget.addLegend(offset=(10, 10))
        self.plot_legend.setBrush(pg.mkBrush('#252526E6'))
        self.plot_legend.setPen(pg.mkPen('#3e3e42'))
        self._connect_plot_hover(self.plot_widget)

        self.plot_container_layout.addWidget(self._create_graph_header())
        self.plot_container_layout.addWidget(self.plot_widget)

    def _create_graph_header(self):
        header = QFrame()
        header.setStyleSheet(
            "QFrame { background:#252526; border-bottom:1px solid #3e3e42; }"
            "QLabel#GraphTitle { color:#ffffff; font-size:12px; font-weight:bold; }"
            "QLabel#GraphDetail { color:#858585; font-size:10px; }"
            "QLabel#GraphStatus { color:#00B37E; font-family:Consolas; font-size:10px; font-weight:bold; }"
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 7, 12, 7)
        header_layout.setSpacing(8)

        title = QLabel("LIVE SPECTRUM")
        title.setObjectName("GraphTitle")
        self.graph_title_label = title
        header_layout.addWidget(title)
        detail = QLabel("Magnitude by frequency  |  0–128 Hz")
        detail.setObjectName("GraphDetail")
        self.graph_detail_label = detail
        header_layout.addWidget(detail)
        header_layout.addStretch()
        self.graph_status_label = QLabel("READY")
        self.graph_status_label.setObjectName("GraphStatus")
        header_layout.addWidget(self.graph_status_label)
        return header

    def _connect_plot_hover(self, plot_widget, channel_idx=None):
        proxy = pg.SignalProxy(
            plot_widget.scene().sigMouseMoved,
            rateLimit=60,
            slot=lambda event: self._on_plot_mouse_moved(event, plot_widget, channel_idx),
        )
        self.plot_signal_proxies.append(proxy)

    def _on_plot_mouse_moved(self, event, plot_widget, channel_idx=None):
        if self.cursor_readout is None:
            return
        position = event[0] if isinstance(event, tuple) else event
        view_box = plot_widget.getPlotItem().vb
        if not view_box.sceneBoundingRect().contains(position):
            self.cursor_readout.setText("Hover a trace for details")
            return

        point = view_box.mapSceneToView(position)
        data_matrix = self._plot_data_matrix()
        max_index = data_matrix.shape[1] - 1
        bin_index = int(np.clip(round(point.x()), 0, max_index))
        if self.display_mode == "Raw Data":
            horizontal_text = f"sample {bin_index:03d}"
        else:
            horizontal_value = HZ_AXIS[bin_index]
            horizontal_text = f"{horizontal_value:.1f} Hz"
        if channel_idx is None:
            active = list(self.curves.keys())
            if not active:
                self.cursor_readout.setText("No active traces")
                return
            if self.display_mode == "Raw Data":
                channel_idx = max(
                    active,
                    key=lambda idx: abs(data_matrix[idx, bin_index] - 32768.0),
                )
            else:
                channel_idx = max(active, key=lambda idx: data_matrix[idx, bin_index])
        magnitude = data_matrix[channel_idx, bin_index]
        label = EEG_10_20_LABELS[channel_idx]
        value_name = "raw" if self.display_mode == "Raw Data" else "mag"
        self.cursor_readout.setText(
            f"{label}  |  {horizontal_text}  |  {magnitude:.2f} {value_name}"
        )

    def _ensure_peak_marker(self, channel_idx, plot_widget):
        key = (id(plot_widget), channel_idx)
        if key in self.peak_lines:
            return
        line = InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen('#00B37E', width=1, style=Qt.DashLine),
        )
        line.setVisible(self.show_peak_markers)
        plot_widget.addItem(line)
        label = pg.TextItem(color='#00B37E', anchor=(0.5, 1.0))
        label.setVisible(self.show_peak_markers)
        plot_widget.addItem(label)
        self.peak_lines[key] = line
        self.peak_labels[key] = label

    def _update_peak_marker(self, channel_idx, data_row, plot_widget):
        if self.display_mode == "Raw Data":
            return
        key = (id(plot_widget), channel_idx)
        self._ensure_peak_marker(channel_idx, plot_widget)
        peak_bin = int(np.argmax(data_row))
        peak_value = float(data_row[peak_bin])
        self.peak_lines[key].setPos(peak_bin)
        if self.display_mode == "Raw Data":
            self.peak_labels[key].setText(f"bin {peak_bin:03d}")
        else:
            self.peak_labels[key].setText(f"{peak_bin * BIN_TO_HZ:.1f} Hz")
        self.peak_labels[key].setPos(peak_bin, max(peak_value, self._y_max() * 0.92))

    def toggle_peak_markers(self):
        self.show_peak_markers = not self.show_peak_markers
        self.peak_btn.setText("⌖ Peaks ON" if self.show_peak_markers else "⌖ Peaks OFF")
        for item in list(self.peak_lines.values()) + list(self.peak_labels.values()):
            item.setVisible(self.show_peak_markers and self.display_mode != "Raw Data")

    def _mode_geometry_config(self):
        if self.display_mode == "Raw Data":
            return {
                "x_label": "X Max (samples):",
                "x_min": 1.0,
                "x_max": float(RAW_WINDOW_BINS),
                "x_decimals": 0,
                "x_step": 1.0,
                "x_value": float(RAW_WINDOW_BINS),
                "y_label": "Trace lanes:",
                "y_min": 0.0,
                "y_max": 65535.0,
                "y_decimals": 0,
                "y_step": 1000.0,
                "y_value": 65535.0,
            }
        return {
            "x_label": "X Max (Hz):",
            "x_min": 1.0,
            "x_max": float(SAMPLE_RATE / 2),
            "x_decimals": 1,
            "x_step": 1.0,
            "x_value": float(SAMPLE_RATE / 2),
            "y_label": "Y Max (mag):",
            "y_min": 1.0,
            "y_max": 100000.0,
            "y_decimals": 0,
            "y_step": 10.0,
            "y_value": 80.0,
        }

    def _style_plot_widget(self, plot_widget, step_hz):
        plot_widget.setBackground('#202024')
        left_label = 'Trace lanes' if self.display_mode == "Raw Data" else 'Magnitude'
        plot_widget.setLabel('left', left_label, **{'color': '#858585', 'font-size': '11px'})
        bottom_label = 'Bin index' if self.display_mode == "Raw Data" else 'Frequency (Hz)'
        plot_widget.setLabel('bottom', bottom_label, **{'color': '#858585', 'font-size': '11px'})
        plot_item = plot_widget.getPlotItem()
        pg.setConfigOptions(antialias=True)
        plot_item.setDownsampling(auto=True)
        plot_widget.showGrid(x=True, y=True, alpha=0.10)
        plot_widget.setXRange(0, self._x_max_bin(), padding=0)
        plot_widget.setYRange(0, self._y_max(), padding=0)
        plot_widget.setMouseEnabled(x=False, y=True)
        plot_widget.setMenuEnabled(False)

        for axis_name in ('left', 'bottom'):
            axis = plot_widget.getAxis(axis_name)
            axis.setPen(pg.mkPen('#3e3e42'))
            axis.setTextPen(pg.mkPen('#858585'))
            axis.setStyle(tickTextOffset=6)

        plot_widget.getAxis('bottom').setTicks([self._build_hz_ticks(step_hz=step_hz)])

    def _build_hz_ticks(self, step_hz=8):
        """Return major tick list [(bin_position, 'N Hz'), ...] for every step_hz Hz."""
        ticks = []
        if self.display_mode == "Raw Data":
            return [(index, str(index)) for index in range(0, RAW_WINDOW_BINS, 64)]
        hz = 0
        while hz <= SAMPLE_RATE / 2:
            ticks.append((self._hz_to_bin(hz), f"{hz}"))
            hz += step_hz
        return ticks

    def set_display_mode(self, mode):
        self.display_mode = mode
        self._send_stream_command(b"R" if mode == "Raw Data" else b"F")
        raw_mode = mode == "Raw Data"
        if raw_mode:
            self._raw_pending = np.empty((NUM_CHANNELS, 0), dtype=np.float32)
            self._raw_sample_accumulator = 0.0
            self.raw_data_matrix.fill(0.0)
        cfg = self._mode_geometry_config()

        self.x_max_label.setText(cfg["x_label"])
        self.y_max_label.setText(cfg["y_label"])

        self.x_max_spin.blockSignals(True)
        self.x_max_spin.setRange(cfg["x_min"], cfg["x_max"])
        self.x_max_spin.setDecimals(cfg["x_decimals"])
        self.x_max_spin.setSingleStep(cfg["x_step"])
        self.x_max_spin.setValue(cfg["x_value"])
        self.x_max_spin.blockSignals(False)

        self.y_max_spin.blockSignals(True)
        self.y_max_spin.setRange(cfg["y_min"], cfg["y_max"])
        self.y_max_spin.setDecimals(cfg["y_decimals"])
        self.y_max_spin.setSingleStep(cfg["y_step"])
        self.y_max_spin.setValue(cfg["y_value"])
        self.y_max_spin.blockSignals(False)

        plots = [self.plot_widget] if not self.is_split_view else [
            wrapper.plot_widget for wrapper in self.individual_widgets.values()
        ]
        for plot_widget in plots:
            plot_widget.setLabel(
                'left',
                'Trace lanes' if raw_mode else 'Magnitude',
                **{'color': '#858585', 'font-size': '11px'},
            )
            plot_widget.setLabel(
                'bottom',
                'Bin index' if raw_mode else 'Frequency (Hz)',
                **{'color': '#858585', 'font-size': '11px'},
            )
            plot_widget.setXRange(0, self._x_max_bin(), padding=0)
            plot_widget.getAxis('bottom').setTicks([self._build_hz_ticks()])

        self.graph_title_label.setText("RAW FRAME DATA" if raw_mode else "LIVE SPECTRUM")
        self.graph_detail_label.setText(
            "Analog sample stream  |  raw waveform view"
            if raw_mode else "Magnitude by frequency  |  0–128 Hz"
        )
        self.plot_legend.setVisible(not raw_mode)
        for item in list(self.peak_lines.values()) + list(self.peak_labels.values()):
            item.setVisible(self.show_peak_markers and not raw_mode)
        self._apply_band_visibility()
        plot_data = self._plot_data_matrix()
        for idx, curve in self.curves.items():
            curve.setPen(pg.mkPen(color=self.channel_colors[idx], width=1 if raw_mode else 2))
            curve.setData(x=self._display_axis(), y=self._plot_channel_values(idx))
        self.update_axis_ranges()

    def _send_stream_command(self, command):
        if not self.ser or not self.ser.is_open:
            return
        try:
            self.ser.write(command)
            self.ser.flush()
        except Exception:
            self._set_status("Stream command failed", "#E25C5C")

    def _configure_split_plot(self, pw):
        self._style_plot_widget(pw, step_hz=16)
        add_band_lines(pw, show=self.show_bands)

    # ══════════════════════════════════════════════════════════════════════════
    # Toolbar actions
    # ══════════════════════════════════════════════════════════════════════════
    def toggle_freeze(self):
        self.is_frozen = not self.is_frozen
        if self.is_frozen:
            self.freeze_btn.setText("▶ Resume")
            self.graph_status_label.setText("FROZEN")
            self.freeze_btn.setStyleSheet(
                "QPushButton { background-color:#094771; border:1px solid #007acc; color:white; }"
                "QPushButton:hover { background-color:#0e639c; }"
            )
        else:
            self.freeze_btn.setText("⏸ Freeze")
            self.graph_status_label.setText("LIVE")
            self.freeze_btn.setStyleSheet(
                "QPushButton { background-color:#252526; border:1px solid #3e3e42; color:#dcdcaa; }"
                "QPushButton:hover { background-color:#333333; border-color:#007acc; }"
            )

    def toggle_bands(self):
        self.show_bands = not self.show_bands
        self.band_btn.setText("⚡ Bands ON" if self.show_bands else "⚡ Bands OFF")
        self._apply_band_visibility()

    def _apply_band_visibility(self):
        plots = [self.plot_widget] if not self.is_split_view else [
            wrapper.plot_widget for wrapper in self.individual_widgets.values()
        ]
        for plot_widget in plots:
            for item in plot_widget.items():
                band_index = getattr(item, '_eeg_band_index', None)
                if band_index is not None:
                    item.setVisible(
                        self.display_mode != "Raw Data"
                        and self.show_bands
                        and self.band_visibility[band_index]
                    )

    def toggle_sidebar(self):
        self.nav_visible = not self.nav_visible
        self.nav_frame.setVisible(self.nav_visible)
        self.sidebar_btn.setText("◀ Hide Panel" if self.nav_visible else "▶ Show Panel")

    def toggle_recording(self):
        if not self._recording:
            dlg = RecordingDialog(EEG_10_20_LABELS, parent=self)
            dlg.set_active_channels(list(self.curves.keys()))
            if dlg.exec_() != QDialog.Accepted:
                return
            path = dlg.output_path
            if not path:
                return

            self._record_channels = dlg.selected_channels
            self._record_duration = dlg.duration_seconds
            self._record_elapsed  = 0.0

            try:
                self._csv_file = open(path, 'w', newline='')
            except OSError as error:
                self._set_status(f"Recording failed: {error}", "#E25C5C")
                return
            self._csv_writer = csv.writer(self._csv_file)
            value_prefix = "raw" if self.display_mode == "Raw Data" else "bin"
            sample_count = RAW_WINDOW_BINS if self.display_mode == "Raw Data" else DISPLAY_BINS
            header = ["timestamp"] + [
                f"{EEG_10_20_LABELS[ch]}_{value_prefix}{b}"
                for ch in self._record_channels
                for b in range(sample_count)
            ]
            self._csv_writer.writerow(header)
            self._recording = True

            # Countdown label suffix
            if self._record_duration:
                self.record_btn.setText(f"⏹ {self._record_duration}s left")
            else:
                self.record_btn.setText("⏹ Stop Rec")
            self.record_btn.setStyleSheet(
                "QPushButton { background-color:#094771; border:1px solid #007acc; color:white; }"
                "QPushButton:hover { background-color:#0e639c; }"
            )
        else:
            self._stop_recording()

    def _stop_recording(self):
        self._recording = False
        if self._csv_file:
            self._csv_file.close()
            self._csv_file   = None
            self._csv_writer = None
        self.record_btn.setText("⏺ Record")
        self.record_btn.setStyleSheet(
            "QPushButton { background-color:#252526; border:1px solid #3e3e42; color:#cccccc; }"
            "QPushButton:hover { background-color:#333333; border-color:#007acc; }"
        )

    def export_plot_image(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Snapshot", "eeg_snapshot.png", "PNG Image (*.png)"
        )
        if path:
            target = self.split_scroll.viewport() if self.is_split_view else self.plot_widget
            if not target.grab().save(path, "PNG"):
                self.telemetry_label.setText("Export Failed")

    def _on_col_count_changed(self, val):
        self.split_columns = val
        if self.is_split_view:
            self._rebuild_grid_positions()

    def _rebuild_grid_positions(self):
        active_count = 0
        for idx in self.channel_render_order:
            if self.checkboxes[idx].isChecked() and idx in self.individual_widgets:
                w   = self.individual_widgets[idx]
                row = active_count // self.split_columns
                col = active_count %  self.split_columns
                self.grid_layout.removeWidget(w)
                self.grid_layout.addWidget(w, row, col)
                active_count += 1

    # ══════════════════════════════════════════════════════════════════════════
    # Axis / display controls
    # ══════════════════════════════════════════════════════════════════════════
    def adjust_plot_sizes(self, val):
        self.current_plot_height = val
        if self.is_split_view:
            for w in self.individual_widgets.values():
                w.setFixedHeight(val)

    def update_axis_ranges(self):
        x_bin = self._x_max_bin()
        y_max = self._y_max()
        if self.is_split_view:
            for w in self.individual_widgets.values():
                w.plot_widget.setXRange(0, x_bin, padding=0)
                w.plot_widget.setYRange(0, y_max, padding=0)
        else:
            self.plot_widget.setXRange(0, x_bin, padding=0)
            self.plot_widget.setYRange(0, y_max, padding=0)

    def set_auto_scale(self, enabled):
        self.auto_scale_y = enabled
        if enabled:
            self._apply_auto_y_range()

    def _apply_auto_y_range(self):
        active = list(self.curves.keys())
        if not active:
            return
        if self.display_mode == "Raw Data":
            self.update_axis_ranges()
            return
        maximum = float(np.max(self.fft_data_matrix[active]))
        y_max = max(1.0, maximum * 1.12)
        plots = [self.plot_widget] if not self.is_split_view else [
            wrapper.plot_widget for wrapper in self.individual_widgets.values()
        ]
        for plot_widget in plots:
            plot_widget.setYRange(0, y_max, padding=0)

    def set_band_visibility(self, band_index, visible):
        self.band_visibility[band_index] = visible
        self._apply_band_visibility()

    def apply_display_preset(self, preset_name):
        if self.display_mode == "Raw Data":
            presets = {
                "Clinical": (DISPLAY_BINS, 1.0, 180),
                "Alpha Focus": (DISPLAY_BINS, 1.0, 210),
                "Full Spectrum": (DISPLAY_BINS, 1.0, 180),
                "Presentation": (DISPLAY_BINS, 1.0, 260),
            }
        else:
            presets = {
                "Clinical": (45.0, 80.0, 180),
                "Alpha Focus": (20.0, 80.0, 210),
                "Full Spectrum": (SAMPLE_RATE / 2, 80.0, 180),
                "Presentation": (45.0, 120.0, 260),
            }
        x_max, y_max, plot_height = presets.get(preset_name, presets["Clinical"])
        self.x_max_spin.blockSignals(True)
        self.y_max_spin.blockSignals(True)
        self.x_max_spin.setValue(x_max)
        self.y_max_spin.setValue(y_max)
        self.x_max_spin.blockSignals(False)
        self.y_max_spin.blockSignals(False)
        self.size_slider.setValue(plot_height)
        show_all_bands = preset_name != "Alpha Focus"
        for band_index, band_check in enumerate(self.band_checks):
            band_check.setChecked(show_all_bands or band_index == 2)
        self.show_bands = True
        self.band_btn.setText("⚡ Bands ON")
        self._apply_band_visibility()

    def fit_axes_to_data(self):
        active = list(self.curves.keys())
        if not active:
            return
        if self.display_mode == "Raw Data":
            self.x_max_spin.blockSignals(True)
            self.y_max_spin.blockSignals(True)
            self.x_max_spin.setValue(RAW_WINDOW_BINS)
            self.y_max_spin.setValue(max(1, len(active)))
            self.x_max_spin.blockSignals(False)
            self.y_max_spin.blockSignals(False)
            self.update_axis_ranges()
            return
        visible = self.fft_data_matrix[active]
        y_min   = float(visible.min())
        y_max   = float(visible.max()) + max((float(visible.max()) - float(visible.min())) * 0.05, 1.0)

        self.x_max_spin.blockSignals(True)
        self.y_max_spin.blockSignals(True)
        self.x_max_spin.setValue(RAW_WINDOW_BINS if self.display_mode == "Raw Data" else SAMPLE_RATE / 2)
        if self.display_mode == "Raw Data":
            self.y_max_spin.setValue(round(max(y_max, 1.0), 2))
        else:
            self.y_max_spin.setValue(round(y_max))
        self.x_max_spin.blockSignals(False)
        self.y_max_spin.blockSignals(False)

        x_bin = self._x_max_bin()
        if self.is_split_view:
            for w in self.individual_widgets.values():
                w.plot_widget.setXRange(0, x_bin, padding=0)
                w.plot_widget.setYRange(y_min, y_max, padding=0)
        else:
            self.plot_widget.setXRange(0, x_bin, padding=0)
            self.plot_widget.setYRange(y_min, y_max, padding=0)

    def update_calibration_scalar(self, val):
        self.calibration_scalar = val

    # ══════════════════════════════════════════════════════════════════════════
    # Channel visibility / drag-drop
    # ══════════════════════════════════════════════════════════════════════════
    def set_active_drag_origin(self, ch_idx):
        self.active_drag_origin_idx = ch_idx

    def set_all_checkboxes(self, state):
        for cb in self.checkboxes:
            cb.blockSignals(True)
            cb.setChecked(state)
            cb.blockSignals(False)
        self.update_channel_visibility()

    def pick_custom_color(self, ch_idx):
        color = QColorDialog.getColor(self.channel_colors[ch_idx], self, "Choose Trace Color")
        if color.isValid():
            self.channel_colors[ch_idx] = color
            self.color_buttons[ch_idx].setStyleSheet(
                f"background-color:{color.name()}; border:1px solid #fff; border-radius:2px;"
            )
            if ch_idx in self.curves:
                self.curves[ch_idx].setPen(pg.mkPen(color=color, width=2))

    def update_channel_visibility(self):
        active_count = 0
        for idx in self.channel_render_order:
            cb = self.checkboxes[idx]
            if cb.isChecked():
                if self.is_split_view:
                    if self.grid_layout is None:
                        continue
                    if idx not in self.individual_widgets:
                        pw = pg.PlotWidget()
                        self._configure_split_plot(pw)
                        self._connect_plot_hover(pw, idx)
                        wrapper = DraggablePlotWrapper(idx, pw, initial_height=self.current_plot_height)
                        row = active_count // self.split_columns
                        col = active_count %  self.split_columns
                        self.grid_layout.addWidget(wrapper, row, col)
                        self.individual_widgets[idx] = wrapper
                        self.curves[idx] = pw.plot(
                            x=self._display_axis(),
                            y=self._plot_channel_values(idx),
                            pen=pg.mkPen(color=self.channel_colors[idx], width=2),
                        )
                        self._ensure_peak_marker(idx, pw)
                    else:
                        if not self.is_animating_layout:
                            w   = self.individual_widgets[idx]
                            row = active_count // self.split_columns
                            col = active_count %  self.split_columns
                            self.grid_layout.addWidget(w, row, col)
                    active_count += 1
                else:
                    if idx not in self.curves:
                        self.curves[idx] = self.plot_widget.plot(
                            x=self._display_axis(),
                            y=self._plot_channel_values(idx),
                            pen=pg.mkPen(color=self.channel_colors[idx], width=2),
                        )
                        self.plot_legend.addItem(
                            self.curves[idx], EEG_10_20_LABELS[idx]
                        )
                        self._ensure_peak_marker(idx, self.plot_widget)
            else:
                if self.is_split_view and idx in self.individual_widgets:
                    if self.grid_layout is not None:
                        self.grid_layout.removeWidget(self.individual_widgets[idx])
                    self.individual_widgets[idx].setParent(None)
                    del self.individual_widgets[idx]
                    if idx in self.curves:
                        del self.curves[idx]
                elif not self.is_split_view and idx in self.curves:
                    self.plot_widget.removeItem(self.curves[idx])
                    self.plot_legend.removeItem(self.curves[idx])
                    del self.curves[idx]

    # ══════════════════════════════════════════════════════════════════════════
    # Hardware
    # ══════════════════════════════════════════════════════════════════════════
    def open_connection_dialog(self):
        dlg = ConnectionDialog(self.ser, parent=self)
        dlg.exec_()

        self.ser = dlg.serial_port

        if self.ser and self.ser.is_open:
            # Every connection change starts a clean protocol session.
            self.ser.reset_input_buffer()
            self._serial_buf.clear()
            self._raw_pending = np.empty((NUM_CHANNELS, 0), dtype=np.float32)
            self._raw_sample_accumulator = 0.0
            self._frame_ready = False
            self._last_seq       = -1
            self._crc_errors     = 0
            self.packet_count    = 0
            self.dropped_packets = 0
            self.session_runtime = 0.0
            self.last_update_time = time.time()
            self._send_stream_command(b"R" if self.display_mode == "Raw Data" else b"F")
            port = dlg.selected_port_name
            baud = dlg.selected_baud
            self.hw_status_label.setText(f"⬤  {port} @ {baud}")
            self.hw_status_label.setStyleSheet(
                "color:#00B37E; font-family:Consolas; font-size:12px; font-weight:bold;"
            )
            self.open_conn_btn.setText("🔌  Manage Connection")
            self._set_status(f"Connected: {port} @ {baud}", "#00B37E")
        else:
            self._serial_buf.clear()
            self.hw_status_label.setText("⬤  Not connected")
            self.hw_status_label.setStyleSheet(
                "color:#E25C5C; font-family:Consolas; font-size:12px; font-weight:bold;"
            )
            self.open_conn_btn.setText("🔌  Open Connection Manager")
            self._set_status("Disconnected — waiting for hardware.", "#FFB800")

    def _set_status(self, text, color):
        self.telemetry_label.setText(text)
        self.telemetry_label.setStyleSheet(
            f"color:{color}; font-family:'Consolas'; font-size:12px; font-weight:bold;"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Split-view / drag-drop machinery
    # ══════════════════════════════════════════════════════════════════════════
    def toggle_split_view(self):
        self.is_split_view = not self.is_split_view

        for i in reversed(range(self.plot_container_layout.count())):
            w = self.plot_container_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

        self.curves.clear()
        self.individual_widgets.clear()
        self.band_region_items.clear()
        self.plot_signal_proxies.clear()
        self.peak_lines.clear()
        self.peak_labels.clear()

        if self.is_split_view:
            self.split_btn.setText("Merge View")
            self.revert_layout_btn.setEnabled(True)
            self.split_scroll = QScrollArea()
            self.split_scroll.setWidgetResizable(True)
            scroll_content = QWidget()
            scroll_content.setStyleSheet("background-color:#121214;")
            scroll_content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.grid_layout = RearrangeableGridLayout(scroll_content)
            self.grid_layout.setSpacing(10)
            self.grid_layout.setSizeConstraint(QGridLayout.SetMinAndMaxSize)
            scroll_content.setAcceptDrops(True)
            scroll_content.dragEnterEvent = self.grid_layout.handle_drag_enter
            scroll_content.dragMoveEvent  = self.grid_layout.handle_drag_move
            scroll_content.dropEvent      = self.grid_layout.handle_drop
            self.split_scroll.setWidget(scroll_content)
            self.plot_container_layout.addWidget(self._create_graph_header())
            self.plot_container_layout.addWidget(self.split_scroll)
        else:
            self.split_btn.setText("Split View")
            self.revert_layout_btn.setEnabled(False)
            self.grid_layout = None
            self.init_plot_display()

        self.update_channel_visibility()

    def live_swap_channels(self, source, target):
        if self.is_animating_layout or self.grid_layout is None:
            return
        si = self.channel_render_order.index(source)
        ti = self.channel_render_order.index(target)
        tw = self.individual_widgets.get(target)
        sw = self.individual_widgets.get(source)
        if not tw or not sw:
            return
        t_orig = tw.pos()
        self.channel_render_order[si], self.channel_render_order[ti] = \
            self.channel_render_order[ti], self.channel_render_order[si]
        self.is_animating_layout = True
        self.update_grid_positions_with_animation(source, target, t_orig)

    def revert_channels_to_original(self):
        if self.is_animating_layout or self.grid_layout is None:
            return
        if self.channel_render_order == list(range(NUM_CHANNELS)):
            return
        self.is_animating_layout = True
        self.grid_layout.setEnabled(False)
        self.channel_render_order = list(range(NUM_CHANNELS))
        active_count = 0
        for idx in self.channel_render_order:
            if self.checkboxes[idx].isChecked() and idx in self.individual_widgets:
                w    = self.individual_widgets[idx]
                row  = active_count // self.split_columns
                col  = active_count %  self.split_columns
                dest = self.grid_layout.cellRect(row, col).topLeft()
                if w.pos() != dest:
                    w.animate_to_pos(dest)
                    w.trigger_drop_flash()
                active_count += 1
        QTimer.singleShot(260, self.re_engage_grid_layout)

    def update_grid_positions_with_animation(self, dragging_id, bumped_id, target_pre_pos):
        if self.grid_layout is None:
            return
        active_count = 0
        self.grid_layout.setEnabled(False)
        for idx in self.channel_render_order:
            if self.checkboxes[idx].isChecked() and idx in self.individual_widgets:
                w    = self.individual_widgets[idx]
                row  = active_count // self.split_columns
                col  = active_count %  self.split_columns
                dest = self.grid_layout.cellRect(row, col).topLeft()
                if idx == bumped_id:
                    w.animate_to_pos(dest)
                    w.trigger_drop_flash()
                elif idx == dragging_id:
                    w.move(dest)
                else:
                    if w.pos() != dest:
                        w.move(dest)
                active_count += 1
        QTimer.singleShot(260, self.re_engage_grid_layout)

    def re_engage_grid_layout(self):
        if self.grid_layout is None:
            return
        self.grid_layout.setEnabled(True)
        active_count = 0
        for idx in self.channel_render_order:
            if self.checkboxes[idx].isChecked() and idx in self.individual_widgets:
                w   = self.individual_widgets[idx]
                row = active_count // self.split_columns
                col = active_count %  self.split_columns
                self.grid_layout.removeWidget(w)
                self.grid_layout.addWidget(w, row, col)
                active_count += 1
        self.is_animating_layout = False

    def finalize_drop_execution(self, source_idx):
        if source_idx in self.individual_widgets:
            self.individual_widgets[source_idx].trigger_drop_flash()
        self.active_drag_origin_idx = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    # ══════════════════════════════════════════════════════════════════════════
    # Main update loop
    # ══════════════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════════════════
    # Binary frame parser
    # ══════════════════════════════════════════════════════════════════════════
    def _parse_serial_frames(self):
        """
        Consume as many complete frames as possible from _serial_buf.

        Frame layout (matches STM32 firmware):
          [0xAA][0x55]  sync
          [seq lo][seq hi]  uint16 LE
          [nch]  uint8
          [nbins lo][nbins hi]  uint16 LE
          [data]  nch * nbins * uint16 LE  (magnitudes 0-65535)
          [crc lo][crc hi]  uint16 LE  CRC-16/CCITT over data only

        Sync-hunt: scan forward byte-by-byte until 0xAA 0x55 found.
        """
        buf = self._serial_buf

        while True:
            # Hunt for sync
            if len(buf) < 2:
                break
            sync_pos = -1
            for i in range(len(buf) - 1):
                if buf[i] == PROTO_SYNC0 and buf[i+1] == PROTO_SYNC1:
                    sync_pos = i
                    break
            if sync_pos == -1:
                # No sync found — keep last byte in case it's the first sync byte
                self._serial_buf = bytearray(buf[-1:])
                return
            if sync_pos > 0:
                # Discard garbage before sync
                buf = buf[sync_pos:]

            if len(buf) < PROTO_FRAME_SIZE:
                break   # Wait for more bytes

            # Parse header
            seq    = struct.unpack_from('<H', buf, 2)[0]
            nch    = buf[4]
            nbins  = struct.unpack_from('<H', buf, 5)[0]

            if nch != NUM_CHANNELS or nbins != DISPLAY_BINS:
                # Header mismatch — skip this sync and keep hunting
                buf = buf[2:]
                continue

            # Extract data region and CRC
            data_start = PROTO_HDR_SIZE
            data_end   = data_start + PROTO_DATA_BYTES
            data_bytes = buf[data_start:data_end]
            rx_crc     = struct.unpack_from('<H', buf, data_end)[0]
            calc_crc   = self._crc16(data_bytes)

            if rx_crc != calc_crc:
                self._crc_errors += 1
                self.dropped_packets += 1
                buf = buf[2:]   # skip sync, try next
                continue

            # Reject duplicates and old frames before updating stream state.
            if self._last_seq >= 0:
                sequence_delta = (seq - self._last_seq) & 0xFFFF
                if sequence_delta == 0 or sequence_delta > 0x8000:
                    buf = buf[PROTO_FRAME_SIZE:]
                    continue
                if sequence_delta > 1:
                    self.dropped_packets += sequence_delta - 1

            # Good frame
            self.packet_count += 1
            self._last_seq = seq

            # Unpack uint16 → float32. In raw-data mode the firmware sends analog
            # waveform samples in the ADC-like 0..65535 range. In FFT mode it sends
            # magnitude values that must be normalised for plotting.
            raw = np.frombuffer(data_bytes, dtype='<u2').astype(np.float32)
            if self.display_mode == "Raw Data":
                raw_frame = raw.reshape(NUM_CHANNELS, DISPLAY_BINS)
                sample_count = RAW_SAMPLES_PER_FRAME
                new_samples = raw_frame[:, -sample_count:]
                self._raw_pending = np.concatenate((self._raw_pending, new_samples), axis=1)
            else:
                raw *= PROTO_SCALE * self.calibration_scalar * 65535.0
                self.fft_data_matrix[:] = raw.reshape(NUM_CHANNELS, DISPLAY_BINS)
            self._frame_ready = True

            buf = buf[PROTO_FRAME_SIZE:]

        self._serial_buf = bytearray(buf)

    @staticmethod
    def _crc16(data: bytes) -> int:
        """CRC-16/CCITT  poly=0x1021  init=0xFFFF — matches firmware."""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                crc = (crc << 1) ^ 0x1021 if crc & 0x8000 else crc << 1
            crc &= 0xFFFF
        return crc

    def _poll_serial(self):
        if not self.ser or not self.ser.is_open:
            return

        try:
            waiting = self.ser.in_waiting
            if waiting > 0:
                self._serial_buf += self.ser.read(waiting)
                self._parse_serial_frames()
        except Exception:
            pass

    def update_plots(self):
        now = time.time()
        self.frame_count += 1

        dt = now - self.last_update_time
        self.session_runtime  += dt
        self.last_update_time  = now

        total_cs    = int(self.session_runtime * 100)
        centiseconds = total_cs % 100
        total_s      = total_cs // 100
        self.runtime_label.setText(
            f"T+ {total_s // 60:02d}:{total_s % 60:02d}.{centiseconds:02d}"
        )

        if hasattr(self, "home_connection_value"):
            connected = bool(self.ser and self.ser.is_open)
            self.home_connection_value.setText("ONLINE" if connected else "OFFLINE")
            self.home_connection_value.setStyleSheet(
                f"color:{'#4EC9B0' if connected else '#F48771'}; font-size:22px; font-weight:bold;"
            )
            self.home_status_label.setText("●  DEVICE ONLINE" if connected else "●  SYSTEM READY")
            self.home_status_label.setStyleSheet(
                f"color:{'#4EC9B0' if connected else '#dcdcaa'};"
                f" background:{'#183b3b' if connected else '#3b351d'};"
                f" border:1px solid {'#245b59' if connected else '#665b2a'};"
                " border-radius:3px; padding:6px 10px; font-size:10px; font-weight:bold;"
            )
            self.home_connection_detail.setText(
                "Streaming from serial device" if connected else "Scan a port to begin"
            )
            active_channels = sum(cb.isChecked() for cb in self.checkboxes)
            self.home_channels_value.setText(f"{active_channels:02d} / {NUM_CHANNELS}")
            self.home_packets_value.setText(f"{self.packet_count:,}")
            self.home_packets_detail.setText(f"CRC errors: {self._crc_errors}")
            self.home_runtime_value.setText(f"{total_s // 3600:02d}:{(total_s % 3600) // 60:02d}:{total_s % 60:02d}")
            self.home_runtime_detail.setText(
                "Acquisition active" if connected else "Ready for acquisition"
            )

        if now - self.last_fps_time >= 1.0:
            if self.ser and self.ser.is_open:
                loss = (self.dropped_packets / max(1, self.packet_count)) * 100
                self.telemetry_label.setText(
                    f"Live | {self.frame_count} Hz | Loss: {loss:.1f}% | CRC err: {self._crc_errors}"
                )
                color = "#E25C5C" if loss > 5 else "#00B37E"
                self.telemetry_label.setStyleSheet(
                    f"color:{color}; font-family:'Consolas'; font-size:12px; font-weight:bold;"
                )
            self.frame_count   = 0
            self.last_fps_time = now

        if self.is_frozen:
            self._frame_ready = False
            self._raw_pending = np.empty((NUM_CHANNELS, 0), dtype=np.float32)
            return

        has_new_frame = self._frame_ready
        if has_new_frame:
            self._frame_ready = False

        if self.display_mode == "Raw Data" and self._raw_pending.shape[1] > 0:
            self._raw_sample_accumulator += dt * SAMPLE_RATE
            sample_count = min(
                int(self._raw_sample_accumulator),
                self._raw_pending.shape[1],
            )
            if sample_count > 0:
                self._raw_sample_accumulator -= sample_count
                self.raw_data_matrix[:, :-sample_count] = self.raw_data_matrix[:, sample_count:]
                self.raw_data_matrix[:, -sample_count:] = self._raw_pending[:, :sample_count]
                self._raw_pending = self._raw_pending[:, sample_count:]
                has_new_frame = True

        if has_new_frame:
            plot_data = self._plot_data_matrix()
            for idx, curve in self.curves.items():
                curve.setData(x=self._display_axis(), y=self._plot_channel_values(idx))

            if self.auto_scale_y:
                self._apply_auto_y_range()
        if hasattr(self, "graph_status_label"):
            source = "LIVE" if self.ser and self.ser.is_open else "PREVIEW"
            self.graph_status_label.setText(f"{source}  /  {len(self.curves):02d} TRACES")

        if has_new_frame and self._recording and self._csv_writer:
            record_matrix = self._plot_data_matrix()
            row = [f"{now:.4f}"] + [
                record_matrix[ch, b]
                for ch in self._record_channels
                for b in range(record_matrix.shape[1])
            ]
            self._csv_writer.writerow(row)
            if self._record_duration is not None:
                self._record_elapsed += dt
                remaining = max(0, self._record_duration - self._record_elapsed)
                self.record_btn.setText(f"⏹ {remaining:.0f}s left")
                if remaining <= 0:
                    self._stop_recording()

        self._peak_tick += 1
        if self._peak_tick >= 6:
            self._peak_tick = 0
            indicator_matrix = self._plot_data_matrix()
            if self.is_split_view:
                for idx, wrapper in self.individual_widgets.items():
                    wrapper.update_peak(indicator_matrix[idx], self.display_mode)
                    wrapper.update_quality(indicator_matrix[idx])
                    self._update_peak_marker(idx, indicator_matrix[idx], wrapper.plot_widget)
            else:
                for idx in self.curves:
                    self._update_peak_marker(idx, indicator_matrix[idx], self.plot_widget)

    def closeEvent(self, event):
        if self._recording:
            self._stop_recording()
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        event.accept()


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_UseOpenGLES)
    app    = QApplication(sys.argv)
    window = MultiChannelFFTApp()
    window.show()
    sys.exit(app.exec_())