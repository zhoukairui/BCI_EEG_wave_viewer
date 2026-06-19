"""
EEG Unified - 多设备兼容脑电波形显示 (BLE直连版)
支持：EEG_001 (1ch→2ch) / EEG_005 (5ch) / EEG_008 (8ch)
功能：扫描/连接蓝牙 → 按所选设备解析数据 → 波形绘制 → 滤波/量程
"""

import sys
import asyncio
import datetime
import os
import numpy as np
from scipy import signal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QListWidget, QListWidgetItem,
                             QLabel, QComboBox, QLineEdit, QGroupBox, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QTime, QDateTime
import pyqtgraph as pg
from bleak import BleakScanner, BleakClient

# ═══════════════════════════════════════════════════════════════════
# 设备配置表（新增设备类型只需在此添加条目）
# ═══════════════════════════════════════════════════════════════════

DEVICE_PROFILES = {
    'EEG_001': {
        'name':       'EEG 001 (1ch→2ch)',
        'n_channels': 2,
        'colors':     ['#e6194b', '#4363d8'],
        'header':     bytes([0xFF, 0xAA, 0x55]),
        'header_len': 3,
        'payload_start': 4,
        'payload_len':   75,
        'n_samples':  25,
        'samples_per_row': 1,        # 按行排列时每行几个样本（reshape用）
        'notify_uuid_prefix': '0000ffe1',
        'write_uuid_prefix':  '0000ffe3',
        'start_cmd':  bytearray([0x62]),
        'stop_cmd':   bytearray([0x63]),
        'save_suffix': '_EEG001_ble_data.txt',
    },
    'EEG_005': {
        'name':       'EEG 005 (5ch)',
        'n_channels': 5,
        'colors':     ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4'],
        'header':     bytes([0xFF]),
        'header_len': 1,
        'payload_start': 2,
        'payload_len':   75,
        'n_samples':  25,
        'samples_per_row': 5,
        'notify_uuid_prefix': '0000ffe1',
        'write_uuid_prefix':  '0000ffe1',
        'start_cmd':  bytearray([0x62]),
        'stop_cmd':   bytearray([0x63]),
        'save_suffix': '_EEG005_ble_data.txt',
    },
    'EEG_008': {
        'name':       'EEG 008 (8ch)',
        'n_channels': 8,
        'colors':     ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
                       '#911eb4', '#42d4f4', '#f032e6', '#469990'],
        'header':     bytes([0x23, 0x23, 0x3E]),
        'header_len': 3,
        'payload_start': 10,
        'payload_len':   120,
        'n_samples':  40,
        'samples_per_row': 8,
        'notify_uuid_prefix': '0000ffe1',
        'write_uuid_prefix':  '0000ffe1',
        'start_cmd':  bytearray([0x41, 0x48, 0x42, 0x2B, 0x53, 0x54,
                                 0x41, 0x52, 0x54, 0x3D, 0x31, 0x0D, 0x0A]),
        'stop_cmd':   bytearray([0x41, 0x48, 0x42, 0x2B, 0x53, 0x54,
                                 0x4F, 0x50, 0x3D, 0x30, 0x0D, 0x0A]),
        'save_suffix': '_EEG008_ble_data.txt',
    },
}

DEFAULT_DEVICE = 'EEG_005'
SAMPLE_RATE = 250
SHOW_SECONDS = 10
PLOT_POINTS = SAMPLE_RATE * SHOW_SECONDS
time_axis = np.linspace(-SHOW_SECONDS, 0, PLOT_POINTS)


# ═══════════════════════════════════════════════════════════════════
# 通用 24-bit 数据解析
# ═══════════════════════════════════════════════════════════════════

def bytes_to_eeghex(b1, b2, b3):
    return (b1 << 16) | (b2 << 8) | b3


def calculate_eegv(b1, b2, b3):
    h = bytes_to_eeghex(b1, b2, b3)
    if b1 >= 0x80:
        h -= 0x1000000
    return (h * 4.5) / (8388607 * 24) * 1000000


def parse_eeg_packet(data_bytes: bytes, cfg: dict):
    """
    通用包解析器。根据 cfg 中的协议参数自动适配。
    """
    header = cfg['header']
    hlen = cfg['header_len']
    pstart = cfg['payload_start']
    plen = cfg['payload_len']
    n_samp = cfg['n_samples']
    sp_row = cfg['samples_per_row']   # EEG_001=1, 005=5, 008=8

    if len(data_bytes) < pstart + plen:
        return None
    if data_bytes[:hlen] != header:
        return None

    raw = []
    for i in range(pstart, pstart + plen, 3):
        if i + 2 < pstart + plen:
            raw.append(calculate_eegv(data_bytes[i], data_bytes[i+1], data_bytes[i+2]))

    if len(raw) < n_samp:
        return None

    data = np.array(raw[:n_samp])
    n_cols = sp_row
    n_rows = n_samp // n_cols

    if sp_row == 1:
        # EEG_001: 1 物理通道，复制为 2 通道
        data_1ch = data.reshape(-1, 1)  # (n_samp, 1)
        return np.append(data_1ch, data_1ch, axis=1)  # (n_rows, 2)
    else:
        return data.reshape(n_rows, n_cols)  # (5, 5) or (5, 8)


# ═══════════════════════════════════════════════════════════════════
# 实时滤波器
# ═══════════════════════════════════════════════════════════════════

class DigitalFilter:
    def __init__(self, b, a):
        self._bs, self._as = b, a
        self._xs = [0.0] * len(b)
        self._ys = [0.0] * (len(a) - 1)

    def process(self, x):
        if np.isnan(x):
            return x
        self._xs.insert(0, x)
        self._xs.pop()
        y = (np.dot(self._bs, self._xs) / self._as[0] -
             np.dot(self._as[1:], self._ys))
        self._ys.insert(0, y)
        self._ys.pop()
        return y

    def __call__(self, x):
        return self.process(x)

    def reset(self):
        self._xs = [0.0] * len(self._bs)
        self._ys = [0.0] * (len(self._as) - 1)


# ═══════════════════════════════════════════════════════════════════
# BLE 工作线程
# ═══════════════════════════════════════════════════════════════════

class BLEWorker(QThread):
    devices_found = pyqtSignal(list)
    connection_changed = pyqtSignal(bool)
    eeg_data_received = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.client = None
        self.loop = asyncio.new_event_loop()
        self.file = None
        self.file_path = None
        self._notify_char_uuid = None
        self._write_char_uuid = None
        self._cfg = None          # 当前设备配置

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def set_config(self, cfg):
        self._cfg = cfg

    # ── 扫描 ──
    def scan_devices(self):
        asyncio.run_coroutine_threadsafe(self._scan(), self.loop)

    async def _scan(self):
        try:
            devices = await BleakScanner.discover(timeout=5.0)
            valid = [d for d in devices if d.name and d.name.strip()]
            self.devices_found.emit(valid)
        except Exception as e:
            print(f"[BLE] 扫描失败: {e}")
            self.devices_found.emit([])

    # ── 连接 ──
    def find_and_connect(self, address):
        asyncio.run_coroutine_threadsafe(self._connect(address), self.loop)

    async def _connect(self, address):
        try:
            self.client = BleakClient(address, loop=self.loop)
            await self.client.connect(timeout=15.0)
            cfg = self._cfg
            if cfg is None:
                raise RuntimeError("设备配置未设置")
            notify_pre = cfg['notify_uuid_prefix']
            write_pre  = cfg['write_uuid_prefix']
            print(f"[BLE] 已连接: {address}")

            for service in self.client.services:
                for char in service.characteristics:
                    cu = char.uuid.lower()
                    if cu.startswith(notify_pre):
                        self._notify_char_uuid = char.uuid
                        await self.client.start_notify(
                            char.uuid, self._notification_handler
                        )
                        print(f"[BLE] 已订阅通知: {char.uuid}")
                    if cu.startswith(write_pre):
                        self._write_char_uuid = char.uuid
                        print(f"[BLE] 写特征: {char.uuid}")

            if self._notify_char_uuid is None:
                print("[BLE] 未找到指定通知特征，搜索可通知特征")
                for service in self.client.services:
                    for char in service.characteristics:
                        if "notify" in char.properties:
                            self._notify_char_uuid = char.uuid
                            if self._write_char_uuid is None:
                                self._write_char_uuid = char.uuid
                            await self.client.start_notify(
                                char.uuid, self._notification_handler
                            )
                            print(f"[BLE] 已订阅通知: {char.uuid}")
                            break
                    if self._notify_char_uuid:
                        break
            if self._write_char_uuid is None:
                self._write_char_uuid = self._notify_char_uuid

            self.connection_changed.emit(True)

        except Exception as e:
            print(f"[BLE] 连接失败: {e}")
            self.connection_changed.emit(False)

    # ── 数据回调 ──
    def _notification_handler(self, sender, data):
        try:
            if not data:
                return
            if self.file is not None:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.file.write(f"{ts},{data.hex()}\n")
                self.file.flush()
            parsed = parse_eeg_packet(data, self._cfg)
            if parsed is not None:
                self.eeg_data_received.emit(parsed)
        except Exception as e:
            print(f"[BLE] 数据错误: {e}")

    # ── 命令 ──
    def send_command(self, cmd):
        if self.client and self.client.is_connected and self._write_char_uuid:
            asyncio.run_coroutine_threadsafe(
                self.client.write_gatt_char(self._write_char_uuid, cmd), self.loop
            )

    def start_collection(self):
        if self._cfg:
            self.send_command(self._cfg['start_cmd'])
            print(f"[BLE] 发送开始指令")

    def stop_collection(self):
        if self._cfg:
            self.send_command(self._cfg['stop_cmd'])
            print(f"[BLE] 发送停止指令")

    # ── 日志 ──
    def create_log_file(self, exp_name=""):
        try:
            today = datetime.datetime.now().strftime("%Y%m%d")
            log_dir = os.path.join("./EEG_data", today)
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            suffix = self._cfg['save_suffix'] if self._cfg else '_ble_data.txt'
            tag = f"_{exp_name}" if exp_name else ""
            self.file_path = os.path.join(log_dir, f"{ts}{tag}{suffix}")
            self.file = open(self.file_path, "w", encoding="utf-8")
            print(f"[BLE] 数据保存至: {self.file_path}")
        except Exception as e:
            print(f"[BLE] 创建日志文件失败: {e}")

    # ── 断开 ──
    def disconnect_device(self):
        if self.client:
            future = asyncio.run_coroutine_threadsafe(self._disconnect(), self.loop)
            try:
                future.result(timeout=3.0)
            except Exception:
                pass

    async def _disconnect(self):
        try:
            if self.client.is_connected:
                if self._notify_char_uuid:
                    await self.client.stop_notify(self._notify_char_uuid)
                await self.client.disconnect()
            self.connection_changed.emit(False)
        except Exception as e:
            print(f"[BLE] 断开失败: {e}")

    def stop(self):
        if self.file:
            self.file.close()
            self.file = None
        self.loop.call_soon_threadsafe(self.loop.stop)


# ═══════════════════════════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════════════════════════

class EEGViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Unified - 多设备脑电波形")
        self.resize(1400, 900)

        # 当前选中的设备
        self.current_device = DEFAULT_DEVICE
        self.cfg = DEVICE_PROFILES[self.current_device].copy()
        self.n_channels = self.cfg['n_channels']

        # 数据缓冲区
        self.eeg_buffers = np.zeros((self.n_channels, PLOT_POINTS))

        # 滤波器
        self.filters = {}
        self.filter_enabled = {"hp": False, "lp": False, "notch": False}
        self.lp_cutoff = 30.0
        self.notch_freq = 50.0

        # BLE
        self.ble_worker = BLEWorker()
        self.ble_worker.set_config(self.cfg)
        self.ble_worker.start()
        self.connected = False
        self.collecting = False

        self._setup_ui()
        self._connect_signals()

        self.start_time = QTime.currentTime()
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self._update_timestamp)
        self.time_timer.start(1000)

        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self._refresh_plots)
        self.plot_timer.start(50)

    # ── UI ──
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # 设备选择 + BLE 控制
        top_layout = QHBoxLayout()

        self.device_combo = QComboBox()
        for key, prof in DEVICE_PROFILES.items():
            self.device_combo.addItem(prof['name'], userData=key)
        idx = self.device_combo.findData(self.current_device)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.device_combo.setMinimumHeight(36)
        top_layout.addWidget(self.device_combo)

        self.device_list = QListWidget()
        self.device_list.setMaximumHeight(80)
        self.device_list.setMinimumHeight(60)
        top_layout.addWidget(self.device_list, stretch=3)

        self.scan_btn = QPushButton("扫描")
        self.scan_btn.setMinimumHeight(36)
        top_layout.addWidget(self.scan_btn)

        self.connect_btn = QPushButton("连接")
        self.connect_btn.setMinimumHeight(36)
        self.connect_btn.setEnabled(False)
        top_layout.addWidget(self.connect_btn)

        self.start_btn = QPushButton("开始")
        self.start_btn.setMinimumHeight(36)
        self.start_btn.setEnabled(False)
        top_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setMinimumHeight(36)
        self.stop_btn.setEnabled(False)
        top_layout.addWidget(self.stop_btn)

        self.disconnect_btn = QPushButton("断开")
        self.disconnect_btn.setMinimumHeight(36)
        self.disconnect_btn.setEnabled(False)
        top_layout.addWidget(self.disconnect_btn)

        main_layout.addLayout(top_layout)

        # 滤波 + 量程
        control_layout = QHBoxLayout()

        filter_group = QGroupBox("滤波")
        fl = QHBoxLayout()
        fl.setContentsMargins(6, 10, 6, 6)
        self.hp_btn = QPushButton("0.5Hz HP")
        self.hp_btn.setCheckable(True)
        fl.addWidget(self.hp_btn)
        fl.addWidget(QLabel("LP:"))
        self.lp_edit = QLineEdit("30")
        self.lp_edit.setFixedWidth(50)
        fl.addWidget(self.lp_edit)
        self.lp_btn = QPushButton("低通")
        self.lp_btn.setCheckable(True)
        fl.addWidget(self.lp_btn)
        self.notch_50 = QCheckBox("50Hz")
        self.notch_50.setChecked(True)
        fl.addWidget(self.notch_50)
        self.notch_60 = QCheckBox("60Hz")
        fl.addWidget(self.notch_60)
        self.notch_btn = QPushButton("Notch")
        self.notch_btn.setCheckable(True)
        fl.addWidget(self.notch_btn)
        filter_group.setLayout(fl)
        control_layout.addWidget(filter_group)

        range_group = QGroupBox("量程 (uV)")
        rl = QHBoxLayout()
        rl.setContentsMargins(6, 10, 6, 6)
        self.ylim_combo = QComboBox()
        for v in ["50", "100", "200", "500", "1000", "2000", "5000", "10000"]:
            self.ylim_combo.addItem(v)
        self.ylim_combo.setCurrentText("500")
        self.ylim_combo.currentIndexChanged.connect(self._update_ylim)
        rl.addWidget(self.ylim_combo)
        rl.addStretch()
        range_group.setLayout(rl)
        control_layout.addWidget(range_group)

        main_layout.addLayout(control_layout)

        # 波形图容器
        self.plot_container = QWidget()
        self.plot_container_layout = QVBoxLayout(self.plot_container)
        self.plot_container_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_container_layout.setSpacing(4)
        self.plot_widgets = []
        self._rebuild_plots()
        main_layout.addWidget(self.plot_container, stretch=1)

        # 状态栏
        status_layout = QHBoxLayout()
        self.timestamp_label = QLabel()
        self.timestamp_label.setAlignment(Qt.AlignLeft)
        status_layout.addWidget(self.timestamp_label)
        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_label)
        self.runtime_label = QLabel()
        self.runtime_label.setAlignment(Qt.AlignRight)
        status_layout.addWidget(self.runtime_label)
        main_layout.addLayout(status_layout)

        self._update_ylim()

    def _rebuild_plots(self):
        """根据当前通道数重建波形图"""
        # 清除旧图
        for pw in self.plot_widgets:
            self.plot_container_layout.removeWidget(pw)
            pw.close()
        self.plot_widgets.clear()

        colors = self.cfg['colors']
        for i in range(self.n_channels):
            pw = pg.PlotWidget(title=f"CH{i+1}")
            pw.setBackground("#f8f8f8")
            pw.showGrid(x=True, y=True, alpha=0.25)
            pw.setLabel("bottom", "时间 (s)")
            pw.setLabel("left", f"CH{i+1} (uV)")
            pw.setMouseEnabled(x=False, y=True)
            pw.setMenuEnabled(False)
            pw.setXRange(-SHOW_SECONDS, 0)
            curve = pw.plot(pen=pg.mkPen(color=colors[i % len(colors)], width=1.5))
            pw.curve = curve
            self.plot_widgets.append(pw)
            self.plot_container_layout.addWidget(pw)

        self._update_ylim()

    # ── 信号 ──
    def _connect_signals(self):
        self.scan_btn.clicked.connect(self.ble_worker.scan_devices)
        self.connect_btn.clicked.connect(self._on_connect_click)
        self.start_btn.clicked.connect(self._on_start_click)
        self.stop_btn.clicked.connect(self._on_stop_click)
        self.disconnect_btn.clicked.connect(self._on_disconnect_click)

        self.hp_btn.toggled.connect(self._on_hp_toggle)
        self.lp_btn.toggled.connect(self._on_lp_toggle)
        self.notch_btn.toggled.connect(self._on_notch_toggle)

        self.ble_worker.devices_found.connect(self._on_devices_found)
        self.ble_worker.connection_changed.connect(self._on_connection_changed)
        self.ble_worker.eeg_data_received.connect(self._on_eeg_data)

    # ── 设备切换 ──
    def _on_device_changed(self, idx):
        key = self.device_combo.itemData(idx)
        if key == self.current_device:
            return

        if self.collecting or self.connected:
            self._on_disconnect_click()

        self.current_device = key
        self.cfg = DEVICE_PROFILES[key].copy()
        self.n_channels = self.cfg['n_channels']

        self.eeg_buffers = np.zeros((self.n_channels, PLOT_POINTS))
        self.filters.clear()
        for k in self.filter_enabled:
            self.filter_enabled[k] = False
        self.hp_btn.setChecked(False)
        self.lp_btn.setChecked(False)
        self.notch_btn.setChecked(False)

        self.ble_worker.set_config(self.cfg)

        # 重建波形图
        self._rebuild_plots()

        self.setWindowTitle(f"EEG Unified - {self.cfg['name']}")
        self.status_label.setText(f"已切换到 {self.cfg['name']}")

    # ── BLE ──
    def _on_devices_found(self, devices):
        self.device_list.clear()
        for d in devices:
            item = QListWidgetItem(f"{d.name}  [{d.address}]")
            item.setData(Qt.UserRole, d.address)
            self.device_list.addItem(item)
        self.connect_btn.setEnabled(len(devices) > 0)
        self.status_label.setText(f"发现 {len(devices)} 个设备")

    def _on_connect_click(self):
        item = self.device_list.currentItem()
        if item:
            self.status_label.setText(f"正在连接...")
            self.ble_worker.find_and_connect(item.data(Qt.UserRole))

    def _on_start_click(self):
        self.ble_worker.create_log_file(self.current_device)
        self.ble_worker.start_collection()
        self.collecting = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("采集进行中...")

    def _on_stop_click(self):
        self.ble_worker.stop_collection()
        self.collecting = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("采集已停止")

    def _on_disconnect_click(self):
        if self.collecting:
            self._on_stop_click()
        self.ble_worker.disconnect_device()

    def _on_connection_changed(self, connected):
        self.connected = connected
        self.device_combo.setEnabled(not connected)
        self.scan_btn.setEnabled(not connected)
        self.connect_btn.setEnabled(not connected)
        self.device_list.setEnabled(not connected)
        self.start_btn.setEnabled(connected)
        self.disconnect_btn.setEnabled(connected)
        if not connected:
            self.stop_btn.setEnabled(False)
            self.start_btn.setEnabled(False)
            self.collecting = False
        self.status_label.setText("蓝牙已连接" if connected else "蓝牙已断开")

    # ── 数据处理 ──
    def _on_eeg_data(self, data):
        n = data.shape[0]
        if n == 0:
            return
        filtered = data.copy()
        for ch in range(self.n_channels):
            for i in range(n):
                v = data[i, ch]
                if self.filter_enabled["hp"] and "hp" in self.filters:
                    v = self.filters["hp"][ch](v)
                if self.filter_enabled["lp"] and "lp" in self.filters:
                    v = self.filters["lp"][ch](v)
                if self.filter_enabled["notch"] and "notch" in self.filters:
                    v = self.filters["notch"][ch](v)
                filtered[i, ch] = v
        max_n = min(n, PLOT_POINTS)
        for ch in range(self.n_channels):
            self.eeg_buffers[ch] = np.roll(self.eeg_buffers[ch], -max_n)
            self.eeg_buffers[ch, -max_n:] = filtered[-max_n:, ch]

    def _refresh_plots(self):
        for ch in range(self.n_channels):
            self.plot_widgets[ch].curve.setData(time_axis, self.eeg_buffers[ch])

    # ── 滤波 ──
    def _create_filter(self, fs, cutoff, btype, order=4):
        nyq = 0.5 * fs
        if cutoff >= nyq:
            cutoff = nyq - 1.0
        b, a = signal.butter(order, cutoff / nyq, btype=btype)
        return DigitalFilter(b, a)

    def _create_notch(self, fs, freq, q=30):
        b, a = signal.iirnotch(freq, q, fs)
        return DigitalFilter(b, a)

    def _on_hp_toggle(self, checked):
        self.filter_enabled["hp"] = checked
        if checked:
            self.filters["hp"] = [self._create_filter(SAMPLE_RATE, 0.5, "high", 2)
                                  for _ in range(self.n_channels)]
        else:
            self.filters.pop("hp", None)

    def _on_lp_toggle(self, checked):
        self.filter_enabled["lp"] = checked
        if checked:
            try:
                self.lp_cutoff = float(self.lp_edit.text())
                if self.lp_cutoff <= 0 or self.lp_cutoff >= SAMPLE_RATE / 2:
                    raise ValueError
            except ValueError:
                self.lp_edit.setText("30")
                self.lp_cutoff = 30.0
                self.lp_btn.setChecked(False)
                return
            self.filters["lp"] = [self._create_filter(SAMPLE_RATE, self.lp_cutoff, "low")
                                  for _ in range(self.n_channels)]
        else:
            self.filters.pop("lp", None)

    def _on_notch_toggle(self, checked):
        self.filter_enabled["notch"] = checked
        if checked:
            self.notch_freq = 50.0 if self.notch_50.isChecked() else 60.0
            self.filters["notch"] = [self._create_notch(SAMPLE_RATE, self.notch_freq)
                                     for _ in range(self.n_channels)]
        else:
            self.filters.pop("notch", None)

    # ── 量程 ──
    def _update_ylim(self):
        ymax = float(self.ylim_combo.currentText())
        for pw in self.plot_widgets:
            pw.setYRange(-ymax, ymax)

    # ── 时间 ──
    def _update_timestamp(self):
        now = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        self.timestamp_label.setText(now)
        elapsed = self.start_time.elapsed() / 1000
        self.runtime_label.setText(f"运行: {elapsed:.0f}s")

    def closeEvent(self, event):
        self.plot_timer.stop()
        self.time_timer.stop()
        if self.collecting:
            self.ble_worker.stop_collection()
        self.ble_worker.disconnect_device()
        self.ble_worker.stop()
        self.ble_worker.wait(2000)
        event.accept()


# ═══════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════

def run():
    app = QApplication(sys.argv)
    window = EEGViewer()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
