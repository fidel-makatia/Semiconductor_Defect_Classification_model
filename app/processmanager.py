"""ProcessManager: QProcess wrapper for the Python inference subprocess.

Mirrors app/src/inferencemanager.h/cpp.
Launches inference_server.py, sends JSON commands via stdin,
reads JSON responses from stdout.
"""

import json

from PySide6.QtCore import QObject, QProcess, Signal


class ProcessManager(QObject):
    ready = Signal()
    responseReceived = Signal(dict)
    errorOccurred = Signal(str)
    processFinished = Signal()

    def __init__(self, python_path: str, script_path: str,
                 project_root: str, support_dir: str = "", parent=None):
        super().__init__(parent)
        self._python_path = python_path
        self._script_path = script_path
        self._project_root = project_root
        self._support_dir = support_dir
        self._buffer = b""

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self._process.readyReadStandardOutput.connect(self._on_ready_read)
        self._process.errorOccurred.connect(self._on_process_error)
        self._process.finished.connect(self._on_process_finished)

    def start(self):
        print(f"Starting inference server: {self._python_path} {self._script_path} "
              f"project_root: {self._project_root}")
        args = [self._script_path, "--project-root", self._project_root]
        if self._support_dir:
            args += ["--support-dir", self._support_dir]
        self._process.start(self._python_path, args)

    def sendCommand(self, cmd: dict):
        if self._process.state() != QProcess.ProcessState.Running:
            self.errorOccurred.emit("Inference server not running")
            return

        data = json.dumps(cmd, ensure_ascii=False) + "\n"
        self._process.write(data.encode("utf-8"))
        self._process.waitForBytesWritten(1000)

    def shutdown(self):
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.write(b'{"cmd":"quit"}\n')
            self._process.waitForBytesWritten(1000)
            self._process.terminate()
            self._process.waitForFinished(3000)

    def _on_ready_read(self):
        data = bytes(self._process.readAllStandardOutput())
        # Also check stderr
        stderr_data = bytes(self._process.readAllStandardError())
        if stderr_data:
            print(f"[server stderr] {stderr_data.decode('utf-8', errors='replace')}", flush=True)
        self._buffer += data

        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                break

            line = self._buffer[:newline].strip()
            self._buffer = self._buffer[newline + 1:]

            if not line:
                continue

            try:
                response = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"JSON parse error: {e} Data: {line[:200]}")
                continue

            if response.get("status") == "ready":
                self.ready.emit()
                continue

            self.responseReceived.emit(response)

    def _on_process_error(self, error):
        self.errorOccurred.emit(f"Process error: {self._process.errorString()}")

    def _on_process_finished(self, exit_code, exit_status):
        stderr_data = bytes(self._process.readAllStandardError())
        if stderr_data:
            print(f"Python stderr: {stderr_data.decode('utf-8', errors='replace')}")

        if exit_status == QProcess.ExitStatus.CrashExit:
            self.errorOccurred.emit("Inference server crashed")

        self.processFinished.emit()
