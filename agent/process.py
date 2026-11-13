"""Windows child lifetime and peak RSS accounting, using only the standard library."""
import os


class ChildGuard:
    def __init__(self, process):
        self.handle = None
        self.peak_rss = None
        if os.name != "nt":
            return
        import ctypes as c
        from ctypes import wintypes as w
        self.c = c
        self.kernel = c.WinDLL("kernel32", use_last_error=True)
        class Basic(c.Structure):
            _fields_ = [("ProcessTime", c.c_longlong), ("JobTime", c.c_longlong),
                        ("LimitFlags", w.DWORD), ("MinWorkingSet", c.c_size_t),
                        ("MaxWorkingSet", c.c_size_t), ("ActiveProcessLimit", w.DWORD),
                        ("Affinity", c.c_size_t), ("PriorityClass", w.DWORD),
                        ("SchedulingClass", w.DWORD)]
        class IO(c.Structure):
            _fields_ = [(name, c.c_ulonglong) for name in
                        ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]
        class Extended(c.Structure):
            _fields_ = [("Basic", Basic), ("IO", IO), ("ProcessLimit", c.c_size_t),
                        ("JobLimit", c.c_size_t), ("PeakProcess", c.c_size_t), ("PeakJob", c.c_size_t)]
        self.kernel.CreateJobObjectW.argtypes = [c.c_void_p, w.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = w.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        handle = self.kernel.CreateJobObjectW(None, None)
        if not handle:
            raise c.WinError(c.get_last_error())
        self.handle = handle
        info = Extended()
        info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(handle, 9, c.byref(info), c.sizeof(info)):
            self.close()
            raise c.WinError(c.get_last_error())
        if not self.kernel.AssignProcessToJobObject(handle, w.HANDLE(int(process._handle))):
            self.close()
            raise c.WinError(c.get_last_error())

    @staticmethod
    def peak_working_set(process=None):
        if os.name != "nt":
            return None
        import ctypes as c
        from ctypes import wintypes as w
        class Counters(c.Structure):
            _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD)] + [
                (name, c.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        kernel = c.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = w.HANDLE
        psapi = c.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, c.POINTER(Counters), w.DWORD]
        info = Counters()
        info.cb = c.sizeof(info)
        handle = w.HANDLE(int(process._handle)) if process else kernel.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, c.byref(info), c.sizeof(info)):
            return None
        return info.PeakWorkingSetSize

    def close(self):
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
