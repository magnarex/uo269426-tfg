import threading


class ReturningThread(threading.Thread):
    def __init__(self, group = None, target = None, name = None, args = (), kwargs = {}, *,daemon = None) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
    
    def run(self):
        try:
            if self._target:
                self._return = self._target(*self._args, **self._kwargs)
        finally:
            del self._target, self._args, self._kwargs
    
    def join(self):
        threading.Thread.join(self)
        return self._return