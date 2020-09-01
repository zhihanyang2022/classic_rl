from .cb_base import CallbackBase

class CallbackHandler(CallbackBase):

    def __init__(self, cbs):
        self.cbs = cbs

    def do_modeling(self, **kwargs):
        for cb in self.cbs: cb.do_modeling(**kwargs)

    def do_learning(self, **kwargs):
        for cb in self.cbs: cb.do_learning(**kwargs)

    def after_learning(self, **kwargs):
        for cb in self.cbs: cb.after_learning(**kwargs)

    def do_planning(self, **kwargs):
        for cb in self.cbs: cb.do_planning(**kwargs)

    def after_planning(self, **kwargs):
        for cb in self.cbs: cb.after_planning(**kwargs)

    def after_each_episode(self, **kwargs):
    	for cb in self.cbs: cb.after_each_episode(**kwargs)