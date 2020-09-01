class CallbackBase:

	def do_modeling(self, **kwargs): pass

	def do_learning(self, **kwargs): pass
	def after_learning(self, **kwargs): pass
	
	def do_planning(self, **kwargs): pass
	def after_planning(self, **kwargs): pass
	
	def after_each_episode(self, **kwargs): pass