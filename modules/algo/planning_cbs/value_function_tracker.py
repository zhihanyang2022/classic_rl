from .cb_base import CallbackBase

class ValueFunctionTracker(CallbackBase):

	def __init__(self):
		
		

	def after_learning(self, **kwargs):

		s = kwargs['s']
		a = kwargs['a']
		s_prime = kwargs['s_prime']

		algo = kwargs['self']

		

	def do_planning(self, **kwargs):
		
		algo = kwargs['algo']
		
		self.qs.append(algo.q.copy())
		self.q_statuses.append(1)