from .cb_base import CallbackBase

class EpisodeLengthTracker(CallbackBase):

	def __init__(self):
		self.values = []
		self.length_of_current_episode = 0

	def do_modeling(self, **kwargs):
		self.length_of_current_episode += 1

	def after_each_episode(self, **kwargs):
		self.values.append(self.length_of_current_episode)
		self.length_of_current_episode = 0