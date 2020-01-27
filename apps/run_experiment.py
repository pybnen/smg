# not needed right know
from sacred import Experiment

ex = Experiment('hello_config')



@ex.config
def config():
    """Default configurations"""
    n_instruments = 5
    batch_size = 16
    beat_resolution = 4

    recipient = "world"
    message = "Hello %s!" % recipient

@ex.named_config
def var1():
    """Configurations"""
    message = "dont worry"
    


@ex.automain
def my_main(message, beat_resolution):
    print(message)
    print(beat_resolution)