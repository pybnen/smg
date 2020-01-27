# use model and generate a fuckn mid

# how you ask?
# well good question! let me tell you how

# start with a piano roll sequence
# should have dimension (batch_size, n_instruments, seq_length, n_pitches)
# batch size is 1, n_instrument is 5, n_pitches is 72

# use model and generate a fuckn mid

# how you ask?
# well good question! let me tell you how

# start with a piano roll sequence
# should have dimension (batch_size, n_instruments, seq_length, n_pitches)
# batch size is 1, n_instrument is 5, n_pitches is 72
from torch.distributions.multinomial import Multinomial


def generate_pianoroll(pianoroll, beat_resolution, lowest_pitch, n_pitches):
    '''
    :param pianoroll: uint8 array, shape = (instruments, time, 128) - min value = 0, max value = 127
    :param beat_resolution: don't touch this unless you know what you are doing
    :return: a multi track piano roll
    '''
    
    INSTRUMENTS = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']
    INSTRUMENT_MIDI_METADATA = {
        0: (0, True),
        1: (0, False),
        2: (25, False),
        3: (33, False),
        4: (48, False)
    }

    tempo = 120.0
    downbeat=None

    assert(np.all(np.logical_and(pianoroll >= 0, pianoroll < 128)))
    pianoroll = pianoroll.astype(np.uint8)

    full_pitch_pianoroll = np.zeros((pianoroll.shape[0], pianoroll.shape[1], 128))
    full_pitch_pianoroll[:, :, lowest_pitch:lowest_pitch + n_pitches] = pianoroll

    # create tracks
    tracks = []
    for i in range(len(full_pitch_pianoroll)):
        program, is_drum = INSTRUMENT_MIDI_METADATA[i]
        tracks.append(
            pp.Track(
                full_pitch_pianoroll[i],
                is_drum=is_drum,
                program=program,
                name=INSTRUMENTS[i]
            )
        )

    # create multitrack piano rolls
    return pp.Multitrack(
        tracks=tracks,
        tempo=tempo * (np.ones(full_pitch_pianoroll.shape[1])),
        downbeat=downbeat,
        beat_resolution=beat_resolution,
        name='generated'
    )


def rand_stub_generator(seq_len):
    n_pitches = 72
    n_instruments = 5
    
    m = Multinomial(total_count=1, probs=torch.ones(1, n_instruments, seq_len, n_pitches))
    
    return lambda : m.sample()


def generate_mid(stub, model, n_iter=10, seq_length=None):
    if seq_length is None:
        seq_length = stub.size(2)
    
    pianoroll = stub
    
    for _ in range(n_iter):
        # generate next timestep
        y_hat = model.forward(stub)
        
        # add timestep to pianoroll
        pianoroll = torch.cat([pianoroll, y_hat], dim=2)
        # slide stub
        stub = pianoroll[:, :, -seq_len:, :]
        

    # convert pianoroll -> pp.Multitrack -> .mid file
    pianoroll_view = pianoroll.view(-1, pianoroll.size(3))
    max_p, argmax_p = pianoroll_view.max(dim=-1)
    
    new_roll = torch.zeros_like(pianoroll_view)
    new_roll[np.arange(new_roll.size(0)), argmax_p.type(torch.LongTensor)] = max_p
    
    pianoroll = new_roll.view(pianoroll.size())
    pianoroll = pianoroll.detach()[0].numpy() * 127
    
    pianoroll[np.logical_and(pianoroll > 1, pianoroll < 60)] += 60

    return generate_pianoroll(pianoroll, beat_resolution=4, lowest_pitch=24, n_pitches=72)

    
def stub_from_file(file, seq_length, random_start=True):
    lowest_pitch = 24
    n_pitches = 72
    n_instruments = 5
    beat_resolution = 4
    
    multitrack_roll = pp.load(file)
    multitrack_roll.downsample(multitrack_roll.beat_resolution // beat_resolution)
    
    stacked = multitrack_roll.get_stacked_pianoroll().transpose(2, 0, 1)
    
    start = 0 # np.random.choice(stacked.shape[1] - seq_length)
    if random_start:
        start = np.random.choice(stacked.shape[1] - seq_length)
     
    print(start)
    stub = stacked[:, start:start+seq_length, lowest_pitch:lowest_pitch+n_pitches]
    
    return torch.from_numpy(stub).type(dtype=torch.FloatTensor).unsqueeze(0) / 127