class WaveNetModel(object):
    '''Implements the WaveNet network for generative audio.
    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        skip_channels = 16      # Not specified in the paper.
        net = WaveNetModel(batch_size, dilations, filter_width,
                           residual_channels, dilation_channels,
                           skip_channels)
        loss = net.loss(input_batch)
    '''

    def __init__(self,
                # number of the input audio file, usually 1
                 batch_size,
                # dilation factors
                 dilations,
                # samples that are included after dilating
                 filter_width,
                # number of filters to learn for the residual.
                 residual_channels,
                # number of filters to learn for the dilation.
                 dilation_channels,
                # number of filters to learn that contribute to the quantized softmax output
                 skip_channels,
                # 8-bit quantization.
                 quantization_channels=2**8,
                # if biased layer exists(set false as default)
                 use_biases=False,
                # something from paper, default false
                 scalar_input=False,
                # inital layer
                 initial_filter_width=32,
                # store histogram or not
                 histograms=False,
                 global_condition_channels=None,
                 global_condition_cardinality=None):
        '''Initializes the WaveNet model.
            global_condition_channels: Number of channels in (embedding
                size) of global conditioning vector. None indicates there is
                no global conditioning.
            global_condition_cardinality: Number of mutually exclusive
                categories to be embedded in global condition embedding. If
                not None, then this implies that global_condition tensor
                specifies an integer selecting which of the N global condition
                categories, where N = global_condition_cardinality. If None,
                then the global_condition tensor is regarded as a vector which
                must have dimension global_condition_channels.
        '''
        self.batch_size = batch_size
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.quantization_channels = quantization_channels
        self.use_biases = use_biases
        self.skip_channels = skip_channels
        self.scalar_input = scalar_input
        self.initial_filter_width = initial_filter_width
        self.histograms = histograms
        self.global_condition_channels = global_condition_channels
        self.global_condition_cardinality = global_condition_cardinality
